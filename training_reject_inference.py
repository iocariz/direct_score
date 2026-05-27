"""Reject-inference utilities.

Pseudo-labels rejected applications using booked-account bad rates by score
band. Used to augment supervised training when the population mode is
``underwriting`` and rejects/canceleds are available.

All inputs must be pre-split (``mis_Date < SPLIT_DATE``) — applying these
helpers post-split would leak future outcomes into training.

Pseudo-labels are assigned **deterministically** via row replication: each
rejected applicant becomes two rows in the augmented training set, one
labelled bad (weight = pseudo_bad_rate · REJECT_SAMPLE_WEIGHT) and one
labelled good (weight = (1 - pseudo_bad_rate) · REJECT_SAMPLE_WEIGHT).
The expected gradient contribution is identical to a Bernoulli draw at
pseudo_bad_rate, but with no Monte Carlo noise and full reproducibility
across seeds. Per the methodology audit, this replaces the previous
``rng.random(n) < pseudo_bad_rate`` assignment that was sensitive to
``RANDOM_STATE`` for finite booster fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from training_constants import (
    MATURITY_CUTOFF,
    RANDOM_STATE,
    REJECT_MAX_RATIO,
    REJECT_MULTIPLIER,
    REJECT_N_BINS,
    REJECT_SAMPLE_WEIGHT,
    REJECT_SCORE_COL,
    SPLIT_DATE,
    TARGET,
)

# Per-row sample-weight column populated by create_reject_pseudo_labels and
# consumed by augment_training_data. Lives on the rejected frame; never on
# booked data (which always has weight 1.0).
REJECT_ROW_WEIGHT_COL = "reject_row_weight"


def compute_score_band_bad_rates(
    df_booked: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    df_mat = df_booked[
        (df_booked["mis_Date"] < SPLIT_DATE)
        & (df_booked["mis_Date"] <= MATURITY_CUTOFF)
        & df_booked[TARGET].notna()
        & df_booked[REJECT_SCORE_COL].notna()
    ].copy()
    if df_mat.empty:
        raise ValueError("Reject inference requires pre-split booked rows with observed targets and scores")
    df_mat[TARGET] = df_mat[TARGET].astype(int)

    bin_edges = np.quantile(df_mat[REJECT_SCORE_COL].values, np.linspace(0, 1, REJECT_N_BINS + 1))
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf

    df_mat["score_band"] = pd.cut(df_mat[REJECT_SCORE_COL], bins=bin_edges, labels=False, include_lowest=True)
    band_stats = (
        df_mat.groupby("score_band")
        .agg(n_booked=(TARGET, "count"), n_bad=(TARGET, "sum"))
        .reset_index()
    )
    band_stats["bad_rate"] = band_stats["n_bad"] / band_stats["n_booked"]
    band_stats.attrs["pre_split_only"] = True
    band_stats.attrs["source_max_date"] = pd.Timestamp(df_mat["mis_Date"].max())
    band_stats.attrs["source_min_date"] = pd.Timestamp(df_mat["mis_Date"].min())

    logger.info("Score-band bad rates (booked, pre-split, matured):")
    for _, row in band_stats.iterrows():
        logger.info("  Band {:2.0f}: n={:>6,}  bad_rate={:.4f}", row["score_band"], int(row["n_booked"]), row["bad_rate"])

    return band_stats, bin_edges


def create_reject_pseudo_labels(
    df_rejected: pd.DataFrame,
    band_stats: pd.DataFrame,
    bin_edges: np.ndarray,
    n_booked_train: int,
) -> pd.DataFrame:
    """Build deterministic, expectation-preserving pseudo-labels for rejects.

    Each rejected applicant is materialized as **two rows** in the returned
    frame: one labelled bad (TARGET=1, weight ``p · REJECT_SAMPLE_WEIGHT``)
    and one labelled good (TARGET=0, weight ``(1-p) · REJECT_SAMPLE_WEIGHT``),
    where ``p`` = pseudo_bad_rate. The pair's total weight equals
    ``REJECT_SAMPLE_WEIGHT`` and its expected gradient contribution matches
    a Bernoulli(p) draw exactly — without the Monte Carlo noise.

    Zero-weight rows (from the boundary cases ``p=0`` or ``p=1``) are
    dropped to avoid wasting fit cycles on sample-weight-zero examples.

    Down-sampling caps the number of *applicants* (not rows) at
    ``n_booked_train · REJECT_MAX_RATIO`` so the row-replication doesn't
    blow past the configured booked:reject influence ratio.
    """
    if not band_stats.attrs.get("pre_split_only", False):
        raise ValueError("band_stats must be computed from pre-split booked rows only")
    if pd.Timestamp(band_stats.attrs["source_max_date"]) >= pd.Timestamp(SPLIT_DATE):
        raise ValueError("band_stats must exclude post-split booked rows")

    df_rej = df_rejected[
        df_rejected[REJECT_SCORE_COL].notna()
        & (df_rejected["mis_Date"] < SPLIT_DATE)
    ].copy()
    if not df_rej.empty and not (df_rej["mis_Date"] < SPLIT_DATE).all():
        raise ValueError("Pseudo-labeled rejects must come from the pre-split period only")

    df_rej["score_band"] = pd.cut(df_rej[REJECT_SCORE_COL], bins=bin_edges, labels=False, include_lowest=True)
    df_rej = df_rej.merge(band_stats[["score_band", "bad_rate"]], on="score_band", how="left")
    if df_rej["bad_rate"].isna().any():
        df_rej = df_rej.loc[df_rej["bad_rate"].notna()].copy()
    df_rej["pseudo_bad_rate"] = (df_rej["bad_rate"] * REJECT_MULTIPLIER).clip(upper=0.50)

    # Down-sample applicants (not rows). Each surviving applicant will produce
    # up to two rows after the deterministic expansion below.
    max_rejects = int(n_booked_train * REJECT_MAX_RATIO)
    if len(df_rej) > max_rejects:
        logger.info(
            "Down-sampling rejects: {:,} -> {:,} applicants (ratio {:.1f}:1)",
            len(df_rej), max_rejects, REJECT_MAX_RATIO,
        )
        df_rej = df_rej.sample(n=max_rejects, random_state=RANDOM_STATE)

    # Deterministic expansion: each applicant -> (bad, good) row pair.
    p = df_rej["pseudo_bad_rate"].astype(float).values

    bad_rows = df_rej.copy()
    bad_rows[TARGET] = 1
    bad_rows[REJECT_ROW_WEIGHT_COL] = p * REJECT_SAMPLE_WEIGHT

    good_rows = df_rej.copy()
    good_rows[TARGET] = 0
    good_rows[REJECT_ROW_WEIGHT_COL] = (1.0 - p) * REJECT_SAMPLE_WEIGHT

    expanded = pd.concat([bad_rows, good_rows], axis=0, ignore_index=True)
    # Drop zero-weight rows; nothing is lost because their gradient
    # contribution is exactly zero. This also avoids surprises from
    # samplers that complain about zero-weight observations.
    expanded = expanded.loc[expanded[REJECT_ROW_WEIGHT_COL] > 0].reset_index(drop=True)

    booked_bad_rate = band_stats["n_bad"].sum() / band_stats["n_booked"].sum()
    n_applicants = len(df_rej)
    expected_bad_rate = float(p.mean()) if n_applicants > 0 else float("nan")
    logger.info(
        "Pseudo-labeled (deterministic): {:,} applicants -> {:,} rows, "
        "{:.2%} expected pseudo-bad (vs {:.2%} booked observed)",
        n_applicants, len(expanded), expected_bad_rate, booked_bad_rate,
    )
    for band in sorted(df_rej["score_band"].dropna().unique()):
        b = df_rej[df_rej["score_band"] == band]
        logger.info(
            "  Band {:2.0f}: n={:>6,}  pseudo_bad_rate={:.4f}",
            band, len(b), b["pseudo_bad_rate"].iloc[0],
        )

    return expanded


def augment_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_reject_labeled: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Combine booked training data with deterministically-labelled rejects.

    Reject row weights come from the per-row ``REJECT_ROW_WEIGHT_COL``
    populated by ``create_reject_pseudo_labels`` (each row's contribution to
    the expected loss is ``p_or_1-p · REJECT_SAMPLE_WEIGHT``). Booked rows
    keep their full weight of 1.0.
    """
    X_rej = df_reject_labeled[feature_cols].copy()
    y_rej = df_reject_labeled[TARGET].astype(int).copy()

    w_booked = np.ones(len(X_train))
    if REJECT_ROW_WEIGHT_COL in df_reject_labeled.columns:
        w_reject = df_reject_labeled[REJECT_ROW_WEIGHT_COL].astype(float).to_numpy()
    else:
        # Backwards-compat path for pre-existing fixtures that hand-build a
        # pseudo-labelled frame without the per-row weight column.
        w_reject = np.full(len(X_rej), REJECT_SAMPLE_WEIGHT)

    X_aug = pd.concat([X_train, X_rej], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train, y_rej], axis=0, ignore_index=True)
    w_aug = np.concatenate([w_booked, w_reject])

    logger.info(
        "Augmented: {:,} booked + {:,} reject rows = {:,}  "
        "(booked {:.2%} bad, rejects {:.2%} weighted-bad, combined {:.2%})",
        len(X_train), len(X_rej), len(X_aug),
        y_train.mean(),
        float(np.average(y_rej.values, weights=w_reject)) if len(w_reject) else 0.0,
        y_aug.mean(),
    )
    return X_aug, y_aug, w_aug
