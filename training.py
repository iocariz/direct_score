"""Training pipeline for basel_bad binary classification.

Converts the notebook 3-model.ipynb into a reproducible training script.

Usage:
    uv run python training.py
    uv run python training.py --data-path data/demand_direct.parquet
    uv run python training.py --optuna-trials 100
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from tqdm.auto import tqdm
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from training_constants import (
    BENCHMARK_MODEL_NAMES,
    CALIBRATION_FRACTION,
    DROP_COLS,
    EARLY_STOPPING_ROUNDS,
    EXPERIMENTAL_STACKING_NAME,
    FEATURE_DISCOVERY_FRACTION,
    MATURITY_CUTOFF,
    MAX_CATEGORIES,
    MIN_LIFT,
    MIN_VALID,
    MISS_CANDIDATES,
    MONOTONE_MAP,
    N_BOOTSTRAP,
    N_ESTIMATORS_CEILING,
    OFFICIAL_MODEL_NAMES,
    POPULATION_MODE_BOOKED_MONITORING,
    POPULATION_MODE_UNDERWRITING,
    RANDOM_STATE,
    RAW_CAT,
    RAW_NUM,
    REJECT_MAX_RATIO,
    REJECT_MULTIPLIER,
    REJECT_N_BINS,
    REJECT_SAMPLE_WEIGHT,
    REJECT_SCORE_COL,
    ROLLING_OOT_MAX_WINDOWS,
    SPLIT_DATE,
    SUMMARY_MODEL_NAMES,
    TARGET,
    UNDERWRITING_DECISION_STATUSES,
)
from model_governance import (
    generate_data_quality_report,
    generate_model_card,
    generate_variable_dictionary,
)
from training_reporting import (
    _compute_midrank,
    _fast_delong,
    _ks_statistic,
    _metric_improvement,
    _score_is_probability,
    _score_metric,
    bootstrap_confidence_intervals,
    build_holdout_score_frame,
    compute_overfit_report,
    create_lift_table,
    create_threshold_analysis,
    delong_auc_test,
    evaluate,
    evaluate_all,
    evaluate_safely,
    extract_feature_importance,
    paired_bootstrap_benchmark_comparisons,
    paired_bootstrap_metric_delta,
    plot_score_distributions,
    sanitize_output_name,
    save_artifacts,
    select_best_model,
    split_leaderboard_results,
)
import training_features as _training_features
from training_features import (
    GROUP_STAT_PAIRS,
    _loo_target_encode,
    _safe_auc,
    add_frequency_encoding,
    add_group_stats,
    add_interactions,
    add_modeling_features,
    build_feature_provenance,
    build_monotone_constraints,
    build_preprocessors,
    engineer_features,
    normalize_interaction_name,
    prune_correlated,
    reduce_cardinality,
    run_rfecv,
    search_interactions,
    select_features,
)


# ── Logging & warnings setup ──────────────────────────────────────────────────

def _configure_logging() -> None:
    """Replace default loguru handler with a clean, timestamped format."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )


def _suppress_warnings() -> None:
    """Silence noisy but harmless warnings from dependencies."""
    # sklearn
    warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    # SAGA solver may not converge on some CV folds with extreme imbalance
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    # LightGBM categorical feature info & early stopping verbosity
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    # XGBoost eval_metric & early stopping
    warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
    # CatBoost
    warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
    # pandas SettingWithCopyWarning & FutureWarning
    warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
    # Optuna experimental warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@contextmanager
def _log_step(step_num: int | str, description: str):
    """Context manager that logs a numbered pipeline step with elapsed time."""
    header = f"[Step {step_num}] {description}"
    logger.info("── {} {}", header, "─" * max(0, 52 - len(header)))
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    if elapsed >= 60:
        logger.info("  done ({:.0f}m {:.0f}s)", elapsed // 60, elapsed % 60)
    else:
        logger.info("  done ({:.1f}s)", elapsed)


# ── Temporal CV ────────────────────────────────────────────────────────────────

class TemporalExpandingCV:
    def __init__(self, dates, n_splits=5):
        dates = pd.to_datetime(pd.Series(dates), errors="raise")
        if dates.isna().any():
            raise ValueError("TemporalExpandingCV dates must not contain missing values")

        unique_dates = pd.Index(np.sort(dates.unique()))
        if len(unique_dates) < n_splits + 1:
            raise ValueError(
                f"TemporalExpandingCV requires at least {n_splits + 1} distinct date blocks, "
                f"got {len(unique_dates)}"
            )

        counts_by_date = dates.value_counts().sort_index()
        cumulative_counts = counts_by_date.cumsum().to_numpy()
        targets = np.linspace(0, len(dates), n_splits + 2)[1:-1]
        boundaries = [0]
        last_boundary = 0
        for fold_idx, target in enumerate(targets, start=1):
            boundary = int(np.searchsorted(cumulative_counts, target, side="left")) + 1
            min_boundary = last_boundary + 1
            max_boundary = len(unique_dates) - (n_splits - fold_idx + 1)
            boundary = min(max(boundary, min_boundary), max_boundary)
            boundaries.append(boundary)
            last_boundary = boundary
        boundaries.append(len(unique_dates))

        self._folds = []
        self.fold_boundaries_ = []
        dates_array = dates.to_numpy()
        for k in range(n_splits):
            train_dates = unique_dates[:boundaries[k + 1]]
            val_dates = unique_dates[boundaries[k + 1]:boundaries[k + 2]]
            if len(train_dates) == 0 or len(val_dates) == 0:
                raise ValueError("TemporalExpandingCV produced an empty training or validation fold")
            train_idx = np.flatnonzero(np.isin(dates_array, train_dates))
            val_idx = np.flatnonzero(np.isin(dates_array, val_dates))
            train_max = pd.Timestamp(train_dates[-1])
            val_min = pd.Timestamp(val_dates[0])
            if not train_max < val_min:
                raise ValueError("TemporalExpandingCV validation dates must be strictly later than training dates")
            self._folds.append((train_idx, val_idx))
            self.fold_boundaries_.append(
                {
                    "fold": k,
                    "train_start": pd.Timestamp(train_dates[0]),
                    "train_end": train_max,
                    "val_start": val_min,
                    "val_end": pd.Timestamp(val_dates[-1]),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                }
            )
        self.n_splits = len(self._folds)

    def split(self, X=None, y=None, groups=None):
        for train_idx, val_idx in self._folds:
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def temporal_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates,
    calibration_fraction: float = CALIBRATION_FRACTION,
    sample_weight: np.ndarray | None = None,
):
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"), index=X.index)
    if dates.isna().any():
        raise ValueError("Calibration dates must not contain missing values")
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError("X, y, and dates must have the same length")
    if sample_weight is not None and len(sample_weight) != len(dates):
        raise ValueError("sample_weight must have the same length as X")

    if sample_weight is not None:
        reference_mask = np.asarray(sample_weight) == 1.0
        reference_dates = dates.loc[reference_mask]
    else:
        reference_dates = dates

    if reference_dates.empty:
        raise ValueError("Calibration split requires at least one booked row")

    counts_by_date = reference_dates.value_counts().sort_index()
    if len(counts_by_date) < 2:
        raise ValueError("Calibration split requires at least 2 distinct date blocks")

    target_rows = max(1, int(np.ceil(len(reference_dates) * calibration_fraction)))
    calibration_start = counts_by_date.index[-1]
    cumulative_rows = 0
    for date_value, count in counts_by_date.sort_index(ascending=False).items():
        cumulative_rows += int(count)
        calibration_start = date_value
        if cumulative_rows >= target_rows:
            break

    calibration_mask = dates >= calibration_start
    fit_mask = ~calibration_mask
    if not calibration_mask.any() or not fit_mask.any():
        raise ValueError("Calibration split must produce non-empty fit and calibration sets")

    X_fit = X.loc[fit_mask].copy()
    X_calib = X.loc[calibration_mask].copy()
    y_fit = y.loc[fit_mask].copy()
    y_calib = y.loc[calibration_mask].copy()
    dates_fit = dates.loc[fit_mask].to_numpy()
    dates_calib = dates.loc[calibration_mask].to_numpy()

    if sample_weight is None:
        return X_fit, X_calib, y_fit, y_calib, dates_fit, dates_calib

    sample_weight = np.asarray(sample_weight)
    w_fit = sample_weight[fit_mask.to_numpy()]
    w_calib = sample_weight[calibration_mask.to_numpy()]
    return X_fit, X_calib, y_fit, y_calib, w_fit, w_calib, dates_fit, dates_calib


def resolve_temporal_feature_discovery_cutoff(
    dates,
    discovery_fraction: float = FEATURE_DISCOVERY_FRACTION,
) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"))
    if dates.isna().any():
        raise ValueError("Feature discovery dates must not contain missing values")
    if not 0 < discovery_fraction < 1:
        raise ValueError("discovery_fraction must be strictly between 0 and 1")

    counts_by_date = dates.value_counts().sort_index()
    if len(counts_by_date) < 2:
        raise ValueError("Feature discovery split requires at least 2 distinct date blocks")

    target_rows = max(1, int(np.ceil(len(dates) * discovery_fraction)))
    discovery_end = counts_by_date.index[0]
    cumulative_rows = 0
    for date_value, count in counts_by_date.items():
        cumulative_rows += int(count)
        discovery_end = date_value
        if cumulative_rows >= target_rows:
            break

    discovery_mask = dates <= discovery_end
    estimation_mask = ~discovery_mask
    if not discovery_mask.any() or not estimation_mask.any():
        raise ValueError("Feature discovery split must produce non-empty discovery and estimation sets")
    return pd.Timestamp(discovery_end)


def temporal_feature_discovery_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates,
    discovery_fraction: float = FEATURE_DISCOVERY_FRACTION,
    discovery_end: str | pd.Timestamp | None = None,
):
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"), index=X.index)
    if dates.isna().any():
        raise ValueError("Feature discovery dates must not contain missing values")
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError("X, y, and dates must have the same length")
    if discovery_end is None:
        discovery_end = resolve_temporal_feature_discovery_cutoff(
            dates,
            discovery_fraction=discovery_fraction,
        )
    else:
        discovery_end = pd.Timestamp(discovery_end)

    discovery_mask = dates <= discovery_end
    estimation_mask = ~discovery_mask
    if not discovery_mask.any() or not estimation_mask.any():
        raise ValueError("Feature discovery split must produce non-empty discovery and estimation sets")

    X_discovery = X.loc[discovery_mask].copy()
    X_estimation = X.loc[estimation_mask].copy()
    y_discovery = y.loc[discovery_mask].copy()
    y_estimation = y.loc[estimation_mask].copy()
    dates_discovery = dates.loc[discovery_mask].to_numpy()
    dates_estimation = dates.loc[estimation_mask].to_numpy()
    return X_discovery, X_estimation, y_discovery, y_estimation, dates_discovery, dates_estimation


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    logger.info("Source: {}", data_path)
    df = pq.read_table(data_path).to_pandas()
    logger.info("Raw: {:,} rows x {} cols", len(df), len(df.columns))

    n_before = len(df)
    df = df[df["status_name"] == "Booked"].copy()
    logger.info(
        "Booked filter: {:,} -> {:,} rows ({:,} rejected/canceled removed)",
        n_before, len(df), n_before - len(df),
    )

    target_counts = df[TARGET].value_counts(dropna=False)
    n_pos = target_counts.get(1.0, 0)
    n_neg = target_counts.get(0.0, 0)
    n_nan = df[TARGET].isna().sum()
    logger.info(
        "Target: {:,} neg / {:,} pos / {:,} immature  ({:.2%} default rate among matured)",
        n_neg, n_pos, n_nan, n_pos / max(n_pos + n_neg, 1),
    )
    logger.info("Date range: {} to {}", df["mis_Date"].min(), df["mis_Date"].max())
    return df


def load_data_with_rejects(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load data and return (booked_df, rejected_df) separately."""
    logger.info("Source: {}", data_path)
    df = pq.read_table(data_path).to_pandas()
    logger.info("Raw: {:,} rows x {} cols", len(df), len(df.columns))

    booked_df = df[df["status_name"] == "Booked"].copy()
    rejected_df = df[df["status_name"].isin(["Rejected", "Canceled"])].copy()

    # Log booked stats
    target_counts = booked_df[TARGET].value_counts(dropna=False)
    n_pos = target_counts.get(1.0, 0)
    n_neg = target_counts.get(0.0, 0)
    n_nan = booked_df[TARGET].isna().sum()
    logger.info(
        "Booked: {:,} rows — {:,} neg / {:,} pos / {:,} immature ({:.2%} default rate)",
        len(booked_df), n_neg, n_pos, n_nan, n_pos / max(n_pos + n_neg, 1),
    )

    # Log reject stats
    score_avail = rejected_df[REJECT_SCORE_COL].notna().mean()
    logger.info(
        "Rejected+Canceled: {:,} rows — {} available for {:.1%}",
        len(rejected_df), REJECT_SCORE_COL, score_avail,
    )

    return booked_df, rejected_df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return _training_features.engineer_features(df)


# ── Interaction search ─────────────────────────────────────────────────────────

def search_interactions(
    df: pd.DataFrame,
    end_before_date: str | pd.Timestamp = SPLIT_DATE,
    return_diagnostics: bool = False,
) -> pd.DataFrame | _training_features.InteractionSearchResult:
    return _training_features.search_interactions(
        df,
        end_before_date=end_before_date,
        return_diagnostics=return_diagnostics,
    )


def add_interactions(df: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    return _training_features.add_interactions(df, interactions)


# ── Feature selection & split ──────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    return _training_features.select_features(df)


def temporal_split(df: pd.DataFrame, feature_cols: list[str]):
    df_model = df[df["mis_Date"] <= MATURITY_CUTOFF].dropna(subset=[TARGET]).copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    train_mask = df_model["mis_Date"] < SPLIT_DATE
    test_mask = df_model["mis_Date"] >= SPLIT_DATE

    X_train = df_model.loc[train_mask, feature_cols]
    y_train = df_model.loc[train_mask, TARGET]
    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, TARGET]

    bench_risk_score_rf = df_model.loc[test_mask, "risk_score_rf"]
    bench_score_RF = df_model.loc[test_mask, "score_RF"]

    train_dates = df_model.loc[train_mask, "mis_Date"].values

    logger.info("Train: {}  ({:.4f} target rate)", X_train.shape, y_train.mean())
    logger.info("Test:  {}  ({:.4f} target rate)", X_test.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, bench_risk_score_rf, bench_score_RF, train_dates


def summarize_population(
    y,
    dates,
    sample_definition: str,
    sample_weight: np.ndarray | None = None,
) -> dict:
    y_array = np.asarray(y, dtype=float)
    dates_array = pd.to_datetime(np.asarray(dates), errors="raise")
    if len(y_array) != len(dates_array):
        raise ValueError("y and dates must have the same length")

    summary = {
        "sample_definition": sample_definition,
        "n_rows": int(len(y_array)),
        "n_pos": int(y_array.sum()) if len(y_array) else 0,
        "target_rate": float(y_array.mean()) if len(y_array) else np.nan,
        "date_start": pd.Timestamp(dates_array.min()).date() if len(dates_array) else None,
        "date_end": pd.Timestamp(dates_array.max()).date() if len(dates_array) else None,
    }

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != len(y_array):
            raise ValueError("sample_weight must have the same length as y")
        booked_mask = sample_weight == 1.0
        summary["n_booked_rows"] = int(booked_mask.sum())
        summary["n_pseudo_labeled_rows"] = int(len(sample_weight) - booked_mask.sum())
        summary["n_booked_pos"] = int(y_array[booked_mask].sum()) if booked_mask.any() else 0

    return summary


def log_population_summary(population_name: str, summary: dict) -> None:
    target_rate = summary["target_rate"]
    rate_str = f"{target_rate:.2%}" if np.isfinite(target_rate) else "—"
    logger.info(
        "{}: {:,} rows [{} to {}] — {} ({:,} pos, rate {})",
        population_name,
        summary["n_rows"],
        summary["date_start"],
        summary["date_end"],
        summary["sample_definition"],
        summary["n_pos"],
        rate_str,
    )
    if "n_booked_rows" in summary:
        logger.info(
            "  booked ground-truth rows={:,}, pseudo-labeled rows={:,}",
            summary["n_booked_rows"],
            summary["n_pseudo_labeled_rows"],
        )


def make_temporal_cv(dates, max_splits: int = 5) -> TemporalExpandingCV:
    distinct_dates = pd.Index(np.sort(pd.to_datetime(np.asarray(dates), errors="raise").unique()))
    if len(distinct_dates) < 3:
        raise ValueError("Temporal CV requires at least 3 distinct date blocks")
    n_splits = min(max_splits, len(distinct_dates) - 1)
    if n_splits != max_splits:
        logger.info(
            "Temporal CV: using {} folds from {} distinct date blocks",
            n_splits,
            len(distinct_dates),
        )
    return TemporalExpandingCV(dates, n_splits=n_splits)


def build_rolling_oot_windows(
    dates,
    max_windows: int = ROLLING_OOT_MAX_WINDOWS,
    min_train_date_blocks: int = 2,
) -> list[dict]:
    dates_series = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"))
    if dates_series.isna().any():
        raise ValueError("Rolling OOT dates must not contain missing values")
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1")
    if min_train_date_blocks < 1:
        raise ValueError("min_train_date_blocks must be at least 1")

    unique_dates = pd.Index(np.sort(dates_series.unique()))
    if len(unique_dates) < min_train_date_blocks + 1:
        raise ValueError(
            "Rolling OOT validation requires at least "
            f"{min_train_date_blocks + 1} distinct date blocks"
        )

    validation_blocks = unique_dates[min_train_date_blocks:]
    n_windows = min(max_windows, len(validation_blocks))
    window_groups = [
        pd.Index(group)
        for group in np.array_split(validation_blocks.to_numpy(), n_windows)
        if len(group) > 0
    ]

    windows = []
    dates_array = dates_series.to_numpy()
    for fold_idx, validation_group in enumerate(window_groups, start=1):
        validation_start = pd.Timestamp(validation_group[0])
        validation_end = pd.Timestamp(validation_group[-1])
        train_dates = unique_dates[unique_dates < validation_start]
        if len(train_dates) < min_train_date_blocks:
            raise ValueError("Rolling OOT validation produced too few training date blocks")

        train_idx = np.flatnonzero(np.isin(dates_array, train_dates))
        validation_idx = np.flatnonzero(np.isin(dates_array, validation_group))
        if len(train_idx) == 0 or len(validation_idx) == 0:
            raise ValueError("Rolling OOT validation produced an empty train or validation window")

        windows.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "validation_idx": validation_idx,
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[-1]),
                "validation_start": validation_start,
                "validation_end": validation_end,
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
            }
        )
    return windows


def reduce_cardinality(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return _training_features.reduce_cardinality(X_train, X_test, cat_cols)


# ── Enhanced feature engineering (post-split) ─────────────────────────────────

def add_frequency_encoding(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return _training_features.add_frequency_encoding(X_train, X_test, cat_cols)


def add_group_stats(
    X_train: pd.DataFrame, X_test: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    return _training_features.add_group_stats(X_train, X_test, num_cols, cat_cols)


def prune_correlated(
    X: pd.DataFrame, num_cols: list[str],
) -> list[str]:
    return _training_features.prune_correlated(X, num_cols)


def add_modeling_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, base_feature_cols: list[str], base_num_cols: list[str], base_cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    return _training_features.add_modeling_features(
        X_train,
        X_test,
        base_feature_cols,
        base_num_cols,
        base_cat_cols,
    )


def build_feature_provenance(
    raw_feature_cols: list[str],
    engineered_feature_cols: list[str],
    interactions: pd.DataFrame,
    freq_cols: list[str],
    group_cols: list[str],
    feature_space_num_cols: list[str],
    feature_space_cat_cols: list[str],
    rfecv_candidate_cols: list[str],
    rfecv_kept_cols: list[str],
) -> pd.DataFrame:
    return _training_features.build_feature_provenance(
        raw_feature_cols,
        engineered_feature_cols,
        interactions,
        freq_cols,
        group_cols,
        feature_space_num_cols,
        feature_space_cat_cols,
        rfecv_candidate_cols,
        rfecv_kept_cols,
    )


# ── Reject inference ──────────────────────────────────────────────────────────

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

    # Down-sample
    max_rejects = int(n_booked_train * REJECT_MAX_RATIO)
    if len(df_rej) > max_rejects:
        logger.info("Down-sampling rejects: {:,} -> {:,} (ratio {:.1f}:1)", len(df_rej), max_rejects, REJECT_MAX_RATIO)
        df_rej = df_rej.sample(n=max_rejects, random_state=RANDOM_STATE)

    # Stochastic pseudo-label assignment
    rng = np.random.RandomState(RANDOM_STATE)
    df_rej[TARGET] = (rng.random(len(df_rej)) < df_rej["pseudo_bad_rate"]).astype(int)

    booked_bad_rate = band_stats["n_bad"].sum() / band_stats["n_booked"].sum()
    logger.info(
        "Pseudo-labeled: {:,} rejects, {:.2%} pseudo-bad (vs {:.2%} booked observed)",
        len(df_rej), df_rej[TARGET].mean(), booked_bad_rate,
    )
    for band in sorted(df_rej["score_band"].dropna().unique()):
        b = df_rej[df_rej["score_band"] == band]
        logger.info("  Band {:2.0f}: n={:>6,}  pseudo_bad_rate={:.4f}  assigned_bad={:,}",
                    band, len(b), b["pseudo_bad_rate"].iloc[0], int(b[TARGET].sum()))

    return df_rej


def augment_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    df_reject_labeled: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Combine booked training data with pseudo-labeled rejects + sample weights."""
    X_rej = df_reject_labeled[feature_cols].copy()
    y_rej = df_reject_labeled[TARGET].copy()

    w_booked = np.ones(len(X_train))
    w_reject = np.full(len(X_rej), REJECT_SAMPLE_WEIGHT)

    X_aug = pd.concat([X_train, X_rej], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train, y_rej], axis=0, ignore_index=True)
    w_aug = np.concatenate([w_booked, w_reject])

    logger.info(
        "Augmented: {:,} booked + {:,} rejects = {:,}  "
        "(booked {:.2%} bad, rejects {:.2%} pseudo-bad, combined {:.2%})",
        len(X_train), len(X_rej), len(X_aug),
        y_train.mean(), y_rej.mean(), y_aug.mean(),
    )
    return X_aug, y_aug, w_aug


# ── Preprocessors ──────────────────────────────────────────────────────────────

def build_preprocessors(num_cols: list[str], cat_cols: list[str]):
    return _training_features.build_preprocessors(num_cols, cat_cols)


def build_monotone_constraints(num_cols: list[str], cat_cols: list[str]) -> list[int]:
    return _training_features.build_monotone_constraints(num_cols, cat_cols)


# ── Feature Selection (temporal stability selection) ──────────────────────────

def run_rfecv(
    X_train: pd.DataFrame, y_train: pd.Series,
    num_cols: list[str], cat_cols: list[str],
    feature_cols: list[str],
    cv,
):
    return _training_features.run_rfecv(X_train, y_train, num_cols, cat_cols, feature_cols, cv)


def normalize_xgboost_monotone_constraints(monotone_constraints):
    if monotone_constraints is None or isinstance(monotone_constraints, (tuple, str, dict)):
        return monotone_constraints
    if isinstance(monotone_constraints, np.ndarray):
        return tuple(monotone_constraints.tolist())
    if isinstance(monotone_constraints, list):
        return tuple(monotone_constraints)
    return monotone_constraints


def train_logistic_regression(X_train, y_train, preprocessor, cv, n_trials: int, sample_weight=None):
    logger.info("Optuna: {} trials x {} folds", n_trials, cv.n_splits)

    # Precompute preprocessed folds once (TargetEncoder is expensive).
    logger.info("Precomputing {} CV folds (TargetEncoder fit)...", cv.n_splits)
    precomputed_folds = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        pre = clone(preprocessor)
        X_tr_t = pre.fit_transform(X_train.iloc[train_idx], y=y_train.iloc[train_idx])
        X_va_t = pre.transform(X_train.iloc[val_idx])
        w_fold = sample_weight[train_idx] if sample_weight is not None else None
        precomputed_folds.append((X_tr_t, y_train.iloc[train_idx], X_va_t, y_train.iloc[val_idx], w_fold))
    logger.info("Optuna: {} trials x {} folds (preprocessed)", n_trials, cv.n_splits)

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)

        fold_scores = []
        for X_tr_t, y_f_tr, X_va_t, y_f_va, w_fold in precomputed_folds:
            clf = LogisticRegression(
                C=C,
                class_weight="balanced", max_iter=5000,
                random_state=RANDOM_STATE, solver="lbfgs",
            )
            clf.fit(X_tr_t, y_f_tr, sample_weight=w_fold)
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))

        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", study_name="lr")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  C={:.4f}", bp["C"])

    lr_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=bp["C"],
            class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])
    lr_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    return lr_model, study


def _lgbm_prauc_eval(y_true, y_raw):
    """Custom LightGBM eval: PR AUC on probabilities (for early stopping)."""
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_raw, -500, 500)))
    return "prauc", float(average_precision_score(y_true, y_prob)), True


def train_lgbm(X_train, y_train, lgbm_preprocessor, lgbm_cat_indices, pos_weight, cv, n_trials: int, sample_weight=None, monotone_constraints=None):
    logger.info("Optuna: {} trials x {} folds, early stopping after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)
    lgbm_subsample_freq = 1

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 0.85),
            "subsample_freq": lgbm_subsample_freq,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "max_bin": trial.suggest_int("max_bin", 63, 127),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-3, 1.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for train_idx, val_idx in tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(lgbm_preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = LGBMClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                scale_pos_weight=pos_weight,
                monotone_constraints=monotone_constraints,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
                **params,
            )
            clf.fit(
                X_tr_t, y_f_tr,
                sample_weight=w_fold,
                eval_set=[(X_va_t, y_f_va)],
                eval_metric=_lgbm_prauc_eval,
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    log_evaluation(-1),
                ],
                categorical_feature=lgbm_cat_indices,
            )
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(normalize_estimator_count(clf.best_iteration_, fallback=N_ESTIMATORS_CEILING))

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", study_name="lgbm")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  n_estimators={} (conservative early stop), lr={:.4f}, leaves={}, depth={}, min_child={}",
                best_n_estimators, bp["learning_rate"], bp["num_leaves"], bp["max_depth"], bp["min_child_samples"])
    logger.info("  subsample={:.2f} (freq={}), colsample_tree={:.2f}, colsample_node={:.2f}, max_bin={}, min_split_gain={:.2e}, alpha={:.2e}, lambda={:.2e}",
                bp["subsample"], lgbm_subsample_freq, bp["colsample_bytree"], bp["colsample_bynode"], bp["max_bin"], bp["min_split_gain"], bp["reg_alpha"], bp["reg_lambda"])

    lgbm_model = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=pos_weight,
            subsample_freq=lgbm_subsample_freq,
            monotone_constraints=monotone_constraints,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
            **study.best_params,
        )),
    ])
    fit_params = {"classifier__categorical_feature": lgbm_cat_indices}
    if sample_weight is not None:
        fit_params["classifier__sample_weight"] = sample_weight
    lgbm_model.fit(X_train, y_train, **fit_params)
    return lgbm_model, study, best_n_estimators


def train_xgboost(X_train, y_train, preprocessor, pos_weight, cv, n_trials: int, sample_weight=None, monotone_constraints=None):
    logger.info("Optuna: {} trials x {} folds, early stopping after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)
    xgb_monotone_constraints = normalize_xgboost_monotone_constraints(monotone_constraints)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for train_idx, val_idx in tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr, y=y_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = XGBClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                scale_pos_weight=pos_weight,
                monotone_constraints=xgb_monotone_constraints,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
                eval_metric="aucpr", **params,
            )
            clf.fit(X_tr_t, y_f_tr, sample_weight=w_fold,
                    eval_set=[(X_va_t, y_f_va)], verbose=False)
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(
                normalize_estimator_count(
                    None if clf.best_iteration is None else clf.best_iteration + 1,
                    fallback=N_ESTIMATORS_CEILING,
                )
            )

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", study_name="xgb")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  n_estimators={} (conservative early stop), lr={:.4f}, depth={}, min_child_w={}",
                best_n_estimators, bp["learning_rate"], bp["max_depth"], bp["min_child_weight"])
    logger.info("  subsample={:.2f}, colsample_tree={:.2f}, colsample_node={:.2f}, gamma={:.2e}, alpha={:.2e}, lambda={:.2e}",
                bp["subsample"], bp["colsample_bytree"], bp["colsample_bynode"], bp["gamma"], bp["reg_alpha"], bp["reg_lambda"])

    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=pos_weight,
            monotone_constraints=xgb_monotone_constraints,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="aucpr", **study.best_params,
        )),
    ])
    if sample_weight is not None:
        xgb_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        xgb_model.fit(X_train, y_train)
    return xgb_model, study, best_n_estimators


def train_catboost(X_train, y_train, lgbm_preprocessor, pos_weight, cv, n_trials: int, sample_weight=None, monotone_constraints=None):
    logger.info("Optuna: {} trials x {} folds, early stopping (PRAUC) after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 3, 5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
            "random_strength": trial.suggest_float("random_strength", 0.1, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for train_idx, val_idx in tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(lgbm_preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = CatBoostClassifier(
                iterations=N_ESTIMATORS_CEILING,
                auto_class_weights="Balanced",
                monotone_constraints=monotone_constraints,
                random_seed=RANDOM_STATE,
                eval_metric="PRAUC",
                verbose=0,
                **params,
            )
            clf.fit(
                X_tr_t, y_f_tr,
                sample_weight=w_fold,
                eval_set=[(X_va_t, y_f_va)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=0,
            )
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(
                normalize_estimator_count(
                    None if clf.best_iteration_ is None else clf.best_iteration_ + 1,
                    fallback=N_ESTIMATORS_CEILING,
                )
            )

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize", study_name="catboost")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  iterations={} (conservative early stop), lr={:.4f}, depth={}, min_data_in_leaf={}",
                best_n_estimators, bp["learning_rate"], bp["depth"], bp["min_data_in_leaf"])
    logger.info("  l2_leaf_reg={:.2e}, random_strength={:.2e}, bagging_temp={:.2f}",
                bp["l2_leaf_reg"], bp["random_strength"], bp["bagging_temperature"])

    catboost_model = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", CatBoostClassifier(
            iterations=best_n_estimators,
            auto_class_weights="Balanced",
            monotone_constraints=monotone_constraints,
            random_seed=RANDOM_STATE, verbose=0,
            **study.best_params,
        )),
    ])
    if sample_weight is not None:
        catboost_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        catboost_model.fit(X_train, y_train)
    return catboost_model, study, best_n_estimators


class TemporalStackingClassifier:
    def __init__(
        self,
        named_estimators_: dict[str, Pipeline],
        final_estimator_: LogisticRegression,
        base_model_names_: list[str],
        meta_feature_names_: list[str],
        meta_training_positions_: np.ndarray,
        fold_training_positions_: list[np.ndarray],
        fold_validation_positions_: list[np.ndarray],
    ):
        self.named_estimators_ = named_estimators_
        self.final_estimator_ = final_estimator_
        self.base_model_names_ = base_model_names_
        self.meta_feature_names_ = meta_feature_names_
        self.meta_training_positions_ = np.asarray(meta_training_positions_, dtype=int)
        self.fold_training_positions_ = [np.asarray(idx, dtype=int) for idx in fold_training_positions_]
        self.fold_validation_positions_ = [np.asarray(idx, dtype=int) for idx in fold_validation_positions_]
        self.classes_ = final_estimator_.classes_

    def _build_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        meta_features = {
            feature_name: self.named_estimators_[model_name].predict_proba(X)[:, 1]
            for model_name, feature_name in zip(self.base_model_names_, self.meta_feature_names_, strict=True)
        }
        return pd.DataFrame(meta_features, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_estimator_.predict_proba(self._build_meta_features(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_estimator_.predict(self._build_meta_features(X))


def fit_pipeline_from_template(
    model_template: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    fitted_model = build_fresh_pipeline_from_fitted(model_template)
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) >= 2:
        safe_target_encoder_cv = int(min(5, len(y_train), class_counts.min()))
        params = fitted_model.get_params()
        if "preprocessor__cat__encoder__cv" in params:
            fitted_model.set_params(preprocessor__cat__encoder__cv=max(2, safe_target_encoder_cv))

    classifier = fitted_model.named_steps["classifier"]
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    if isinstance(classifier, LGBMClassifier):
        preprocessor = fitted_model.named_steps["preprocessor"]
        num_cols = list(preprocessor.transformers[0][2])
        cat_cols = list(preprocessor.transformers[1][2])
        fit_kwargs["classifier__categorical_feature"] = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    fitted_model.fit(X_train, y_train, **fit_kwargs)
    return fitted_model


def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_models: dict[str, Pipeline],
    cv,
    sample_weight: np.ndarray | None = None,
):
    if not base_models:
        raise ValueError("Temporal stacking requires at least one base model")

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = y_train.copy()

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must have the same length as X_train")

    base_model_names = list(base_models)
    meta_feature_names = [f"stack__{sanitize_output_name(name)}" for name in base_model_names]
    oof_meta_features = np.full((len(X_train), len(base_model_names)), np.nan, dtype=float)
    fold_training_positions: list[np.ndarray] = []
    fold_validation_positions: list[np.ndarray] = []

    logger.info(
        "{} base learners -> LR meta-learner, {} temporal folds",
        len(base_model_names),
        cv.n_splits,
    )

    for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        y_fold_train = y_train.iloc[train_idx]
        class_counts = pd.Series(y_fold_train).value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            logger.warning(
                "Stacking fold {} skipped: fit window has insufficient class support ({})",
                fold_number,
                class_counts.to_dict(),
            )
            continue

        X_fold_train = X_train.iloc[train_idx].copy()
        X_fold_validation = X_train.iloc[val_idx].copy()
        w_fold_train = sample_weight[train_idx] if sample_weight is not None else None

        for model_idx, model_name in enumerate(base_model_names):
            fold_model = fit_pipeline_from_template(
                base_models[model_name],
                X_fold_train,
                y_fold_train,
                sample_weight=w_fold_train,
            )
            oof_meta_features[val_idx, model_idx] = fold_model.predict_proba(X_fold_validation)[:, 1]

        fold_training_positions.append(np.asarray(train_idx, dtype=int))
        fold_validation_positions.append(np.asarray(val_idx, dtype=int))

    meta_training_mask = np.isfinite(oof_meta_features).all(axis=1)
    if not meta_training_mask.any():
        raise ValueError("Temporal stacking produced no out-of-fold predictions")

    y_meta = y_train.iloc[meta_training_mask]
    meta_class_counts = pd.Series(y_meta).value_counts()
    if len(meta_class_counts) < 2:
        raise ValueError("Temporal stacking meta-learner requires at least 2 classes in OOF predictions")

    meta_training_frame = pd.DataFrame(
        oof_meta_features[meta_training_mask],
        columns=meta_feature_names,
        index=X_train.index[meta_training_mask],
    )
    meta_model = LogisticRegression(max_iter=20_000, random_state=RANDOM_STATE)
    meta_fit_kwargs = {}
    if sample_weight is not None:
        meta_fit_kwargs["sample_weight"] = sample_weight[meta_training_mask]
    meta_model.fit(meta_training_frame, y_meta, **meta_fit_kwargs)

    logger.info(
        "Temporal stacking meta-learner fit on {:,} OOF rows across {} folds ({} rows excluded)",
        len(meta_training_frame),
        len(fold_validation_positions),
        len(X_train) - len(meta_training_frame),
    )

    return TemporalStackingClassifier(
        named_estimators_=dict(base_models),
        final_estimator_=meta_model,
        base_model_names_=base_model_names,
        meta_feature_names_=meta_feature_names,
        meta_training_positions_=np.flatnonzero(meta_training_mask),
        fold_training_positions_=fold_training_positions,
        fold_validation_positions_=fold_validation_positions,
    )


def safe_stratified_n_splits(y, max_splits: int = 5) -> int:
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Stratified cross-validation requires at least 2 examples in each class")
    return int(min(max_splits, len(y), class_counts.min()))


def normalize_estimator_count(value, fallback: int = 1) -> int:
    if value is None or pd.isna(value):
        return fallback
    return max(int(value), fallback)


def select_conservative_boosting_rounds(
    best_iterations: list[int],
    fallback: int = N_ESTIMATORS_CEILING,
    quantile: float = 0.25,
) -> int:
    if not best_iterations:
        return normalize_estimator_count(fallback)
    normalized = np.asarray(
        [normalize_estimator_count(value, fallback=fallback) for value in best_iterations],
        dtype=float,
    )
    conservative_value = np.floor(np.quantile(normalized, quantile))
    return normalize_estimator_count(conservative_value, fallback=int(normalized.min()))


def build_fresh_pipeline_from_fitted(model: Pipeline) -> Pipeline:
    preprocessor = clone(model.named_steps["preprocessor"])
    classifier = model.named_steps["classifier"]

    if isinstance(classifier, LogisticRegression):
        fresh_classifier = LogisticRegression(**classifier.get_params())
    elif isinstance(classifier, LGBMClassifier):
        fresh_classifier = LGBMClassifier(**classifier.get_params())
    elif isinstance(classifier, XGBClassifier):
        classifier_params = classifier.get_params()
        classifier_params["monotone_constraints"] = normalize_xgboost_monotone_constraints(
            classifier_params.get("monotone_constraints")
        )
        fresh_classifier = XGBClassifier(**classifier_params)
    elif isinstance(classifier, CatBoostClassifier):
        classifier_params = classifier.get_params()
        monotone_constraints = classifier_params.get("monotone_constraints")
        if isinstance(monotone_constraints, np.ndarray):
            classifier_params["monotone_constraints"] = monotone_constraints.tolist()
        fresh_classifier = CatBoostClassifier(**classifier_params)
    else:
        fresh_classifier = clone(classifier)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", fresh_classifier),
    ])


def build_population_summary_df(
    booked_df: pd.DataFrame,
    rejected_df: pd.DataFrame | None = None,
    population_mode: str = POPULATION_MODE_BOOKED_MONITORING,
) -> pd.DataFrame:
    frames = [booked_df.assign(_population_group="booked")]
    if rejected_df is not None and not rejected_df.empty:
        frames.append(rejected_df.assign(_population_group="decisioned_non_booked"))
    applications = pd.concat(frames, axis=0, ignore_index=False).copy()
    if applications.empty:
        return pd.DataFrame(
            columns=[
                "population_mode", "split", "status_name", "population_group", "n_rows",
                "n_with_observed_target", "n_bad_observed", "date_start", "date_end",
            ]
        )

    records = []
    for split_name, split_mask in (
        ("pre_split", applications["mis_Date"] < pd.Timestamp(SPLIT_DATE)),
        ("post_split", applications["mis_Date"] >= pd.Timestamp(SPLIT_DATE)),
    ):
        split_df = applications.loc[split_mask].copy()
        if split_df.empty:
            continue
        grouped = split_df.groupby(["status_name", "_population_group"], dropna=False)
        for (status_name, population_group), group_df in grouped:
            observed = group_df[TARGET].notna()
            records.append(
                {
                    "population_mode": population_mode,
                    "split": split_name,
                    "status_name": status_name,
                    "population_group": population_group,
                    "n_rows": len(group_df),
                    "n_with_observed_target": int(observed.sum()),
                    "n_bad_observed": int(group_df.loc[observed, TARGET].sum()) if observed.any() else 0,
                    "date_start": pd.Timestamp(group_df["mis_Date"].min()).date(),
                    "date_end": pd.Timestamp(group_df["mis_Date"].max()).date(),
                }
            )
    return pd.DataFrame(records).sort_values(["split", "status_name"]).reset_index(drop=True)


def build_applicant_score_frame(
    booked_df: pd.DataFrame,
    rejected_df: pd.DataFrame | None,
    X_training_reference_base: pd.DataFrame,
    base_feature_cols: list[str],
    base_num_cols: list[str],
    base_cat_cols: list[str],
    frozen_feature_cols: list[str],
    models: dict[str, object],
) -> pd.DataFrame:
    frames = [booked_df]
    if rejected_df is not None and not rejected_df.empty:
        frames.append(rejected_df)
    applicant_df = pd.concat(frames, axis=0, ignore_index=False).copy()
    applicant_df = applicant_df[
        applicant_df["status_name"].isin(UNDERWRITING_DECISION_STATUSES)
        & (applicant_df["mis_Date"] >= pd.Timestamp(SPLIT_DATE))
    ].copy()
    if applicant_df.empty:
        return pd.DataFrame(
            columns=[
                "authorization_id", "mis_Date", "status_name", TARGET,
                "has_observed_target", "target_source", "risk_score_rf", "score_RF",
            ]
        )

    _, X_applicant, _, _, _, _, _ = add_modeling_features(
        X_training_reference_base,
        applicant_df[base_feature_cols],
        base_feature_cols,
        base_num_cols,
        base_cat_cols,
    )
    X_applicant = X_applicant[frozen_feature_cols].copy()

    score_frame = applicant_df.loc[:, [
        "authorization_id", "mis_Date", "status_name", TARGET, "risk_score_rf", "score_RF",
    ]].copy()
    score_frame["has_observed_target"] = score_frame[TARGET].notna()
    score_frame["target_source"] = np.where(
        score_frame["has_observed_target"],
        "observed_booked",
        "unobserved_application",
    )
    for name, model in models.items():
        score_frame[f"score__{sanitize_output_name(name)}"] = model.predict_proba(X_applicant)[:, 1]
    return score_frame.sort_values(["mis_Date", "authorization_id"]).reset_index(drop=True)


def run_rolling_out_of_time_validation(
    X_booked_base: pd.DataFrame,
    y_booked: pd.Series,
    dates,
    bench_risk_score_rf: pd.Series,
    bench_score_RF: pd.Series,
    base_feature_cols: list[str],
    base_num_cols: list[str],
    base_cat_cols: list[str],
    frozen_feature_cols: list[str],
    frozen_num_cols: list[str],
    frozen_cat_cols: list[str],
    base_models: dict[str, Pipeline],
    max_windows: int = ROLLING_OOT_MAX_WINDOWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if not base_models:
        empty_results = pd.DataFrame(
            columns=[
                "fold", "Model", "train_start", "train_end", "calibration_start", "calibration_end",
                "validation_start", "validation_end", "n_fit", "n_calibration", "n_validation", "n_validation_pos",
                "ROC AUC", "Gini", "KS", "PR AUC", "Brier", "is_calibrated",
            ]
        )
        empty_summary = pd.DataFrame(
            columns=[
                "Model", "n_folds", "mean_ROC_AUC", "std_ROC_AUC", "mean_PR_AUC", "std_PR_AUC",
                "mean_Brier", "std_Brier", "validation_start_min", "validation_end_max",
            ]
        )
        return empty_results, empty_summary

    rolling_windows = build_rolling_oot_windows(dates, max_windows=max_windows, min_train_date_blocks=2)
    lgbm_cat_indices = list(range(len(frozen_num_cols), len(frozen_num_cols) + len(frozen_cat_cols)))
    dates_array = pd.to_datetime(np.asarray(dates), errors="raise")
    bench_risk_series = pd.Series(bench_risk_score_rf, index=X_booked_base.index)
    bench_score_series = pd.Series(bench_score_RF, index=X_booked_base.index)
    records = []

    for window in rolling_windows:
        fold = window["fold"]
        train_idx = window["train_idx"]
        validation_idx = window["validation_idx"]
        X_window_train_base = X_booked_base.iloc[train_idx].copy()
        y_window_train = y_booked.iloc[train_idx].copy()
        train_dates = dates_array[train_idx]
        X_window_validation_base = X_booked_base.iloc[validation_idx].copy()
        y_window_validation = y_booked.iloc[validation_idx].copy()

        X_fit_base, X_calibration_base, y_fit, y_calibration, fit_dates, calibration_dates = temporal_calibration_split(
            X_window_train_base,
            y_window_train,
            train_dates,
            calibration_fraction=CALIBRATION_FRACTION,
        )

        X_fit, X_validation, _, _, _, _, _ = add_modeling_features(
            X_fit_base, X_window_validation_base, base_feature_cols, base_num_cols, base_cat_cols,
        )
        _, X_calibration, _, _, _, _, _ = add_modeling_features(
            X_fit_base, X_calibration_base, base_feature_cols, base_num_cols, base_cat_cols,
        )

        X_fit = X_fit[frozen_feature_cols].copy()
        X_validation = X_validation[frozen_feature_cols].copy()
        X_calibration = X_calibration[frozen_feature_cols].copy()

        logger.info(
            "Rolling OOT fold {}: fit {:,} rows [{} to {}], calibration {:,} rows [{} to {}], validation {:,} rows [{} to {}]",
            fold,
            len(X_fit),
            pd.Timestamp(pd.to_datetime(fit_dates).min()).date(),
            pd.Timestamp(pd.to_datetime(fit_dates).max()).date(),
            len(X_calibration),
            pd.Timestamp(pd.to_datetime(calibration_dates).min()).date(),
            pd.Timestamp(pd.to_datetime(calibration_dates).max()).date(),
            len(X_validation),
            pd.Timestamp(window["validation_start"]).date(),
            pd.Timestamp(window["validation_end"]).date(),
        )

        fold_models: dict[str, object] = {}
        class_counts = pd.Series(y_fit).value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            logger.warning(
                "Rolling OOT fold {} skipped: fit window has insufficient class support ({})",
                fold,
                class_counts.to_dict(),
            )
            continue
        safe_target_encoder_cv = (
            int(min(5, len(y_fit), class_counts.min()))
            if len(class_counts) >= 2
            else None
        )
        for name, model in base_models.items():
            fold_model = build_fresh_pipeline_from_fitted(model)
            if safe_target_encoder_cv is not None:
                params = fold_model.get_params()
                if "preprocessor__cat__encoder__cv" in params:
                    fold_model.set_params(
                        preprocessor__cat__encoder__cv=max(2, safe_target_encoder_cv),
                    )
            fit_kwargs = {}
            if name == "Logistic Regression":
                fit_kwargs["classifier__sample_weight"] = None
            elif name == "LightGBM":
                fit_kwargs["classifier__categorical_feature"] = lgbm_cat_indices
            fold_model.fit(X_fit, y_fit, **fit_kwargs)
            fold_models[name] = fold_model

            fold_scores = fold_model.predict_proba(X_validation)[:, 1]
            metrics = evaluate_safely(name, y_window_validation.values, fold_scores)
            records.append(
                {
                    "fold": fold,
                    **metrics,
                    "train_start": pd.Timestamp(pd.to_datetime(fit_dates).min()).date(),
                    "train_end": pd.Timestamp(pd.to_datetime(fit_dates).max()).date(),
                    "calibration_start": pd.Timestamp(pd.to_datetime(calibration_dates).min()).date(),
                    "calibration_end": pd.Timestamp(pd.to_datetime(calibration_dates).max()).date(),
                    "validation_start": pd.Timestamp(window["validation_start"]).date(),
                    "validation_end": pd.Timestamp(window["validation_end"]).date(),
                    "n_fit": len(X_fit),
                    "n_calibration": len(X_calibration),
                    "n_validation": len(X_validation),
                    "n_validation_pos": int(y_window_validation.sum()),
                    "is_calibrated": False,
                }
            )

            if len(np.unique(y_calibration)) >= 2:
                calibration_cv = safe_stratified_n_splits(y_calibration)
                calibrated_model = CalibratedClassifierCV(
                    FrozenEstimator(fold_model),
                    method="sigmoid",
                    cv=calibration_cv,
                )
                calibrated_model.fit(X_calibration, y_calibration)
                calibrated_scores = calibrated_model.predict_proba(X_validation)[:, 1]
                calibrated_name = f"{name} (calibrated)"
                calibrated_metrics = evaluate_safely(calibrated_name, y_window_validation.values, calibrated_scores)
                records.append(
                    {
                        "fold": fold,
                        **calibrated_metrics,
                        "train_start": pd.Timestamp(pd.to_datetime(fit_dates).min()).date(),
                        "train_end": pd.Timestamp(pd.to_datetime(fit_dates).max()).date(),
                        "calibration_start": pd.Timestamp(pd.to_datetime(calibration_dates).min()).date(),
                        "calibration_end": pd.Timestamp(pd.to_datetime(calibration_dates).max()).date(),
                        "validation_start": pd.Timestamp(window["validation_start"]).date(),
                        "validation_end": pd.Timestamp(window["validation_end"]).date(),
                        "n_fit": len(X_fit),
                        "n_calibration": len(X_calibration),
                        "n_validation": len(X_validation),
                        "n_validation_pos": int(y_window_validation.sum()),
                        "is_calibrated": True,
                    }
                )

        benchmark_risk_scores = -bench_risk_series.loc[X_window_validation_base.index].to_numpy()
        benchmark_score_rf_scores = -bench_score_series.loc[X_window_validation_base.index].to_numpy()
        for benchmark_name, benchmark_scores in zip(
            BENCHMARK_MODEL_NAMES,
            [benchmark_risk_scores, benchmark_score_rf_scores],
            strict=True,
        ):
            benchmark_metrics = evaluate_safely(
                benchmark_name,
                y_window_validation.values,
                benchmark_scores,
                is_probability=False,
            )
            records.append(
                {
                    "fold": fold,
                    **benchmark_metrics,
                    "train_start": pd.Timestamp(pd.to_datetime(fit_dates).min()).date(),
                    "train_end": pd.Timestamp(pd.to_datetime(fit_dates).max()).date(),
                    "calibration_start": pd.Timestamp(pd.to_datetime(calibration_dates).min()).date(),
                    "calibration_end": pd.Timestamp(pd.to_datetime(calibration_dates).max()).date(),
                    "validation_start": pd.Timestamp(window["validation_start"]).date(),
                    "validation_end": pd.Timestamp(window["validation_end"]).date(),
                    "n_fit": len(X_fit),
                    "n_calibration": len(X_calibration),
                    "n_validation": len(X_validation),
                    "n_validation_pos": int(y_window_validation.sum()),
                    "is_calibrated": False,
                }
            )

    rolling_results_df = pd.DataFrame(records)
    if rolling_results_df.empty:
        rolling_summary_df = pd.DataFrame(
            columns=[
                "Model", "n_folds", "mean_ROC_AUC", "std_ROC_AUC", "mean_PR_AUC", "std_PR_AUC",
                "mean_Brier", "std_Brier", "validation_start_min", "validation_end_max",
            ]
        )
        return rolling_results_df, rolling_summary_df

    rolling_summary_df = (
        rolling_results_df
        .groupby("Model", dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_ROC_AUC=("ROC AUC", "mean"),
            std_ROC_AUC=("ROC AUC", "std"),
            mean_PR_AUC=("PR AUC", "mean"),
            std_PR_AUC=("PR AUC", "std"),
            mean_Brier=("Brier", "mean"),
            std_Brier=("Brier", "std"),
            validation_start_min=("validation_start", "min"),
            validation_end_max=("validation_end", "max"),
        )
        .reset_index()
        .sort_values("mean_PR_AUC", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return rolling_results_df, rolling_summary_df


def build_ablation_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])


def fit_phase3_ablation_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
    sample_weight=None,
) -> Pipeline:
    model = Pipeline([
        ("preprocessor", build_ablation_preprocessor(num_cols, cat_cols)),
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def prepare_feature_subset(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    apply_cardinality: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    subset_cat_cols = [c for c in cat_cols if c in feature_cols]
    subset_num_cols = [c for c in feature_cols if c not in subset_cat_cols]
    X_train_subset = X_train[feature_cols].copy()
    X_other_subset = X_other[feature_cols].copy()
    if apply_cardinality and subset_cat_cols:
        X_train_subset, X_other_subset, _ = reduce_cardinality(X_train_subset, X_other_subset, subset_cat_cols)
    return X_train_subset, X_other_subset, subset_num_cols, subset_cat_cols


def run_phase3_ablations(
    X_booked_base: pd.DataFrame,
    y_booked: pd.Series,
    booked_dates,
    X_test_base: pd.DataFrame,
    y_test: pd.Series,
    raw_feature_cols: list[str],
    engineered_feature_cols: list[str],
    interaction_feature_cols: list[str],
    base_feature_cols: list[str],
    base_num_cols: list[str],
    base_cat_cols: list[str],
    rfecv_candidate_feature_cols: list[str],
    frozen_feature_cols: list[str],
    frozen_num_cols: list[str],
    frozen_cat_cols: list[str],
    X_augmented_base: pd.DataFrame | None = None,
    y_augmented: pd.Series | None = None,
    augmented_dates=None,
    augmented_sample_weight=None,
) -> pd.DataFrame:
    records = []
    X_booked_fit_base, X_booked_calib_base, y_booked_fit, y_booked_calib, booked_fit_dates, booked_calib_dates = temporal_calibration_split(
        X_booked_base, y_booked, booked_dates, calibration_fraction=CALIBRATION_FRACTION,
    )

    def add_record(
        component: str,
        variant: str,
        feature_cols: list[str],
        num_cols: list[str],
        cat_cols: list[str],
        model,
        X_test_variant: pd.DataFrame,
        n_train_rows: int,
        n_calibration_rows: int,
        uses_rfecv: bool,
        uses_calibration: bool,
        uses_reject_inference: bool,
    ) -> None:
        metrics = evaluate("Phase 3 Ablation", y_test.values, model.predict_proba(X_test_variant)[:, 1])
        records.append(
            {
                "component": component,
                "variant": variant,
                "model": "Logistic Regression",
                "n_features": len(feature_cols),
                "n_num": len(num_cols),
                "n_cat": len(cat_cols),
                "n_train": n_train_rows,
                "n_calibration": n_calibration_rows,
                "n_test": len(y_test),
                "uses_rfecv": uses_rfecv,
                "uses_calibration": uses_calibration,
                "uses_reject_inference": uses_reject_inference,
                "ROC AUC": metrics["ROC AUC"],
                "PR AUC": metrics["PR AUC"],
                "Brier": metrics["Brier"],
            }
        )

    raw_only_cols = [c for c in raw_feature_cols if c in X_booked_base.columns]
    engineered_space_cols = [c for c in raw_feature_cols + engineered_feature_cols if c in X_booked_base.columns]
    interaction_space_cols = [c for c in raw_feature_cols + engineered_feature_cols + interaction_feature_cols if c in X_booked_base.columns]

    for component, variant, feature_cols in [
        ("raw_features", "raw_only", raw_only_cols),
        ("engineered_features", "raw_plus_engineered", engineered_space_cols),
        ("interaction_search", "with_discovery_interactions", interaction_space_cols),
    ]:
        X_fit_variant, X_test_variant, num_variant, cat_variant = prepare_feature_subset(
            X_booked_fit_base, X_test_base, feature_cols, base_cat_cols, apply_cardinality=True,
        )
        model = fit_phase3_ablation_model(X_fit_variant, y_booked_fit, num_variant, cat_variant)
        add_record(
            component,
            variant,
            feature_cols,
            num_variant,
            cat_variant,
            model,
            X_test_variant,
            len(X_fit_variant),
            0,
            False,
            False,
            False,
        )

    X_booked_fit_full, X_test_full, _, _, _, _, _ = add_modeling_features(
        X_booked_fit_base, X_test_base, base_feature_cols, base_num_cols, base_cat_cols,
    )
    X_booked_fit_full_for_calib, X_booked_calib_full, _, _, _, _, _ = add_modeling_features(
        X_booked_fit_base, X_booked_calib_base, base_feature_cols, base_num_cols, base_cat_cols,
    )

    candidate_cat_cols = [c for c in base_cat_cols if c in rfecv_candidate_feature_cols]
    X_fit_candidate, X_test_candidate, num_candidate, cat_candidate = prepare_feature_subset(
        X_booked_fit_full, X_test_full, rfecv_candidate_feature_cols, candidate_cat_cols, apply_cardinality=False,
    )
    model_candidate = fit_phase3_ablation_model(X_fit_candidate, y_booked_fit, num_candidate, cat_candidate)
    add_record(
        "rfecv",
        "candidate_feature_space",
        rfecv_candidate_feature_cols,
        num_candidate,
        cat_candidate,
        model_candidate,
        X_test_candidate,
        len(X_fit_candidate),
        0,
        False,
        False,
        False,
    )

    X_fit_frozen, X_test_frozen, _, _ = prepare_feature_subset(
        X_booked_fit_full, X_test_full, frozen_feature_cols, frozen_cat_cols, apply_cardinality=False,
    )
    X_calib_frozen = X_booked_calib_full[frozen_feature_cols].copy()

    model_frozen = fit_phase3_ablation_model(X_fit_frozen, y_booked_fit, frozen_num_cols, frozen_cat_cols)
    add_record(
        "rfecv",
        "frozen_feature_space",
        frozen_feature_cols,
        frozen_num_cols,
        frozen_cat_cols,
        model_frozen,
        X_test_frozen,
        len(X_fit_frozen),
        0,
        True,
        False,
        False,
    )
    add_record(
        "calibration",
        "uncalibrated",
        frozen_feature_cols,
        frozen_num_cols,
        frozen_cat_cols,
        model_frozen,
        X_test_frozen,
        len(X_fit_frozen),
        0,
        True,
        False,
        False,
    )

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model_frozen), method="sigmoid")
    calibrated_model.fit(X_calib_frozen, y_booked_calib)
    add_record(
        "calibration",
        "sigmoid_calibrated",
        frozen_feature_cols,
        frozen_num_cols,
        frozen_cat_cols,
        calibrated_model,
        X_test_frozen,
        len(X_fit_frozen),
        len(X_calib_frozen),
        True,
        True,
        False,
    )

    if X_augmented_base is not None and y_augmented is not None and augmented_dates is not None and augmented_sample_weight is not None:
        X_aug_fit_base, X_aug_calib_base, y_aug_fit, y_aug_calib, w_aug_fit, w_aug_calib, _, _ = temporal_calibration_split(
            X_augmented_base,
            y_augmented,
            augmented_dates,
            calibration_fraction=CALIBRATION_FRACTION,
            sample_weight=augmented_sample_weight,
        )
        booked_calib_mask = w_aug_calib == 1.0
        X_aug_fit_full, X_test_reject_full, _, _, _, _, _ = add_modeling_features(
            X_aug_fit_base, X_test_base, base_feature_cols, base_num_cols, base_cat_cols,
        )
        X_aug_fit_full_for_calib, X_aug_calib_full, _, _, _, _, _ = add_modeling_features(
            X_aug_fit_base, X_aug_calib_base, base_feature_cols, base_num_cols, base_cat_cols,
        )
        X_fit_reject = X_aug_fit_full[frozen_feature_cols].copy()
        X_test_reject = X_test_reject_full[frozen_feature_cols].copy()
        X_calib_reject = X_aug_calib_full.loc[booked_calib_mask, frozen_feature_cols].copy()
        y_calib_reject = y_aug_calib.loc[booked_calib_mask]
        model_reject = fit_phase3_ablation_model(
            X_fit_reject,
            y_aug_fit,
            frozen_num_cols,
            frozen_cat_cols,
            sample_weight=w_aug_fit,
        )
        calibrated_reject = CalibratedClassifierCV(FrozenEstimator(model_reject), method="sigmoid")
        calibrated_reject.fit(X_calib_reject, y_calib_reject)
        add_record(
            "reject_inference",
            "booked_plus_rejects",
            frozen_feature_cols,
            frozen_num_cols,
            frozen_cat_cols,
            calibrated_reject,
            X_test_reject,
            len(X_fit_reject),
            len(X_calib_reject),
            True,
            True,
            True,
        )
    return pd.DataFrame(records)


# ── SHAP Explainability ───────────────────────────────────────────────────────

def compute_shap_analysis(
    models: dict,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    output_dir: Path,
    preferred_model_name: str | None = None,
) -> pd.DataFrame | None:
    """SHAP values for the best tree model: summary, importance, dependence plots."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping (pip install shap)")
        return None

    candidate_names = [preferred_model_name] if preferred_model_name is not None else []
    candidate_names.extend(["LightGBM", "XGBoost", "CatBoost", "Logistic Regression"])
    tree_class_names = {"LGBMClassifier", "XGBClassifier", "CatBoostClassifier"}

    # Build ranked list of (model, explainer_type) — we may need to skip models
    # that fail due to shap/library version incompatibilities.
    ranked_candidates = []
    for name in dict.fromkeys(candidate_names):
        if name not in models or not hasattr(models[name], "named_steps"):
            continue
        candidate_model = models[name]
        candidate_clf = candidate_model.named_steps["classifier"]
        if candidate_clf.__class__.__name__ in tree_class_names:
            ranked_candidates.append((name, candidate_model, "tree"))
        elif isinstance(candidate_clf, LogisticRegression):
            ranked_candidates.append((name, candidate_model, "linear"))

    if not ranked_candidates:
        logger.warning("No supported model available for SHAP")
        return None

    feature_names = num_cols + cat_cols

    for name, model, explainer_type in ranked_candidates:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]

        X_t = pre.transform(X_test)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()

        # Subsample for speed on large test sets
        max_shap = 5000
        if X_t.shape[0] > max_shap:
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(X_t.shape[0], max_shap, replace=False)
            X_t = X_t[idx]
            logger.info("Subsampled {:,} -> {:,} for SHAP", X_test.shape[0], max_shap)

        try:
            if explainer_type == "linear":
                background = X_t
                if X_t.shape[0] > 1000:
                    rng = np.random.RandomState(RANDOM_STATE)
                    bg_idx = rng.choice(X_t.shape[0], 1000, replace=False)
                    background = X_t[bg_idx]
                explainer = shap.LinearExplainer(clf, background)
            else:
                explainer = shap.TreeExplainer(clf)
            break
        except (ValueError, TypeError, Exception) as exc:
            logger.warning("SHAP explainer failed for {} ({}), trying next model: {}", name, explainer_type, exc)
            continue
    else:
        logger.warning("All SHAP explainer attempts failed — skipping")
        return None
    shap_values = explainer.shap_values(X_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values, X_t, feature_names=feature_names,
        show=False, max_display=20,
    )
    plt.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    shap.summary_plot(
        shap_values, X_t, feature_names=feature_names,
        plot_type="bar", show=False, max_display=20,
    )
    plt.savefig(plots_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:6]
    n_dep = min(6, len(top_idx))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, ax in zip(top_idx[:n_dep], axes.flatten()[:n_dep]):
        shap.dependence_plot(
            int(i), shap_values, X_t,
            feature_names=feature_names, ax=ax, show=False,
        )
    for j in range(n_dep, 6):
        axes.flatten()[j].set_visible(False)
    fig.suptitle(f"SHAP Dependence — {name} (top {n_dep})", fontsize=14)
    fig.tight_layout()
    fig.savefig(plots_dir / "shap_dependence.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(output_dir / "shap_values.csv", index=False, float_format="%.6f")

    summary = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(output_dir / "shap_importance.csv", index=False, float_format="%.6f")

    logger.info("SHAP ({}): {} features", name, len(feature_names))
    for _, r in summary.head(10).iterrows():
        logger.info("  {:<35s} mean|SHAP|={:.4f}", r["feature"], r["mean_abs_shap"])

    return summary


def _psi_component(expected_pct: np.ndarray, actual_pct: np.ndarray) -> np.ndarray:
    eps = 1e-6
    e = np.clip(expected_pct, eps, None)
    a = np.clip(actual_pct, eps, None)
    return (a - e) * np.log(a / e)


def compute_psi(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    n_bins: int = 10,
) -> float:
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(train_scores[np.isfinite(train_scores)], quantiles)
    bin_edges[0], bin_edges[-1] = -np.inf, np.inf
    train_pct = np.histogram(train_scores, bins=bin_edges)[0].astype(float)
    test_pct = np.histogram(test_scores, bins=bin_edges)[0].astype(float)
    train_pct /= train_pct.sum()
    test_pct /= test_pct.sum()
    return float(_psi_component(train_pct, test_pct).sum())


def compute_csi(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    records = []
    for col in num_cols:
        tr = X_train[col].dropna().values.astype(float)
        te = X_test[col].dropna().values.astype(float)
        if len(tr) < 50 or len(te) < 50:
            continue
        bin_edges = np.unique(np.percentile(tr, np.linspace(0, 100, n_bins + 1)))
        if len(bin_edges) < 3:
            continue
        bin_edges[0], bin_edges[-1] = -np.inf, np.inf
        tr_pct = np.histogram(tr, bins=bin_edges)[0].astype(float)
        te_pct = np.histogram(te, bins=bin_edges)[0].astype(float)
        tr_pct /= tr_pct.sum()
        te_pct /= te_pct.sum()
        records.append(
            {
                "feature": col,
                "type": "numerical",
                "csi": float(_psi_component(tr_pct, te_pct).sum()),
                "n_bins": len(bin_edges) - 1,
            }
        )
    for col in cat_cols:
        cats = sorted(
            set(X_train[col].dropna().unique()) | set(X_test[col].dropna().unique()),
            key=str,
        )
        if not cats:
            continue
        tr_vc = X_train[col].value_counts()
        te_vc = X_test[col].value_counts()
        tr_pct = np.array([tr_vc.get(c, 0) for c in cats], dtype=float)
        te_pct = np.array([te_vc.get(c, 0) for c in cats], dtype=float)
        if tr_pct.sum() == 0 or te_pct.sum() == 0:
            continue
        tr_pct /= tr_pct.sum()
        te_pct /= te_pct.sum()
        records.append(
            {
                "feature": col,
                "type": "categorical",
                "csi": float(_psi_component(tr_pct, te_pct).sum()),
                "n_bins": len(cats),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["feature", "type", "csi", "n_bins"])
    return df.sort_values("csi", ascending=False).reset_index(drop=True)


def run_stability_analysis(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    num_cols: list[str],
    cat_cols: list[str],
    output_dir: Path,
) -> None:
    psi_records = []
    for name in SUMMARY_MODEL_NAMES:
        if name not in train_scores or name not in test_scores:
            continue
        psi = compute_psi(train_scores[name], test_scores[name])
        psi_records.append({"model": name, "psi": psi})
        flag = "OK" if psi < 0.10 else ("MODERATE" if psi < 0.25 else "HIGH DRIFT")
        logger.info("  PSI {:<25s} = {:.4f}  [{}]", name, psi, flag)
    pd.DataFrame(psi_records).to_csv(output_dir / "psi.csv", index=False, float_format="%.6f")
    csi_df = compute_csi(X_train, X_test, num_cols, cat_cols)
    csi_df.to_csv(output_dir / "csi.csv", index=False, float_format="%.6f")
    n_high = int((csi_df["csi"] >= 0.25).sum())
    n_mod = int(((csi_df["csi"] >= 0.10) & (csi_df["csi"] < 0.25)).sum())
    n_stable = len(csi_df) - n_high - n_mod
    logger.info(
        "  CSI: {} features — {} high drift, {} moderate, {} stable",
        len(csi_df),
        n_high,
        n_mod,
        n_stable,
    )
    if n_high > 0:
        for _, row in csi_df[csi_df["csi"] >= 0.25].iterrows():
            logger.info("    HIGH: {:<30s} CSI={:.4f}", row["feature"], row["csi"])


def compute_woe_iv(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_good = int((y == 0).sum())
    total_bad = int((y == 1).sum())
    if total_good == 0 or total_bad == 0:
        return pd.DataFrame(), pd.DataFrame()
    eps = 1e-6
    woe_records = []
    for col in num_cols:
        valid = X[col].notna()
        x_col = X.loc[valid, col]
        y_col = y.loc[valid]
        col_good = int((y_col == 0).sum())
        col_bad = int((y_col == 1).sum())
        if col_good == 0 or col_bad == 0:
            continue
        try:
            bins = pd.qcut(x_col, q=n_bins, duplicates="drop")
        except Exception:
            continue
        for bin_label in bins.cat.categories:
            mask = bins == bin_label
            n_good = int((y_col[mask] == 0).sum())
            n_bad = int((y_col[mask] == 1).sum())
            dist_good = max(n_good / col_good, eps)
            dist_bad = max(n_bad / col_bad, eps)
            woe = np.log(dist_good / dist_bad)
            woe_records.append(
                {
                    "feature": col,
                    "bin": str(bin_label),
                    "type": "numerical",
                    "n_total": int(mask.sum()),
                    "n_good": n_good,
                    "n_bad": n_bad,
                    "event_rate": n_bad / max(int(mask.sum()), 1),
                    "woe": woe,
                    "iv": (dist_good - dist_bad) * woe,
                }
            )
    for col in cat_cols:
        valid = X[col].notna()
        x_col = X.loc[valid, col]
        y_col = y.loc[valid]
        col_good = int((y_col == 0).sum())
        col_bad = int((y_col == 1).sum())
        if col_good == 0 or col_bad == 0:
            continue
        for cat_val in sorted(x_col.unique(), key=str):
            mask = x_col == cat_val
            n_good = int((y_col[mask] == 0).sum())
            n_bad = int((y_col[mask] == 1).sum())
            dist_good = max(n_good / col_good, eps)
            dist_bad = max(n_bad / col_bad, eps)
            woe = np.log(dist_good / dist_bad)
            woe_records.append(
                {
                    "feature": col,
                    "bin": str(cat_val),
                    "type": "categorical",
                    "n_total": int(mask.sum()),
                    "n_good": n_good,
                    "n_bad": n_bad,
                    "event_rate": n_bad / max(int(mask.sum()), 1),
                    "woe": woe,
                    "iv": (dist_good - dist_bad) * woe,
                }
            )
    woe_df = pd.DataFrame(woe_records)
    if woe_df.empty:
        return woe_df, pd.DataFrame(columns=["feature", "iv"])
    iv_df = (
        woe_df.groupby("feature")["iv"]
        .sum()
        .reset_index()
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )
    return woe_df, iv_df


def main(
    data_path: str = "data/demand_direct.parquet",
    optuna_trials: int = 50,
    output_dir: str = "output",
    feature_discovery_fraction: float = FEATURE_DISCOVERY_FRACTION,
    reject_inference: bool = False,
    enable_experimental_stacking: bool = False,
    population_mode: str = POPULATION_MODE_UNDERWRITING,
):
    _configure_logging()
    _suppress_warnings()
    if population_mode not in {POPULATION_MODE_BOOKED_MONITORING, POPULATION_MODE_UNDERWRITING}:
        raise ValueError(f"Unsupported population_mode: {population_mode}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pipeline_t0 = time.perf_counter()
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║            basel_bad Training Pipeline                 ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(
        "Config: split={}, maturity={}, seed={}, optuna_trials={}, output={}, feature_discovery_fraction={}, experimental_stacking={}, population_mode={}",
        SPLIT_DATE,
        MATURITY_CUTOFF,
        RANDOM_STATE,
        optuna_trials,
        output_dir,
        feature_discovery_fraction,
        enable_experimental_stacking,
        population_mode,
    )
    if population_mode == POPULATION_MODE_UNDERWRITING:
        logger.info("Target population: underwriting-stage loan applications")
        logger.warning(
            "Observed outcomes exist only for booked accounts; booked-only test metrics remain an accepted-population proxy"
        )
        logger.info(
            "Sample definitions: feature discovery=earlier pre-test booked rows with observed outcomes, "
            "development=later pre-test booked training rows, calibration=latest pre-test booked holdout, "
            "proxy test=post-split booked matured rows, applicant scoring=all post-split decisioned applications"
        )
    else:
        logger.info("Target population: booked accounts only")
        logger.info(
            "Sample definitions: feature discovery=earlier pre-test booked rows, "
            "development=later pre-test training rows, calibration=latest pre-test booked holdout, "
            "test=post-split booked matured rows"
        )
    if not reject_inference and not enable_experimental_stacking:
        logger.info("Run mode: OFFICIAL baseline")
    else:
        logger.warning("Run mode: EXPERIMENTAL")
    if reject_inference:
        logger.warning("Reject inference is experimental and excluded from official benchmark comparisons")
        logger.warning(
            "Reject inference ENABLED: multiplier={}, max_ratio={}, weight={}",
            REJECT_MULTIPLIER,
            REJECT_MAX_RATIO,
            REJECT_SAMPLE_WEIGHT,
        )
    if enable_experimental_stacking:
        logger.warning("Stacking is experimental and excluded from official baseline comparisons")

    with _log_step(1, "Load data"):
        if population_mode == POPULATION_MODE_UNDERWRITING or reject_inference:
            df, rejected_df = load_data_with_rejects(data_path)
        else:
            df = load_data(data_path)
            rejected_df = None
        population_summary_df = build_population_summary_df(
            df,
            rejected_df,
            population_mode=population_mode,
        )

    with _log_step(2, "Feature engineering"):
        raw_feature_cols, _, _ = select_features(df)
        df = engineer_features(df)
        if rejected_df is not None:
            rejected_df = engineer_features(rejected_df)
        base_feature_cols_no_interactions, _, _ = select_features(df)
        engineered_feature_cols = [
            feature for feature in base_feature_cols_no_interactions if feature not in raw_feature_cols
        ]

    with _log_step(3, "Feature discovery workflow"):
        feature_discovery_result = _training_features.run_feature_discovery_workflow(
            df=df,
            rejected_df=rejected_df,
            raw_feature_cols=raw_feature_cols,
            engineered_feature_cols=engineered_feature_cols,
            base_feature_cols_no_interactions=base_feature_cols_no_interactions,
            feature_discovery_fraction=feature_discovery_fraction,
            temporal_split_fn=temporal_split,
            resolve_temporal_feature_discovery_cutoff_fn=resolve_temporal_feature_discovery_cutoff,
            temporal_feature_discovery_split_fn=temporal_feature_discovery_split,
            summarize_population_fn=summarize_population,
            log_population_summary_fn=log_population_summary,
            make_temporal_cv_fn=make_temporal_cv,
        )
        df = feature_discovery_result.df
        rejected_df = feature_discovery_result.rejected_df
        interaction_feature_cols = feature_discovery_result.interaction_feature_cols
        interaction_leaderboard_df = feature_discovery_result.interaction_leaderboard_df
        feature_discovery_boundary_df = feature_discovery_result.feature_discovery_boundary_df
        base_feature_cols = feature_discovery_result.base_feature_cols
        base_num_cols = feature_discovery_result.base_num_cols
        base_cat_cols = feature_discovery_result.base_cat_cols
        X_estimation_base = feature_discovery_result.X_estimation_base
        y_estimation = feature_discovery_result.y_estimation
        estimation_dates = feature_discovery_result.estimation_dates
        X_test_base = feature_discovery_result.X_test_base
        y_test = feature_discovery_result.y_test
        benchmark_risk_score_test = feature_discovery_result.benchmark_risk_score_test
        benchmark_score_test = feature_discovery_result.benchmark_score_test
        feature_cols = feature_discovery_result.feature_cols
        num_cols = feature_discovery_result.num_cols
        cat_cols = feature_discovery_result.cat_cols
        rfecv_candidate_feature_cols = feature_discovery_result.rfecv_candidate_feature_cols
        feature_provenance_df = feature_discovery_result.feature_provenance_df

    with _log_step(4, "Freeze feature set & prepare official matrices"):
        X_development_base = X_estimation_base.copy()
        y_development = y_estimation.copy()
        development_dates = estimation_dates.copy()
        benchmark_risk_score_estimation = df.loc[X_estimation_base.index, "risk_score_rf"].copy()
        benchmark_score_estimation = df.loc[X_estimation_base.index, "score_RF"].copy()
        sample_weight = None
        X_augmented_base_for_ablation = None
        y_augmented_for_ablation = None
        augmented_dates_for_ablation = None
        augmented_sample_weight_for_ablation = None
        log_population_summary(
            "Official estimation sample",
            summarize_population(
                y_development,
                development_dates,
                "later pre-test booked rows used for final model estimation after feature freezing",
            ),
        )
        if reject_inference and rejected_df is not None:
            band_stats, bin_edges = compute_score_band_bad_rates(df)
            estimation_start = pd.Timestamp(pd.to_datetime(estimation_dates).min())
            reject_pool = rejected_df[rejected_df["mis_Date"] >= estimation_start].copy()
            reject_labeled = create_reject_pseudo_labels(
                reject_pool,
                band_stats,
                bin_edges,
                n_booked_train=len(X_development_base),
            )
            X_development_base, y_development, sample_weight = augment_training_data(
                X_development_base,
                y_development,
                reject_labeled,
                base_feature_cols,
            )
            development_dates = np.concatenate([development_dates, reject_labeled["mis_Date"].values])
            X_augmented_base_for_ablation = X_development_base.copy()
            y_augmented_for_ablation = y_development.copy()
            augmented_dates_for_ablation = development_dates.copy()
            augmented_sample_weight_for_ablation = sample_weight.copy()
            log_population_summary(
                "Development sample after reject inference",
                summarize_population(
                    y_development,
                    development_dates,
                    "later pre-test development rows used for modeling (booked + pseudo-labeled rejects)",
                    sample_weight=sample_weight,
                ),
            )
        X_development, X_test, _, _, _, _, _ = add_modeling_features(
            X_development_base,
            X_test_base,
            base_feature_cols,
            base_num_cols,
            base_cat_cols,
        )
        X_development = X_development[feature_cols].copy()
        X_test = X_test[feature_cols].copy()
        logger.info(
            "Official matrices with frozen features: {} train rows x {} cols, {} test rows x {} cols",
            len(X_development),
            len(feature_cols),
            len(X_test),
            len(feature_cols),
        )

    preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(num_cols, cat_cols)
    monotone_constraints = build_monotone_constraints(num_cols, cat_cols)

    if sample_weight is not None:
        X_development_fit, X_calibration_holdout, y_development_fit, y_calibration_holdout, w_development_fit, w_calibration_holdout, development_fit_dates, calibration_holdout_dates = temporal_calibration_split(
            X_development,
            y_development,
            development_dates,
            calibration_fraction=CALIBRATION_FRACTION,
            sample_weight=sample_weight,
        )
        calibration_booked_mask = w_calibration_holdout == 1.0
        X_calibration_booked = X_calibration_holdout.loc[calibration_booked_mask]
        y_calibration_booked = y_calibration_holdout.loc[calibration_booked_mask]
        calibration_booked_dates = calibration_holdout_dates[calibration_booked_mask]
        log_population_summary(
            "Calibration holdout",
            summarize_population(
                y_calibration_holdout,
                calibration_holdout_dates,
                "latest pre-test holdout rows reserved from model fitting",
                sample_weight=w_calibration_holdout,
            ),
        )
    else:
        X_development_fit, X_calibration_holdout, y_development_fit, y_calibration_holdout, development_fit_dates, calibration_holdout_dates = temporal_calibration_split(
            X_development,
            y_development,
            development_dates,
            calibration_fraction=CALIBRATION_FRACTION,
        )
        w_development_fit = None
        w_calibration_holdout = None
        X_calibration_booked = X_calibration_holdout
        y_calibration_booked = y_calibration_holdout
        calibration_booked_dates = calibration_holdout_dates

    if len(X_calibration_booked) == 0:
        raise ValueError("Calibration split produced no booked ground-truth rows")

    log_population_summary(
        "Development fit sample",
        summarize_population(
            y_development_fit,
            development_fit_dates,
            "earlier pre-test rows used for model fitting",
            sample_weight=w_development_fit,
        ),
    )
    log_population_summary(
        "Calibration ground-truth sample",
        summarize_population(
            y_calibration_booked,
            calibration_booked_dates,
            "booked subset of the latest pre-test holdout used for calibration",
        ),
    )

    pos_weight = (y_development_fit == 0).sum() / (y_development_fit == 1).sum()
    cv = make_temporal_cv(development_fit_dates)
    fit_start = pd.Timestamp(pd.to_datetime(development_fit_dates).min()).date()
    fit_end = pd.Timestamp(pd.to_datetime(development_fit_dates).max()).date()
    calib_start = pd.Timestamp(pd.to_datetime(calibration_holdout_dates).min()).date()
    calib_end = pd.Timestamp(pd.to_datetime(calibration_holdout_dates).max()).date()
    logger.info(
        "Development/calibration split: {:,} fit [{} to {}] + {:,} holdout [{} to {}] ({:,} booked ground-truth, {:,} pos)  (imbalance {:.0f}:1, temporal CV {} folds)",
        len(X_development_fit),
        fit_start,
        fit_end,
        len(X_calibration_holdout),
        calib_start,
        calib_end,
        len(X_calibration_booked),
        int(y_calibration_booked.sum()),
        pos_weight,
        cv.n_splits,
    )

    with _log_step(7, "Logistic Regression — development fit sample"):
        lr_model, lr_study = train_logistic_regression(
            X_development_fit,
            y_development_fit,
            preprocessor,
            cv,
            optuna_trials,
            sample_weight=w_development_fit,
        )
    with _log_step(8, "LightGBM — development fit sample"):
        lgbm_model, lgbm_study, lgbm_best_n = train_lgbm(
            X_development_fit,
            y_development_fit,
            lgbm_preprocessor,
            lgbm_cat_indices,
            pos_weight,
            cv,
            optuna_trials,
            sample_weight=w_development_fit,
            monotone_constraints=monotone_constraints,
        )
    with _log_step(9, "XGBoost — development fit sample"):
        xgb_model, xgb_study, xgb_best_n = train_xgboost(
            X_development_fit,
            y_development_fit,
            preprocessor,
            pos_weight,
            cv,
            optuna_trials,
            sample_weight=w_development_fit,
            monotone_constraints=monotone_constraints,
        )
    with _log_step(10, "CatBoost — development fit sample"):
        catboost_model, catboost_study, catboost_best_n = train_catboost(
            X_development_fit,
            y_development_fit,
            lgbm_preprocessor,
            pos_weight,
            cv,
            optuna_trials,
            sample_weight=w_development_fit,
            monotone_constraints=monotone_constraints,
        )

    stack_model = None
    if enable_experimental_stacking:
        with _log_step(11, "Stacking ensemble (experimental)"):
            stack_model = train_stacking(
                X_development_fit,
                y_development_fit,
                {
                    "Logistic Regression": lr_model,
                    "LightGBM": lgbm_model,
                    "XGBoost": xgb_model,
                    "CatBoost": catboost_model,
                },
                cv,
                sample_weight=w_development_fit,
            )

    with _log_step(12, "Calibration — booked ground-truth holdout"):
        models = {
            "Logistic Regression": lr_model,
            "LightGBM": lgbm_model,
            "XGBoost": xgb_model,
            "CatBoost": catboost_model,
        }
        if stack_model is not None:
            models[EXPERIMENTAL_STACKING_NAME] = stack_model
        for name in OFFICIAL_MODEL_NAMES:
            cal = CalibratedClassifierCV(FrozenEstimator(models[name]), method="sigmoid")
            cal.fit(X_calibration_booked, y_calibration_booked)
            models[f"{name} (calibrated)"] = cal
        logger.info(
            "Sigmoid (Platt) calibration on {:,} booked held-out samples ({:,} pos)",
            len(y_calibration_booked),
            y_calibration_booked.sum(),
        )

    with _log_step("12b", "Rolling OOT validation — pre-test estimation sample"):
        rolling_base_models = {name: models[name] for name in OFFICIAL_MODEL_NAMES}
        rolling_oot_results_df, rolling_oot_summary_df = run_rolling_out_of_time_validation(
            X_estimation_base,
            y_estimation,
            estimation_dates,
            benchmark_risk_score_estimation,
            benchmark_score_estimation,
            base_feature_cols,
            base_num_cols,
            base_cat_cols,
            feature_cols,
            num_cols,
            cat_cols,
            rolling_base_models,
        )
        if population_mode == POPULATION_MODE_UNDERWRITING:
            if not rolling_oot_results_df.empty:
                rolling_oot_results_df["population_mode"] = population_mode
                rolling_oot_results_df["evaluation_population"] = "booked_proxy"
            if not rolling_oot_summary_df.empty:
                rolling_oot_summary_df["population_mode"] = population_mode
                rolling_oot_summary_df["evaluation_population"] = "booked_proxy"
        if not rolling_oot_summary_df.empty:
            logger.info("Rolling OOT summary:")
            for _, row in rolling_oot_summary_df.iterrows():
                logger.info(
                    "  {:<30s} folds={} mean PR AUC={:.4f} mean AUC={:.4f}",
                    row["Model"],
                    int(row["n_folds"]),
                    row["mean_PR_AUC"],
                    row["mean_ROC_AUC"],
                )

    with _log_step(13, "Evaluation — booked test sample"):
        results_df, test_scores = evaluate_all(
            X_test,
            y_test,
            models,
            benchmark_risk_score_test,
            benchmark_score_test,
        )
        official_results_df, experimental_results_df = split_leaderboard_results(
            results_df,
            reject_inference=reject_inference,
        )
        if population_mode == POPULATION_MODE_UNDERWRITING:
            results_df["population_mode"] = population_mode
            results_df["evaluation_population"] = "booked_proxy"
            official_results_df["population_mode"] = population_mode
            official_results_df["evaluation_population"] = "booked_proxy"
            if not experimental_results_df.empty:
                experimental_results_df["population_mode"] = population_mode
                experimental_results_df["evaluation_population"] = "booked_proxy"
        if not experimental_results_df.empty:
            logger.warning(
                "Experimental rows excluded from primary leaderboard: {}",
                ", ".join(experimental_results_df.index),
            )

    if population_mode == POPULATION_MODE_UNDERWRITING:
        with _log_step("13b", "Score post-split underwriting applications"):
            applicant_scores_df = build_applicant_score_frame(
                df,
                rejected_df,
                X_development_base,
                base_feature_cols,
                base_num_cols,
                base_cat_cols,
                feature_cols,
                models,
            )
            observed_count = int(applicant_scores_df["has_observed_target"].sum()) if not applicant_scores_df.empty else 0
            logger.info(
                "Post-split applicant scoring: {:,} rows ({:,} booked rows with observed outcomes)",
                len(applicant_scores_df),
                observed_count,
            )
    else:
        applicant_scores_df = None

    holdout_scores_df = build_holdout_score_frame(
        y_test,
        test_scores,
        population_mode=population_mode if population_mode == POPULATION_MODE_UNDERWRITING else None,
        evaluation_population="booked_proxy" if population_mode == POPULATION_MODE_UNDERWRITING else None,
    )

    if w_development_fit is not None:
        development_fit_booked_mask = w_development_fit == 1.0
        X_development_fit_booked = X_development_fit.loc[development_fit_booked_mask]
        y_development_fit_booked = y_development_fit.loc[development_fit_booked_mask]
    else:
        X_development_fit_booked = X_development_fit
        y_development_fit_booked = y_development_fit
    train_scores = {}
    for name, mdl in models.items():
        if "(calibrated)" not in name:
            train_scores[name] = mdl.predict_proba(X_development_fit_booked)[:, 1]

    with _log_step(14, "Bootstrap confidence intervals — booked test sample"):
        ci_df = bootstrap_confidence_intervals(y_test.values, test_scores)
        ci_df.to_csv(output_path / "confidence_intervals.csv", float_format="%.6f")
        official_candidate_names = [name for name in official_results_df.index if name not in BENCHMARK_MODEL_NAMES]
        experimental_candidate_names = list(experimental_results_df.index)
        benchmark_comparisons_df = paired_bootstrap_benchmark_comparisons(
            y_test.values,
            test_scores,
            official_candidate_names,
        )
        experimental_benchmark_comparisons_df = paired_bootstrap_benchmark_comparisons(
            y_test.values,
            test_scores,
            experimental_candidate_names,
        )
        if population_mode == POPULATION_MODE_UNDERWRITING:
            if not benchmark_comparisons_df.empty:
                benchmark_comparisons_df["population_mode"] = population_mode
                benchmark_comparisons_df["evaluation_population"] = "booked_proxy"
            if not experimental_benchmark_comparisons_df.empty:
                experimental_benchmark_comparisons_df["population_mode"] = population_mode
                experimental_benchmark_comparisons_df["evaluation_population"] = "booked_proxy"
        logger.info("95% CIs ({:,} stratified bootstrap iterations):", N_BOOTSTRAP)
        for model_name, row in ci_df.iterrows():
            brier_str = (
                f"  Brier={row['Brier']:.4f} [{row['Brier_lo']:.4f}, {row['Brier_hi']:.4f}]"
                if not np.isnan(row["Brier"])
                else ""
            )
            logger.info(
                "  {:<30s} AUC={:.4f} [{:.4f}, {:.4f}]  PR={:.4f} [{:.4f}, {:.4f}]{}",
                model_name,
                row["AUC"],
                row["AUC_lo"],
                row["AUC_hi"],
                row["PR_AUC"],
                row["PR_AUC_lo"],
                row["PR_AUC_hi"],
                brier_str,
            )
        if not benchmark_comparisons_df.empty:
            logger.info(
                "Paired benchmark comparisons: {} official candidate/reference pairs",
                len(benchmark_comparisons_df),
            )
        if not experimental_benchmark_comparisons_df.empty:
            logger.warning(
                "Experimental benchmark comparisons saved separately: {} candidate/reference pairs",
                len(experimental_benchmark_comparisons_df),
            )

    with _log_step("14b", "Phase 3 ablations"):
        ablation_results_df = run_phase3_ablations(
            X_estimation_base,
            y_estimation,
            estimation_dates,
            X_test_base,
            y_test,
            raw_feature_cols,
            engineered_feature_cols,
            interaction_feature_cols,
            base_feature_cols,
            base_num_cols,
            base_cat_cols,
            rfecv_candidate_feature_cols,
            feature_cols,
            num_cols,
            cat_cols,
            X_augmented_base=X_augmented_base_for_ablation,
            y_augmented=y_augmented_for_ablation,
            augmented_dates=augmented_dates_for_ablation,
            augmented_sample_weight=augmented_sample_weight_for_ablation,
        )
        logger.info("Phase 3 ablations completed: {} rows", len(ablation_results_df))
    if population_mode == POPULATION_MODE_UNDERWRITING and not ablation_results_df.empty:
        ablation_results_df["population_mode"] = population_mode
        ablation_results_df["evaluation_population"] = "booked_proxy"

    with _log_step(15, "SHAP explainability"):
        compute_shap_analysis(models, X_test, num_cols, cat_cols, output_path)

    with _log_step(16, "WoE / IV analysis"):
        woe_df, iv_df = compute_woe_iv(X_development_fit, y_development_fit, num_cols, cat_cols)
        woe_df.to_csv(output_path / "woe_detail.csv", index=False, float_format="%.6f")
        iv_df.to_csv(output_path / "iv_summary.csv", index=False, float_format="%.6f")
        logger.info("{} features analyzed ({:,} fit rows)", len(iv_df), len(X_development_fit))
        for _, row in iv_df.head(15).iterrows():
            strength = (
                "Useless"
                if row["iv"] < 0.02
                else "Weak"
                if row["iv"] < 0.1
                else "Medium"
                if row["iv"] < 0.3
                else "Strong"
                if row["iv"] < 0.5
                else "Suspicious"
            )
            logger.info("  {:<35s} IV={:.4f}  [{}]", row["feature"], row["iv"], strength)

    with _log_step(17, "PSI / CSI stability"):
        run_stability_analysis(
            X_development_fit_booked,
            X_test,
            train_scores,
            test_scores,
            num_cols,
            cat_cols,
            output_path,
        )

    with _log_step("17b", "Lift table and threshold analysis"):
        lift_tables = []
        threshold_tables = []
        for name in OFFICIAL_MODEL_NAMES:
            if name in test_scores:
                lift_tables.append(create_lift_table(y_test.values, test_scores[name], name))
                threshold_tables.append(create_threshold_analysis(y_test.values, test_scores[name], name))
        lift_table_df = pd.concat(lift_tables, ignore_index=True) if lift_tables else pd.DataFrame()
        threshold_analysis_df = pd.concat(threshold_tables, ignore_index=True) if threshold_tables else pd.DataFrame()
        if not lift_table_df.empty:
            lift_table_df.to_csv(output_path / "lift_table.csv", index=False, float_format="%.6f")
            logger.info("Lift table: {} rows across {} models", len(lift_table_df), lift_table_df["model"].nunique())
        if not threshold_analysis_df.empty:
            threshold_analysis_df.to_csv(output_path / "threshold_analysis.csv", index=False, float_format="%.6f")
            logger.info(
                "Threshold analysis: {} rows across {} models",
                len(threshold_analysis_df),
                threshold_analysis_df["model"].nunique(),
            )

    with _log_step("17c", "Overfitting diagnostics"):
        overfit_df = compute_overfit_report(
            y_development_fit_booked.values,
            y_test.values,
            train_scores,
            test_scores,
            model_names=OFFICIAL_MODEL_NAMES,
        )
        if not overfit_df.empty:
            overfit_df.to_csv(output_path / "overfit_report.csv", index=False, float_format="%.6f")
            logger.info("Train vs test performance comparison:")
            logger.info(
                "{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}",
                "Model",
                "Train AUC",
                "Test AUC",
                "AUC Δ",
                "Train PR",
                "Test PR",
                "PR Δ",
                "Flag",
            )
            logger.info("{}", "─" * 97)
            for _, row in overfit_df.iterrows():
                logger.info(
                    "{:<25s} {:>10.4f} {:>10.4f} {:>+10.4f} {:>10.4f} {:>10.4f} {:>+10.4f} {:>6s}",
                    row["model"],
                    row["train_auc"],
                    row["test_auc"],
                    row["auc_delta"],
                    row["train_pr_auc"],
                    row["test_pr_auc"],
                    row["pr_auc_delta"],
                    row["overfit_flag"],
                )
            n_flagged = int((overfit_df["overfit_flag"] == "YES").sum())
            if n_flagged > 0:
                logger.warning(
                    "{} model(s) flagged for potential overfitting (AUC or PR AUC delta > 0.03)",
                    n_flagged,
                )
            else:
                logger.info("No models flagged for overfitting")

    with _log_step("17d", "Model selection"):
        model_selection_df = select_best_model(
            official_results_df,
            overfit_df=overfit_df if not overfit_df.empty else None,
            rolling_oot_summary_df=rolling_oot_summary_df if not rolling_oot_summary_df.empty else None,
            candidate_names=OFFICIAL_MODEL_NAMES,
            benchmark_comparisons_df=benchmark_comparisons_df if not benchmark_comparisons_df.empty else None,
        )
        if not model_selection_df.empty:
            model_selection_df.to_csv(output_path / "model_selection.csv", index=False, float_format="%.1f")
            logger.info("Model selection scorecard (weights: discrimination=35%, stability=20%, calibration=15%, generalization=15%, lift=15%):")
            logger.info(
                "{:<25s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s} {:>5s}",
                "Model",
                "Disc.",
                "Calib.",
                "Stab.",
                "Gen.",
                "Lift",
                "Overall",
                "Rec.",
            )
            logger.info("{}", "─" * 87)
            for _, row in model_selection_df.iterrows():
                logger.info(
                    "{:<25s} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>8.1f} {:>5s}",
                    row["model"],
                    row["discrimination_score"],
                    row["calibration_score"],
                    row["stability_score"],
                    row["generalization_score"],
                    row["lift_score"],
                    row["weighted_score"],
                    "<<<" if row["recommended"] else "",
                )
            recommended = model_selection_df.loc[model_selection_df["recommended"]].iloc[0]
            logger.info(
                "Recommended model: {} (weighted score: {:.1f})",
                recommended["model"],
                recommended["weighted_score"],
            )
            compute_shap_analysis(
                models,
                X_test,
                num_cols,
                cat_cols,
                output_path,
                preferred_model_name=str(recommended["model"]),
            )

    with _log_step("17e", "Model governance artifacts"):
        iv_df_for_dict = None
        iv_path = output_path / "iv_summary.csv"
        if iv_path.exists():
            iv_df_for_dict = pd.read_csv(iv_path)
        generate_model_card(
            official_results_df,
            model_selection_df=model_selection_df if not model_selection_df.empty else None,
            overfit_df=overfit_df if not overfit_df.empty else None,
            benchmark_comparisons_df=benchmark_comparisons_df if not benchmark_comparisons_df.empty else None,
            population_summary_df=population_summary_df if not population_summary_df.empty else None,
            feature_provenance_df=feature_provenance_df if not feature_provenance_df.empty else None,
            output_path=output_path,
        )
        generate_variable_dictionary(
            feature_cols,
            num_cols,
            cat_cols,
            feature_provenance_df=feature_provenance_df if not feature_provenance_df.empty else None,
            iv_df=iv_df_for_dict,
            output_path=output_path,
        )
        generate_data_quality_report(
            X_development_fit,
            num_cols,
            cat_cols,
            output_path,
            label="development_fit",
        )
        generate_data_quality_report(
            X_test,
            num_cols,
            cat_cols,
            output_path,
            label="test",
        )

    with _log_step(18, "Save artifacts"):
        feat_imp = extract_feature_importance(models, num_cols, cat_cols)
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_score_distributions(
            y_test.values,
            test_scores,
            plots_dir / "score_dist_test.png",
            title_prefix="Test",
        )
        plot_score_distributions(
            y_development_fit_booked.values,
            train_scores,
            plots_dir / "score_dist_train.png",
            title_prefix="Train",
        )
        save_artifacts(
            models,
            official_results_df,
            feat_imp,
            output_path,
            experimental_results_df=experimental_results_df,
            benchmark_comparisons_df=benchmark_comparisons_df,
            experimental_benchmark_comparisons_df=experimental_benchmark_comparisons_df,
            feature_provenance_df=feature_provenance_df,
            interaction_leaderboard_df=interaction_leaderboard_df,
            feature_discovery_boundary_df=feature_discovery_boundary_df,
            ablation_results_df=ablation_results_df,
            rolling_oot_results_df=rolling_oot_results_df,
            rolling_oot_summary_df=rolling_oot_summary_df,
            population_summary_df=population_summary_df,
            applicant_scores_df=applicant_scores_df,
            holdout_scores_df=holdout_scores_df,
        )

    total_elapsed = time.perf_counter() - pipeline_t0
    logger.info("Pipeline finished in {:.0f}m {:.0f}s", total_elapsed // 60, total_elapsed % 60)

    return models, results_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train basel_bad classifier")
    parser.add_argument("--data-path", default="data/demand_direct.parquet", help="Path to parquet data file")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials per model")
    parser.add_argument("--output-dir", default="output", help="Directory for saved models and artifacts")
    parser.add_argument(
        "--feature-discovery-fraction",
        type=float,
        default=FEATURE_DISCOVERY_FRACTION,
        help="Fraction of pre-test booked rows reserved for feature discovery",
    )
    parser.add_argument("--reject-inference", action="store_true", help="Enable reject inference via score-band parceling")
    parser.add_argument("--enable-experimental-stacking", action="store_true", help="Train and evaluate the experimental stacking ensemble")
    parser.add_argument(
        "--population-mode",
        default=POPULATION_MODE_UNDERWRITING,
        choices=[POPULATION_MODE_UNDERWRITING, POPULATION_MODE_BOOKED_MONITORING],
        help="Population design: underwriting applicant scoring or booked-only monitoring",
    )
    return parser

def cli(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    main(
        data_path=args.data_path,
        optuna_trials=args.optuna_trials,
        output_dir=args.output_dir,
        feature_discovery_fraction=args.feature_discovery_fraction,
        reject_inference=args.reject_inference,
        enable_experimental_stacking=args.enable_experimental_stacking,
        population_mode=args.population_mode,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(cli())
