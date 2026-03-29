from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder

from training_constants import (
    DROP_COLS,
    INTERACTION_SEARCH_TOP_K_CAT,
    INTERACTION_SEARCH_TOP_K_NUM,
    MAX_CATEGORIES,
    MIN_LIFT,
    MIN_VALID,
    MISS_CANDIDATES,
    MONOTONE_MAP,
    RANDOM_STATE,
    RAW_CAT,
    RAW_NUM,
    SPLIT_DATE,
    STABILITY_SELECTION_C_VALUES,
    STABILITY_SELECTION_L1_RATIOS,
    STABILITY_SELECTION_MIN_FEATURES,
    STABILITY_SELECTION_THRESHOLD,
    TARGET,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df.columns)

    df["TOTAL_PRODUCTS"] = df["TOTAL_CARD_NBR"] + df["TOTAL_LOAN_NBR"]

    df["HAS_CODEBTOR"] = df["INCOME_T2"].notna().astype(int)
    df["HOUSEHOLD_INCOME"] = df["INCOME_T1"] + df["INCOME_T2"].fillna(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["CODEBTOR_INCOME_SHARE"] = df["INCOME_T2"] / df["HOUSEHOLD_INCOME"].replace(0, np.nan)

    df["INSTALLMENT_TO_INCOME"] = df["INSTALLMENT_AMT"] / df["INCOME_T1"].replace(0, np.nan)
    df["TOTAL_AMT_TO_INCOME"] = df["TOTAL_AMT"] / df["INCOME_T1"].replace(0, np.nan)
    df["AMT_PER_MONTH"] = df["TOTAL_AMT"] / df["TENOR"].replace(0, np.nan)

    df["INSTALLMENT_TO_HOUSEHOLD"] = df["INSTALLMENT_AMT"] / df["HOUSEHOLD_INCOME"].replace(0, np.nan)
    df["TOTAL_AMT_TO_HOUSEHOLD"] = df["TOTAL_AMT"] / df["HOUSEHOLD_INCOME"].replace(0, np.nan)

    df["CODEBTOR_X_INST_TO_INC"] = df["HAS_CODEBTOR"] * df["INSTALLMENT_TO_INCOME"].fillna(0)
    df["CODEBTOR_X_AMT_TO_INC"] = df["HAS_CODEBTOR"] * df["TOTAL_AMT_TO_INCOME"].fillna(0)
    df["CODEBTOR_X_AMT_PER_MONTH"] = df["HAS_CODEBTOR"] * df["AMT_PER_MONTH"].fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        df["BOOK_RATIO_LOAN"] = df["BOOK_LOAN_NBR"] / df["TOTAL_LOAN_NBR"].replace(0, np.nan)
        df["BOOK_RATIO_CARD"] = df["BOOK_CARD_NBR"] / df["TOTAL_CARD_NBR"].replace(0, np.nan)
    df["HAS_CARDS"] = (df["TOTAL_CARD_NBR"] > 0).astype(int)
    df["HAS_LOANS"] = (df["TOTAL_LOAN_NBR"] > 0).astype(int)

    df["LOG_INCOME_T1"] = np.log1p(df["INCOME_T1"].clip(lower=0))
    df["LOG_TOTAL_AMT"] = np.log1p(df["TOTAL_AMT"].clip(lower=0))
    df["LOG_MAX_CREDIT"] = np.log1p(df["MAX_CREDIT_TJ_AV"].clip(lower=0))

    df["PRODTYPE3_X_HOUSE"] = df["product_type_3"] + "_" + df["HOUSE_TYPE"]
    df["PRODTYPE3_X_CUSTTYPE"] = df["product_type_3"] + "_" + df["CUSTOMER_TYPE"]
    df["CUSTTYPE_X_HOUSE"] = df["CUSTOMER_TYPE"] + "_" + df["HOUSE_TYPE"]

    miss_flags = []
    for col in MISS_CANDIDATES:
        miss_rate = df[col].isna().mean()
        if miss_rate > 0.01:
            flag_name = f"MISS_{col}"
            df[flag_name] = df[col].isna().astype(int)
            miss_flags.append(f"{col} ({miss_rate:.1%})")

    n_added = len(df.columns) - n_before
    logger.info("{} features added  (3 codebtor, 3 afford, 2 household, 3 codebtor_x_afford, 4 portfolio, 3 log, 1 count, 3 cat interact, {} miss flags)", n_added, len(miss_flags))
    if miss_flags:
        logger.info("Missing flags: {}", ", ".join(miss_flags))
    return df


def _loo_target_encode(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out target encoding to avoid in-sample leakage."""
    df_temp = pd.DataFrame({"group": groups, "y": y})
    group_sum = df_temp.groupby("group")["y"].transform("sum")
    group_count = df_temp.groupby("group")["y"].transform("count")
    loo_mean = (group_sum - df_temp["y"]) / (group_count - 1)
    loo_mean = loo_mean.fillna(y.mean())
    return loo_mean.values


def _safe_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(s)
    n = valid.sum()
    if n < MIN_VALID or y[valid].sum() < 10:
        return np.nan, n
    return roc_auc_score(y[valid], s[valid]), n


def _cv_target_encode_auc(
    groups: np.ndarray, y: np.ndarray, n_splits: int = 5,
) -> tuple[float, int]:
    """Pooled leave-one-out target-encoded AUC.

    Used as a fallback when temporal splits are unavailable (too few date
    blocks).  Earlier versions used StratifiedKFold(shuffle=True) here, but
    that introduced random splitting inside a temporal pipeline.  The LOO
    encoder already eliminates in-sample bias without needing CV folds.
    """
    encoded = _loo_target_encode(groups, y)
    return _safe_auc(y, encoded)


def _build_temporal_validation_splits(
    dates,
    min_train_date_blocks: int = 2,
    max_windows: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dates_series = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"))
    unique_dates = pd.Index(np.sort(dates_series.unique()))
    if len(unique_dates) < min_train_date_blocks + 1:
        return []

    validation_blocks = unique_dates[min_train_date_blocks:]
    n_windows = min(max_windows, len(validation_blocks))
    window_groups = [
        pd.Index(group)
        for group in np.array_split(validation_blocks.to_numpy(), n_windows)
        if len(group) > 0
    ]

    dates_array = dates_series.to_numpy()
    splits = []
    for validation_group in window_groups:
        train_dates = unique_dates[unique_dates < validation_group[0]]
        train_idx = np.flatnonzero(np.isin(dates_array, train_dates))
        val_idx = np.flatnonzero(np.isin(dates_array, validation_group))
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        splits.append((train_idx, val_idx))
    return splits


def _temporal_numeric_auc(
    scores: np.ndarray,
    y: np.ndarray,
    dates,
    temporal_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, int]:
    splits = temporal_splits if temporal_splits is not None else _build_temporal_validation_splits(dates)
    if not splits:
        return _safe_auc(y, scores)

    validation_scores = np.full(len(y), np.nan)
    for _, val_idx in splits:
        validation_scores[val_idx] = scores[val_idx]
    return _safe_auc(y, validation_scores)


def _temporal_target_encode_auc(
    groups: np.ndarray,
    y: np.ndarray,
    dates,
    temporal_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, int]:
    splits = temporal_splits if temporal_splits is not None else _build_temporal_validation_splits(dates)
    if not splits:
        return _cv_target_encode_auc(groups, y)

    encoded = np.full(len(y), np.nan)
    for train_idx, val_idx in splits:
        df_fold = pd.DataFrame({"group": groups[train_idx], "y": y[train_idx]})
        means = df_fold.groupby("group")["y"].mean()
        global_mean = y[train_idx].mean()
        encoded[val_idx] = pd.Series(groups[val_idx]).map(means).fillna(global_mean).values
    return _safe_auc(y, encoded)


def normalize_interaction_name(name: str) -> str:
    return name.replace("/", "_DIV_").replace("*", "_X_")


def _apply_binned_numeric_labels(values: np.ndarray, bin_edges) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    bin_labels = np.full(len(values), "missing", dtype=object)
    valid = np.isfinite(values)
    if valid.sum() == 0:
        return bin_labels

    edges = np.unique(np.asarray(bin_edges, dtype=float))
    if len(edges) < 2:
        return bin_labels
    edges[0] = -np.inf
    edges[-1] = np.inf

    binned = pd.cut(values[valid], bins=edges, include_lowest=True)
    bin_labels[valid] = np.asarray(binned.astype(str), dtype=object)
    return bin_labels


def _fit_binned_numeric_labels(
    values: np.ndarray,
    q: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return None, None

    try:
        _, raw_bin_edges = pd.qcut(
            pd.Series(values[valid]),
            q=q,
            retbins=True,
            duplicates="drop",
        )
    except Exception:
        return None, None

    bin_edges = np.unique(np.asarray(raw_bin_edges, dtype=float))
    if len(bin_edges) < 2:
        return None, None
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    return _apply_binned_numeric_labels(values, bin_edges), bin_edges


@dataclass
class InteractionSearchResult:
    selected_interactions: pd.DataFrame
    interaction_leaderboard_df: pd.DataFrame
    interaction_search_summary_df: pd.DataFrame


def search_interactions(
    df: pd.DataFrame,
    end_before_date: str | pd.Timestamp = SPLIT_DATE,
    return_diagnostics: bool = False,
) -> pd.DataFrame | InteractionSearchResult:
    cutoff = pd.Timestamp(end_before_date)
    df_search = df[df["mis_Date"] < cutoff].dropna(subset=[TARGET]).copy()
    df_search[TARGET] = df_search[TARGET].astype(int)
    y_search = df_search[TARGET].values
    dates_search = df_search["mis_Date"].to_numpy()
    temporal_splits = _build_temporal_validation_splits(dates_search)
    numeric_scoring_strategy = "temporal_validation" if temporal_splits else "fallback_pooled_auc"
    categorical_scoring_strategy = "temporal_target_encode" if temporal_splits else "fallback_cv_target_encode"

    logger.info(
        "Search set: {:,} rows ({:,} pos) — raw space {} num + {} cat features",
        len(df_search), y_search.sum(),
        len(RAW_NUM),
        len(RAW_CAT),
    )

    base_auc = {}
    for col in RAW_NUM:
        auc, _ = _temporal_numeric_auc(df_search[col].values, y_search, dates_search, temporal_splits=temporal_splits)
        if not np.isnan(auc):
            base_auc[col] = abs(auc - 0.5)
    for col in RAW_CAT:
        auc, _ = _temporal_target_encode_auc(
            df_search[col].astype(str).values,
            y_search,
            dates_search,
            temporal_splits=temporal_splits,
        )
        if not np.isnan(auc):
            base_auc[col] = abs(auc - 0.5)

    logger.info("Base AUCs: {} features ({} num, {} cat)", len(base_auc),
                 sum(1 for c in RAW_NUM if c in base_auc),
                 sum(1 for c in RAW_CAT if c in base_auc))

    screened_num_cols = [
        col
        for col, _ in sorted(
            ((col, base_auc[col]) for col in RAW_NUM if col in base_auc),
            key=lambda item: item[1],
            reverse=True,
        )[:INTERACTION_SEARCH_TOP_K_NUM]
    ]
    screened_cat_cols = [
        col
        for col, _ in sorted(
            ((col, base_auc[col]) for col in RAW_CAT if col in base_auc),
            key=lambda item: item[1],
            reverse=True,
        )[:INTERACTION_SEARCH_TOP_K_CAT]
    ]
    num_pairs = list(combinations(screened_num_cols, 2))
    cat_pairs = list(combinations(screened_cat_cols, 2))
    logger.info(
        "Interaction gating: {} / {} num features -> {} pairs; {} / {} cat features -> {} pairs",
        len(screened_num_cols), len(RAW_NUM), len(num_pairs),
        len(screened_cat_cols), len(RAW_CAT), len(cat_pairs),
    )

    results = []
    for a, b in tqdm(num_pairs, desc="Num pairs (ratio+product)", leave=False):
        va = df_search[a].values.astype(float)
        vb = df_search[b].values.astype(float)
        feat_a_power = base_auc.get(a, 0)
        feat_b_power = base_auc.get(b, 0)
        parent_power = max(feat_a_power, feat_b_power)

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = va / vb
        auc_r, n_r = _temporal_numeric_auc(ratio, y_search, dates_search, temporal_splits=temporal_splits)
        if not np.isnan(auc_r):
            power = abs(auc_r - 0.5)
            lift = power - parent_power
            results.append({
                "name": f"{a}/{b}", "type": "ratio",
                "auc": auc_r, "lift": lift,
                "feat_a": a, "feat_b": b,
                "feat_a_power": feat_a_power,
                "feat_b_power": feat_b_power,
                "parent_power": parent_power,
                "power": power,
                "selected": lift >= MIN_LIFT,
                "scoring_strategy": numeric_scoring_strategy,
            })

        product = va * vb
        auc_p, n_p = _temporal_numeric_auc(product, y_search, dates_search, temporal_splits=temporal_splits)
        if not np.isnan(auc_p):
            power = abs(auc_p - 0.5)
            lift = power - parent_power
            results.append({
                "name": f"{a}*{b}", "type": "product",
                "auc": auc_p, "lift": lift,
                "feat_a": a, "feat_b": b,
                "feat_a_power": feat_a_power,
                "feat_b_power": feat_b_power,
                "parent_power": parent_power,
                "power": power,
                "selected": lift >= MIN_LIFT,
                "scoring_strategy": numeric_scoring_strategy,
            })

    for a, b in tqdm(cat_pairs, desc="Cat pairs (CV encode)", leave=False):
        combo = (df_search[a].astype(str) + "_" + df_search[b].astype(str)).values
        auc_c, n_c = _temporal_target_encode_auc(combo, y_search, dates_search, temporal_splits=temporal_splits)
        if not np.isnan(auc_c):
            feat_a_power = base_auc.get(a, 0)
            feat_b_power = base_auc.get(b, 0)
            parent_power = max(feat_a_power, feat_b_power)
            power = abs(auc_c - 0.5)
            lift = power - parent_power
            results.append({
                "name": f"{a}_x_{b}", "type": "cat_concat",
                "auc": auc_c, "lift": lift,
                "feat_a": a, "feat_b": b,
                "feat_a_power": feat_a_power,
                "feat_b_power": feat_b_power,
                "parent_power": parent_power,
                "power": power,
                "selected": lift >= MIN_LIFT,
                "scoring_strategy": categorical_scoring_strategy,
            })

    # Binned numerical × categorical interactions (captures threshold effects)
    binned_pairs = [
        (num, cat)
        for num in screened_num_cols
        for cat in screened_cat_cols
    ]
    if binned_pairs:
        logger.info("Screening {} binned num x cat pairs", len(binned_pairs))
    for num, cat in tqdm(binned_pairs, desc="Binned num x cat", leave=False):
        vals = df_search[num].values.astype(float)
        if np.isfinite(vals).sum() < 50:
            continue

        bin_labels, bin_edges = _fit_binned_numeric_labels(vals, q=5)
        if bin_labels is None or bin_edges is None:
            continue
        combo = np.char.add(np.char.add(bin_labels, "_"), df_search[cat].astype(str).values)
        auc_bc, _ = _temporal_target_encode_auc(combo, y_search, dates_search, temporal_splits=temporal_splits)
        if not np.isnan(auc_bc):
            feat_num_power = base_auc.get(num, 0)
            feat_cat_power = base_auc.get(cat, 0)
            parent_power = max(feat_num_power, feat_cat_power)
            power = abs(auc_bc - 0.5)
            lift = power - parent_power
            results.append({
                "name": f"BIN_{num}_x_{cat}", "type": "binned_num_cat",
                "auc": auc_bc, "lift": lift,
                "feat_a": num, "feat_b": cat,
                "feat_a_power": feat_num_power,
                "feat_b_power": feat_cat_power,
                "parent_power": parent_power,
                "power": power,
                "selected": lift >= MIN_LIFT,
                "scoring_strategy": categorical_scoring_strategy,
                "bin_edges": tuple(float(edge) for edge in bin_edges),
            })

    leaderboard_cols = [
        "name", "type", "auc", "lift", "feat_a", "feat_b",
        "feat_a_power", "feat_b_power", "parent_power", "power",
        "selected", "scoring_strategy", "bin_edges",
    ]
    all_results = pd.DataFrame(results, columns=leaderboard_cols)
    selected_results = all_results.loc[all_results["selected"]].copy() if not all_results.empty else all_results.copy()
    if not all_results.empty:
        all_results = all_results.sort_values(["selected", "lift", "auc"], ascending=[False, False, False]).reset_index(drop=True)
        selected_results = all_results.loc[all_results["selected"]].reset_index(drop=True)
        n_ratio = int((selected_results["type"] == "ratio").sum())
        n_product = int((selected_results["type"] == "product").sum())
        n_cat = int((selected_results["type"] == "cat_concat").sum())
        n_binned = int((selected_results["type"] == "binned_num_cat").sum())
        logger.info(
            "Found {} interactions (>= {:.0%} lift): {} ratio, {} product, {} cat, {} binned",
            len(selected_results), MIN_LIFT, n_ratio, n_product, n_cat, n_binned,
        )
        top = selected_results.head(5)
        for _, r in top.iterrows():
            logger.info("  {:<40s}  AUC={:.4f}  lift={:+.4f}", r["name"], r["auc"], r["lift"])
    else:
        logger.info("No interactions found with >= {:.0%} lift", MIN_LIFT)

    interaction_search_summary_df = pd.DataFrame([
        {
            "interaction_search_cutoff": cutoff.date().isoformat(),
            "search_rows": len(df_search),
            "search_positives": int(y_search.sum()),
            "numeric_scoring_strategy": numeric_scoring_strategy,
            "categorical_scoring_strategy": categorical_scoring_strategy,
            "raw_num_features": len(RAW_NUM),
            "raw_cat_features": len(RAW_CAT),
            "screened_num_features": len(screened_num_cols),
            "screened_cat_features": len(screened_cat_cols),
            "screened_num_pairs": len(num_pairs),
            "screened_cat_pairs": len(cat_pairs),
            "scored_candidates": len(all_results),
            "selected_interactions": len(selected_results),
        }
    ])

    if return_diagnostics:
        return InteractionSearchResult(
            selected_interactions=selected_results,
            interaction_leaderboard_df=all_results,
            interaction_search_summary_df=interaction_search_summary_df,
        )
    return selected_results


def add_interactions(df: pd.DataFrame, all_results: pd.DataFrame) -> pd.DataFrame:
    existing = set(df.columns)
    added = []

    for _, row in all_results.iterrows():
        col_name = normalize_interaction_name(row["name"])
        if col_name in existing:
            continue

        a, b = row["feat_a"], row["feat_b"]
        if row["type"] == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                df[col_name] = df[a] / df[b].replace(0, np.nan)
        elif row["type"] == "product":
            df[col_name] = df[a] * df[b]
        elif row["type"] == "cat_concat":
            df[col_name] = df[a].astype(str) + "_" + df[b].astype(str)
        elif row["type"] == "binned_num_cat":
            vals = df[a].values.astype(float)
            stored_edges = row.get("bin_edges", pd.NA)
            if isinstance(stored_edges, str) and stored_edges.strip():
                try:
                    stored_edges = ast.literal_eval(stored_edges)
                except (SyntaxError, ValueError):
                    stored_edges = pd.NA

            bin_edges = None
            if isinstance(stored_edges, (list, tuple, np.ndarray)):
                parsed_edges = np.asarray(stored_edges, dtype=float)
                if parsed_edges.ndim == 1 and len(parsed_edges) >= 2:
                    bin_edges = parsed_edges

            if bin_edges is not None:
                bin_labels = _apply_binned_numeric_labels(vals, bin_edges)
            else:
                bin_labels, _ = _fit_binned_numeric_labels(vals, q=5)
                if bin_labels is None:
                    bin_labels = np.full(len(vals), "missing", dtype=object)

            df[col_name] = np.char.add(np.char.add(bin_labels, "_"), df[b].astype(str).values)
        added.append(col_name)
        existing.add(col_name)

    logger.info("Added {} interaction columns -> {} total cols", len(added), len(df.columns))
    return df


def select_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or df[c].dtype.name == "str"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    logger.info("Features — {} numerical, {} categorical, {} total", len(num_cols), len(cat_cols), len(feature_cols))
    return feature_cols, num_cols, cat_cols


def reduce_cardinality(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    cardinality_maps = {}
    reduced = []

    for col in cat_cols:
        top_cats = X_train[col].value_counts().nlargest(MAX_CATEGORIES).index
        cardinality_maps[col] = set(top_cats)
        n_before = X_train[col].nunique()
        X_train[col] = X_train[col].where(X_train[col].isin(top_cats), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top_cats), "Other")
        n_after = X_train[col].nunique()
        if n_before != n_after:
            reduced.append(f"{col} ({n_before}->{n_after})")

    if reduced:
        logger.info("Capped {} cat features to max {} levels: {}", len(reduced), MAX_CATEGORIES, ", ".join(reduced))
    else:
        logger.info("All {} cat features already <= {} levels", len(cat_cols), MAX_CATEGORIES)
    return X_train, X_test, cardinality_maps


GROUP_STAT_PAIRS = [
    ("INCOME_T1", "CSP"),
    ("INCOME_T1", "product_type_3"),
    ("TOTAL_AMT", "product_type_3"),
    ("MAX_CREDIT_TJ_AV", "HOUSE_TYPE"),
    ("INSTALLMENT_AMT", "CSP"),
    ("AGE_T1", "HOUSE_TYPE"),
]


def add_frequency_encoding(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Frequency encoding: training-set category proportions as numerical features."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    added = []
    for col in cat_cols:
        freq = X_train[col].value_counts(normalize=True)
        col_name = f"FREQ_{col}"
        X_train[col_name] = X_train[col].map(freq).fillna(0).astype(float)
        X_test[col_name] = X_test[col].map(freq).fillna(0).astype(float)
        added.append(col_name)
    logger.info("Added {} frequency-encoded features", len(added))
    return X_train, X_test, added


def add_group_stats(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    num_cols: list[str], cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Group statistics: individual value / group median (train stats only)."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    added = []
    for num, cat in GROUP_STAT_PAIRS:
        if num not in num_cols or cat not in cat_cols:
            continue
        col_name = f"{num}_VS_{cat}"
        group_med = X_train.groupby(cat)[num].median()
        global_med = X_train[num].median()
        if global_med == 0 or np.isnan(global_med):
            global_med = 1e-8

        train_med = X_train[cat].map(group_med).fillna(global_med)
        test_med = X_test[cat].map(group_med).fillna(global_med)

        with np.errstate(divide="ignore", invalid="ignore"):
            X_train[col_name] = X_train[num] / train_med.replace(0, np.nan)
            X_test[col_name] = X_test[num] / test_med.replace(0, np.nan)
        added.append(col_name)
    logger.info("Added {} group-statistic features: {}", len(added), ", ".join(added))
    return X_train, X_test, added


def prune_correlated(
    X_train: pd.DataFrame, num_cols: list[str], threshold: float = 0.95,
) -> list[str]:
    """Identify near-duplicate numerical features by pairwise correlation."""
    if len(num_cols) < 2:
        return []
    corr = X_train[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        logger.info("Pruning {} correlated features (|r| > {:.2f}): {}",
                    len(to_drop), threshold, ", ".join(to_drop))
    else:
        logger.info("No features exceed correlation threshold {:.2f}", threshold)
    return to_drop


def add_modeling_features(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], list[str], list[str]]:
    X_train_out, X_other_out, _ = reduce_cardinality(X_train, X_other, cat_cols)
    X_train_out, X_other_out, freq_cols = add_frequency_encoding(X_train_out, X_other_out, cat_cols)
    X_train_out, X_other_out, group_cols = add_group_stats(X_train_out, X_other_out, num_cols, cat_cols)
    new_num = freq_cols + group_cols
    return (
        X_train_out,
        X_other_out,
        feature_cols + new_num,
        num_cols + new_num,
        cat_cols,
        freq_cols,
        group_cols,
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
    raw_set = set(raw_feature_cols)
    engineered_set = set(engineered_feature_cols)
    freq_set = set(freq_cols)
    group_set = set(group_cols)
    num_set = set(feature_space_num_cols)
    cat_set = set(feature_space_cat_cols)
    candidate_set = set(rfecv_candidate_cols)
    kept_set = set(rfecv_kept_cols)
    interaction_meta = {
        normalize_interaction_name(row["name"]): {
            "interaction_type": row["type"],
            "feat_a": row["feat_a"],
            "feat_b": row["feat_b"],
        }
        for _, row in interactions.iterrows()
    }

    ordered_features = list(
        dict.fromkeys(
            raw_feature_cols
            + engineered_feature_cols
            + list(interaction_meta)
            + freq_cols
            + group_cols
        )
    )

    records = []
    for feature in ordered_features:
        interaction_info = interaction_meta.get(feature, {})
        if feature in interaction_meta:
            provenance = "interaction"
        elif feature in group_set:
            provenance = "group_stat"
        elif feature in freq_set:
            provenance = "frequency"
        elif feature in engineered_set:
            provenance = "engineered"
        elif feature in raw_set:
            provenance = "raw"
        else:
            provenance = "unknown"

        if feature in num_set:
            data_type = "numerical"
        elif feature in cat_set:
            data_type = "categorical"
        else:
            data_type = pd.NA

        records.append(
            {
                "feature": feature,
                "provenance": provenance,
                "data_type": data_type,
                "rfecv_candidate": feature in candidate_set,
                "rfecv_kept": feature in kept_set,
                "interaction_type": interaction_info.get("interaction_type", pd.NA),
                "feat_a": interaction_info.get("feat_a", pd.NA),
                "feat_b": interaction_info.get("feat_b", pd.NA),
            }
        )

    return pd.DataFrame(records).sort_values(["provenance", "feature"]).reset_index(drop=True)


def build_preprocessors(num_cols: list[str], cat_cols: list[str], target_encoder_smooth="auto"):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # Keep TargetEncoder deterministic and non-shuffled to preserve temporal discipline.
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", TargetEncoder(smooth=target_encoder_smooth, cv=5, shuffle=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    lgbm_num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    lgbm_cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    lgbm_preprocessor = ColumnTransformer([
        ("num", lgbm_num_transformer, num_cols),
        ("cat", lgbm_cat_transformer, cat_cols),
    ])
    lgbm_cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    return preprocessor, lgbm_preprocessor, lgbm_cat_indices


def build_monotone_constraints(num_cols: list[str], cat_cols: list[str]) -> list[int]:
    """Monotonicity constraint vector matching ColumnTransformer output order."""
    constraints = [MONOTONE_MAP.get(c, 0) for c in num_cols]
    constraints.extend([0] * len(cat_cols))
    n_neg = sum(1 for c in constraints if c == -1)
    n_pos = sum(1 for c in constraints if c == 1)
    logger.info("Monotone constraints: {}/{} features constrained ({} neg, {} pos)",
                n_neg + n_pos, len(constraints), n_neg, n_pos)
    return constraints


def _safe_average_precision(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> float:
    y_array = np.asarray(y_true)
    if len(np.unique(y_array)) < 2:
        return np.nan
    return average_precision_score(y_array, scores)


def _build_stability_selection_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
    target_encoder_cv: int,
) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", TargetEncoder(smooth="auto", cv=target_encoder_cv, shuffle=False)),
        ]), cat_cols),
    ])


def _fit_stability_selection_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[LogisticRegression | None, float, tuple[float, float] | None]:
    min_class_count = int(y_fit.value_counts().min())
    if min_class_count < 2:
        return None, np.nan, None

    target_encoder_cv = max(2, min(5, min_class_count))
    selector_preprocessor = _build_stability_selection_preprocessor(num_cols, cat_cols, target_encoder_cv)
    X_fit_prepared = selector_preprocessor.fit_transform(X_fit, y_fit)
    X_val_prepared = selector_preprocessor.transform(X_val)

    best_model = None
    best_score = -np.inf
    best_params = None
    best_n_selected = np.inf
    for c_value in STABILITY_SELECTION_C_VALUES:
        for l1_ratio in STABILITY_SELECTION_L1_RATIOS:
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                C=c_value,
                l1_ratio=l1_ratio,
                class_weight="balanced",
                max_iter=5000,
                tol=1e-3,
                random_state=RANDOM_STATE,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(X_fit_prepared, y_fit)
            score = _safe_average_precision(y_val, model.predict_proba(X_val_prepared)[:, 1])
            if np.isnan(score):
                continue
            n_selected = int((np.abs(model.coef_).ravel() > 1e-8).sum())
            if (
                score > best_score + 1e-8
                or (abs(score - best_score) <= 1e-8 and n_selected < best_n_selected)
            ):
                best_model = model
                best_score = score
                best_params = (c_value, l1_ratio)
                best_n_selected = n_selected

    return best_model, best_score, best_params


def run_rfecv(
    X_train: pd.DataFrame, y_train: pd.Series,
    num_cols: list[str], cat_cols: list[str],
    feature_cols: list[str],
    cv,
):
    all_feature_names = num_cols + cat_cols
    logger.info(
        "Temporal stability selection: {} candidate features ({} num, {} cat, {} folds)",
        len(all_feature_names),
        len(num_cols),
        len(cat_cols),
        cv.get_n_splits() if hasattr(cv, "get_n_splits") else "?",
    )
    if not all_feature_names:
        return [], [], []

    selection_counts = pd.Series(0.0, index=all_feature_names)
    coefficient_sums = pd.Series(0.0, index=all_feature_names)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        X_fit = X_train.iloc[train_idx][all_feature_names].copy()
        y_fit = y_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx][all_feature_names].copy()
        y_val = y_train.iloc[val_idx].copy()
        if y_fit.nunique() < 2 or y_val.nunique() < 2:
            logger.warning(
                "Stability selection fold {} skipped due to insufficient class diversity (train uniques={}, val uniques={})",
                fold_idx,
                y_fit.nunique(),
                y_val.nunique(),
            )
            continue

        best_model, best_score, best_params = _fit_stability_selection_fold(
            X_fit,
            y_fit,
            X_val,
            y_val,
            num_cols,
            cat_cols,
        )
        if best_model is None or best_params is None or np.isnan(best_score):
            logger.warning("Stability selection fold {} produced no valid elastic-net model", fold_idx)
            continue

        coefficient_abs = np.abs(best_model.coef_).ravel()
        selected_mask = coefficient_abs > 1e-8
        selection_counts += selected_mask.astype(float)
        coefficient_sums += coefficient_abs
        fold_scores.append(best_score)
        logger.info(
            "  Fold {} best AP={:.4f} with C={} l1_ratio={} selecting {} features",
            fold_idx,
            best_score,
            best_params[0],
            best_params[1],
            int(selected_mask.sum()),
        )

    valid_folds = len(fold_scores)
    if valid_folds == 0:
        logger.warning("Temporal stability selection had no valid folds; retaining all {} candidate features", len(all_feature_names))
        return feature_cols, num_cols, cat_cols

    selection_frequency = selection_counts / valid_folds
    mean_abs_coef = coefficient_sums / valid_folds
    selected_set = {
        feature
        for feature in all_feature_names
        if selection_frequency.loc[feature] >= STABILITY_SELECTION_THRESHOLD
    }

    min_features = min(STABILITY_SELECTION_MIN_FEATURES, len(all_feature_names))
    if len(selected_set) < min_features:
        ranked_features = sorted(
            all_feature_names,
            key=lambda feature: (selection_frequency.loc[feature], mean_abs_coef.loc[feature]),
            reverse=True,
        )
        selected_set = set(ranked_features[:min_features])

    eliminated = [feature for feature in all_feature_names if feature not in selected_set]
    logger.info(
        "Temporal stability selection kept {} / {} features across {} valid folds (threshold {:.0%}, mean fold AP {:.4f})",
        len(selected_set),
        len(all_feature_names),
        valid_folds,
        STABILITY_SELECTION_THRESHOLD,
        float(np.mean(fold_scores)),
    )
    if eliminated:
        logger.info("Dropped {}: {}", len(eliminated), ", ".join(eliminated))

    num_cols = [c for c in num_cols if c in selected_set]
    cat_cols = [c for c in cat_cols if c in selected_set]
    feature_cols = [c for c in feature_cols if c in selected_set]

    return feature_cols, num_cols, cat_cols


@dataclass
class FeatureDiscoveryResult:
    df: pd.DataFrame
    rejected_df: pd.DataFrame | None
    interaction_feature_cols: list[str]
    interaction_leaderboard_df: pd.DataFrame
    feature_discovery_boundary_df: pd.DataFrame
    base_feature_cols: list[str]
    base_num_cols: list[str]
    base_cat_cols: list[str]
    X_estimation_base: pd.DataFrame
    y_estimation: pd.Series
    estimation_dates: np.ndarray
    X_test_base: pd.DataFrame
    y_test: pd.Series
    benchmark_risk_score_test: pd.Series
    benchmark_score_test: pd.Series
    feature_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    rfecv_candidate_feature_cols: list[str]
    feature_provenance_df: pd.DataFrame


def run_feature_discovery_workflow(
    *,
    df: pd.DataFrame,
    rejected_df: pd.DataFrame | None,
    raw_feature_cols: list[str],
    engineered_feature_cols: list[str],
    base_feature_cols_no_interactions: list[str],
    feature_discovery_fraction: float,
    temporal_split_fn: Callable[..., tuple[object, ...]],
    resolve_temporal_feature_discovery_cutoff_fn: Callable[..., pd.Timestamp],
    temporal_feature_discovery_split_fn: Callable[..., tuple[object, ...]],
    summarize_population_fn: Callable[..., dict],
    log_population_summary_fn: Callable[..., None],
    make_temporal_cv_fn: Callable[..., object],
    search_interactions_fn: Callable[..., object] = search_interactions,
    add_interactions_fn: Callable[..., pd.DataFrame] = add_interactions,
    select_features_fn: Callable[..., tuple[list[str], list[str], list[str]]] = select_features,
    add_modeling_features_fn: Callable[..., tuple[object, ...]] = add_modeling_features,
    prune_correlated_fn: Callable[..., list[str]] = prune_correlated,
    run_rfecv_fn: Callable[..., tuple[list[str], list[str], list[str]]] = run_rfecv,
    build_feature_provenance_fn: Callable[..., pd.DataFrame] = build_feature_provenance,
    normalize_interaction_name_fn: Callable[[str], str] = normalize_interaction_name,
) -> FeatureDiscoveryResult:
    X_discovery_source, y_discovery_source, _, _, _, _, discovery_source_dates = temporal_split_fn(
        df, base_feature_cols_no_interactions,
    )
    feature_discovery_end = resolve_temporal_feature_discovery_cutoff_fn(
        discovery_source_dates,
        discovery_fraction=feature_discovery_fraction,
    )
    X_feature_discovery_seed, X_feature_estimation_seed, y_feature_discovery_seed, y_feature_estimation_seed, feature_discovery_seed_dates, feature_estimation_seed_dates = temporal_feature_discovery_split_fn(
        X_discovery_source,
        y_discovery_source,
        discovery_source_dates,
        discovery_end=feature_discovery_end,
    )
    feature_discovery_seed_index = X_feature_discovery_seed.index.copy()
    feature_estimation_seed_index = X_feature_estimation_seed.index.copy()
    interaction_search_cutoff = pd.Timestamp(pd.to_datetime(feature_estimation_seed_dates).min())
    logger.info(
        "Feature discovery boundary: discovery_end={}, estimation_start={}, discovery_fraction={:.0%}",
        feature_discovery_end.date(),
        interaction_search_cutoff.date(),
        feature_discovery_fraction,
    )
    log_population_summary_fn(
        "Feature discovery sample",
        summarize_population_fn(
            y_feature_discovery_seed,
            feature_discovery_seed_dates,
            "earlier pre-test booked rows reserved for interaction search and RFECV",
        ),
    )
    log_population_summary_fn(
        "Feature estimation seed sample",
        summarize_population_fn(
            y_feature_estimation_seed,
            feature_estimation_seed_dates,
            "later pre-test booked rows reserved for final estimation after feature freezing",
        ),
    )

    interaction_search_result = search_interactions_fn(
        df,
        end_before_date=interaction_search_cutoff,
        return_diagnostics=True,
    )
    if isinstance(interaction_search_result, InteractionSearchResult):
        interactions = interaction_search_result.selected_interactions
        interaction_leaderboard_df = interaction_search_result.interaction_leaderboard_df
        interaction_search_summary_df = interaction_search_result.interaction_search_summary_df
    else:
        interactions = interaction_search_result
        interaction_leaderboard_df = interactions.copy()
        if not interaction_leaderboard_df.empty and "selected" not in interaction_leaderboard_df.columns:
            interaction_leaderboard_df["selected"] = True
        interaction_search_summary_df = pd.DataFrame([
            {
                "interaction_search_cutoff": interaction_search_cutoff.date().isoformat(),
                "search_rows": len(df[df["mis_Date"] < interaction_search_cutoff].dropna(subset=[TARGET])),
                "search_positives": int(df.loc[(df["mis_Date"] < interaction_search_cutoff) & df[TARGET].notna(), TARGET].astype(int).sum()),
                "numeric_scoring_strategy": pd.NA,
                "categorical_scoring_strategy": pd.NA,
                "raw_num_features": len(RAW_NUM),
                "raw_cat_features": len(RAW_CAT),
                "screened_num_features": pd.NA,
                "screened_cat_features": pd.NA,
                "screened_num_pairs": pd.NA,
                "screened_cat_pairs": pd.NA,
                "scored_candidates": len(interaction_leaderboard_df),
                "selected_interactions": len(interactions),
            }
        ])
    interaction_feature_cols = (
        [normalize_interaction_name_fn(name) for name in interactions["name"].tolist()]
        if not interactions.empty
        else []
    )
    df = add_interactions_fn(df, interactions)
    if rejected_df is not None:
        rejected_df = add_interactions_fn(rejected_df, interactions)

    base_feature_cols, base_num_cols, base_cat_cols = select_features_fn(df)
    X_booked_development_base, y_booked_development, X_test_base, y_test, benchmark_risk_score_test, benchmark_score_test, booked_development_dates = temporal_split_fn(
        df, base_feature_cols,
    )
    test_dates = df.loc[X_test_base.index, "mis_Date"].values
    log_population_summary_fn(
        "Development sample",
        summarize_population_fn(y_booked_development, booked_development_dates, "pre-test booked matured rows"),
    )
    log_population_summary_fn(
        "Test sample",
        summarize_population_fn(y_test, test_dates, "post-split booked matured rows"),
    )

    feature_discovery_boundary_record = {
        "feature_discovery_fraction": feature_discovery_fraction,
        "feature_discovery_end": feature_discovery_end.date().isoformat(),
        "interaction_search_cutoff": interaction_search_cutoff.date().isoformat(),
        "discovery_seed_rows": len(X_feature_discovery_seed),
        "discovery_seed_positives": int(y_feature_discovery_seed.sum()),
        "estimation_seed_rows": len(X_feature_estimation_seed),
        "estimation_seed_positives": int(y_feature_estimation_seed.sum()),
    }
    if not interaction_search_summary_df.empty:
        feature_discovery_boundary_record.update(interaction_search_summary_df.iloc[0].to_dict())
    feature_discovery_boundary_df = pd.DataFrame([feature_discovery_boundary_record])

    missing_discovery_rows = feature_discovery_seed_index.difference(X_booked_development_base.index)
    missing_estimation_rows = feature_estimation_seed_index.difference(X_booked_development_base.index)
    if not missing_discovery_rows.empty or not missing_estimation_rows.empty:
        raise ValueError("Feature discovery partition drifted after interaction expansion")
    X_feature_discovery_base = X_booked_development_base.loc[feature_discovery_seed_index].copy()
    X_estimation_base = X_booked_development_base.loc[feature_estimation_seed_index].copy()
    y_feature_discovery = y_booked_development.loc[feature_discovery_seed_index].copy()
    y_estimation = y_booked_development.loc[feature_estimation_seed_index].copy()
    feature_discovery_dates = feature_discovery_seed_dates.copy()
    estimation_dates = feature_estimation_seed_dates.copy()

    X_feature_discovery_space, _, feature_space_cols, feature_space_num_cols, feature_space_cat_cols, freq_cols, group_cols = add_modeling_features_fn(
        X_feature_discovery_base,
        X_estimation_base,
        base_feature_cols,
        base_num_cols,
        base_cat_cols,
    )
    corr_drop = prune_correlated_fn(X_feature_discovery_space, feature_space_num_cols)
    rfecv_candidate_feature_cols = [feature for feature in feature_space_cols if feature not in corr_drop]
    rfecv_candidate_num_cols = [feature for feature in feature_space_num_cols if feature not in corr_drop]
    rfecv_candidate_cat_cols = [feature for feature in feature_space_cat_cols if feature in rfecv_candidate_feature_cols]
    rfe_cv = make_temporal_cv_fn(feature_discovery_dates)
    feature_cols, num_cols, cat_cols = run_rfecv_fn(
        X_feature_discovery_space[rfecv_candidate_feature_cols],
        y_feature_discovery,
        rfecv_candidate_num_cols,
        rfecv_candidate_cat_cols,
        rfecv_candidate_feature_cols,
        cv=rfe_cv,
    )
    feature_provenance_df = build_feature_provenance_fn(
        raw_feature_cols,
        engineered_feature_cols,
        interactions,
        freq_cols,
        group_cols,
        feature_space_num_cols,
        feature_space_cat_cols,
        rfecv_candidate_feature_cols,
        feature_cols,
    )
    logger.info("Frozen feature set: {} num + {} cat = {}", len(num_cols), len(cat_cols), len(feature_cols))

    return FeatureDiscoveryResult(
        df=df,
        rejected_df=rejected_df,
        interaction_feature_cols=interaction_feature_cols,
        interaction_leaderboard_df=interaction_leaderboard_df,
        feature_discovery_boundary_df=feature_discovery_boundary_df,
        base_feature_cols=base_feature_cols,
        base_num_cols=base_num_cols,
        base_cat_cols=base_cat_cols,
        X_estimation_base=X_estimation_base,
        y_estimation=y_estimation,
        estimation_dates=estimation_dates,
        X_test_base=X_test_base,
        y_test=y_test,
        benchmark_risk_score_test=benchmark_risk_score_test,
        benchmark_score_test=benchmark_score_test,
        feature_cols=feature_cols,
        num_cols=num_cols,
        cat_cols=cat_cols,
        rfecv_candidate_feature_cols=rfecv_candidate_feature_cols,
        feature_provenance_df=feature_provenance_df,
    )
