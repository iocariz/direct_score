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
from typing import Any, TypedDict
from contextlib import contextmanager
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
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier

from training_constants import (
    BENCHMARK_MODEL_NAMES,
    CALIBRATION_FRACTION,
    CALIBRATION_METHOD_BY_MODEL,
    CONCEPT_DRIFT_DELTA_THRESHOLD,
    DEFAULT_CALIBRATION_METHOD,
    DEFAULT_TIER_THRESHOLDS,
    DROP_COLS,
    EARLY_STOPPING_ROUNDS,
    EXPERIMENTAL_STACKING_NAME,
    FEATURE_DISCOVERY_FRACTION,
    MATURITY_CUTOFF,
    MAX_CATEGORIES,
    MIN_LIFT,
    MIN_VALID,
    MISS_CANDIDATES,
    MODEL_SELECTION_DATE,
    MONOTONE_MAP,
    N_BOOTSTRAP,
    N_ESTIMATORS_CEILING,
    OFFICIAL_MODEL_NAMES,
    OVERFIT_DELTA_THRESHOLD,
    POPULATION_MODE_BOOKED_MONITORING,
    POPULATION_MODE_UNDERWRITING,
    PSI_HIGH_DRIFT_THRESHOLD,
    PSI_MODERATE_THRESHOLD,
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
    compute_adverse_impact_analysis,
    compute_concept_drift_report,
    compute_overfit_report,
    compute_population_ks_test,
    compute_selection_bias_correlation,
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
    train_post_hoc_ensemble,
)
from training_models import (
    EnsembleModel,
    normalize_estimator_count,
    normalize_xgboost_monotone_constraints,
    safe_stratified_n_splits,
    save_optuna_study as _save_optuna_study,
    select_conservative_boosting_rounds,
    train_catboost,
    train_ebm,
    train_lgbm,
    train_logistic_regression,
    train_xgboost,
)
from training_reject_inference import (
    augment_training_data,
    compute_score_band_bad_rates,
    create_reject_pseudo_labels,
)
from training_stacking import (
    TemporalStackingClassifier,
    build_fresh_pipeline_from_fitted,
    compute_temporal_oof_scores,
    fit_pipeline_from_template,
    train_stacking,
)
from training_temporal import (
    TemporalExpandingCV,
    build_rolling_oot_windows,
    make_temporal_cv,
    resolve_temporal_feature_discovery_cutoff,
    temporal_calibration_split,
    temporal_feature_discovery_split,
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
    check_calibration_holdout_size,
    enforce_matured_target,
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


# EnsembleModel and _save_optuna_study live in training_models.py and are
# imported (and re-exported) below. _save_optuna_study is exposed under its
# legacy private name for backwards compatibility with internal callers.


# ── Temporal CV ────────────────────────────────────────────────────────────────
# TemporalExpandingCV, temporal_calibration_split, resolve_temporal_feature_discovery_cutoff,
# temporal_feature_discovery_split live in training_temporal.py and are imported below.


# ── Data loading ───────────────────────────────────────────────────────────────

def _sort_by_mis_date(df: pd.DataFrame) -> pd.DataFrame:
    """Defensive temporal ordering at the data-load boundary.

    Several downstream operations are stable under row order today
    (TemporalExpandingCV uses dates directly; add_modeling_features uses
    .map / .loc with index alignment), but they would be fragile to an
    upstream parquet that re-ordered rows. Sorting once at the load
    boundary makes the temporal contract explicit and reproducible.
    Stable sort preserves intra-date order so the artifact is byte-stable
    when the input is.
    """
    return df.sort_values("mis_Date", kind="stable").reset_index(drop=True)


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
    df = _sort_by_mis_date(df)

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

    booked_df = _sort_by_mis_date(df[df["status_name"] == "Booked"].copy())
    rejected_df = _sort_by_mis_date(
        df[df["status_name"].isin(["Rejected", "Canceled"])].copy()
    )

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


# ── Feature selection & split ──────────────────────────────────────────────────
# engineer_features, search_interactions, add_interactions, select_features
# are imported directly from training_features above — no wrappers needed.


def temporal_split(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series, np.ndarray]:
    """Train/test split by mis_Date. Returns:
    (X_train, y_train, X_test, y_test, bench_risk_score_rf, bench_score_RF, train_dates).

    The maturity invariant (basel_bad requires 12 months on book) is enforced
    by enforce_matured_target, which also raises if upstream ever populates
    immature targets — see training_features.enforce_matured_target.
    """
    df_model = enforce_matured_target(df)
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


def split_holdout_for_model_selection(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    bench_risk_score_test: pd.Series,
    bench_score_test: pd.Series,
    test_dates: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split post-split holdout into selection and untouched final windows."""
    test_dates_ts = pd.to_datetime(np.asarray(test_dates), errors="raise")
    selection_mask = test_dates_ts < pd.Timestamp(MODEL_SELECTION_DATE)
    final_mask = ~selection_mask
    if not selection_mask.any() or not final_mask.any():
        raise ValueError(
            "Model-selection split produced an empty partition; "
            "adjust MODEL_SELECTION_DATE or data date coverage."
        )

    return (
        X_test.loc[selection_mask],
        y_test.loc[selection_mask],
        bench_risk_score_test.loc[selection_mask],
        bench_score_test.loc[selection_mask],
        X_test.loc[final_mask],
        y_test.loc[final_mask],
        bench_risk_score_test.loc[final_mask],
        bench_score_test.loc[final_mask],
    )


class PreparedStageResult(TypedDict):
    df: pd.DataFrame
    rejected_df: pd.DataFrame | None
    population_summary_df: pd.DataFrame
    raw_feature_cols: list[str]
    engineered_feature_cols: list[str]
    feature_discovery_result: Any
    X_development_base: pd.DataFrame
    y_development: pd.Series
    development_dates: np.ndarray
    benchmark_risk_score_estimation: pd.Series
    benchmark_score_estimation: pd.Series
    sample_weight: np.ndarray | None
    X_augmented_base_for_ablation: pd.DataFrame | None
    y_augmented_for_ablation: pd.Series | None
    augmented_dates_for_ablation: np.ndarray | None
    augmented_sample_weight_for_ablation: np.ndarray | None
    X_development: pd.DataFrame
    X_test: pd.DataFrame
    preprocessor: ColumnTransformer
    lgbm_preprocessor: ColumnTransformer
    lgbm_cat_indices: list[int]
    monotone_constraints: list[int]
    X_development_fit: pd.DataFrame
    X_calibration_holdout: pd.DataFrame
    y_development_fit: pd.Series
    y_calibration_holdout: pd.Series
    w_development_fit: np.ndarray | None
    w_calibration_holdout: np.ndarray | None
    development_fit_dates: np.ndarray
    calibration_holdout_dates: np.ndarray
    X_calibration_booked: pd.DataFrame
    y_calibration_booked: pd.Series
    pos_weight: float
    cv: Any


class TrainingStageResult(TypedDict):
    models: dict[str, Any]
    development_oof_scores: dict[str, np.ndarray]
    rolling_oot_results_df: pd.DataFrame
    rolling_oot_summary_df: pd.DataFrame


class EvaluationStageResult(TypedDict):
    results_df: pd.DataFrame
    models: dict[str, Any]
    official_results_df: pd.DataFrame
    experimental_results_df: pd.DataFrame
    benchmark_comparisons_df: pd.DataFrame
    experimental_benchmark_comparisons_df: pd.DataFrame
    feature_provenance_df: pd.DataFrame
    interaction_leaderboard_df: pd.DataFrame
    feature_discovery_boundary_df: pd.DataFrame
    ablation_results_df: pd.DataFrame
    rolling_oot_results_df: pd.DataFrame
    rolling_oot_summary_df: pd.DataFrame
    population_summary_df: pd.DataFrame
    applicant_scores_df: pd.DataFrame | None
    holdout_scores_df: pd.DataFrame
    test_scores: dict[str, np.ndarray]
    y_test: pd.Series
    train_scores: dict[str, np.ndarray]
    y_development_fit_booked: pd.Series
    num_cols: list[str]
    cat_cols: list[str]


def _run_data_preparation_stages(
    data_path: str,
    population_mode: str,
    reject_inference: bool,
    feature_discovery_fraction: float,
) -> PreparedStageResult:
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

    with _log_step(4, "Freeze feature set & prepare official matrices"):
        X_development_base = feature_discovery_result.X_estimation_base.copy()
        y_development = feature_discovery_result.y_estimation.copy()
        development_dates = feature_discovery_result.estimation_dates.copy()
        benchmark_risk_score_estimation = df.loc[feature_discovery_result.X_estimation_base.index, "risk_score_rf"].copy()
        benchmark_score_estimation = df.loc[feature_discovery_result.X_estimation_base.index, "score_RF"].copy()
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
            estimation_start = pd.Timestamp(pd.to_datetime(development_dates).min())
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
                feature_discovery_result.base_feature_cols,
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
            feature_discovery_result.X_test_base,
            feature_discovery_result.base_feature_cols,
            feature_discovery_result.base_num_cols,
            feature_discovery_result.base_cat_cols,
        )
        X_development = X_development[feature_discovery_result.feature_cols].copy()
        X_test = X_test[feature_discovery_result.feature_cols].copy()
        logger.info(
            "Official matrices with frozen features: {} train rows x {} cols, {} test rows x {} cols",
            len(X_development),
            len(feature_discovery_result.feature_cols),
            len(X_test),
            len(feature_discovery_result.feature_cols),
        )

    preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(
        feature_discovery_result.num_cols,
        feature_discovery_result.cat_cols,
    )
    monotone_constraints = build_monotone_constraints(
        feature_discovery_result.num_cols,
        feature_discovery_result.cat_cols,
    )

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

    # Warn (don't fail) when the holdout has too few positives for stable
    # isotonic calibration. See training_features.check_calibration_holdout_size.
    check_calibration_holdout_size(y_calibration_booked)

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

    return {
        "df": df,
        "rejected_df": rejected_df,
        "population_summary_df": population_summary_df,
        "raw_feature_cols": raw_feature_cols,
        "engineered_feature_cols": engineered_feature_cols,
        "feature_discovery_result": feature_discovery_result,
        "X_development_base": X_development_base,
        "y_development": y_development,
        "development_dates": development_dates,
        "benchmark_risk_score_estimation": benchmark_risk_score_estimation,
        "benchmark_score_estimation": benchmark_score_estimation,
        "sample_weight": sample_weight,
        "X_augmented_base_for_ablation": X_augmented_base_for_ablation,
        "y_augmented_for_ablation": y_augmented_for_ablation,
        "augmented_dates_for_ablation": augmented_dates_for_ablation,
        "augmented_sample_weight_for_ablation": augmented_sample_weight_for_ablation,
        "X_development": X_development,
        "X_test": X_test,
        "preprocessor": preprocessor,
        "lgbm_preprocessor": lgbm_preprocessor,
        "lgbm_cat_indices": lgbm_cat_indices,
        "monotone_constraints": monotone_constraints,
        "X_development_fit": X_development_fit,
        "X_calibration_holdout": X_calibration_holdout,
        "y_development_fit": y_development_fit,
        "y_calibration_holdout": y_calibration_holdout,
        "w_development_fit": w_development_fit,
        "w_calibration_holdout": w_calibration_holdout,
        "development_fit_dates": development_fit_dates,
        "calibration_holdout_dates": calibration_holdout_dates,
        "X_calibration_booked": X_calibration_booked,
        "y_calibration_booked": y_calibration_booked,
        "pos_weight": pos_weight,
        "cv": cv,
    }


def _run_model_training_stages(
    prepared: PreparedStageResult,
    optuna_trials: int,
    output_path: Path,
    enable_experimental_stacking: bool,
    population_mode: str,
) -> TrainingStageResult:
    feature_discovery_result = prepared["feature_discovery_result"]
    with _log_step(7, "Logistic Regression — development fit sample"):
        lr_model, lr_study = train_logistic_regression(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            prepared["preprocessor"],
            prepared["cv"],
            optuna_trials,
            sample_weight=prepared["w_development_fit"],
            num_cols=feature_discovery_result.num_cols,
            cat_cols=feature_discovery_result.cat_cols,
        )
    with _log_step("7b", "EBM — development fit sample"):
        ebm_model, ebm_study = train_ebm(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            prepared["preprocessor"],
            prepared["cv"],
            optuna_trials,
            sample_weight=prepared["w_development_fit"],
        )
    with _log_step(8, "LightGBM — development fit sample"):
        lgbm_model, lgbm_study, _ = train_lgbm(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            prepared["lgbm_preprocessor"],
            prepared["lgbm_cat_indices"],
            prepared["pos_weight"],
            prepared["cv"],
            optuna_trials,
            sample_weight=prepared["w_development_fit"],
            monotone_constraints=prepared["monotone_constraints"],
        )
    with _log_step(9, "XGBoost — development fit sample"):
        xgb_model, xgb_study, _ = train_xgboost(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            prepared["preprocessor"],
            prepared["pos_weight"],
            prepared["cv"],
            optuna_trials,
            sample_weight=prepared["w_development_fit"],
            monotone_constraints=prepared["monotone_constraints"],
        )
    with _log_step(10, "CatBoost — development fit sample"):
        catboost_model, catboost_study, _ = train_catboost(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            prepared["lgbm_preprocessor"],
            prepared["pos_weight"],
            prepared["cv"],
            optuna_trials,
            sample_weight=prepared["w_development_fit"],
            monotone_constraints=prepared["monotone_constraints"],
        )

    for study_name, study_obj in [
        ("Logistic Regression", lr_study),
        ("EBM", ebm_study),
        ("LightGBM", lgbm_study),
        ("XGBoost", xgb_study),
        ("CatBoost", catboost_study),
    ]:
        _save_optuna_study(study_obj, output_path, study_name)

    stack_model = None
    if enable_experimental_stacking:
        with _log_step(11, "Stacking ensemble (experimental)"):
            stack_model = train_stacking(
                prepared["X_development_fit"],
                prepared["y_development_fit"],
                {
                    "Logistic Regression": lr_model,
                    "EBM": ebm_model,
                    "LightGBM": lgbm_model,
                    "XGBoost": xgb_model,
                    "CatBoost": catboost_model,
                },
                prepared["cv"],
                sample_weight=prepared["w_development_fit"],
            )

    with _log_step("11b", "Temporal OOF development scores"):
        development_oof_scores = compute_temporal_oof_scores(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            {
                "Logistic Regression": lr_model,
                "EBM": ebm_model,
                "LightGBM": lgbm_model,
                "XGBoost": xgb_model,
                "CatBoost": catboost_model,
            },
            prepared["cv"],
            sample_weight=prepared["w_development_fit"],
        )

    with _log_step(12, "Calibration — booked ground-truth holdout"):
        models = {
            "Logistic Regression": lr_model,
            "EBM": ebm_model,
            "LightGBM": lgbm_model,
            "XGBoost": xgb_model,
            "CatBoost": catboost_model,
        }
        if stack_model is not None:
            models[EXPERIMENTAL_STACKING_NAME] = stack_model
        # Per-model calibration method follows Niculescu-Mizil & Caruana (2005);
        # see CALIBRATION_METHOD_BY_MODEL in training_constants.py for the full
        # rationale. Sigmoid for LR/EBM (log-odds-like outputs), isotonic for
        # tree ensembles (sigmoid-shaped reliability diagrams).
        for name in OFFICIAL_MODEL_NAMES:
            method = CALIBRATION_METHOD_BY_MODEL.get(name, DEFAULT_CALIBRATION_METHOD)
            cal = CalibratedClassifierCV(FrozenEstimator(models[name]), method=method)
            cal.fit(prepared["X_calibration_booked"], prepared["y_calibration_booked"])
            models[f"{name} (calibrated)"] = cal
        logger.info(
            "Calibration on {:,} booked held-out samples ({:,} pos) — sigmoid for LR/EBM, isotonic for tree models",
            len(prepared["y_calibration_booked"]),
            prepared["y_calibration_booked"].sum(),
        )

    with _log_step("12b", "Rolling OOT validation — pre-test estimation sample"):
        rolling_base_models = {name: models[name] for name in OFFICIAL_MODEL_NAMES}
        rolling_oot_results_df, rolling_oot_summary_df = run_rolling_out_of_time_validation(
            feature_discovery_result.X_estimation_base,
            feature_discovery_result.y_estimation,
            feature_discovery_result.estimation_dates,
            prepared["benchmark_risk_score_estimation"],
            prepared["benchmark_score_estimation"],
            feature_discovery_result.base_feature_cols,
            feature_discovery_result.base_num_cols,
            feature_discovery_result.base_cat_cols,
            feature_discovery_result.feature_cols,
            feature_discovery_result.num_cols,
            feature_discovery_result.cat_cols,
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

    with _log_step("12c", "Concept drift detection"):
        concept_drift_df = compute_concept_drift_report(rolling_oot_results_df, model_names=OFFICIAL_MODEL_NAMES)
        if not concept_drift_df.empty:
            concept_drift_df.to_csv(output_path / "concept_drift.csv", index=False, float_format="%.6f")
            logger.info("Concept drift analysis (PR AUC trend across OOT folds):")
            for _, row in concept_drift_df.iterrows():
                logger.info(
                    "  {:<25s} first={:.4f} last={:.4f} slope={:+.4f}/fold  [{}]",
                    row["model"], row["pr_auc_first"], row["pr_auc_last"],
                    row["pr_auc_slope_per_fold"], row["concept_drift_flag"],
                )
            n_drift = int((concept_drift_df["concept_drift_flag"] == "YES").sum())
            if n_drift > 0:
                logger.warning("{} model(s) show concept drift (PR AUC declining > 0.02 across OOT folds)", n_drift)

    with _log_step("12d", "Post-hoc ensemble"):
        calib_scores_for_ensemble = {}
        for name in OFFICIAL_MODEL_NAMES:
            calib_scores_for_ensemble[name] = models[name].predict_proba(prepared["X_calibration_booked"])[:, 1]
        ensemble_result = train_post_hoc_ensemble(prepared["y_calibration_booked"].values, calib_scores_for_ensemble)
        if ensemble_result:
            lr_w = ensemble_result["lr_weight"]
            tree_w = ensemble_result["tree_weight"]
            tree_name = ensemble_result["tree_name"]
            logger.info(
                "Best ensemble: {:.0%} {} + {:.0%} {} → calibration PR AUC {:.4f}",
                lr_w, ensemble_result["lr_name"], tree_w, tree_name, ensemble_result["pr_auc"],
            )
            ensemble_name = f"Ensemble ({ensemble_result['lr_name']} + {tree_name})"
            models[ensemble_name] = EnsembleModel(
                models[ensemble_result["lr_name"]], models[tree_name], lr_w, tree_w,
                name=ensemble_name,
            )
            pd.DataFrame([ensemble_result]).to_csv(output_path / "ensemble_weights.csv", index=False, float_format="%.4f")
        else:
            logger.info("Post-hoc ensemble: no valid combination found")

    return {
        "models": models,
        "development_oof_scores": development_oof_scores,
        "rolling_oot_results_df": rolling_oot_results_df,
        "rolling_oot_summary_df": rolling_oot_summary_df,
    }


def _run_evaluation_and_diagnostics_stages(
    prepared: PreparedStageResult,
    training_stage: TrainingStageResult,
    reject_inference: bool,
    population_mode: str,
    output_path: Path,
) -> EvaluationStageResult:
    df = prepared["df"]
    rejected_df = prepared["rejected_df"]
    feature_discovery_result = prepared["feature_discovery_result"]
    models = training_stage["models"]
    development_oof_scores = training_stage["development_oof_scores"]
    rolling_oot_summary_df = training_stage["rolling_oot_summary_df"]
    X_test = prepared["X_test"]
    y_test_full = feature_discovery_result.y_test
    benchmark_risk_score_test = feature_discovery_result.benchmark_risk_score_test
    benchmark_score_test = feature_discovery_result.benchmark_score_test
    X_test_base = feature_discovery_result.X_test_base

    test_dates = df.loc[X_test_base.index, "mis_Date"].values
    (
        X_selection,
        y_selection,
        benchmark_risk_score_selection,
        benchmark_score_selection,
        X_final,
        y_final,
        benchmark_risk_score_final,
        benchmark_score_final,
    ) = split_holdout_for_model_selection(
        X_test=X_test,
        y_test=y_test_full,
        bench_risk_score_test=benchmark_risk_score_test,
        bench_score_test=benchmark_score_test,
        test_dates=test_dates,
    )
    X_test = X_selection
    y_test = y_selection

    with _log_step(13, "Evaluation — selection holdout sample"):
        results_df, test_scores = evaluate_all(
            X_selection,
            y_selection,
            models,
            benchmark_risk_score_selection,
            benchmark_score_selection,
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
                prepared["X_development_base"],
                feature_discovery_result.base_feature_cols,
                feature_discovery_result.base_num_cols,
                feature_discovery_result.base_cat_cols,
                feature_discovery_result.feature_cols,
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
        y_selection,
        test_scores,
        population_mode=population_mode if population_mode == POPULATION_MODE_UNDERWRITING else None,
        evaluation_population="booked_proxy" if population_mode == POPULATION_MODE_UNDERWRITING else "booked_selection",
    )

    with _log_step("13c", "Evaluation — final untouched holdout sample"):
        final_results_df, final_scores = evaluate_all(
            X_final,
            y_final,
            models,
            benchmark_risk_score_final,
            benchmark_score_final,
        )
        if population_mode == POPULATION_MODE_UNDERWRITING:
            final_results_df["population_mode"] = population_mode
            final_results_df["evaluation_population"] = "booked_proxy_final"
        final_results_df.to_csv(output_path / "final_holdout_results.csv", float_format="%.6f")
        final_holdout_scores_df = build_holdout_score_frame(
            y_final,
            final_scores,
            population_mode=population_mode if population_mode == POPULATION_MODE_UNDERWRITING else None,
            evaluation_population="booked_proxy_final" if population_mode == POPULATION_MODE_UNDERWRITING else "booked_final",
        )
        final_holdout_scores_df.to_csv(output_path / "final_holdout_scores.csv", index=False, float_format="%.6f")

    w_development_fit = prepared["w_development_fit"]
    if w_development_fit is not None:
        development_fit_booked_mask = w_development_fit == 1.0
        X_development_fit_booked = prepared["X_development_fit"].loc[development_fit_booked_mask]
        y_development_fit_booked = prepared["y_development_fit"].loc[development_fit_booked_mask]
    else:
        X_development_fit_booked = prepared["X_development_fit"]
        y_development_fit_booked = prepared["y_development_fit"]

    train_scores = {
        name: scores[development_fit_booked_mask] if w_development_fit is not None else scores
        for name, scores in development_oof_scores.items()
    }

    with _log_step(14, "Bootstrap confidence intervals — booked test sample"):
        ci_df = bootstrap_confidence_intervals(
            y_selection.values,
            test_scores,
            dates=df.loc[X_selection.index, "mis_Date"].values,
        )
        ci_df.to_csv(output_path / "confidence_intervals.csv", float_format="%.6f")
        official_candidate_names = [name for name in official_results_df.index if name not in BENCHMARK_MODEL_NAMES]
        experimental_candidate_names = list(experimental_results_df.index)
        benchmark_comparisons_df = paired_bootstrap_benchmark_comparisons(
            y_selection.values,
            test_scores,
            official_candidate_names,
        )
        experimental_benchmark_comparisons_df = paired_bootstrap_benchmark_comparisons(
            y_selection.values,
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

    with _log_step("14b", "Phase 3 ablations"):
        ablation_results_df = run_phase3_ablations(
            feature_discovery_result.X_estimation_base,
            feature_discovery_result.y_estimation,
            feature_discovery_result.estimation_dates,
            feature_discovery_result.X_test_base,
            y_test_full,
            prepared["raw_feature_cols"],
            prepared["engineered_feature_cols"],
            feature_discovery_result.interaction_feature_cols,
            feature_discovery_result.base_feature_cols,
            feature_discovery_result.base_num_cols,
            feature_discovery_result.base_cat_cols,
            feature_discovery_result.rfecv_candidate_feature_cols,
            feature_discovery_result.feature_cols,
            feature_discovery_result.num_cols,
            feature_discovery_result.cat_cols,
            X_augmented_base=prepared["X_augmented_base_for_ablation"],
            y_augmented=prepared["y_augmented_for_ablation"],
            augmented_dates=prepared["augmented_dates_for_ablation"],
            augmented_sample_weight=prepared["augmented_sample_weight_for_ablation"],
        )
        logger.info("Phase 3 ablations completed: {} rows", len(ablation_results_df))
    if population_mode == POPULATION_MODE_UNDERWRITING and not ablation_results_df.empty:
        ablation_results_df["population_mode"] = population_mode
        ablation_results_df["evaluation_population"] = "booked_proxy"

    with _log_step(15, "SHAP explainability"):
        compute_shap_analysis(models, X_test, feature_discovery_result.num_cols, feature_discovery_result.cat_cols, output_path)

    with _log_step(16, "WoE / IV analysis"):
        woe_df, iv_df = compute_woe_iv(
            prepared["X_development_fit"],
            prepared["y_development_fit"],
            feature_discovery_result.num_cols,
            feature_discovery_result.cat_cols,
        )
        woe_df.to_csv(output_path / "woe_detail.csv", index=False, float_format="%.6f")
        iv_df.to_csv(output_path / "iv_summary.csv", index=False, float_format="%.6f")

    with _log_step(17, "PSI / CSI stability"):
        run_stability_analysis(
            X_development_fit_booked,
            X_test,
            train_scores,
            test_scores,
            feature_discovery_result.num_cols,
            feature_discovery_result.cat_cols,
            output_path,
        )

    overfit_df, model_selection_df = _run_diagnostics_and_governance(
        y_test=y_test,
        y_development_fit_booked=y_development_fit_booked,
        test_scores=test_scores,
        train_scores=train_scores,
        models=models,
        X_test=X_test,
        X_development_fit=prepared["X_development_fit"],
        num_cols=feature_discovery_result.num_cols,
        cat_cols=feature_discovery_result.cat_cols,
        feature_cols=feature_discovery_result.feature_cols,
        official_results_df=official_results_df,
        rolling_oot_summary_df=rolling_oot_summary_df,
        benchmark_comparisons_df=benchmark_comparisons_df,
        population_summary_df=prepared["population_summary_df"],
        feature_provenance_df=feature_discovery_result.feature_provenance_df,
        applicant_scores_df=applicant_scores_df,
        age_values=df.loc[X_test.index, "AGE_T1"].values if "AGE_T1" in df.columns else None,
        output_path=output_path,
    )

    return {
        "results_df": results_df,
        "models": models,
        "official_results_df": official_results_df,
        "experimental_results_df": experimental_results_df,
        "benchmark_comparisons_df": benchmark_comparisons_df,
        "experimental_benchmark_comparisons_df": experimental_benchmark_comparisons_df,
        "feature_provenance_df": feature_discovery_result.feature_provenance_df,
        "interaction_leaderboard_df": feature_discovery_result.interaction_leaderboard_df,
        "feature_discovery_boundary_df": feature_discovery_result.feature_discovery_boundary_df,
        "ablation_results_df": ablation_results_df,
        "rolling_oot_results_df": training_stage["rolling_oot_results_df"],
        "rolling_oot_summary_df": rolling_oot_summary_df,
        "population_summary_df": prepared["population_summary_df"],
        "applicant_scores_df": applicant_scores_df,
        "holdout_scores_df": holdout_scores_df,
        "test_scores": test_scores,
        "y_test": y_test,
        "train_scores": train_scores,
        "y_development_fit_booked": y_development_fit_booked,
        "num_cols": feature_discovery_result.num_cols,
        "cat_cols": feature_discovery_result.cat_cols,
    }


def _run_artifact_persistence_stage(
    eval_stage: EvaluationStageResult,
    output_path: Path,
) -> None:
    with _log_step(18, "Save artifacts"):
        feat_imp = extract_feature_importance(eval_stage["models"], eval_stage["num_cols"], eval_stage["cat_cols"])
        plots_dir = output_path / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_score_distributions(
            eval_stage["y_test"].values,
            eval_stage["test_scores"],
            plots_dir / "score_dist_test.png",
            title_prefix="Test",
        )
        plot_score_distributions(
            eval_stage["y_development_fit_booked"].values,
            eval_stage["train_scores"],
            plots_dir / "score_dist_train.png",
            title_prefix="Train",
        )
        save_artifacts(
            eval_stage["models"],
            eval_stage["official_results_df"],
            feat_imp,
            output_path,
            experimental_results_df=eval_stage["experimental_results_df"],
            benchmark_comparisons_df=eval_stage["benchmark_comparisons_df"],
            experimental_benchmark_comparisons_df=eval_stage["experimental_benchmark_comparisons_df"],
            feature_provenance_df=eval_stage["feature_provenance_df"],
            interaction_leaderboard_df=eval_stage["interaction_leaderboard_df"],
            feature_discovery_boundary_df=eval_stage["feature_discovery_boundary_df"],
            ablation_results_df=eval_stage["ablation_results_df"],
            rolling_oot_results_df=eval_stage["rolling_oot_results_df"],
            rolling_oot_summary_df=eval_stage["rolling_oot_summary_df"],
            population_summary_df=eval_stage["population_summary_df"],
            applicant_scores_df=eval_stage["applicant_scores_df"],
            holdout_scores_df=eval_stage["holdout_scores_df"],
        )

        # Regenerate stakeholder charts from the freshly-written CSVs so the
        # PNGs cannot drift behind the metrics they visualise. Wrapped
        # defensively — a chart failure must not invalidate the rest of the
        # run's artifacts (the CSVs are the source of truth).
        try:
            from stakeholder_charts import generate_stakeholder_charts

            generated = generate_stakeholder_charts(output_path)
            logger.info("Regenerated {} stakeholder charts in {}", len(generated), output_path / "plots")
        except Exception as exc:
            logger.warning(
                "Stakeholder chart regeneration failed: {}. CSVs are still authoritative; "
                "run `uv run stakeholder_charts.py --output-dir {}` to retry.",
                exc, output_path,
            )


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


# make_temporal_cv and build_rolling_oot_windows live in training_temporal.py.


# reduce_cardinality, add_frequency_encoding, add_group_stats,
# prune_correlated, add_modeling_features, and build_feature_provenance are
# imported directly from training_features above.


# ── Reject inference ──────────────────────────────────────────────────────────
# compute_score_band_bad_rates, create_reject_pseudo_labels, augment_training_data
# live in training_reject_inference.py and are imported below.


# ── Preprocessors ──────────────────────────────────────────────────────────────
# build_preprocessors and build_monotone_constraints are imported from
# training_features above.

# ── Feature Selection (temporal stability selection) ──────────────────────────
# run_rfecv is imported from training_features above.



# Temporal stacking (TemporalStackingClassifier, fit_pipeline_from_template,
# compute_temporal_oof_scores, train_stacking, build_fresh_pipeline_from_fitted)
# lives in training_stacking.py and is imported below. safe_stratified_n_splits,
# normalize_estimator_count, select_conservative_boosting_rounds live in
# training_models.py.


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
                "applicant_index", "mis_Date", "status_name", TARGET,
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

    # Sort by mis_Date + original authorization_id for deterministic ordering,
    # then drop the PII linkage key. The output frame uses an opaque
    # applicant_index so downstream artifacts in output/ cannot link a score
    # back to a specific applicant. AGE_T1 is also omitted (not consumed by
    # downstream reports) to minimise PII surface.
    applicant_df = applicant_df.sort_values(["mis_Date", "authorization_id"])
    X_applicant = X_applicant.loc[applicant_df.index]

    score_frame = applicant_df.loc[:, [
        "mis_Date", "status_name", TARGET, "risk_score_rf", "score_RF",
    ]].copy()
    score_frame.insert(0, "applicant_index", np.arange(1, len(score_frame) + 1))
    score_frame["has_observed_target"] = score_frame[TARGET].notna()
    score_frame["target_source"] = np.where(
        score_frame["has_observed_target"],
        "observed_booked",
        "unobserved_application",
    )
    for name, model in models.items():
        score_frame[f"score__{sanitize_output_name(name)}"] = model.predict_proba(X_applicant)[:, 1]
    return score_frame.reset_index(drop=True)


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
                # Use the same per-model method table as the official path so
                # rolling-OOT calibration metrics are comparable to evaluation.
                # (Pre-fix this site treated EBM as a tree model, which gave it
                # isotonic in OOT but sigmoid in the official run.)
                cal_method = CALIBRATION_METHOD_BY_MODEL.get(name, DEFAULT_CALIBRATION_METHOD)
                calibrated_model = CalibratedClassifierCV(
                    FrozenEstimator(fold_model),
                    method=cal_method,
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
        shap = None

    candidate_names = [preferred_model_name] if preferred_model_name is not None else []
    candidate_names.extend(OFFICIAL_MODEL_NAMES)
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
        elif (
            hasattr(candidate_clf, "explain_global")
            and hasattr(candidate_clf, "term_importances")
            and hasattr(candidate_clf, "eval_terms")
        ):
            ranked_candidates.append((name, candidate_model, "ebm"))

    if not ranked_candidates:
        logger.warning("No supported model available for SHAP")
        return None

    feature_names = num_cols + cat_cols
    selected_explainer = None
    selected_global_explanation = None
    selected_term_scores = None

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
                if shap is None:
                    raise ImportError("shap not installed")
                background = X_t
                if X_t.shape[0] > 1000:
                    rng = np.random.RandomState(RANDOM_STATE)
                    bg_idx = rng.choice(X_t.shape[0], 1000, replace=False)
                    background = X_t[bg_idx]
                selected_explainer = shap.LinearExplainer(clf, background)
            elif explainer_type == "ebm":
                selected_global_explanation = clf.explain_global()
                selected_term_scores = clf.eval_terms(X_t)
            else:
                if shap is None:
                    raise ImportError("shap not installed")
                selected_explainer = shap.TreeExplainer(clf)
            break
        except (ValueError, TypeError, Exception) as exc:
            logger.warning("SHAP explainer failed for {} ({}), trying next model: {}", name, explainer_type, exc)
            continue
    else:
        logger.warning("All SHAP explainer attempts failed — skipping")
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if explainer_type == "ebm":
        overall_data = selected_global_explanation.data()
        term_names = [str(term_name) for term_name in overall_data.get("names", [])]
        term_importances = np.asarray(overall_data.get("scores", []), dtype=float)
        if len(term_names) == 0 or term_importances.size == 0:
            logger.warning("EBM global explanation is empty — skipping")
            return None

        order = np.argsort(np.abs(term_importances))[::-1]
        ordered_names = [term_names[idx] for idx in order]
        ordered_importances = np.abs(term_importances[order])
        summary = pd.DataFrame({
            "feature": ordered_names,
            "mean_abs_shap": ordered_importances,
        })

        top_n = min(20, len(summary))
        plot_df = summary.head(top_n).iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#7b2cbf")
        ax.set_xlabel("Global importance")
        ax.set_ylabel("Term")
        ax.set_title(f"Global term importance — {name}")
        fig.tight_layout()
        fig.savefig(plots_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color="#7b2cbf")
        ax.set_xlabel("Global importance")
        ax.set_ylabel("Term")
        ax.set_title(f"Global term importance — {name}")
        fig.tight_layout()
        fig.savefig(plots_dir / "shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        plotted = 0
        for term_idx in order:
            if plotted >= 6:
                break
            term_data = selected_global_explanation.data(int(term_idx))
            scores = np.asarray(term_data.get("scores", []), dtype=float)
            if scores.size == 0:
                continue
            names = [str(label) for label in term_data.get("names", [])]
            if len(names) == len(scores) + 1:
                labels = names[1:]
            else:
                labels = names[: len(scores)]
            ax = axes[plotted]
            positions = np.arange(len(scores))
            ax.bar(positions, scores, color="#7b2cbf", alpha=0.85)
            if labels and len(labels) == len(scores) and len(labels) <= 12:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticks([])
            ax.set_title(term_names[int(term_idx)])
            plotted += 1
        for j in range(plotted, 6):
            axes[j].set_visible(False)
        fig.suptitle(f"EBM term effects — {name} (top {plotted})", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / "shap_dependence.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        if selected_term_scores is not None:
            term_score_names = term_names[: selected_term_scores.shape[1]]
            shap_df = pd.DataFrame(selected_term_scores, columns=term_score_names)
            shap_df.to_csv(output_dir / "shap_values.csv", index=False, float_format="%.6f")

        summary.to_csv(output_dir / "shap_importance.csv", index=False, float_format="%.6f")
        logger.info("Explainability ({}): {} terms", name, len(summary))
        for _, r in summary.head(10).iterrows():
            logger.info("  {:<35s} mean|effect|={:.4f}", r["feature"], r["mean_abs_shap"])

        return summary

    shap_values = selected_explainer.shap_values(X_t)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

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
        if psi < PSI_MODERATE_THRESHOLD:
            flag = "OK"
        elif psi < PSI_HIGH_DRIFT_THRESHOLD:
            flag = "MODERATE"
        else:
            flag = "HIGH DRIFT"
        logger.info("  PSI {:<25s} = {:.4f}  [{}]", name, psi, flag)
    pd.DataFrame(psi_records).to_csv(output_dir / "psi.csv", index=False, float_format="%.6f")
    csi_df = compute_csi(X_train, X_test, num_cols, cat_cols)
    csi_df.to_csv(output_dir / "csi.csv", index=False, float_format="%.6f")
    n_high = int((csi_df["csi"] >= PSI_HIGH_DRIFT_THRESHOLD).sum())
    n_mod = int(((csi_df["csi"] >= PSI_MODERATE_THRESHOLD) & (csi_df["csi"] < PSI_HIGH_DRIFT_THRESHOLD)).sum())
    n_stable = len(csi_df) - n_high - n_mod
    logger.info(
        "  CSI: {} features — {} high drift, {} moderate, {} stable",
        len(csi_df),
        n_high,
        n_mod,
        n_stable,
    )
    if n_high > 0:
        for _, row in csi_df[csi_df["csi"] >= PSI_HIGH_DRIFT_THRESHOLD].iterrows():
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


def _run_diagnostics_and_governance(
    *,
    y_test: pd.Series,
    y_development_fit_booked: pd.Series,
    test_scores: dict[str, np.ndarray],
    train_scores: dict[str, np.ndarray],
    models: dict,
    X_test: pd.DataFrame,
    X_development_fit: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    feature_cols: list[str],
    official_results_df: pd.DataFrame,
    rolling_oot_summary_df: pd.DataFrame,
    benchmark_comparisons_df: pd.DataFrame,
    population_summary_df: pd.DataFrame,
    feature_provenance_df: pd.DataFrame,
    applicant_scores_df: pd.DataFrame | None,
    age_values: np.ndarray | None,
    output_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Steps 17b-17f: lift table, overfitting, model selection, governance, population bias.

    Returns (overfit_df, model_selection_df) for use by the save-artifacts step.
    """

    # 17b. Lift table and threshold analysis
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
            logger.info("Threshold analysis: {} rows across {} models", len(threshold_analysis_df), threshold_analysis_df["model"].nunique())

    # 17c. Overfitting diagnostics
    with _log_step("17c", "Overfitting diagnostics"):
        overfit_df = compute_overfit_report(
            y_development_fit_booked.values, y_test.values, train_scores, test_scores,
            model_names=OFFICIAL_MODEL_NAMES,
        )
        if not overfit_df.empty:
            overfit_df.to_csv(output_path / "overfit_report.csv", index=False, float_format="%.6f")
            logger.info("Train vs test performance comparison:")
            logger.info("{:<25s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>6s}",
                        "Model", "Train AUC", "Test AUC", "AUC Δ", "Train PR", "Test PR", "PR Δ", "Flag")
            logger.info("{}", "─" * 97)
            for _, row in overfit_df.iterrows():
                logger.info(
                    "{:<25s} {:>10.4f} {:>10.4f} {:>+10.4f} {:>10.4f} {:>10.4f} {:>+10.4f} {:>6s}",
                    row["model"], row["train_auc"], row["test_auc"], row["auc_delta"],
                    row["train_pr_auc"], row["test_pr_auc"], row["pr_auc_delta"], row["overfit_flag"],
                )
            n_flagged = int((overfit_df["overfit_flag"] == "YES").sum())
            if n_flagged > 0:
                logger.warning(
                    "{} model(s) flagged for potential overfitting (AUC or PR AUC delta > {:.2f})",
                    n_flagged, OVERFIT_DELTA_THRESHOLD,
                )
            else:
                logger.info("No models flagged for overfitting")

    # 17d. Model selection
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
            logger.info("{:<25s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>8s} {:>5s}",
                        "Model", "Disc.", "Calib.", "Stab.", "Gen.", "Lift", "Overall", "Rec.")
            logger.info("{}", "─" * 87)
            for _, row in model_selection_df.iterrows():
                logger.info(
                    "{:<25s} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.1f} {:>8.1f} {:>5s}",
                    row["model"], row["discrimination_score"], row["calibration_score"],
                    row["stability_score"], row["generalization_score"],
                    row["lift_score"], row["weighted_score"],
                    "<<<" if row["recommended"] else "",
                )
            recommended = model_selection_df.loc[model_selection_df["recommended"]].iloc[0]
            logger.info("Recommended model: {} (weighted score: {:.1f})", recommended["model"], recommended["weighted_score"])
            compute_shap_analysis(models, X_test, num_cols, cat_cols, output_path, preferred_model_name=str(recommended["model"]))

    # 17e. Model governance artifacts
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
            feature_cols, num_cols, cat_cols,
            feature_provenance_df=feature_provenance_df if not feature_provenance_df.empty else None,
            iv_df=iv_df_for_dict, output_path=output_path,
        )
        generate_data_quality_report(X_development_fit, num_cols, cat_cols, output_path, label="development_fit")
        generate_data_quality_report(X_test, num_cols, cat_cols, output_path, label="test")

    # 17f. Population bias analysis
    with _log_step("17f", "Population bias analysis"):
        if applicant_scores_df is not None and not applicant_scores_df.empty:
            pop_ks_df = compute_population_ks_test(applicant_scores_df, model_names=OFFICIAL_MODEL_NAMES)
            if not pop_ks_df.empty:
                pop_ks_df.to_csv(output_path / "population_ks_test.csv", index=False, float_format="%.6f")
                logger.info("Booked vs non-booked score distribution KS test:")
                for _, row in pop_ks_df.iterrows():
                    logger.info("  {:<25s} KS={:.4f}  p={:.2e}  booked_mean={:.4f}  non_booked_mean={:.4f}",
                                row["model"], row["ks_statistic"], row["ks_p_value"],
                                row["booked_mean_score"], row["non_booked_mean_score"])
            sel_bias_df = compute_selection_bias_correlation(applicant_scores_df, model_names=OFFICIAL_MODEL_NAMES)
            if not sel_bias_df.empty:
                sel_bias_df.to_csv(output_path / "selection_bias_correlation.csv", index=False, float_format="%.6f")
                logger.info("Selection bias — correlation with risk_score_rf:")
                for _, row in sel_bias_df.iterrows():
                    logger.info("  {:<25s} Pearson={:+.4f}  Spearman={:+.4f}  [{}]",
                                row["model"], row["pearson_corr"], row["spearman_corr"], row["selection_bias_flag"])
                n_high = int((sel_bias_df["selection_bias_flag"] == "HIGH").sum())
                if n_high > 0:
                    logger.warning("{} model(s) show HIGH correlation with existing risk_score_rf — may be recapitulating the selection mechanism", n_high)
        else:
            logger.info("Applicant scores not available (booked-monitoring mode) — skipping population KS and selection bias")

        ai_tables = []
        ai_population_label = None
        if (
            applicant_scores_df is not None
            and not applicant_scores_df.empty
            and "AGE_T1" in applicant_scores_df.columns
        ):
            for name in OFFICIAL_MODEL_NAMES:
                score_col = f"score__{sanitize_output_name(name)}"
                if score_col not in applicant_scores_df.columns:
                    continue
                ai_table = compute_adverse_impact_analysis(
                    applicant_scores_df[TARGET].values,
                    applicant_scores_df[score_col].values,
                    applicant_scores_df["AGE_T1"].values,
                    name,
                )
                if not ai_table.empty:
                    ai_table["analysis_population"] = "underwriting_applicants"
                    ai_tables.append(ai_table)
            ai_population_label = "post-split underwriting applicants"
        elif age_values is not None:
            for name in OFFICIAL_MODEL_NAMES:
                if name not in test_scores:
                    continue
                ai_table = compute_adverse_impact_analysis(y_test.values, test_scores[name], age_values, name)
                if not ai_table.empty:
                    ai_table["analysis_population"] = "booked_holdout"
                    ai_tables.append(ai_table)
            ai_population_label = "booked holdout"
        else:
            logger.info("AGE_T1 not available — skipping adverse impact analysis")

        ai_df = pd.concat(ai_tables, ignore_index=True) if ai_tables else pd.DataFrame()
        if not ai_df.empty:
            ai_df.to_csv(output_path / "adverse_impact_age.csv", index=False, float_format="%.6f")
            logger.info(
                "Adverse impact analysis by age band ({}, 10%% rejection threshold):",
                ai_population_label,
            )
            for name in OFFICIAL_MODEL_NAMES:
                model_ai = ai_df.loc[ai_df["model"] == name]
                if model_ai.empty:
                    continue
                n_fail = int((model_ai["air_flag"] == "FAIL").sum())
                if n_fail > 0:
                    failing = model_ai.loc[model_ai["air_flag"] == "FAIL"]
                    logger.warning("  {}: {} age band(s) below 80%% AIR threshold: {}",
                                   name, n_fail, ", ".join(f"{r['age_band']} (AIR={r['adverse_impact_ratio']:.2f})" for _, r in failing.iterrows()))
                else:
                    logger.info("  {}: all age bands pass 80%% AIR threshold", name)

    return overfit_df, model_selection_df


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

    prepared = _run_data_preparation_stages(
        data_path=data_path,
        population_mode=population_mode,
        reject_inference=reject_inference,
        feature_discovery_fraction=feature_discovery_fraction,
    )
    df = prepared["df"]
    rejected_df = prepared["rejected_df"]
    population_summary_df = prepared["population_summary_df"]
    feature_discovery_result = prepared["feature_discovery_result"]
    raw_feature_cols = prepared["raw_feature_cols"]
    engineered_feature_cols = prepared["engineered_feature_cols"]
    X_development_base = prepared["X_development_base"]
    y_development = prepared["y_development"]
    development_dates = prepared["development_dates"]
    X_augmented_base_for_ablation = prepared["X_augmented_base_for_ablation"]
    y_augmented_for_ablation = prepared["y_augmented_for_ablation"]
    augmented_dates_for_ablation = prepared["augmented_dates_for_ablation"]
    augmented_sample_weight_for_ablation = prepared["augmented_sample_weight_for_ablation"]
    X_test = prepared["X_test"]
    X_development_fit = prepared["X_development_fit"]
    y_development_fit = prepared["y_development_fit"]
    w_development_fit = prepared["w_development_fit"]
    X_calibration_booked = prepared["X_calibration_booked"]
    y_calibration_booked = prepared["y_calibration_booked"]
    X_test_base = feature_discovery_result.X_test_base
    y_test = feature_discovery_result.y_test
    benchmark_risk_score_test = feature_discovery_result.benchmark_risk_score_test
    benchmark_score_test = feature_discovery_result.benchmark_score_test
    interaction_feature_cols = feature_discovery_result.interaction_feature_cols
    interaction_leaderboard_df = feature_discovery_result.interaction_leaderboard_df
    feature_discovery_boundary_df = feature_discovery_result.feature_discovery_boundary_df
    base_feature_cols = feature_discovery_result.base_feature_cols
    base_num_cols = feature_discovery_result.base_num_cols
    base_cat_cols = feature_discovery_result.base_cat_cols
    X_estimation_base = feature_discovery_result.X_estimation_base
    y_estimation = feature_discovery_result.y_estimation
    estimation_dates = feature_discovery_result.estimation_dates
    feature_cols = feature_discovery_result.feature_cols
    num_cols = feature_discovery_result.num_cols
    cat_cols = feature_discovery_result.cat_cols
    rfecv_candidate_feature_cols = feature_discovery_result.rfecv_candidate_feature_cols
    feature_provenance_df = feature_discovery_result.feature_provenance_df

    training_stage = _run_model_training_stages(
        prepared=prepared,
        optuna_trials=optuna_trials,
        output_path=output_path,
        enable_experimental_stacking=enable_experimental_stacking,
        population_mode=population_mode,
    )
    models = training_stage["models"]
    development_oof_scores = training_stage["development_oof_scores"]
    rolling_oot_results_df = training_stage["rolling_oot_results_df"]
    rolling_oot_summary_df = training_stage["rolling_oot_summary_df"]

    eval_stage = _run_evaluation_and_diagnostics_stages(
        prepared=prepared,
        training_stage=training_stage,
        reject_inference=reject_inference,
        population_mode=population_mode,
        output_path=output_path,
    )
    results_df = eval_stage["results_df"]
    models = eval_stage["models"]
    _run_artifact_persistence_stage(eval_stage=eval_stage, output_path=output_path)

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
