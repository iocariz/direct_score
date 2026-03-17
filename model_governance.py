"""Model governance artifacts: model card, variable dictionary, data quality report."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from training_constants import (
    BENCHMARK_MODEL_NAMES,
    CALIBRATION_FRACTION,
    DROP_COLS,
    EARLY_STOPPING_ROUNDS,
    FEATURE_DISCOVERY_FRACTION,
    MATURITY_CUTOFF,
    MONOTONE_MAP,
    N_BOOTSTRAP,
    N_ESTIMATORS_CEILING,
    OFFICIAL_MODEL_NAMES,
    RANDOM_STATE,
    RAW_CAT,
    RAW_NUM,
    SPLIT_DATE,
    TARGET,
)


def generate_model_card(
    results_df: pd.DataFrame,
    model_selection_df: pd.DataFrame | None,
    overfit_df: pd.DataFrame | None,
    benchmark_comparisons_df: pd.DataFrame | None,
    population_summary_df: pd.DataFrame | None,
    feature_provenance_df: pd.DataFrame | None,
    output_path: Path,
) -> Path:
    """Generate a structured model card summarizing the model for governance review."""
    lines = []

    def _section(title):
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 70}\n")

    def _subsection(title):
        lines.append(f"\n--- {title} ---\n")

    # Header
    lines.append("MODEL CARD — basel_bad Credit Scoring Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Random seed: {RANDOM_STATE}")

    # 1. Model overview
    _section("1. MODEL OVERVIEW")
    lines.append(f"Target variable:        {TARGET} (12-month default flag)")
    lines.append(f"Task:                   Binary classification")
    lines.append(f"Maturity cutoff:        {MATURITY_CUTOFF}")
    lines.append(f"Temporal split:         Train < {SPLIT_DATE}, Test >= {SPLIT_DATE}")
    lines.append(f"Calibration holdout:    {CALIBRATION_FRACTION:.0%} of development set (latest temporal block)")
    lines.append(f"Feature discovery:      {FEATURE_DISCOVERY_FRACTION:.0%} of pre-test data (earliest block)")

    # 2. Recommended model
    _section("2. RECOMMENDED MODEL")
    if model_selection_df is not None and not model_selection_df.empty:
        rec = model_selection_df.loc[model_selection_df["recommended"]].iloc[0]
        lines.append(f"Model:                  {rec['model']}")
        lines.append(f"Weighted score:         {rec['weighted_score']:.1f} / 100")
        lines.append(f"Test ROC AUC:           {rec['test_auc']:.4f}")
        lines.append(f"Test PR AUC:            {rec['test_pr_auc']:.4f}")
        if not np.isnan(rec["test_brier"]):
            lines.append(f"Test Brier:             {rec['test_brier']:.4f}")
        lines.append("")
        lines.append("Selection criteria weights:")
        lines.append("  Discrimination (PR AUC):    35%")
        lines.append("  Stability (rolling OOT):    20%")
        lines.append("  Calibration (Brier):        15%")
        lines.append("  Generalization (overfit):   15%")
        lines.append("  Benchmark lift (AUC delta): 15%")
    else:
        lines.append("No model selection available.")

    # 3. Performance summary
    _section("3. PERFORMANCE SUMMARY")
    if results_df is not None:
        _subsection("Held-out test set results")
        candidate_df = results_df.loc[
            results_df.index.isin(OFFICIAL_MODEL_NAMES)
        ] if hasattr(results_df.index, 'isin') else results_df
        lines.append(f"{'Model':<30s} {'ROC AUC':>8s} {'PR AUC':>8s} {'KS':>6s} {'Brier':>8s}")
        lines.append("-" * 64)
        for name, row in candidate_df.iterrows():
            brier_str = f"{row['Brier']:.4f}" if not np.isnan(row["Brier"]) else "  N/A"
            lines.append(
                f"{str(name):<30s} {row['ROC AUC']:>8.4f} {row['PR AUC']:>8.4f} "
                f"{row['KS']:>6.4f} {brier_str:>8s}"
            )

    # 4. Overfitting assessment
    _section("4. OVERFITTING ASSESSMENT")
    if overfit_df is not None and not overfit_df.empty:
        lines.append(f"{'Model':<25s} {'Train AUC':>10s} {'Test AUC':>10s} {'Delta':>8s} {'Flag':>6s}")
        lines.append("-" * 63)
        for _, row in overfit_df.iterrows():
            lines.append(
                f"{row['model']:<25s} {row['train_auc']:>10.4f} {row['test_auc']:>10.4f} "
                f"{row['auc_delta']:>+8.4f} {row['overfit_flag']:>6s}"
            )
        n_flagged = int((overfit_df["overfit_flag"] == "YES").sum())
        lines.append("")
        if n_flagged == 0:
            lines.append("ASSESSMENT: No models show significant overfitting (AUC delta <= 0.03).")
        else:
            lines.append(f"WARNING: {n_flagged} model(s) flagged for potential overfitting.")
    else:
        lines.append("Overfitting diagnostics not available.")

    # 5. Benchmark comparisons
    _section("5. BENCHMARK COMPARISONS")
    if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
        lines.append(f"{'Candidate':<25s} {'vs Benchmark':<30s} {'AUC Δ':>8s} {'p-value':>10s} {'p-adj':>10s}")
        lines.append("-" * 87)
        for _, row in benchmark_comparisons_df.iterrows():
            p_adj_col = "auc_p_adjusted" if "auc_p_adjusted" in benchmark_comparisons_df.columns else None
            p_adj = f"{row[p_adj_col]:.4f}" if p_adj_col and not np.isnan(row[p_adj_col]) else "  N/A"
            lines.append(
                f"{row['candidate_model']:<25s} {row['reference_model']:<30s} "
                f"{row['auc_improvement']:>+8.4f} {row['auc_delong_p_value']:>10.4f} {p_adj:>10s}"
            )
    else:
        lines.append("No benchmark comparisons available.")

    # 6. Population coverage
    _section("6. POPULATION COVERAGE")
    if population_summary_df is not None and not population_summary_df.empty:
        for _, row in population_summary_df.iterrows():
            lines.append(f"  {row['split']:>12s} | {row['status_name']:<15s} | n={int(row['n_rows']):>8,}")
    else:
        lines.append("Population summary not available.")

    # 7. Feature inventory
    _section("7. FEATURE INVENTORY")
    if feature_provenance_df is not None and not feature_provenance_df.empty:
        n_rfecv_kept = int(feature_provenance_df["rfecv_kept"].sum()) if "rfecv_kept" in feature_provenance_df.columns else 0
        n_total = len(feature_provenance_df)
        lines.append(f"Total candidate features: {n_total}")
        lines.append(f"RFECV-selected features:  {n_rfecv_kept}")
        if "source" in feature_provenance_df.columns:
            source_counts = feature_provenance_df.loc[
                feature_provenance_df.get("rfecv_kept", pd.Series(dtype=bool)).fillna(False).astype(bool)
            ]["source"].value_counts()
            for source, count in source_counts.items():
                lines.append(f"  {source}: {count}")
    else:
        lines.append(f"Raw numerical features:   {len(RAW_NUM)}")
        lines.append(f"Raw categorical features: {len(RAW_CAT)}")
        lines.append(f"Excluded columns:         {len(DROP_COLS)}")

    # 8. Known limitations
    _section("8. KNOWN LIMITATIONS AND CAVEATS")
    lines.append("1. Evaluation uses booked-proxy population only. Rejected and canceled")
    lines.append("   applicants have no observed repayment outcome.")
    lines.append("2. Target variable (basel_bad) requires 12 months on book to mature.")
    lines.append(f"   Only accounts with mis_Date <= {MATURITY_CUTOFF} are used.")
    lines.append("3. Calibration performed via Platt (sigmoid) scaling on held-out")
    lines.append("   booked ground-truth samples.")
    lines.append("4. Monotonicity constraints applied to tree models where domain")
    lines.append(f"   knowledge dictates direction ({len(MONOTONE_MAP)} features constrained).")

    # 9. Technical configuration
    _section("9. TECHNICAL CONFIGURATION")
    lines.append(f"Random seed:              {RANDOM_STATE}")
    lines.append(f"N bootstrap iterations:   {N_BOOTSTRAP}")
    lines.append(f"Early stopping rounds:    {EARLY_STOPPING_ROUNDS}")
    lines.append(f"Max estimators:           {N_ESTIMATORS_CEILING}")
    lines.append(f"Calibration fraction:     {CALIBRATION_FRACTION}")
    lines.append(f"Feature discovery split:  {FEATURE_DISCOVERY_FRACTION}")

    card_path = output_path / "model_card.txt"
    card_path.write_text("\n".join(lines))
    logger.info("Saved model card: {}", card_path)
    return card_path


def generate_variable_dictionary(
    feature_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    feature_provenance_df: pd.DataFrame | None,
    iv_df: pd.DataFrame | None,
    output_path: Path,
) -> Path:
    """Export a variable dictionary with feature metadata."""
    records = []
    provenance_lookup = {}
    if feature_provenance_df is not None and not feature_provenance_df.empty:
        for _, row in feature_provenance_df.iterrows():
            provenance_lookup[row["feature"]] = row.to_dict()

    iv_lookup = {}
    if iv_df is not None and not iv_df.empty:
        for _, row in iv_df.iterrows():
            iv_lookup[row["feature"]] = float(row["iv"])

    for col in feature_cols:
        dtype = "numerical" if col in num_cols else "categorical" if col in cat_cols else "unknown"
        monotone = MONOTONE_MAP.get(col, None)
        monotone_str = (
            "decreasing" if monotone == -1
            else "increasing" if monotone == 1
            else "none"
        ) if monotone is not None else "none"

        prov = provenance_lookup.get(col, {})
        source = prov.get("source", "raw" if col in RAW_NUM or col in RAW_CAT else "engineered")
        iv = iv_lookup.get(col, np.nan)

        records.append({
            "feature": col,
            "type": dtype,
            "source": source,
            "monotone_constraint": monotone_str,
            "information_value": iv,
            "in_raw_num": col in RAW_NUM,
            "in_raw_cat": col in RAW_CAT,
            "in_monotone_map": col in MONOTONE_MAP,
        })

    df = pd.DataFrame(records)
    dict_path = output_path / "variable_dictionary.csv"
    df.to_csv(dict_path, index=False, float_format="%.6f")
    logger.info("Saved variable dictionary: {} ({} features)", dict_path, len(df))
    return dict_path


def generate_data_quality_report(
    X: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    output_path: Path,
    label: str = "development",
) -> Path:
    """Per-feature data quality summary: missingness, outliers, cardinality."""
    records = []
    n = len(X)

    for col in num_cols:
        if col not in X.columns:
            continue
        vals = X[col]
        missing_pct = float(vals.isna().mean())
        valid = vals.dropna()
        if len(valid) > 0:
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            n_outliers = int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum())
        else:
            q1, q3, n_outliers = np.nan, np.nan, 0
        records.append({
            "feature": col,
            "type": "numerical",
            "n_rows": n,
            "missing_count": int(vals.isna().sum()),
            "missing_pct": missing_pct,
            "n_unique": int(valid.nunique()),
            "mean": float(valid.mean()) if len(valid) > 0 else np.nan,
            "std": float(valid.std()) if len(valid) > 0 else np.nan,
            "min": float(valid.min()) if len(valid) > 0 else np.nan,
            "q25": float(q1),
            "q75": float(q3),
            "max": float(valid.max()) if len(valid) > 0 else np.nan,
            "n_outliers_iqr": n_outliers,
        })

    for col in cat_cols:
        if col not in X.columns:
            continue
        vals = X[col]
        missing_pct = float(vals.isna().mean())
        n_unique = int(vals.dropna().nunique())
        top_cat = vals.mode().iloc[0] if not vals.mode().empty else None
        top_pct = float((vals == top_cat).mean()) if top_cat is not None else np.nan
        records.append({
            "feature": col,
            "type": "categorical",
            "n_rows": n,
            "missing_count": int(vals.isna().sum()),
            "missing_pct": missing_pct,
            "n_unique": n_unique,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "max": np.nan,
            "n_outliers_iqr": 0,
        })

    df = pd.DataFrame(records)
    report_path = output_path / f"data_quality_{label}.csv"
    df.to_csv(report_path, index=False, float_format="%.6f")
    logger.info("Saved data quality report ({}): {} ({} features)", label, report_path, len(df))
    return report_path
