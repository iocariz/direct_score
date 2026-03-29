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


def _format_metric(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}f}"


def _format_int(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def _feature_flag_series(feature_provenance_df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in feature_provenance_df.columns:
        return feature_provenance_df[column_name].fillna(False).astype(bool)
    return pd.Series(False, index=feature_provenance_df.index, dtype=bool)


def _feature_source_column(feature_provenance_df: pd.DataFrame) -> str | None:
    if "provenance" in feature_provenance_df.columns:
        return "provenance"
    if "source" in feature_provenance_df.columns:
        return "source"
    return None


def _benchmark_p_column(benchmark_comparisons_df: pd.DataFrame) -> str | None:
    for column_name in ("auc_p_adjusted", "auc_delong_p_value", "auc_p_value"):
        if column_name in benchmark_comparisons_df.columns:
            return column_name
    return None


def _benchmark_result_label(auc_improvement, p_value) -> str:
    if auc_improvement is None or pd.isna(auc_improvement):
        return "n/a"
    if p_value is None or pd.isna(p_value) or float(p_value) >= 0.05:
        return "ns"
    return "win" if float(auc_improvement) > 0 else "loss" if float(auc_improvement) < 0 else "tie"


def _sorted_candidate_results(
    results_df: pd.DataFrame,
    model_selection_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    candidate_names = [name for name in OFFICIAL_MODEL_NAMES if name in results_df.index]
    candidate_df = results_df.loc[candidate_names].copy() if candidate_names else results_df.copy()
    candidate_df["model"] = candidate_df.index.astype(str)
    candidate_df["weighted_score"] = np.nan
    candidate_df["recommended"] = False

    if model_selection_df is not None and not model_selection_df.empty and "model" in model_selection_df.columns:
        selection_lookup = model_selection_df.set_index("model")
        if "weighted_score" in selection_lookup.columns:
            candidate_df["weighted_score"] = candidate_df["model"].map(selection_lookup["weighted_score"])
        if "recommended" in selection_lookup.columns:
            candidate_df["recommended"] = candidate_df["model"].map(selection_lookup["recommended"]).fillna(False).astype(bool)

    sort_columns = [column_name for column_name in ["weighted_score", "PR AUC", "ROC AUC"] if column_name in candidate_df.columns]
    if sort_columns:
        candidate_df = candidate_df.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    return candidate_df.reset_index(drop=True)


def _recommended_results_row(results_df: pd.DataFrame, recommended_model: str | None) -> pd.Series | None:
    if recommended_model is None or results_df is None or results_df.empty:
        return None
    if recommended_model not in results_df.index:
        return None
    return results_df.loc[recommended_model]


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
    candidate_df = _sorted_candidate_results(results_df, model_selection_df)
    recommended_row = None
    recommended_model = None
    if model_selection_df is not None and not model_selection_df.empty and "recommended" in model_selection_df.columns:
        recommended_candidates = model_selection_df.loc[model_selection_df["recommended"].fillna(False).astype(bool)]
        if not recommended_candidates.empty:
            recommended_row = recommended_candidates.iloc[0]
            recommended_model = str(recommended_row["model"])
    if recommended_row is None and not candidate_df.empty:
        recommended_model = str(candidate_df.iloc[0]["model"])
    recommended_results_row = _recommended_results_row(results_df, recommended_model)

    def _section(title):
        lines.append(f"\n{'=' * 70}")
        lines.append(f"  {title}")
        lines.append(f"{'=' * 70}\n")

    def _subsection(title):
        lines.append(f"\n--- {title} ---\n")

    # Header
    lines.append("MODEL CARD — Basel Bad Credit Scoring Model")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Random seed: {RANDOM_STATE}")

    _section("0. EXECUTIVE SUMMARY")
    if recommended_model is not None:
        lines.append(f"Recommended production candidate: {recommended_model}")
        if recommended_row is not None and "weighted_score" in recommended_row.index:
            lines.append(f"Overall model-selection score:   {_format_metric(recommended_row['weighted_score'], digits=1)} / 100")
        if results_df is not None and recommended_model in results_df.index and "N" in results_df.columns:
            lines.append(f"Hold-out evaluation rows:        {_format_int(results_df.loc[recommended_model, 'N'])}")

        lines.append("")
        lines.append("Headline findings:")
        if recommended_row is not None:
            headline_auc = recommended_results_row.get("ROC AUC") if recommended_results_row is not None else recommended_row.get("test_auc")
            headline_pr_auc = recommended_results_row.get("PR AUC") if recommended_results_row is not None else recommended_row.get("test_pr_auc")
            headline_brier = recommended_results_row.get("Brier") if recommended_results_row is not None else recommended_row.get("test_brier_calibrated", recommended_row.get("test_brier_raw", recommended_row.get("test_brier")))
            lines.append(
                "- Held-out performance: "
                f"ROC AUC {_format_metric(headline_auc)}, "
                f"PR AUC {_format_metric(headline_pr_auc)}, "
                f"Brier {_format_metric(headline_brier)}."
            )
            remaining_candidates = model_selection_df.loc[
                model_selection_df["model"] != recommended_model
            ] if model_selection_df is not None and not model_selection_df.empty and "model" in model_selection_df.columns else pd.DataFrame()
            if not remaining_candidates.empty and "weighted_score" in remaining_candidates.columns:
                runner_up = remaining_candidates.sort_values("weighted_score", ascending=False).iloc[0]
                margin = float(recommended_row["weighted_score"] - runner_up["weighted_score"])
                lines.append(
                    f"- Selection margin: {_format_metric(margin, digits=1, signed=True)} points over {runner_up['model']}."
                )

        if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty and recommended_model is not None:
            benchmark_df = benchmark_comparisons_df.loc[
                benchmark_comparisons_df["candidate_model"].isin(OFFICIAL_MODEL_NAMES)
            ].copy() if "candidate_model" in benchmark_comparisons_df.columns else benchmark_comparisons_df.copy()
            recommended_benchmarks = benchmark_df.loc[
                benchmark_df["candidate_model"] == recommended_model
            ] if "candidate_model" in benchmark_df.columns else pd.DataFrame()
            if not recommended_benchmarks.empty:
                p_column = _benchmark_p_column(recommended_benchmarks)
                best_benchmark = recommended_benchmarks.sort_values("auc_improvement", ascending=False).iloc[0]
                p_value = best_benchmark[p_column] if p_column is not None else np.nan
                lines.append(
                    "- Benchmark comparison: "
                    f"best lift versus {best_benchmark['reference_model']} is AUC Δ "
                    f"{_format_metric(best_benchmark['auc_improvement'], signed=True)} "
                    f"({ _benchmark_result_label(best_benchmark['auc_improvement'], p_value) }, "
                    f"p={_format_metric(p_value)})."
                )

        if overfit_df is not None and not overfit_df.empty and recommended_model is not None and "model" in overfit_df.columns:
            recommended_overfit = overfit_df.loc[overfit_df["model"] == recommended_model]
            if not recommended_overfit.empty:
                row = recommended_overfit.iloc[0]
                lines.append(
                    "- Generalization check: "
                    f"{row['overfit_flag']} with ROC Δ {_format_metric(row.get('auc_delta'), signed=True)}"
                    f" and PR Δ {_format_metric(row.get('pr_auc_delta'), signed=True)}."
                )
    else:
        lines.append("No recommended model summary available.")

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
    if recommended_row is not None:
        display_auc = recommended_results_row.get("ROC AUC") if recommended_results_row is not None else recommended_row.get("test_auc")
        display_pr_auc = recommended_results_row.get("PR AUC") if recommended_results_row is not None else recommended_row.get("test_pr_auc")
        display_brier = recommended_results_row.get("Brier") if recommended_results_row is not None else recommended_row.get("test_brier_calibrated", recommended_row.get("test_brier_raw", recommended_row.get("test_brier")))
        lines.append(f"Model:                  {recommended_row['model']}")
        lines.append(f"Weighted score:         {_format_metric(recommended_row['weighted_score'], digits=1)} / 100")
        lines.append(f"Test ROC AUC:           {_format_metric(display_auc)}")
        lines.append(f"Test PR AUC:            {_format_metric(display_pr_auc)}")
        if not pd.isna(display_brier):
            lines.append(f"Test Brier:             {_format_metric(display_brier)}")
        if overfit_df is not None and not overfit_df.empty and recommended_model is not None and "model" in overfit_df.columns:
            recommended_overfit = overfit_df.loc[overfit_df["model"] == recommended_model]
            if not recommended_overfit.empty:
                row = recommended_overfit.iloc[0]
                lines.append(
                    f"Generalization flag:    {row['overfit_flag']} (ROC Δ {_format_metric(row.get('auc_delta'), signed=True)}, PR Δ {_format_metric(row.get('pr_auc_delta'), signed=True)})"
                )
        if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty and recommended_model is not None:
            benchmark_df = benchmark_comparisons_df.loc[
                benchmark_comparisons_df["candidate_model"].isin(OFFICIAL_MODEL_NAMES)
            ].copy() if "candidate_model" in benchmark_comparisons_df.columns else benchmark_comparisons_df.copy()
            recommended_benchmarks = benchmark_df.loc[
                benchmark_df["candidate_model"] == recommended_model
            ] if "candidate_model" in benchmark_df.columns else pd.DataFrame()
            if not recommended_benchmarks.empty:
                p_column = _benchmark_p_column(recommended_benchmarks)
                best_benchmark = recommended_benchmarks.sort_values("auc_improvement", ascending=False).iloc[0]
                p_value = best_benchmark[p_column] if p_column is not None else np.nan
                lines.append(
                    f"Best benchmark lift:    {best_benchmark['reference_model']} | AUC Δ {_format_metric(best_benchmark['auc_improvement'], signed=True)} | p={_format_metric(p_value)}"
                )
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
    if not candidate_df.empty:
        _subsection("Held-out test set results")
        if "N" in candidate_df.columns and not candidate_df["N"].dropna().empty:
            lines.append(f"Hold-out sample size: {_format_int(candidate_df['N'].dropna().iloc[0])} rows")
            lines.append("")
        lines.append(f"{'Rank':<4s} {'Model':<22s} {'Overall':>7s} {'ROC':>7s} {'PR':>7s} {'KS':>7s} {'Brier':>7s} {'Rec':>4s}")
        lines.append("-" * 76)
        for rank, (_, row) in enumerate(candidate_df.iterrows(), start=1):
            lines.append(
                f"{rank:<4d} {str(row['model']):<22.22s} {_format_metric(row.get('weighted_score'), digits=1):>7s} "
                f"{_format_metric(row.get('ROC AUC')):>7s} {_format_metric(row.get('PR AUC')):>7s} "
                f"{_format_metric(row.get('KS')):>7s} {_format_metric(row.get('Brier')):>7s} "
                f"{'yes' if bool(row.get('recommended', False)) else '':>4s}"
            )
    else:
        lines.append("Held-out test set results not available.")

    # 4. Overfitting assessment
    _section("4. OVERFITTING ASSESSMENT")
    if overfit_df is not None and not overfit_df.empty:
        overfit_display_df = overfit_df.copy()
        if "model" in overfit_display_df.columns and not candidate_df.empty:
            order_lookup = {model_name: idx for idx, model_name in enumerate(candidate_df["model"].tolist())}
            overfit_display_df["sort_key"] = overfit_display_df["model"].map(order_lookup).fillna(len(order_lookup))
            overfit_display_df = overfit_display_df.sort_values(["sort_key", "auc_delta"], ascending=[True, False]).drop(columns=["sort_key"])
        lines.append(f"{'Model':<22s} {'Train ROC':>9s} {'Test ROC':>8s} {'Δ ROC':>7s} {'Δ PR':>7s} {'Flag':>6s}")
        lines.append("-" * 68)
        for _, row in overfit_display_df.iterrows():
            lines.append(
                f"{row['model']:<22.22s} {_format_metric(row.get('train_auc')):>9s} {_format_metric(row.get('test_auc')):>8s} "
                f"{_format_metric(row.get('auc_delta'), signed=True):>7s} {_format_metric(row.get('pr_auc_delta'), signed=True):>7s} {row['overfit_flag']:>6s}"
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
        benchmark_df = benchmark_comparisons_df.loc[
            benchmark_comparisons_df["candidate_model"].isin(OFFICIAL_MODEL_NAMES)
        ].copy() if "candidate_model" in benchmark_comparisons_df.columns else benchmark_comparisons_df.copy()
        p_column = _benchmark_p_column(benchmark_df)

        if recommended_model is not None and "candidate_model" in benchmark_df.columns:
            recommended_benchmarks = benchmark_df.loc[
                benchmark_df["candidate_model"] == recommended_model
            ].sort_values("auc_improvement", ascending=False)
            if not recommended_benchmarks.empty:
                _subsection("Recommended model vs benchmarks")
                for _, row in recommended_benchmarks.iterrows():
                    p_value = row[p_column] if p_column is not None else np.nan
                    lines.append(
                        f"{row['reference_model']:<28.28s} AUC Δ {_format_metric(row['auc_improvement'], signed=True)} | "
                        f"95% CI [{_format_metric(row.get('auc_improvement_lo'), signed=True)}, {_format_metric(row.get('auc_improvement_hi'), signed=True)}] | "
                        f"p={_format_metric(p_value)} | {_benchmark_result_label(row['auc_improvement'], p_value)}"
                    )

        if "candidate_model" in benchmark_df.columns:
            _subsection("Best benchmark comparison per candidate")
            lines.append(f"{'Candidate':<22s} {'Best benchmark':<24s} {'AUC Δ':>8s} {'95% CI':>25s} {'Result':>7s}")
            lines.append("-" * 92)
            candidate_order = candidate_df["model"].tolist() if not candidate_df.empty else OFFICIAL_MODEL_NAMES
            for candidate_name in candidate_order:
                candidate_rows = benchmark_df.loc[benchmark_df["candidate_model"] == candidate_name]
                if candidate_rows.empty:
                    continue
                row = candidate_rows.sort_values("auc_improvement", ascending=False).iloc[0]
                p_value = row[p_column] if p_column is not None else np.nan
                ci_text = f"[{_format_metric(row.get('auc_improvement_lo'), signed=True)}, {_format_metric(row.get('auc_improvement_hi'), signed=True)}]"
                lines.append(
                    f"{candidate_name:<22.22s} {str(row['reference_model']):<24.24s} "
                    f"{_format_metric(row['auc_improvement'], signed=True):>8s} {ci_text:>25s} "
                    f"{_benchmark_result_label(row['auc_improvement'], p_value):>7s}"
                )
    else:
        lines.append("No benchmark comparisons available.")

    # 6. Population coverage
    _section("6. POPULATION COVERAGE")
    if population_summary_df is not None and not population_summary_df.empty:
        population_df = population_summary_df.copy()
        split_order = {"pre_split": 0, "post_split": 1}
        status_order = {"Booked": 0, "Rejected": 1, "Canceled": 2}
        if "split" in population_df.columns:
            population_df["split_order"] = population_df["split"].map(split_order).fillna(99)
        if "status_name" in population_df.columns:
            population_df["status_order"] = population_df["status_name"].map(status_order).fillna(99)
        population_df = population_df.sort_values(["split_order", "status_order"]).drop(columns=[col for col in ["split_order", "status_order"] if col in population_df.columns])
        lines.append(f"{'Split':<10s} {'Status':<10s} {'Rows':>10s} {'Observed':>10s} {'Bad rate':>10s} {'Window':<24s}")
        lines.append("-" * 82)
        for _, row in population_df.iterrows():
            observed_n = row.get("n_with_observed_target")
            booked_row = str(row.get("status_name", "")) == "Booked"
            bad_rate = np.nan
            if booked_row and observed_n is not None and not pd.isna(observed_n) and float(observed_n) > 0:
                bad_rate = float(row.get("n_bad_observed", np.nan)) / float(observed_n)
            window = ""
            if "date_start" in row.index and "date_end" in row.index:
                window = f"{row['date_start']} to {row['date_end']}"
            bad_rate_display = f"{bad_rate * 100:.2f}%" if not pd.isna(bad_rate) else "N/A"
            lines.append(
                f"{str(row['split']):<10.10s} {str(row['status_name']):<10.10s} {_format_int(row['n_rows']):>10s} "
                f"{_format_int(observed_n):>10s} {bad_rate_display:>10s} {window:<24.24s}"
            )
    else:
        lines.append("Population summary not available.")

    # 7. Feature inventory
    _section("7. FEATURE INVENTORY")
    if feature_provenance_df is not None and not feature_provenance_df.empty:
        n_total = len(feature_provenance_df)
        candidate_mask = _feature_flag_series(feature_provenance_df, "rfecv_candidate")
        selected_mask = _feature_flag_series(feature_provenance_df, "rfecv_kept")
        n_candidates = int(candidate_mask.sum()) if candidate_mask.any() else n_total
        n_selected = int(selected_mask.sum()) if selected_mask.any() else n_candidates
        selected_df = feature_provenance_df.loc[selected_mask].copy() if selected_mask.any() else feature_provenance_df.copy()
        source_column = _feature_source_column(feature_provenance_df)

        lines.append(f"Total feature inventory:         {n_total}")
        lines.append(f"Model-selection candidates:      {n_candidates}")
        lines.append(f"Selected modeling features:      {n_selected}")
        if "data_type" in selected_df.columns:
            data_type_counts = selected_df["data_type"].value_counts()
            lines.append(f"Selected numerical features:     {int(data_type_counts.get('numerical', 0))}")
            lines.append(f"Selected categorical features:   {int(data_type_counts.get('categorical', 0))}")
        lines.append("Selection method:                temporal elastic-net stability selection")
        if source_column is not None:
            lines.append("")
            lines.append("Selected feature mix by provenance:")
            for source, count in selected_df[source_column].value_counts().items():
                lines.append(f"  {source}: {int(count)}")
        if "interaction_type" in selected_df.columns and selected_df["interaction_type"].notna().any():
            lines.append("")
            lines.append("Selected interactions by type:")
            for interaction_type, count in selected_df["interaction_type"].dropna().value_counts().items():
                lines.append(f"  {interaction_type}: {int(count)}")
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
    lines.append("3. Calibration is model-dependent on held-out booked ground-truth")
    lines.append("   samples: sigmoid scaling for additive/log-odds models and")
    lines.append("   isotonic calibration for tree ensembles.")
    lines.append("4. Monotonicity constraints applied to tree models where domain")
    lines.append(f"   knowledge dictates direction ({len(MONOTONE_MAP)} features constrained).")

    # 9. Technical configuration
    _section("9. TECHNICAL CONFIGURATION")
    lines.append(f"Random seed:              {RANDOM_STATE}")
    lines.append(f"N bootstrap iterations:   {N_BOOTSTRAP}")
    lines.append(f"Early stopping rounds:    {EARLY_STOPPING_ROUNDS}")
    lines.append(f"Max estimators:           {N_ESTIMATORS_CEILING}")
    lines.append(f"Calibration fraction:     {CALIBRATION_FRACTION:.0%}")
    lines.append("Feature selector:         temporal elastic-net stability selection")
    lines.append(f"Feature discovery split:  {FEATURE_DISCOVERY_FRACTION:.0%}")

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
