import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from training_constants import (
    BENCHMARK_MODEL_NAMES,
    CONCEPT_DRIFT_DELTA_THRESHOLD,
    DEFAULT_THRESHOLD_PCTS,
    DEFAULT_TIER_THRESHOLDS,
    MODEL_SELECTION_QUALITY_FLOOR,
    N_BOOTSTRAP,
    OVERFIT_DELTA_THRESHOLD,
    RANDOM_STATE,
    SUMMARY_MODEL_NAMES,
    TARGET,
)


def sanitize_output_name(name: str) -> str:
    """Normalize a model name into a filesystem-safe basename.

    Strips path-traversal characters (/, \\, ..) and any leading/trailing
    separators so the result is safe to concatenate onto an output directory.
    """
    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("\x00", "")
    )
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    all_scores = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_cdf = np.searchsorted(pos, all_scores, side="right") / len(pos)
    neg_cdf = np.searchsorted(neg, all_scores, side="right") / len(neg)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


def calibration_diagnostics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    """Compute Expected/Maximum Calibration Error and a per-bin reliability table.

    Bins are equal-frequency over predicted probabilities (deciles by default).
    Returns:
        ece — Expected Calibration Error: weighted mean |predicted − observed|
              across bins, the standard regulator-facing calibration metric.
        mce — Maximum Calibration Error: worst-bin |predicted − observed|,
              the upper-bound view that surfaces tail miscalibration.
        bins_df — DataFrame with one row per bin: count, mean predicted
              probability, observed default rate, gap, weight.

    Bins with fewer than 2 rows are dropped from the ECE/MCE computation but
    retained in the bins_df for transparency.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) == 0:
        return float("nan"), float("nan"), pd.DataFrame(
            columns=["bin", "n", "mean_predicted", "observed_default_rate", "gap", "weight"]
        )

    # Equal-frequency binning via score quantiles. duplicates="drop" silently
    # collapses ties; the resulting bin count may be smaller than n_bins for
    # heavily-tied scores.
    try:
        bin_edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
        if len(bin_edges) < 2:
            bin_edges = np.array([s.min(), s.max() + 1e-9])
        bin_idx = np.clip(np.digitize(s, bin_edges[1:-1]), 0, len(bin_edges) - 2)
    except ValueError:
        return float("nan"), float("nan"), pd.DataFrame(
            columns=["bin", "n", "mean_predicted", "observed_default_rate", "gap", "weight"]
        )

    records = []
    n_total = len(y)
    for b in sorted(np.unique(bin_idx)):
        sel = bin_idx == b
        n_b = int(sel.sum())
        if n_b == 0:
            continue
        mean_pred = float(s[sel].mean())
        obs_rate = float(y[sel].mean())
        records.append({
            "bin": int(b),
            "n": n_b,
            "mean_predicted": mean_pred,
            "observed_default_rate": obs_rate,
            "gap": mean_pred - obs_rate,
            "weight": n_b / n_total,
        })
    bins_df = pd.DataFrame(records)

    usable = bins_df.loc[bins_df["n"] >= 2]
    if usable.empty:
        return float("nan"), float("nan"), bins_df

    abs_gap = usable["gap"].abs()
    ece = float((usable["weight"] * abs_gap).sum() / usable["weight"].sum())
    mce = float(abs_gap.max())
    return ece, mce, bins_df


def compute_calibration_diagnostics_report(
    y_true: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model ECE/MCE + per-bin reliability tables.

    Returns:
        summary_df — one row per model: ece, mce, n, n_bins_used.
        details_df — long form, one row per (model, bin) with predicted vs
                     observed gap, suitable for plotting reliability diagrams.
    """
    summary_records: list[dict] = []
    detail_records: list[dict] = []
    for name, scores in score_arrays.items():
        s = np.asarray(scores, dtype=float)
        # Skip non-probability scores (e.g. legacy benchmarks negated for AUC):
        # ECE is only meaningful when the score lives in [0, 1].
        finite = s[np.isfinite(s)]
        if len(finite) == 0 or finite.min() < -1e-6 or finite.max() > 1 + 1e-6:
            continue
        ece, mce, bins_df = calibration_diagnostics(y_true, s, n_bins=n_bins)
        summary_records.append({
            "model": name,
            "ece": ece,
            "mce": mce,
            "n": int(np.isfinite(s).sum()),
            "n_bins_used": int((bins_df["n"] >= 2).sum()) if not bins_df.empty else 0,
        })
        for _, row in bins_df.iterrows():
            detail_records.append({"model": name, **row.to_dict()})
    summary_df = pd.DataFrame(summary_records)
    details_df = pd.DataFrame(detail_records)
    return summary_df, details_df


def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, is_probability: bool = True) -> dict:
    mask = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    roc_auc = roc_auc_score(y_true, y_score)
    result = {
        "Model": name,
        "ROC AUC": roc_auc,
        "Gini": 2 * roc_auc - 1,
        "KS": _ks_statistic(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "N": mask.sum(),
    }
    result["Brier"] = brier_score_loss(y_true, np.clip(y_score, 0, 1)) if is_probability else np.nan
    return result


def evaluate_safely(name: str, y_true: np.ndarray, y_score: np.ndarray, is_probability: bool = True) -> dict:
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask]
    y_score = y_score[mask]

    result = {
        "Model": name,
        "ROC AUC": np.nan,
        "Gini": np.nan,
        "KS": np.nan,
        "PR AUC": np.nan,
        "N": int(mask.sum()),
        "Brier": np.nan,
    }
    if len(y_true) == 0 or np.unique(y_true).size < 2:
        return result

    roc_auc = roc_auc_score(y_true, y_score)
    result["ROC AUC"] = roc_auc
    result["Gini"] = 2 * roc_auc - 1
    result["KS"] = _ks_statistic(y_true, y_score)
    result["PR AUC"] = average_precision_score(y_true, y_score)
    if is_probability:
        result["Brier"] = brier_score_loss(y_true, np.clip(y_score, 0, 1))
    return result


def build_holdout_score_frame(
    y_true: pd.Series | np.ndarray,
    score_arrays: dict[str, np.ndarray],
    population_mode: str | None = None,
    evaluation_population: str | None = None,
) -> pd.DataFrame:
    score_frame = pd.DataFrame({TARGET: np.asarray(y_true, dtype=float)})
    for name, scores in score_arrays.items():
        score_frame[f"score__{sanitize_output_name(name)}"] = np.asarray(scores, dtype=float)
    if population_mode is not None:
        score_frame["population_mode"] = population_mode
    if evaluation_population is not None:
        score_frame["evaluation_population"] = evaluation_population
    return score_frame


def evaluate_all(
    X_test,
    y_test,
    models: dict,
    bench_risk_score_rf,
    bench_score_RF,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    logger.info("Test set: {:,} rows ({:.2%} positive rate)", len(y_test), y_test.mean())
    results = []
    score_arrays: dict[str, np.ndarray] = {}

    for name, mdl in models.items():
        y_proba = mdl.predict_proba(X_test)[:, 1]
        score_arrays[name] = y_proba
        results.append(evaluate(name, y_test.values, y_proba))

    for name, scores in zip(BENCHMARK_MODEL_NAMES, [bench_risk_score_rf, bench_score_RF], strict=True):
        score_arrays[name] = -scores.values
        results.append(evaluate(name, y_test.values, -scores.values, is_probability=False))

    results_df = pd.DataFrame(results).set_index("Model")
    results_df = results_df.sort_values("PR AUC", ascending=False)

    logger.info("")
    logger.info("{:<35s} {:>8s} {:>6s} {:>6s} {:>8s} {:>8s}", "Model", "ROC AUC", "Gini", "KS", "PR AUC", "Brier")
    logger.info("{}", "─" * 77)
    for model_name, row in results_df.iterrows():
        brier_str = f"{row['Brier']:.4f}" if not np.isnan(row["Brier"]) else "   —"
        logger.info(
            "{:<35s} {:>8.4f} {:>6.4f} {:>6.4f} {:>8.4f} {:>8s}",
            model_name,
            row["ROC AUC"],
            row["Gini"],
            row["KS"],
            row["PR AUC"],
            brier_str,
        )
    logger.info("")

    return results_df, score_arrays


def _score_is_probability(scores: np.ndarray) -> bool:
    finite_scores = np.asarray(scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    return len(finite_scores) > 0 and float(finite_scores.min()) >= 0 and float(finite_scores.max()) <= 1.0 + 1e-9


def _score_metric(y_true: np.ndarray, scores: np.ndarray, metric_name: str, is_probability: bool) -> float:
    try:
        if metric_name == "AUC":
            return roc_auc_score(y_true, scores)
        if metric_name == "PR_AUC":
            return average_precision_score(y_true, scores)
        if metric_name == "Brier":
            if not is_probability:
                return np.nan
            return brier_score_loss(y_true, np.clip(scores, 0, 1))
    except ValueError:
        return np.nan
    raise ValueError(f"Unsupported metric: {metric_name}")


def _metric_improvement(metric_name: str, candidate_value: float, reference_value: float) -> float:
    if np.isnan(candidate_value) or np.isnan(reference_value):
        return np.nan
    if metric_name == "Brier":
        return reference_value - candidate_value
    return candidate_value - reference_value


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    sort_order = np.argsort(x)
    sorted_x = x[sort_order]
    sorted_midranks = np.zeros(len(x), dtype=float)

    i = 0
    while i < len(sorted_x):
        j = i
        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1
        sorted_midranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j

    midranks = np.empty(len(x), dtype=float)
    midranks[sort_order] = sorted_midranks
    return midranks


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int) -> tuple[np.ndarray, np.ndarray]:
    n_classifiers, n_examples = predictions_sorted_transposed.shape
    m = label_1_count
    n = n_examples - m

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((n_classifiers, m), dtype=float)
    ty = np.empty((n_classifiers, n), dtype=float)
    tz = np.empty((n_classifiers, n_examples), dtype=float)

    for classifier_idx in range(n_classifiers):
        tx[classifier_idx] = _compute_midrank(positive_examples[classifier_idx])
        ty[classifier_idx] = _compute_midrank(negative_examples[classifier_idx])
        tz[classifier_idx] = _compute_midrank(predictions_sorted_transposed[classifier_idx])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    return aucs, sx / m + sy / n


def delong_auc_test(
    y_true: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
) -> dict:
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    y_true = np.asarray(y_true)
    mask = np.isfinite(y_true) & np.isfinite(candidate_scores) & np.isfinite(reference_scores)
    y = y_true[mask].astype(int)
    candidate = candidate_scores[mask]
    reference = reference_scores[mask]

    if len(y) == 0 or np.unique(y).size < 2:
        return {
            "n": len(y),
            "n_pos": int((y == 1).sum()),
            "n_neg": int((y == 0).sum()),
            "candidate_auc": np.nan,
            "reference_auc": np.nan,
            "auc_improvement": np.nan,
            "auc_se": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
        }

    candidate_auc = roc_auc_score(y, candidate)
    reference_auc = roc_auc_score(y, reference)
    auc_improvement = candidate_auc - reference_auc
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos < 2 or n_neg < 2:
        return {
            "n": len(y),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "candidate_auc": candidate_auc,
            "reference_auc": reference_auc,
            "auc_improvement": auc_improvement,
            "auc_se": np.nan,
            "z_score": np.nan,
            "p_value": np.nan,
        }

    order = np.argsort(-y)
    predictions_sorted = np.vstack([candidate, reference])[:, order]
    aucs, delong_cov = _fast_delong(predictions_sorted, n_pos)
    auc_improvement = float(aucs[0] - aucs[1])
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ delong_cov @ contrast.T)
    variance = max(variance, 0.0)
    auc_se = np.sqrt(variance)

    if auc_se == 0:
        z_score = 0.0 if auc_improvement == 0 else np.sign(auc_improvement) * np.inf
        p_value = 1.0 if auc_improvement == 0 else 0.0
    else:
        z_score = auc_improvement / auc_se
        p_value = 2 * (1 - NormalDist().cdf(abs(z_score)))

    return {
        "n": len(y),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "candidate_auc": float(aucs[0]),
        "reference_auc": float(aucs[1]),
        "auc_improvement": auc_improvement,
        "auc_se": auc_se,
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def paired_bootstrap_metric_delta(
    y_true: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    metric_name: str,
    candidate_is_probability: bool | None = None,
    reference_is_probability: bool | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> dict:
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    reference_scores = np.asarray(reference_scores, dtype=float)
    y_true = np.asarray(y_true)
    mask = np.isfinite(y_true) & np.isfinite(candidate_scores) & np.isfinite(reference_scores)
    y = y_true[mask].astype(int)
    candidate = candidate_scores[mask]
    reference = reference_scores[mask]

    if candidate_is_probability is None:
        candidate_is_probability = _score_is_probability(candidate)
    if reference_is_probability is None:
        reference_is_probability = _score_is_probability(reference)

    candidate_value = _score_metric(y, candidate, metric_name, candidate_is_probability)
    reference_value = _score_metric(y, reference, metric_name, reference_is_probability)
    observed_improvement = _metric_improvement(metric_name, candidate_value, reference_value)

    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(y) == 0 or len(idx_pos) == 0 or len(idx_neg) == 0:
        return {
            "n": len(y),
            "candidate_value": candidate_value,
            "reference_value": reference_value,
            "improvement": observed_improvement,
            "improvement_lo": np.nan,
            "improvement_hi": np.nan,
            "p_value": np.nan,
            "n_bootstrap": 0,
        }

    alpha = (1 - ci) / 2
    rng = np.random.RandomState(RANDOM_STATE)
    improvements = []
    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        sampled_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sampled_idx = np.concatenate([sampled_pos, sampled_neg])
        y_sample = y[sampled_idx]
        candidate_sample = candidate[sampled_idx]
        reference_sample = reference[sampled_idx]
        candidate_metric = _score_metric(y_sample, candidate_sample, metric_name, candidate_is_probability)
        reference_metric = _score_metric(y_sample, reference_sample, metric_name, reference_is_probability)
        improvement = _metric_improvement(metric_name, candidate_metric, reference_metric)
        if not np.isnan(improvement):
            improvements.append(improvement)

    if not improvements:
        return {
            "n": len(y),
            "candidate_value": candidate_value,
            "reference_value": reference_value,
            "improvement": observed_improvement,
            "improvement_lo": np.nan,
            "improvement_hi": np.nan,
            "p_value": np.nan,
            "n_bootstrap": 0,
        }

    improvements_array = np.asarray(improvements, dtype=float)
    lower_tail = (np.sum(improvements_array <= 0) + 1) / (len(improvements_array) + 1)
    upper_tail = (np.sum(improvements_array >= 0) + 1) / (len(improvements_array) + 1)
    p_value = min(1.0, 2 * min(lower_tail, upper_tail))

    return {
        "n": len(y),
        "candidate_value": candidate_value,
        "reference_value": reference_value,
        "improvement": observed_improvement,
        "improvement_lo": np.percentile(improvements_array, 100 * alpha),
        "improvement_hi": np.percentile(improvements_array, 100 * (1 - alpha)),
        "p_value": p_value,
        "n_bootstrap": len(improvements_array),
    }


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    valid = np.isfinite(p)
    adjusted = np.full(n, np.nan)
    if not valid.any():
        return adjusted
    idx = np.where(valid)[0]
    p_valid = p[idx]
    order = np.argsort(p_valid)
    sorted_p = p_valid[order]
    corrected = np.empty(len(sorted_p))
    cummax = 0.0
    for i, pv in enumerate(sorted_p):
        corrected_p = pv * (len(sorted_p) - i)
        cummax = max(cummax, corrected_p)
        corrected[i] = min(cummax, 1.0)
    # Undo sort
    result = np.empty(len(sorted_p))
    result[order] = corrected
    adjusted[idx] = result
    return adjusted


def paired_bootstrap_benchmark_comparisons(
    y_true: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    candidate_model_names: list[str],
    reference_model_names: list[str] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> pd.DataFrame:
    if reference_model_names is None:
        reference_model_names = BENCHMARK_MODEL_NAMES

    records = []
    for candidate_name in candidate_model_names:
        if candidate_name not in score_arrays:
            continue
        candidate_scores = score_arrays[candidate_name]
        candidate_is_probability = _score_is_probability(candidate_scores)
        for reference_name in reference_model_names:
            if reference_name not in score_arrays:
                continue
            reference_scores = score_arrays[reference_name]
            reference_is_probability = _score_is_probability(reference_scores)
            delong_stats = delong_auc_test(y_true, candidate_scores, reference_scores)

            auc_stats = paired_bootstrap_metric_delta(
                y_true,
                candidate_scores,
                reference_scores,
                "AUC",
                candidate_is_probability=candidate_is_probability,
                reference_is_probability=reference_is_probability,
                n_bootstrap=n_bootstrap,
                ci=ci,
            )
            pr_stats = paired_bootstrap_metric_delta(
                y_true,
                candidate_scores,
                reference_scores,
                "PR_AUC",
                candidate_is_probability=candidate_is_probability,
                reference_is_probability=reference_is_probability,
                n_bootstrap=n_bootstrap,
                ci=ci,
            )
            brier_stats = paired_bootstrap_metric_delta(
                y_true,
                candidate_scores,
                reference_scores,
                "Brier",
                candidate_is_probability=candidate_is_probability,
                reference_is_probability=reference_is_probability,
                n_bootstrap=n_bootstrap,
                ci=ci,
            )

            # Effect size: AUC improvement as % of gap to perfect (1.0)
            ref_auc = auc_stats["reference_value"]
            auc_gap = 1.0 - ref_auc if not np.isnan(ref_auc) else np.nan
            auc_effect_pct = (auc_stats["improvement"] / auc_gap * 100) if auc_gap > 0 else np.nan

            records.append(
                {
                    "candidate_model": candidate_name,
                    "reference_model": reference_name,
                    "n": auc_stats["n"],
                    "n_pos": delong_stats["n_pos"],
                    "n_neg": delong_stats["n_neg"],
                    "candidate_auc": auc_stats["candidate_value"],
                    "reference_auc": auc_stats["reference_value"],
                    "auc_improvement": auc_stats["improvement"],
                    "auc_improvement_lo": auc_stats["improvement_lo"],
                    "auc_improvement_hi": auc_stats["improvement_hi"],
                    "auc_improvement_pct_of_max": auc_effect_pct,
                    "auc_p_value": auc_stats["p_value"],
                    "auc_delong_se": delong_stats["auc_se"],
                    "auc_delong_z": delong_stats["z_score"],
                    "auc_delong_p_value": delong_stats["p_value"],
                    "candidate_pr_auc": pr_stats["candidate_value"],
                    "reference_pr_auc": pr_stats["reference_value"],
                    "pr_auc_improvement": pr_stats["improvement"],
                    "pr_auc_improvement_lo": pr_stats["improvement_lo"],
                    "pr_auc_improvement_hi": pr_stats["improvement_hi"],
                    "pr_auc_p_value": pr_stats["p_value"],
                    "candidate_brier": brier_stats["candidate_value"],
                    "reference_brier": brier_stats["reference_value"],
                    "brier_improvement": brier_stats["improvement"],
                    "brier_improvement_lo": brier_stats["improvement_lo"],
                    "brier_improvement_hi": brier_stats["improvement_hi"],
                    "brier_p_value": brier_stats["p_value"],
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "candidate_model", "reference_model", "n", "n_pos", "n_neg",
                "candidate_auc", "reference_auc", "auc_improvement", "auc_improvement_lo", "auc_improvement_hi",
                "auc_improvement_pct_of_max", "auc_p_value",
                "auc_delong_se", "auc_delong_z", "auc_delong_p_value",
                "candidate_pr_auc", "reference_pr_auc", "pr_auc_improvement", "pr_auc_improvement_lo", "pr_auc_improvement_hi", "pr_auc_p_value",
                "candidate_brier", "reference_brier", "brier_improvement", "brier_improvement_lo", "brier_improvement_hi", "brier_p_value",
                "auc_p_adjusted", "pr_auc_p_adjusted", "brier_p_adjusted",
            ]
        )

    df = pd.DataFrame(records).sort_values(["reference_model", "auc_improvement"], ascending=[True, False]).reset_index(drop=True)

    # Holm-Bonferroni correction across all tests per metric
    for p_col, adj_col in [
        ("auc_p_value", "auc_p_adjusted"),
        ("pr_auc_p_value", "pr_auc_p_adjusted"),
        ("brier_p_value", "brier_p_adjusted"),
    ]:
        df[adj_col] = _holm_bonferroni(df[p_col].values)

    return df


def split_leaderboard_results(
    results_df: pd.DataFrame,
    reject_inference: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_rows = []
    experimental_rows = []
    for model_name in results_df.index:
        if reject_inference and model_name not in BENCHMARK_MODEL_NAMES:
            experimental_rows.append(model_name)
        elif "(experimental)" in model_name:
            experimental_rows.append(model_name)
        else:
            official_rows.append(model_name)
    return results_df.loc[official_rows].copy(), results_df.loc[experimental_rows].copy()


def extract_feature_importance(
    models: dict,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    feature_names = num_cols + cat_cols
    records = []

    for name, model in models.items():
        if "(calibrated)" in name or name.startswith("Stacking") or name.startswith("Ensemble"):
            continue
        if not hasattr(model, "named_steps") or model.named_steps is None:
            continue

        clf = model.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            importance_names = feature_names
            imp_type = "split_importance"
        elif hasattr(clf, "coef_"):
            importances = clf.coef_[0]
            importance_names = feature_names
            imp_type = "coefficient"
        elif hasattr(clf, "term_importances"):
            importances = clf.term_importances()
            importance_names = list(getattr(clf, "term_names_", feature_names[: len(importances)]))
            imp_type = "term_importance"
        else:
            continue

        for feat, imp in zip(importance_names, importances):
            records.append({"model": name, "feature": feat, "importance": imp, "type": imp_type})

    df = pd.DataFrame(records)
    if not df.empty:
        df["abs_importance"] = df["importance"].abs()
        df = df.sort_values(["model", "abs_importance"], ascending=[True, False])
    return df


def plot_score_distributions(
    y_true: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    output_path: Path,
    title_prefix: str = "Test",
) -> None:
    main_models = SUMMARY_MODEL_NAMES
    to_plot = [(name, score_arrays[name]) for name in main_models if name in score_arrays]

    if not to_plot:
        return

    n = len(to_plot)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (name, scores) in zip(axes, to_plot):
        mask = ~np.isnan(scores)
        s, y = scores[mask], y_true[mask]

        ax.hist(s[y == 0], bins=50, alpha=0.5, density=True, color="steelblue", label=f"Good (n={int((y == 0).sum()):,})")
        ax.hist(s[y == 1], bins=50, alpha=0.5, density=True, color="tomato", label=f"Bad (n={int((y == 1).sum()):,})")
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Predicted P(default)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    for i in range(len(to_plot), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"{title_prefix} Set — Score Distributions by Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: {}", output_path)


def save_artifacts(
    models: dict,
    results_df: pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    output_dir: Path,
    experimental_results_df: pd.DataFrame | None = None,
    benchmark_comparisons_df: pd.DataFrame | None = None,
    experimental_benchmark_comparisons_df: pd.DataFrame | None = None,
    feature_provenance_df: pd.DataFrame | None = None,
    interaction_leaderboard_df: pd.DataFrame | None = None,
    feature_discovery_boundary_df: pd.DataFrame | None = None,
    ablation_results_df: pd.DataFrame | None = None,
    rolling_oot_results_df: pd.DataFrame | None = None,
    rolling_oot_summary_df: pd.DataFrame | None = None,
    population_summary_df: pd.DataFrame | None = None,
    applicant_scores_df: pd.DataFrame | None = None,
    holdout_scores_df: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.csv"
    results_df.to_csv(results_path, float_format="%.6f")
    logger.info("Saved results: {}", results_path)

    if experimental_results_df is not None and not experimental_results_df.empty:
        experimental_results_path = output_dir / "results_experimental.csv"
        experimental_results_df.to_csv(experimental_results_path, float_format="%.6f")
        logger.info("Saved experimental results: {}", experimental_results_path)

    if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
        benchmark_comparisons_path = output_dir / "benchmark_comparisons.csv"
        benchmark_comparisons_df.to_csv(benchmark_comparisons_path, index=False, float_format="%.6f")
        logger.info("Saved benchmark comparisons: {}", benchmark_comparisons_path)

    if experimental_benchmark_comparisons_df is not None and not experimental_benchmark_comparisons_df.empty:
        experimental_benchmark_comparisons_path = output_dir / "benchmark_comparisons_experimental.csv"
        experimental_benchmark_comparisons_df.to_csv(
            experimental_benchmark_comparisons_path,
            index=False,
            float_format="%.6f",
        )
        logger.info("Saved experimental benchmark comparisons: {}", experimental_benchmark_comparisons_path)

    if not feat_imp_df.empty:
        imp_path = output_dir / "feature_importance.csv"
        feat_imp_df.to_csv(imp_path, index=False, float_format="%.6f")
        logger.info("Saved feature importance: {} ({} rows)", imp_path, len(feat_imp_df))

    # Calibration diagnostics — Expected Calibration Error and per-bin
    # reliability table for every probabilistic model present in the holdout
    # score frame. ECE is the standard regulator-facing calibration metric;
    # the per-bin frame supports reliability-diagram plotting.
    if holdout_scores_df is not None and not holdout_scores_df.empty and TARGET in holdout_scores_df.columns:
        score_cols = [c for c in holdout_scores_df.columns if c.startswith("score__")]
        if score_cols:
            y_holdout = holdout_scores_df[TARGET].astype(int).to_numpy()
            score_arrays = {
                col[len("score__"):]: holdout_scores_df[col].to_numpy()
                for col in score_cols
            }
            cal_summary, cal_details = compute_calibration_diagnostics_report(y_holdout, score_arrays)
            if not cal_summary.empty:
                cal_summary_path = output_dir / "calibration_diagnostics.csv"
                cal_summary.to_csv(cal_summary_path, index=False, float_format="%.6f")
                logger.info(
                    "Saved calibration diagnostics: {} ({} models)", cal_summary_path, len(cal_summary)
                )
            if not cal_details.empty:
                cal_details_path = output_dir / "calibration_reliability_bins.csv"
                cal_details.to_csv(cal_details_path, index=False, float_format="%.6f")
                logger.info(
                    "Saved per-bin reliability table: {} ({} rows)", cal_details_path, len(cal_details)
                )

    if feature_provenance_df is not None and not feature_provenance_df.empty:
        feature_provenance_path = output_dir / "feature_provenance.csv"
        feature_provenance_df.to_csv(feature_provenance_path, index=False)
        logger.info("Saved feature provenance: {} ({} rows)", feature_provenance_path, len(feature_provenance_df))

    if interaction_leaderboard_df is not None and not interaction_leaderboard_df.empty:
        interaction_leaderboard_path = output_dir / "interaction_leaderboard.csv"
        interaction_leaderboard_df.to_csv(interaction_leaderboard_path, index=False, float_format="%.6f")
        logger.info("Saved interaction leaderboard: {} ({} rows)", interaction_leaderboard_path, len(interaction_leaderboard_df))

    if feature_discovery_boundary_df is not None and not feature_discovery_boundary_df.empty:
        feature_discovery_boundary_path = output_dir / "feature_discovery_boundary.csv"
        feature_discovery_boundary_df.to_csv(feature_discovery_boundary_path, index=False)
        logger.info("Saved feature discovery boundary: {} ({} rows)", feature_discovery_boundary_path, len(feature_discovery_boundary_df))

    if ablation_results_df is not None and not ablation_results_df.empty:
        ablation_results_path = output_dir / "ablation_results.csv"
        ablation_results_df.to_csv(ablation_results_path, index=False, float_format="%.6f")
        logger.info("Saved ablation results: {} ({} rows)", ablation_results_path, len(ablation_results_df))

    if rolling_oot_results_df is not None and not rolling_oot_results_df.empty:
        rolling_oot_results_path = output_dir / "rolling_oot_results.csv"
        rolling_oot_results_df.to_csv(rolling_oot_results_path, index=False, float_format="%.6f")
        logger.info("Saved rolling OOT results: {} ({} rows)", rolling_oot_results_path, len(rolling_oot_results_df))

    if rolling_oot_summary_df is not None and not rolling_oot_summary_df.empty:
        rolling_oot_summary_path = output_dir / "rolling_oot_summary.csv"
        rolling_oot_summary_df.to_csv(rolling_oot_summary_path, index=False, float_format="%.6f")
        logger.info("Saved rolling OOT summary: {} ({} rows)", rolling_oot_summary_path, len(rolling_oot_summary_df))

    if population_summary_df is not None and not population_summary_df.empty:
        population_summary_path = output_dir / "population_summary.csv"
        population_summary_df.to_csv(population_summary_path, index=False)
        logger.info("Saved population summary: {} ({} rows)", population_summary_path, len(population_summary_df))

    if applicant_scores_df is not None and not applicant_scores_df.empty:
        applicant_scores_path = output_dir / "applicant_scores_post_split.csv"
        applicant_scores_df.to_csv(applicant_scores_path, index=False, float_format="%.6f")
        logger.info("Saved applicant scores: {} ({} rows)", applicant_scores_path, len(applicant_scores_df))

    if holdout_scores_df is not None and not holdout_scores_df.empty:
        holdout_scores_path = output_dir / "holdout_test_scores.csv"
        holdout_scores_df.to_csv(holdout_scores_path, index=False, float_format="%.6f")
        logger.info("Saved hold-out test scores: {} ({} rows)", holdout_scores_path, len(holdout_scores_df))

    # Risk tier thresholds — close the artifact contract with scoring.py.
    # ScoringService.from_output_dir reads risk_tiers.json if present; without
    # this write the file was a dead load.
    tier_path = output_dir / "risk_tiers.json"
    with tier_path.open("w", encoding="utf-8") as f:
        json.dump(
            [list(entry) for entry in DEFAULT_TIER_THRESHOLDS],
            f,
            indent=2,
        )
    logger.info("Saved risk tiers: {}", tier_path)

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    checksums: dict[str, str] = {}
    for name, model in models.items():
        safe_name = sanitize_output_name(name)
        path = models_dir / f"{safe_name}.joblib"
        joblib.dump(model, path)
        checksums[f"{safe_name}.joblib"] = hashlib.sha256(path.read_bytes()).hexdigest()
    checksums_path = models_dir / "checksums.json"
    with checksums_path.open("w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)
    logger.info("Saved {} models to {} (checksums: {})", len(models), models_dir, checksums_path)


def _block_jackknife_metrics(
    y: np.ndarray,
    s: np.ndarray,
    blocks: list[np.ndarray],
    is_prob: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-block-out jackknife statistics.

    Returns three arrays (auc_jack, pr_jack, brier_jack) — one entry per block,
    each entry being the metric on the dataset with that block removed. Skips
    blocks whose removal leaves a degenerate (single-class) sample.
    """
    auc_jack: list[float] = []
    pr_jack: list[float] = []
    brier_jack: list[float] = []
    n = len(y)
    for block_indices in blocks:
        if len(block_indices) >= n:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[block_indices] = False
        y_keep = y[keep_mask]
        s_keep = s[keep_mask]
        if len(np.unique(y_keep)) < 2:
            continue
        try:
            auc_jack.append(roc_auc_score(y_keep, s_keep))
            pr_jack.append(average_precision_score(y_keep, s_keep))
            if is_prob:
                brier_jack.append(brier_score_loss(y_keep, np.clip(s_keep, 0, 1)))
        except ValueError:
            continue
    return np.asarray(auc_jack), np.asarray(pr_jack), np.asarray(brier_jack)


def _bca_bounds(
    boot_values: list[float],
    jackknife_values: np.ndarray,
    observed: float,
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap CI bounds.

    Falls back to percentile bounds if BCa is degenerate (jackknife has too
    few values, all bootstraps are below/above the observed value, or the
    BCa-adjusted quantile points collapse).

    Args:
        boot_values: Bootstrap-resampled metric values.
        jackknife_values: Leave-one-block-out metric values for acceleration.
        observed: The point estimate on the full sample.
        alpha: Half-width (e.g. 0.025 for 95% CI).
    """
    if not boot_values:
        return float("nan"), float("nan")
    boot = np.asarray(boot_values, dtype=float)
    boot = boot[np.isfinite(boot)]
    if len(boot) < 2:
        return float("nan"), float("nan")

    def _percentile_fallback() -> tuple[float, float]:
        return (
            float(np.percentile(boot, 100 * alpha)),
            float(np.percentile(boot, 100 * (1 - alpha))),
        )

    jack = np.asarray(jackknife_values, dtype=float)
    jack = jack[np.isfinite(jack)]
    if len(jack) < 2:
        return _percentile_fallback()

    # Bias correction z0 = Φ⁻¹(P(boot < observed)). Clip to avoid ±∞ when
    # all bootstraps lie on one side of the observed value.
    p_below = float(np.mean(boot < observed))
    p_below = float(np.clip(p_below, 1e-6, 1 - 1e-6))
    z0 = NormalDist().inv_cdf(p_below)

    # Acceleration a from jackknife.
    diff = jack.mean() - jack
    denom_sq = float(np.sum(diff ** 2))
    if denom_sq <= 0:
        a = 0.0
    else:
        a = float(np.sum(diff ** 3)) / (6.0 * denom_sq ** 1.5)

    z_lo = NormalDist().inv_cdf(alpha)
    z_hi = NormalDist().inv_cdf(1 - alpha)

    def _adjusted(z: float) -> float | None:
        denom = 1.0 - a * (z0 + z)
        if denom <= 0:
            return None
        return NormalDist().cdf(z0 + (z0 + z) / denom)

    alpha_lo_adj = _adjusted(z_lo)
    alpha_hi_adj = _adjusted(z_hi)
    if (
        alpha_lo_adj is None
        or alpha_hi_adj is None
        or not (0 < alpha_lo_adj < alpha_hi_adj < 1)
    ):
        return _percentile_fallback()

    return (
        float(np.percentile(boot, 100 * alpha_lo_adj)),
        float(np.percentile(boot, 100 * alpha_hi_adj)),
    )


def _build_jackknife_blocks(
    n: int,
    block_to_idx: dict[str, np.ndarray] | None,
    max_blocks: int = 100,
) -> list[np.ndarray]:
    """Build leave-one-block-out indices for jackknife.

    When monthly blocks are supplied (block_to_idx not None), the jackknife
    drops one month at a time — matching the bootstrap structure. Otherwise
    rows are chunked into ``min(n, max_blocks)`` equal-size blocks; this is a
    block jackknife approximation that's much cheaper than true LOO and
    asymptotically valid for the BCa acceleration term.
    """
    if block_to_idx is not None:
        return [block_to_idx[block] for block in sorted(block_to_idx)]
    if n <= 0:
        return []
    n_blocks = min(n, max_blocks)
    chunks = np.array_split(np.arange(n), n_blocks)
    return [np.asarray(chunk, dtype=int) for chunk in chunks if len(chunk) > 0]


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
    dates: np.ndarray | None = None,
) -> pd.DataFrame:
    """Bootstrap CIs for AUC, PR AUC and Brier using the BCa method.

    Bias-corrected and accelerated (BCa) CIs are robust to skewed bootstrap
    distributions, which matters for PR AUC on imbalanced credit-default
    samples. When ``dates`` is provided, rows are bootstrapped in monthly
    blocks (preserves temporal autocorrelation) and the jackknife is also
    block-level. Without dates, bootstrap is stratified by class and the
    jackknife uses 100 equal-sized index chunks.
    """
    alpha = (1 - ci) / 2
    rng = np.random.RandomState(RANDOM_STATE)
    records = []

    for name, scores in score_arrays.items():
        mask = ~np.isnan(scores)
        y, s = y_true[mask], scores[mask]
        is_prob = float(s.min()) >= 0 and float(s.max()) <= 1.01

        boot_auc: list[float] = []
        boot_pr: list[float] = []
        boot_brier: list[float] = []
        block_to_idx: dict[str, np.ndarray] | None = None

        if dates is not None:
            d = pd.to_datetime(np.asarray(dates)[mask], errors="raise").to_period("M")
            block_values = d.astype(str)
            unique_blocks = np.unique(block_values)
            block_to_idx = {block: np.flatnonzero(block_values == block) for block in unique_blocks}
            if len(unique_blocks) == 0:
                continue
            for _ in range(n_bootstrap):
                sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
                bi = np.concatenate([block_to_idx[block] for block in sampled_blocks])
                y_b, s_b = y[bi], s[bi]
                try:
                    boot_auc.append(roc_auc_score(y_b, s_b))
                    boot_pr.append(average_precision_score(y_b, s_b))
                    if is_prob:
                        boot_brier.append(brier_score_loss(y_b, np.clip(s_b, 0, 1)))
                except ValueError:
                    continue
        else:
            idx_p = np.where(y == 1)[0]
            idx_n = np.where(y == 0)[0]
            for _ in range(n_bootstrap):
                bp = rng.choice(idx_p, size=len(idx_p), replace=True)
                bn = rng.choice(idx_n, size=len(idx_n), replace=True)
                bi = np.concatenate([bp, bn])
                y_b, s_b = y[bi], s[bi]
                try:
                    boot_auc.append(roc_auc_score(y_b, s_b))
                    boot_pr.append(average_precision_score(y_b, s_b))
                    if is_prob:
                        boot_brier.append(brier_score_loss(y_b, np.clip(s_b, 0, 1)))
                except ValueError:
                    continue

        # Use observed test-set statistics as point estimates, not bootstrap median.
        obs_auc = roc_auc_score(y, s)
        obs_pr = average_precision_score(y, s)
        obs_brier = brier_score_loss(y, np.clip(s, 0, 1)) if is_prob else np.nan

        # Block jackknife for BCa acceleration. Re-uses monthly blocks when
        # dates were supplied so the jackknife matches the bootstrap design.
        jack_blocks = _build_jackknife_blocks(len(y), block_to_idx)
        auc_jack, pr_jack, brier_jack = _block_jackknife_metrics(y, s, jack_blocks, is_prob)

        auc_lo, auc_hi = _bca_bounds(boot_auc, auc_jack, obs_auc, alpha)
        pr_lo, pr_hi = _bca_bounds(boot_pr, pr_jack, obs_pr, alpha)
        if is_prob:
            brier_lo, brier_hi = _bca_bounds(boot_brier, brier_jack, obs_brier, alpha)
        else:
            brier_lo, brier_hi = float("nan"), float("nan")

        records.append(
            {
                "Model": name,
                "AUC": obs_auc,
                "AUC_lo": auc_lo,
                "AUC_hi": auc_hi,
                "PR_AUC": obs_pr,
                "PR_AUC_lo": pr_lo,
                "PR_AUC_hi": pr_hi,
                "Brier": obs_brier,
                "Brier_lo": brier_lo,
                "Brier_hi": brier_hi,
            }
        )

    return pd.DataFrame(records).set_index("Model")


def create_lift_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Decile-based lift table for a single model."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n = len(y_sorted)
    total_bads = int(y_sorted.sum())
    overall_rate = y_sorted.mean()

    decile_size = n // n_deciles
    records = []
    cum_bads = 0
    cum_n = 0

    for d in range(1, n_deciles + 1):
        start = (d - 1) * decile_size
        end = d * decile_size if d < n_deciles else n
        chunk = y_sorted[start:end]
        n_chunk = len(chunk)
        bads_chunk = int(chunk.sum())
        cum_n += n_chunk
        cum_bads += bads_chunk
        default_rate = bads_chunk / n_chunk if n_chunk > 0 else 0.0
        cum_default_rate = cum_bads / cum_n if cum_n > 0 else 0.0
        lift = default_rate / overall_rate if overall_rate > 0 else 0.0
        cum_lift = cum_default_rate / overall_rate if overall_rate > 0 else 0.0
        capture_rate = cum_bads / total_bads if total_bads > 0 else 0.0

        records.append({
            "model": model_name,
            "decile": d,
            "n_accounts": n_chunk,
            "n_defaults": bads_chunk,
            "default_rate": default_rate,
            "cum_default_rate": cum_default_rate,
            "lift": lift,
            "cum_lift": cum_lift,
            "capture_rate": capture_rate,
        })

    return pd.DataFrame(records)


def create_threshold_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    model_name: str,
    thresholds_pct: list[float] | None = None,
) -> pd.DataFrame:
    """Confusion-matrix metrics at business-relevant rejection thresholds.

    Each threshold is expressed as "reject the top X% riskiest by score".
    """
    if thresholds_pct is None:
        thresholds_pct = list(DEFAULT_THRESHOLD_PCTS)

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    total_bads = int(y_true.sum())

    records = []
    for pct in thresholds_pct:
        cutoff_idx = int(np.ceil(n * pct / 100.0))
        score_threshold = np.sort(y_score)[::-1][min(cutoff_idx, n - 1)]

        predicted_bad = y_score >= score_threshold
        tp = int((predicted_bad & (y_true == 1)).sum())
        fp = int((predicted_bad & (y_true == 0)).sum())
        fn = int((~predicted_bad & (y_true == 1)).sum())
        tn = int((~predicted_bad & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        capture_rate = tp / total_bads if total_bads > 0 else 0.0

        records.append({
            "model": model_name,
            "reject_pct": pct,
            "score_threshold": float(score_threshold),
            "n_rejected": int(predicted_bad.sum()),
            "n_total": n,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "capture_rate": capture_rate,
        })

    return pd.DataFrame(records)


def compute_overfit_report(
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compare train vs test metrics to detect overfitting.

    Returns one row per model with train/test AUC, PR AUC, Brier,
    and the delta (train - test). Large positive deltas indicate overfitting.
    """
    if model_names is None:
        model_names = [n for n in test_scores if n in train_scores]
    records = []
    for name in model_names:
        if name not in train_scores or name not in test_scores:
            continue
        tr = np.asarray(train_scores[name], dtype=float)
        te = np.asarray(test_scores[name], dtype=float)
        is_prob = _score_is_probability(te)

        train_metrics = evaluate_safely(f"{name} (train)", y_train, tr, is_probability=is_prob)
        test_metrics = evaluate_safely(f"{name} (test)", y_test, te, is_probability=is_prob)

        auc_delta = train_metrics["ROC AUC"] - test_metrics["ROC AUC"]
        pr_delta = train_metrics["PR AUC"] - test_metrics["PR AUC"]
        brier_delta = (
            (test_metrics["Brier"] - train_metrics["Brier"])
            if is_prob and not (np.isnan(train_metrics["Brier"]) or np.isnan(test_metrics["Brier"]))
            else np.nan
        )

        records.append({
            "model": name,
            "train_auc": train_metrics["ROC AUC"],
            "test_auc": test_metrics["ROC AUC"],
            "auc_delta": auc_delta,
            "train_pr_auc": train_metrics["PR AUC"],
            "test_pr_auc": test_metrics["PR AUC"],
            "pr_auc_delta": pr_delta,
            "train_brier": train_metrics["Brier"],
            "test_brier": test_metrics["Brier"],
            "brier_delta": brier_delta,
            "train_ks": train_metrics["KS"],
            "test_ks": test_metrics["KS"],
            "ks_delta": train_metrics["KS"] - test_metrics["KS"],
            "train_n": train_metrics["N"],
            "test_n": test_metrics["N"],
            "overfit_flag": (
                "YES"
                if auc_delta > OVERFIT_DELTA_THRESHOLD or pr_delta > OVERFIT_DELTA_THRESHOLD
                else "NO"
            ),
        })

    return pd.DataFrame(records)


def select_best_model(
    results_df: pd.DataFrame,
    overfit_df: pd.DataFrame | None = None,
    rolling_oot_summary_df: pd.DataFrame | None = None,
    candidate_names: list[str] | None = None,
    benchmark_comparisons_df: pd.DataFrame | None = None,
    quality_floor: float = MODEL_SELECTION_QUALITY_FLOOR,
) -> pd.DataFrame:
    """Score and rank candidate models on multiple criteria.

    Criteria (each 0-100, higher is better):
      1. Discrimination (PR AUC on test) — 35%
      2. Calibration (Brier of calibrated variant on test) — 15%
      3. Stability (mean rolling OOT PR AUC) — 20%
      4. Generalization (100 - overfit penalty) — 15%
      5. Benchmark lift (AUC improvement vs best benchmark) — 15%

    Calibrated models (e.g. "LightGBM (calibrated)") are not separate
    candidates — they are post-hoc probability adjustments of the base model.
    Selection picks the best base architecture; the calibrated variant of the
    winner is the production deployment candidate.  The calibration score uses
    the calibrated Brier when available, since that reflects production quality.

    Returns a DataFrame sorted by weighted_score descending, with the top
    row being the recommended model.
    """
    if candidate_names is None:
        candidate_names = SUMMARY_MODEL_NAMES

    available = [n for n in candidate_names if n in results_df.index]
    if not available:
        return pd.DataFrame()

    benchmark_p_col = None
    if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
        if "auc_p_adjusted" in benchmark_comparisons_df.columns:
            benchmark_p_col = "auc_p_adjusted"
        elif "auc_p_value" in benchmark_comparisons_df.columns:
            benchmark_p_col = "auc_p_value"

    records = []
    for name in available:
        row = results_df.loc[name]

        # 1. Discrimination — normalize PR AUC to 0-100
        pr_auc = float(row["PR AUC"])
        pr_auc_scores = results_df.loc[available, "PR AUC"].astype(float)
        pr_min, pr_max = pr_auc_scores.min(), pr_auc_scores.max()
        discrimination = (
            ((pr_auc - pr_min) / (pr_max - pr_min) * 100) if pr_max > pr_min else 50.0
        )

        # 2. Calibration — use calibrated Brier when available (reflects
        #    production probability quality), fall back to raw Brier.
        calibrated_name = f"{name} (calibrated)"
        if calibrated_name in results_df.index and not np.isnan(results_df.loc[calibrated_name, "Brier"]):
            brier = float(results_df.loc[calibrated_name, "Brier"])
        else:
            brier = float(row["Brier"]) if not np.isnan(row["Brier"]) else 1.0
        # Collect calibrated Brier for all candidates for min-max scaling
        brier_values = []
        for cand in available:
            cal_name = f"{cand} (calibrated)"
            if cal_name in results_df.index and not np.isnan(results_df.loc[cal_name, "Brier"]):
                brier_values.append(float(results_df.loc[cal_name, "Brier"]))
            elif not np.isnan(results_df.loc[cand, "Brier"]):
                brier_values.append(float(results_df.loc[cand, "Brier"]))
        if brier_values:
            b_min, b_max = min(brier_values), max(brier_values)
            calibration = ((b_max - brier) / (b_max - b_min) * 100) if b_max > b_min else 50.0
        else:
            calibration = 50.0

        # 3. Stability — rolling OOT mean PR AUC
        stability = 50.0
        if rolling_oot_summary_df is not None and not rolling_oot_summary_df.empty:
            oot_row = rolling_oot_summary_df.loc[rolling_oot_summary_df["Model"] == name]
            if not oot_row.empty:
                oot_pr = float(oot_row["mean_PR_AUC"].iloc[0])
                oot_all = rolling_oot_summary_df.loc[
                    rolling_oot_summary_df["Model"].isin(available), "mean_PR_AUC"
                ].astype(float)
                oot_min, oot_max = oot_all.min(), oot_all.max()
                stability = (
                    ((oot_pr - oot_min) / (oot_max - oot_min) * 100)
                    if oot_max > oot_min else 50.0
                )

        # 4. Generalization — penalize overfitting using the LARGER of the
        # AUC and PR-AUC train-vs-test deltas. PR-AUC delta is more informative
        # for imbalanced credit-default samples (positive rate ~3-5%), so
        # ignoring it would hide tree-model overfit that is visible in the
        # precision-recall curve but masked in ROC. compute_overfit_report
        # already computes both deltas; we just take the maximum here.
        generalization = 100.0
        if overfit_df is not None and not overfit_df.empty:
            of_row = overfit_df.loc[overfit_df["model"] == name]
            if not of_row.empty:
                auc_delta = float(of_row["auc_delta"].iloc[0])
                pr_auc_delta_raw = (
                    float(of_row["pr_auc_delta"].iloc[0])
                    if "pr_auc_delta" in of_row.columns
                    else float("-inf")
                )
                effective_delta = max(auc_delta, pr_auc_delta_raw)
                # Penalty curve: 0 if delta <= 0.01, linearly ramping to 100
                # at delta = 0.10. Keeps the same shape as before so model-
                # selection rankings only change for models whose PR-AUC
                # delta exceeds their AUC delta.
                penalty = max(0.0, min(100.0, (effective_delta - 0.01) / 0.09 * 100))
                generalization = 100.0 - penalty

        # 5. Benchmark lift — best AUC improvement vs any benchmark
        lift_score = 50.0
        conservative_auc_lift = np.nan
        benchmark_evidence_tier = 0
        benchmark_evidence = "none"
        if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
            cand_rows = benchmark_comparisons_df.loc[
                benchmark_comparisons_df["candidate_model"] == name
            ]
            if not cand_rows.empty:
                best_lift = cand_rows["auc_improvement"].astype(float).max()
                if "auc_improvement_lo" in cand_rows.columns:
                    conservative_auc_lift = cand_rows["auc_improvement_lo"].astype(float).max()
                all_lifts = benchmark_comparisons_df.loc[
                    benchmark_comparisons_df["candidate_model"].isin(available),
                    "auc_improvement",
                ].astype(float)
                l_min, l_max = all_lifts.min(), all_lifts.max()
                lift_score = (
                    ((best_lift - l_min) / (l_max - l_min) * 100)
                    if l_max > l_min else 50.0
                )

                positive_lower = bool(np.isfinite(conservative_auc_lift) and conservative_auc_lift > 0)
                significant_positive = False
                if benchmark_p_col is not None:
                    p_values = cand_rows[benchmark_p_col].astype(float)
                    if "auc_improvement_lo" in cand_rows.columns:
                        significant_positive = bool(
                            ((p_values <= 0.05) & (cand_rows["auc_improvement_lo"].astype(float) > 0)).any()
                        )
                    else:
                        significant_positive = bool(
                            ((p_values <= 0.05) & (cand_rows["auc_improvement"].astype(float) > 0)).any()
                        )

                if significant_positive:
                    benchmark_evidence_tier = 3
                    benchmark_evidence = "strong"
                elif positive_lower:
                    benchmark_evidence_tier = 2
                    benchmark_evidence = "supported"
                elif best_lift > 0:
                    benchmark_evidence_tier = 1
                    benchmark_evidence = "point_only"

        weighted = (
            0.35 * discrimination
            + 0.15 * calibration
            + 0.20 * stability
            + 0.15 * generalization
            + 0.15 * lift_score
        )

        calibrated_name = f"{name} (calibrated)"
        raw_brier = float(row["Brier"]) if not np.isnan(row["Brier"]) else np.nan
        cal_brier = (
            float(results_df.loc[calibrated_name, "Brier"])
            if calibrated_name in results_df.index and not np.isnan(results_df.loc[calibrated_name, "Brier"])
            else np.nan
        )
        records.append({
            "model": name,
            "discrimination_score": round(discrimination, 1),
            "calibration_score": round(calibration, 1),
            "stability_score": round(stability, 1),
            "generalization_score": round(generalization, 1),
            "lift_score": round(lift_score, 1),
            "weighted_score": round(weighted, 1),
            "conservative_auc_lift": conservative_auc_lift,
            "benchmark_evidence_tier": benchmark_evidence_tier,
            "benchmark_evidence": benchmark_evidence,
            "test_pr_auc": pr_auc,
            "test_auc": float(row["ROC AUC"]),
            "test_brier_raw": raw_brier,
            "test_brier_calibrated": cal_brier,
            "recommended": False,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["selection_band"] = False
    if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
        max_weighted = float(df["weighted_score"].max())
        df["selection_band"] = df["weighted_score"] >= max_weighted - 2.0
        winner_idx = (
            df.loc[df["selection_band"]]
            .sort_values(
                [
                    "benchmark_evidence_tier",
                    "conservative_auc_lift",
                    "generalization_score",
                    "calibration_score",
                    "discrimination_score",
                    "stability_score",
                    "weighted_score",
                    "test_pr_auc",
                    "test_auc",
                ],
                ascending=[False, False, False, False, False, False, False, False, False],
                na_position="last",
            )
            .index[0]
        )
    else:
        winner_idx = df["weighted_score"].idxmax()

    # Absolute quality floor (audit fix K). weighted_score is min-max-scaled
    # per dimension across the candidate set, so one model always wins by
    # being "least bad" even when every candidate is poor. If the proposed
    # winner doesn't clear the floor, no model is recommended — the artifact
    # records meets_quality_floor=False so a regulator can see the system
    # withheld a recommendation rather than silently endorsing a weak model.
    df["meets_quality_floor"] = df["weighted_score"] >= quality_floor
    winner_meets_floor = bool(df.loc[winner_idx, "meets_quality_floor"])
    df.loc[:, "recommended"] = False
    if winner_meets_floor:
        df.loc[winner_idx, "recommended"] = True
    else:
        winner_score = float(df.loc[winner_idx, "weighted_score"])
        winner_name = str(df.loc[winner_idx, "model"])
        logger.warning(
            "No model recommended: best candidate {!r} scored {:.1f}/100, "
            "below the quality floor of {:.1f}. The pipeline will produce "
            "models and metrics for review but will not endorse a default "
            "deployment choice.",
            winner_name,
            winner_score,
            quality_floor,
        )
    return (
        df.sort_values(
            [
                "recommended",
                "weighted_score",
                "benchmark_evidence_tier",
                "conservative_auc_lift",
                "generalization_score",
                "calibration_score",
                "discrimination_score",
                "stability_score",
                "test_pr_auc",
                "test_auc",
            ],
            ascending=[False, False, False, False, False, False, False, False, False, False],
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ── Population Bias Analysis ─────────────────────────────────────────────────


def compute_population_ks_test(
    applicant_scores_df: pd.DataFrame,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Two-sample KS test comparing score distributions between booked and rejected populations.

    A large KS statistic / small p-value means the model assigns very
    different score distributions to booked vs rejected applicants, which is
    expected (the existing decisioning already separated them).  But if the
    model's separation *perfectly mirrors* the old decision, it may be
    recapitulating selection rather than learning new signal.
    """
    from scipy.stats import ks_2samp

    if applicant_scores_df is None or applicant_scores_df.empty:
        return pd.DataFrame()
    if "status_name" not in applicant_scores_df.columns:
        return pd.DataFrame()

    booked = applicant_scores_df.loc[applicant_scores_df["status_name"] == "Booked"]
    non_booked = applicant_scores_df.loc[applicant_scores_df["status_name"] != "Booked"]
    if booked.empty or non_booked.empty:
        return pd.DataFrame()

    if model_names is None:
        model_names = [
            c.replace("score__", "")
            for c in applicant_scores_df.columns
            if c.startswith("score__") and "(calibrated)" not in c
        ]

    records = []
    for name in model_names:
        col = f"score__{sanitize_output_name(name)}"
        if col not in applicant_scores_df.columns:
            continue
        booked_scores = booked[col].dropna().values
        non_booked_scores = non_booked[col].dropna().values
        if len(booked_scores) < 10 or len(non_booked_scores) < 10:
            continue
        stat, p_value = ks_2samp(booked_scores, non_booked_scores)
        records.append({
            "model": name,
            "n_booked": len(booked_scores),
            "n_non_booked": len(non_booked_scores),
            "booked_mean_score": float(np.mean(booked_scores)),
            "non_booked_mean_score": float(np.mean(non_booked_scores)),
            "ks_statistic": float(stat),
            "ks_p_value": float(p_value),
        })

    return pd.DataFrame(records)


def compute_selection_bias_correlation(
    applicant_scores_df: pd.DataFrame,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Correlation between model PD and the original risk_score_rf.

    High correlation means the model mostly recapitulates the existing
    decisioning mechanism rather than learning new signal.  Moderate
    correlation is expected and healthy; very high (>0.90) warrants scrutiny.
    """
    if applicant_scores_df is None or applicant_scores_df.empty:
        return pd.DataFrame()
    if "risk_score_rf" not in applicant_scores_df.columns:
        return pd.DataFrame()

    risk_score = applicant_scores_df["risk_score_rf"]

    if model_names is None:
        model_names = [
            c.replace("score__", "")
            for c in applicant_scores_df.columns
            if c.startswith("score__") and "(calibrated)" not in c
        ]

    records = []
    for name in model_names:
        col = f"score__{sanitize_output_name(name)}"
        if col not in applicant_scores_df.columns:
            continue
        model_score = applicant_scores_df[col]
        valid = risk_score.notna() & model_score.notna()
        if valid.sum() < 20:
            continue
        # Model PD is higher = riskier; risk_score_rf is lower = riskier
        # (negated in evaluate_all), so correlate with -risk_score_rf for
        # an intuitive positive correlation when they agree on ranking.
        pearson = float(np.corrcoef(model_score[valid], -risk_score[valid])[0, 1])
        spearman = float(
            pd.Series(model_score[valid].values).corr(
                pd.Series(-risk_score[valid].values), method="spearman"
            )
        )
        flag = (
            "HIGH" if abs(pearson) > 0.90
            else "MODERATE" if abs(pearson) > 0.70
            else "LOW"
        )
        records.append({
            "model": name,
            "n_valid": int(valid.sum()),
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "selection_bias_flag": flag,
        })

    return pd.DataFrame(records)


def compute_adverse_impact_analysis(
    y_true: np.ndarray,
    y_score: np.ndarray,
    age: np.ndarray,
    model_name: str,
    age_bins: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    age = np.asarray(age, dtype=float)

    valid = np.isfinite(y_score) & np.isfinite(age)
    y_true, y_score, age = y_true[valid], y_score[valid], age[valid]
    if len(y_score) == 0:
        return pd.DataFrame()

    if age_bins is None:
        age_bins = [(18, 25), (25, 35), (35, 45), (45, 55), (55, 65), (65, 100)]

    # Global 10% rejection threshold
    threshold_pct = 10.0
    cutoff_idx = int(np.ceil(len(y_score) * threshold_pct / 100.0))
    score_threshold = np.sort(y_score)[::-1][min(cutoff_idx, len(y_score) - 1)]

    records = []
    for lo, hi in age_bins:
        mask = (age >= lo) & (age < hi)
        n = int(mask.sum())
        if n < 10:
            continue
        y_band = y_true[mask]
        s_band = y_score[mask]
        observed_mask = np.isfinite(y_band)
        n_observed = int(observed_mask.sum())
        n_defaults = int(np.nansum(y_band[observed_mask])) if n_observed > 0 else 0
        observed_rate = float(np.nanmean(y_band[observed_mask])) if n_observed > 0 else np.nan
        predicted_pd = float(s_band.mean())
        approved = s_band < score_threshold
        approval_rate = float(approved.mean())

        records.append({
            "model": model_name,
            "age_band": f"{int(lo)}-{int(hi)}",
            "n": n,
            "n_observed_outcomes": n_observed,
            "n_defaults": n_defaults,
            "observed_default_rate": observed_rate,
            "mean_predicted_pd": predicted_pd,
            "approval_rate_at_10pct_reject": approval_rate,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Adverse impact ratio: approval rate / best approval rate
    best_approval = df["approval_rate_at_10pct_reject"].max()
    df["adverse_impact_ratio"] = (
        df["approval_rate_at_10pct_reject"] / best_approval if best_approval > 0 else np.nan
    )
    df["air_flag"] = np.where(df["adverse_impact_ratio"] < 0.80, "FAIL", "PASS")
    return df


def compute_concept_drift_report(
    rolling_oot_results_df: pd.DataFrame,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    """Detect concept drift: is the score-to-outcome relationship degrading over time?

    Compares calibration quality (Brier) and discrimination (PR AUC) across
    rolling OOT folds. A monotonically worsening trend signals concept drift —
    the patterns the model learned are changing in the population.

    Returns one row per model with trend statistics and a drift flag.
    """
    if rolling_oot_results_df is None or rolling_oot_results_df.empty:
        return pd.DataFrame()

    # Use uncalibrated results for consistency
    df = rolling_oot_results_df.copy()
    if "is_calibrated" in df.columns:
        df = df.loc[df["is_calibrated"] == False].copy()

    if model_names is not None:
        df = df.loc[df["Model"].isin(model_names)].copy()

    if df.empty or "fold" not in df.columns:
        return pd.DataFrame()

    records = []
    for name in df["Model"].unique():
        mdf = df.loc[df["Model"] == name].sort_values("fold")
        if len(mdf) < 2:
            continue

        folds = mdf["fold"].values
        pr_auc = mdf["PR AUC"].astype(float).values
        roc_auc = mdf["ROC AUC"].astype(float).values
        brier = mdf["Brier"].astype(float).values if "Brier" in mdf.columns else np.full(len(mdf), np.nan)

        # Linear trend: negative slope in PR AUC = degradation
        if len(folds) >= 3 and np.all(np.isfinite(pr_auc)):
            pr_slope = float(np.polyfit(folds.astype(float), pr_auc, 1)[0])
            roc_slope = float(np.polyfit(folds.astype(float), roc_auc, 1)[0])
        else:
            pr_slope = float(pr_auc[-1] - pr_auc[0]) / max(len(folds) - 1, 1)
            roc_slope = float(roc_auc[-1] - roc_auc[0]) / max(len(folds) - 1, 1)

        brier_slope = np.nan
        if np.all(np.isfinite(brier)) and len(folds) >= 3:
            brier_slope = float(np.polyfit(folds.astype(float), brier, 1)[0])

        pr_range = float(pr_auc.max() - pr_auc.min())
        first_last_delta = float(pr_auc[-1] - pr_auc[0])

        # Flag concept drift when PR AUC is declining AND first→last delta is
        # at least CONCEPT_DRIFT_DELTA_THRESHOLD negative.
        drift_flag = (
            "YES"
            if pr_slope < 0 and first_last_delta < CONCEPT_DRIFT_DELTA_THRESHOLD
            else "NO"
        )

        records.append({
            "model": name,
            "n_folds": len(mdf),
            "pr_auc_first": float(pr_auc[0]),
            "pr_auc_last": float(pr_auc[-1]),
            "pr_auc_slope_per_fold": pr_slope,
            "roc_auc_slope_per_fold": roc_slope,
            "brier_slope_per_fold": brier_slope,
            "pr_auc_range": pr_range,
            "first_last_delta": first_last_delta,
            "concept_drift_flag": drift_flag,
        })

    return pd.DataFrame(records)


def train_post_hoc_ensemble(
    y_calibration: np.ndarray,
    calibration_scores: dict[str, np.ndarray],
    lr_name: str = "Logistic Regression",
    tree_names: list[str] | None = None,
) -> dict:
    """Find optimal weights for a simple LR + best-tree weighted average.

    Uses the calibration holdout to optimize weights via grid search on
    PR AUC. Returns the best weight, component names, and the blending
    function.

    This is simpler than stacking — no meta-learner, just a convex
    combination of two probability estimates.
    """
    if tree_names is None:
        tree_names = ["LightGBM", "XGBoost", "CatBoost"]

    if lr_name not in calibration_scores:
        return {}

    lr_scores = calibration_scores[lr_name]
    best_result = None

    for tree_name in tree_names:
        if tree_name not in calibration_scores:
            continue
        tree_scores = calibration_scores[tree_name]

        # Grid search over LR weight from 0.0 to 1.0
        for lr_weight in np.arange(0.0, 1.05, 0.05):
            tree_weight = 1.0 - lr_weight
            blended = lr_weight * lr_scores + tree_weight * tree_scores
            try:
                pr_auc = average_precision_score(y_calibration, blended)
            except ValueError:
                continue
            if best_result is None or pr_auc > best_result["pr_auc"]:
                best_result = {
                    "lr_name": lr_name,
                    "tree_name": tree_name,
                    "lr_weight": float(lr_weight),
                    "tree_weight": float(tree_weight),
                    "pr_auc": float(pr_auc),
                }

    return best_result if best_result is not None else {}
