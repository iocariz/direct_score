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
    N_BOOTSTRAP,
    RANDOM_STATE,
    SUMMARY_MODEL_NAMES,
    TARGET,
)


def sanitize_output_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def _ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = np.sort(y_score[y_true == 1])
    neg = np.sort(y_score[y_true == 0])
    all_scores = np.sort(np.unique(np.concatenate([pos, neg])))
    pos_cdf = np.searchsorted(pos, all_scores, side="right") / len(pos)
    neg_cdf = np.searchsorted(neg, all_scores, side="right") / len(neg)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))


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
        if "(calibrated)" in name or name.startswith("Stacking"):
            continue
        if not hasattr(model, "named_steps"):
            continue

        clf = model.named_steps["classifier"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            imp_type = "split_importance"
        elif hasattr(clf, "coef_"):
            importances = clf.coef_[0]
            imp_type = "coefficient"
        else:
            continue

        for feat, imp in zip(feature_names, importances):
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

    if feature_provenance_df is not None and not feature_provenance_df.empty:
        feature_provenance_path = output_dir / "feature_provenance.csv"
        feature_provenance_df.to_csv(feature_provenance_path, index=False)
        logger.info("Saved feature provenance: {} ({} rows)", feature_provenance_path, len(feature_provenance_df))

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

    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for name, model in models.items():
        safe_name = sanitize_output_name(name)
        path = models_dir / f"{safe_name}.joblib"
        joblib.dump(model, path)
    logger.info("Saved {} models to {}", len(models), models_dir)


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    score_arrays: dict[str, np.ndarray],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
) -> pd.DataFrame:
    alpha = (1 - ci) / 2
    rng = np.random.RandomState(RANDOM_STATE)
    records = []

    for name, scores in score_arrays.items():
        mask = ~np.isnan(scores)
        y, s = y_true[mask], scores[mask]
        idx_p = np.where(y == 1)[0]
        idx_n = np.where(y == 0)[0]
        is_prob = float(s.min()) >= 0 and float(s.max()) <= 1.01

        boot_auc, boot_pr, boot_brier = [], [], []
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

        def _ci_bounds(arr):
            if not arr:
                return np.nan, np.nan
            a = np.array(arr)
            return np.percentile(a, 100 * alpha), np.percentile(a, 100 * (1 - alpha))

        # Use observed test-set statistics as point estimates, not bootstrap median
        obs_auc = roc_auc_score(y, s)
        obs_pr = average_precision_score(y, s)
        obs_brier = brier_score_loss(y, np.clip(s, 0, 1)) if is_prob else np.nan

        auc_lo, auc_hi = _ci_bounds(boot_auc)
        pr_lo, pr_hi = _ci_bounds(boot_pr)
        brier_lo, brier_hi = _ci_bounds(boot_brier)

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
        thresholds_pct = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

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

        train_metrics = evaluate(f"{name} (train)", y_train, tr, is_probability=is_prob)
        test_metrics = evaluate(f"{name} (test)", y_test, te, is_probability=is_prob)

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
            "overfit_flag": "YES" if auc_delta > 0.03 or pr_delta > 0.03 else "NO",
        })

    return pd.DataFrame(records)


def select_best_model(
    results_df: pd.DataFrame,
    overfit_df: pd.DataFrame | None = None,
    rolling_oot_summary_df: pd.DataFrame | None = None,
    candidate_names: list[str] | None = None,
    benchmark_comparisons_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score and rank candidate models on multiple criteria.

    Criteria (each 0-100, higher is better):
      1. Discrimination (PR AUC on test) — 35%
      2. Calibration (1 - Brier on test) — 15%
      3. Stability (mean rolling OOT PR AUC) — 20%
      4. Generalization (100 - overfit penalty) — 15%
      5. Benchmark lift (AUC improvement vs best benchmark) — 15%

    Returns a DataFrame sorted by weighted_score descending, with the top
    row being the recommended model.
    """
    if candidate_names is None:
        candidate_names = SUMMARY_MODEL_NAMES

    available = [n for n in candidate_names if n in results_df.index]
    if not available:
        return pd.DataFrame()

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

        # 2. Calibration — lower Brier is better; scale to 0-100
        brier = float(row["Brier"]) if not np.isnan(row["Brier"]) else 1.0
        brier_scores = results_df.loc[available, "Brier"].astype(float).dropna()
        if len(brier_scores) > 0:
            b_min, b_max = brier_scores.min(), brier_scores.max()
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

        # 4. Generalization — penalize overfitting
        generalization = 100.0
        if overfit_df is not None and not overfit_df.empty:
            of_row = overfit_df.loc[overfit_df["model"] == name]
            if not of_row.empty:
                auc_delta = float(of_row["auc_delta"].iloc[0])
                # Penalty: 0 if delta <= 0.01, linearly increasing up to 100 at delta=0.10
                penalty = max(0.0, min(100.0, (auc_delta - 0.01) / 0.09 * 100))
                generalization = 100.0 - penalty

        # 5. Benchmark lift — best AUC improvement vs any benchmark
        lift_score = 50.0
        if benchmark_comparisons_df is not None and not benchmark_comparisons_df.empty:
            cand_rows = benchmark_comparisons_df.loc[
                benchmark_comparisons_df["candidate_model"] == name
            ]
            if not cand_rows.empty:
                best_lift = cand_rows["auc_improvement"].astype(float).max()
                all_lifts = benchmark_comparisons_df.loc[
                    benchmark_comparisons_df["candidate_model"].isin(available),
                    "auc_improvement",
                ].astype(float)
                l_min, l_max = all_lifts.min(), all_lifts.max()
                lift_score = (
                    ((best_lift - l_min) / (l_max - l_min) * 100)
                    if l_max > l_min else 50.0
                )

        weighted = (
            0.35 * discrimination
            + 0.15 * calibration
            + 0.20 * stability
            + 0.15 * generalization
            + 0.15 * lift_score
        )

        records.append({
            "model": name,
            "discrimination_score": round(discrimination, 1),
            "calibration_score": round(calibration, 1),
            "stability_score": round(stability, 1),
            "generalization_score": round(generalization, 1),
            "lift_score": round(lift_score, 1),
            "weighted_score": round(weighted, 1),
            "test_pr_auc": pr_auc,
            "test_auc": float(row["ROC AUC"]),
            "test_brier": float(row["Brier"]) if not np.isnan(row["Brier"]) else np.nan,
            "recommended": False,
        })

    df = pd.DataFrame(records).sort_values("weighted_score", ascending=False).reset_index(drop=True)
    if not df.empty:
        df.loc[0, "recommended"] = True
    return df
