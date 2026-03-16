from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

TARGET = "basel_bad"

MODEL_ORDER = [
    "risk_score_rf (benchmark)",
    "Logistic Regression",
    "XGBoost",
    "CatBoost",
    "LightGBM",
    "score_RF (benchmark)",
]

CANDIDATE_ORDER = [
    "Logistic Regression",
    "XGBoost",
    "CatBoost",
    "LightGBM",
]

STATUS_ORDER = ["Booked", "Rejected", "Canceled"]

MODEL_COLORS = {
    "risk_score_rf (benchmark)": "#1f4e79",
    "Logistic Regression": "#2a9d8f",
    "XGBoost": "#577590",
    "CatBoost": "#f4a261",
    "LightGBM": "#8d99ae",
    "score_RF (benchmark)": "#d62828",
}

STATUS_COLORS = {
    "Booked": "#2a9d8f",
    "Rejected": "#577590",
    "Canceled": "#f4a261",
}

KPI_TILE_COLORS = [
    "#1f4e79",
    "#2a9d8f",
    "#577590",
    "#f4a261",
    "#8d99ae",
    "#264653",
]

TEXT_DARK = "#18324b"


def _read_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def _maybe_read_csv(output_dir: Path, name: str) -> pd.DataFrame | None:
    path = output_dir / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _int_label(value: float | int) -> str:
    return f"{int(round(value)):,}"


def _metric_label(value: float) -> str:
    return f"{value:.4f}"


def _sanitize_output_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")


def _score_column_name(model_name: str) -> str:
    return f"score__{_sanitize_output_name(model_name)}"


def _millions_formatter() -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: f"{value / 1_000_000:.1f}M")


def _non_calibrated_results(results_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~results_df["Model"].str.contains("(calibrated)", regex=False, na=False)
    df = results_df.loc[mask].copy()
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    return df.sort_values("Model")


def _non_calibrated_comparisons(comparisons_df: pd.DataFrame) -> pd.DataFrame:
    mask = ~comparisons_df["candidate_model"].str.contains("(calibrated)", regex=False, na=False)
    df = comparisons_df.loc[mask].copy()
    df["candidate_model"] = pd.Categorical(df["candidate_model"], categories=CANDIDATE_ORDER, ordered=True)
    return df.sort_values(["reference_model", "candidate_model"])


def create_kpi_chart(
    results_df: pd.DataFrame,
    population_summary_df: pd.DataFrame,
    benchmark_comparisons_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    post_split = population_summary_df.loc[population_summary_df["split"] == "post_split"].copy()
    counts = post_split.set_index("status_name")["n_rows"].to_dict()
    held_out_row = results_df.loc[results_df["Model"] == "Logistic Regression"].iloc[0]
    comparison_row = benchmark_comparisons_df.loc[
        (benchmark_comparisons_df["candidate_model"] == "Logistic Regression")
        & (benchmark_comparisons_df["reference_model"] == "score_RF (benchmark)")
    ].iloc[0]

    tiles = [
        ("Decisioned applications scored", int(post_split["n_rows"].sum()), "Full post-split\ndecisioned population"),
        ("Booked applications", int(counts.get("Booked", 0)), "Observed outcomes\navailable"),
        ("Rejected applications", int(counts.get("Rejected", 0)), "Scored without\nobserved repayment"),
        ("Canceled applications", int(counts.get("Canceled", 0)), "Scored without\nobserved repayment"),
        ("Held-out booked proxy rows", int(held_out_row["N"]), "Official test sample"),
        ("Observed defaults in hold-out", int(comparison_row["n_pos"]), "Booked defaults in\nthe test sample"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.88, bottom=0.10, wspace=0.18, hspace=0.18)

    for ax, tile, color in zip(axes.flatten(), tiles, KPI_TILE_COLORS):
        label, value, subtitle = tile
        ax.set_axis_off()
        patch = FancyBboxPatch(
            (0.03, 0.06),
            0.94,
            0.88,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(0.08, 0.79, label, transform=ax.transAxes, fontsize=13, fontweight="bold", color="white")
        ax.text(0.08, 0.47, _int_label(value), transform=ax.transAxes, fontsize=28, fontweight="bold", color="white")
        ax.text(0.08, 0.17, subtitle, transform=ax.transAxes, fontsize=10.5, color="white", linespacing=1.15)

    fig.suptitle("Underwriting-stage scoring headline KPIs", fontsize=20, fontweight="bold", color=TEXT_DARK)
    fig.text(0.5, 0.02, "Source: population_summary.csv and benchmark_comparisons.csv", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_process_chart(output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.22, 0.92, "Before", fontsize=18, fontweight="bold", color=TEXT_DARK, ha="center")
    ax.text(0.78, 0.92, "Now", fontsize=18, fontweight="bold", color=TEXT_DARK, ha="center")

    left_boxes = [
        (0.08, 0.66, 0.28, 0.16, "Booked accounts only"),
        (0.08, 0.42, 0.28, 0.16, "Monitoring-style benchmark comparisons"),
        (0.08, 0.18, 0.28, 0.16, "No persisted applicant-stage score frame"),
    ]
    right_boxes = [
        (0.64, 0.66, 0.28, 0.16, "Explicit underwriting mode for Booked, Rejected, and Canceled"),
        (0.64, 0.42, 0.28, 0.16, "Full decisioned-population score export for post-split applications"),
        (0.64, 0.18, 0.28, 0.16, "Booked-proxy performance labels kept explicit for governance"),
    ]

    for x, y, w, h, text in left_boxes:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.6", fc="#577590", ec="none"),
        )

    for x, y, w, h, text in right_boxes:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.6", fc="#2a9d8f", ec="none"),
        )

    arrow = FancyArrowPatch((0.40, 0.50), (0.60, 0.50), arrowstyle="simple", mutation_scale=40, color="#1f4e79", alpha=0.9)
    ax.add_patch(arrow)
    ax.text(0.50, 0.58, "Refactored for underwriting", ha="center", fontsize=13, fontweight="bold", color=TEXT_DARK)
    ax.text(0.50, 0.40, "Model now scores the full decisioned funnel", ha="center", fontsize=11, color="#4b5563")
    fig.suptitle("From booked-only monitoring to underwriting-stage scoring", fontsize=20, fontweight="bold", color=TEXT_DARK)
    return _save_figure(fig, output_path)


def create_population_chart(population_summary_df: pd.DataFrame, output_path: Path) -> Path:
    df = population_summary_df.pivot(index="split", columns="status_name", values="n_rows").fillna(0)
    df = df.reindex(["pre_split", "post_split"]).fillna(0)
    labels = ["Pre-split", "Post-split"]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    bottom = np.zeros(len(df))

    for status in STATUS_ORDER:
        values = df[status].to_numpy()
        ax.bar(labels, values, bottom=bottom, color=STATUS_COLORS[status], width=0.62, label=status)
        for idx, (base, value) in enumerate(zip(bottom, values, strict=False)):
            if value > 75_000:
                ax.text(idx, base + value / 2, _int_label(value), ha="center", va="center", fontsize=11, color="white", fontweight="bold")
        bottom = bottom + values

    totals = df.sum(axis=1).to_numpy()
    for idx, total in enumerate(totals):
        ax.text(idx, total + totals.max() * 0.025, f"Total\n{_int_label(total)}", ha="center", va="bottom", fontsize=11, fontweight="bold", color=TEXT_DARK)

    ax.set_title("Decisioned population coverage by split", fontsize=18, fontweight="bold", color=TEXT_DARK)
    ax.set_ylabel("Applications")
    ax.yaxis.set_major_formatter(_millions_formatter())
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.01, "Most of the underwriting funnel is non-booked, which is why applicant-stage coverage matters.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_holdout_chart(results_df: pd.DataFrame, output_path: Path) -> Path:
    df = _non_calibrated_results(results_df).copy()
    y = np.arange(len(df))
    colors = [MODEL_COLORS[str(model)] for model in df["Model"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.15, wspace=0.20)

    for ax, metric, span in zip(axes, ["ROC AUC", "PR AUC"], [0.08, 0.045], strict=False):
        values = df[metric].astype(float).to_numpy()
        ax.barh(y, values, color=colors, height=0.62)
        ax.set_title(metric, fontsize=16, fontweight="bold", color=TEXT_DARK)
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(max(0, values.min() - span * 0.6), values.max() + span * 0.35)
        for idx, value in enumerate(values):
            ax.text(value + span * 0.02, idx, _metric_label(value), va="center", fontsize=10, color=TEXT_DARK)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["Model"].astype(str).tolist(), fontsize=11)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()
    fig.suptitle("Held-out booked proxy benchmark comparison", fontsize=20, fontweight="bold", color=TEXT_DARK)
    fig.text(0.5, 0.02, "Logistic Regression is the strongest current in-house candidate on the held-out booked proxy test.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_auc_lift_chart(benchmark_comparisons_df: pd.DataFrame, output_path: Path) -> Path:
    df = _non_calibrated_comparisons(benchmark_comparisons_df)
    references = ["score_RF (benchmark)", "risk_score_rf (benchmark)"]
    titles = ["Lift versus score_RF", "Lift versus risk_score_rf"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.88, bottom=0.15, wspace=0.20)

    for ax, reference, title in zip(axes, references, titles, strict=False):
        plot_df = df.loc[df["reference_model"] == reference].copy()
        plot_df = plot_df.sort_values("candidate_model")
        y = np.arange(len(plot_df))
        point = plot_df["auc_improvement"].astype(float).to_numpy()
        lo = plot_df["auc_improvement_lo"].astype(float).to_numpy()
        hi = plot_df["auc_improvement_hi"].astype(float).to_numpy()
        p_values = plot_df["auc_delong_p_value"].astype(float).to_numpy()
        colors = [MODEL_COLORS[str(model)] for model in plot_df["candidate_model"]]

        ax.axvline(0.0, color="#6b7280", linestyle="--", linewidth=1.5)
        ax.errorbar(point, y, xerr=np.vstack([point - lo, hi - point]), fmt="none", ecolor="#9ca3af", elinewidth=2, capsize=4, zorder=1)
        ax.scatter(point, y, s=90, color=colors, zorder=2)
        ax.set_title(title, fontsize=16, fontweight="bold", color=TEXT_DARK)
        ax.set_xlabel("AUC improvement")
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(min(lo.min(), -0.04) - 0.005, max(hi.max(), 0.06) + 0.005)
        for idx, (p_value, value) in enumerate(zip(p_values, point, strict=False)):
            ax.text(value + 0.0025, idx + 0.14, f"p={p_value:.3g}", fontsize=9, color="#4b5563")

        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["candidate_model"].astype(str).tolist(), fontsize=11)
        ax.invert_yaxis()

    axes[1].set_yticks(axes[0].get_yticks())
    axes[1].set_yticklabels([])
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle("Paired AUC lift with confidence intervals", fontsize=20, fontweight="bold", color=TEXT_DARK)
    fig.text(0.5, 0.02, "Intervals entirely to the right of zero indicate a positive and stable benchmark lift.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_rolling_oot_chart(rolling_oot_results_df: pd.DataFrame, output_path: Path) -> Path:
    df = rolling_oot_results_df.loc[rolling_oot_results_df["is_calibrated"] == False].copy()
    df = df.loc[df["Model"].isin(MODEL_ORDER)].copy()
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["Model", "fold"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, ["ROC AUC", "PR AUC"], strict=False):
        for model in MODEL_ORDER:
            model_df = df.loc[df["Model"] == model]
            ax.plot(
                model_df["fold"],
                model_df[metric],
                marker="o",
                linewidth=2.2,
                markersize=6,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.set_title(metric, fontsize=16, fontweight="bold", color=TEXT_DARK)
        ax.set_xlabel("Forward validation window")
        ax.set_ylabel(metric)
        ax.set_xticks(sorted(df["fold"].unique()))
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    fig.suptitle("Rolling out-of-time performance by forward window", fontsize=20, fontweight="bold", color=TEXT_DARK)
    fig.text(0.5, 0.02, "The ranking is directionally stable across the four forward validation windows.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_calibration_chart(results_df: pd.DataFrame, output_path: Path) -> Path:
    raw_df = results_df.loc[results_df["Model"].isin(CANDIDATE_ORDER), ["Model", "Brier"]].copy()
    calibrated_df = results_df.loc[results_df["Model"].str.contains("(calibrated)", regex=False, na=False), ["Model", "Brier"]].copy()
    calibrated_df["base_model"] = calibrated_df["Model"].str.replace(" (calibrated)", "", regex=False)
    merged = raw_df.merge(calibrated_df[["base_model", "Brier"]], left_on="Model", right_on="base_model", suffixes=("_raw", "_calibrated"))
    merged["Model"] = pd.Categorical(merged["Model"], categories=CANDIDATE_ORDER, ordered=True)
    merged = merged.sort_values("Model").reset_index(drop=True)

    x = np.arange(len(merged))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, merged["Brier_raw"], width=width, color="#8d99ae", label="Raw")
    ax.bar(x + width / 2, merged["Brier_calibrated"], width=width, color="#2a9d8f", label="Calibrated")

    for idx, row in enumerate(merged.itertuples(index=False)):
        ax.text(idx - width / 2, row.Brier_raw + 0.004, _metric_label(row.Brier_raw), ha="center", fontsize=10, color=TEXT_DARK, rotation=90)
        ax.text(idx + width / 2, row.Brier_calibrated + 0.004, _metric_label(row.Brier_calibrated), ha="center", fontsize=10, color=TEXT_DARK, rotation=90)

    ax.set_xticks(x, merged["Model"].astype(str).tolist())
    ax.set_ylabel("Brier score")
    ax.set_title("Calibration improves probability quality", fontsize=19, fontweight="bold", color=TEXT_DARK)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02, "Calibrated model probabilities are more usable for thresholds and expected-loss analysis.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def create_holdout_curves_chart(results_df: pd.DataFrame, holdout_scores_df: pd.DataFrame, output_path: Path) -> Path:
    results_lookup = _non_calibrated_results(results_df).set_index("Model")
    model_names = [
        model_name
        for model_name in MODEL_ORDER
        if model_name in results_lookup.index and _score_column_name(model_name) in holdout_scores_df.columns
    ]

    if not model_names:
        raise ValueError("No model score columns found for ROC/PR curve chart generation")

    y_true = holdout_scores_df[TARGET].astype(int).to_numpy()
    positive_rate = float(y_true.mean())

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.80, top=0.88, bottom=0.14, wspace=0.18)

    roc_ax, pr_ax = axes
    roc_ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#9ca3af")
    pr_ax.axhline(positive_rate, linestyle="--", linewidth=1.5, color="#9ca3af")

    legend_handles: list[Line2D] = []

    for model_name in model_names:
        scores = holdout_scores_df[_score_column_name(model_name)].to_numpy(dtype=float)
        mask = np.isfinite(scores)
        model_y = y_true[mask]
        model_scores = scores[mask]

        fpr, tpr, _ = roc_curve(model_y, model_scores)
        precision, recall, _ = precision_recall_curve(model_y, model_scores)
        color = MODEL_COLORS[model_name]

        roc_ax.plot(fpr, tpr, linewidth=2.3, color=color)
        pr_ax.plot(recall[::-1], precision[::-1], linewidth=2.3, color=color)

        result_row = results_lookup.loc[model_name]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.8,
                label=f"{model_name} | ROC {result_row['ROC AUC']:.3f} | PR {result_row['PR AUC']:.3f}",
            )
        )

    roc_ax.set_title("ROC curve", fontsize=16, fontweight="bold", color=TEXT_DARK)
    roc_ax.set_xlabel("False positive rate")
    roc_ax.set_ylabel("True positive rate")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.grid(alpha=0.25)
    roc_ax.spines[["top", "right"]].set_visible(False)

    pr_ax.set_title("Precision-recall curve", fontsize=16, fontweight="bold", color=TEXT_DARK)
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(bottom=0)
    pr_ax.grid(alpha=0.25)
    pr_ax.spines[["top", "right"]].set_visible(False)

    fig.legend(handles=legend_handles, frameon=False, loc="center left", bbox_to_anchor=(0.81, 0.50), fontsize=10)
    fig.suptitle("Held-out booked proxy ROC and precision-recall curves", fontsize=20, fontweight="bold", color=TEXT_DARK)
    fig.text(0.5, 0.02, "Curves use the official booked-proxy test sample. The PR baseline reflects the low default rate.", ha="center", fontsize=10, color="#4b5563")
    return _save_figure(fig, output_path)


def generate_stakeholder_charts(output_dir: Path) -> list[Path]:
    results_df = _read_csv(output_dir, "results.csv")
    benchmark_comparisons_df = _read_csv(output_dir, "benchmark_comparisons.csv")
    rolling_oot_results_df = _read_csv(output_dir, "rolling_oot_results.csv")
    population_summary_df = _read_csv(output_dir, "population_summary.csv")
    holdout_scores_df = _maybe_read_csv(output_dir, "holdout_test_scores.csv")
    plots_dir = output_dir / "plots"

    generated = [
        create_kpi_chart(results_df, population_summary_df, benchmark_comparisons_df, plots_dir / "stakeholder_kpis.png"),
        create_process_chart(plots_dir / "stakeholder_before_after.png"),
        create_population_chart(population_summary_df, plots_dir / "stakeholder_population_coverage.png"),
        create_holdout_chart(results_df, plots_dir / "stakeholder_holdout_benchmarks.png"),
        create_auc_lift_chart(benchmark_comparisons_df, plots_dir / "stakeholder_auc_lift.png"),
        create_rolling_oot_chart(rolling_oot_results_df, plots_dir / "stakeholder_rolling_oot.png"),
        create_calibration_chart(results_df, plots_dir / "stakeholder_calibration.png"),
    ]

    if holdout_scores_df is not None:
        generated.append(
            create_holdout_curves_chart(results_df, holdout_scores_df, plots_dir / "stakeholder_roc_pr_curves.png")
        )

    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    generated = generate_stakeholder_charts(Path(args.output_dir))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
