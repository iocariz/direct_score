from __future__ import annotations

import stakeholder_charts as stakeholder_charts_module
import pandas as pd

from stakeholder_charts import (
    _non_calibrated_comparisons,
    _non_calibrated_results,
    create_kpi_chart,
    create_holdout_curves_chart,
)
from training_constants import BENCHMARK_MODEL_NAMES, TARGET


class TestStakeholderCharts:
    def test_non_calibrated_results_keeps_ebm(self):
        results_df = pd.DataFrame(
            [
                {"Model": BENCHMARK_MODEL_NAMES[0], "ROC AUC": 0.58, "PR AUC": 0.07},
                {"Model": "Logistic Regression", "ROC AUC": 0.61, "PR AUC": 0.09},
                {"Model": "EBM", "ROC AUC": 0.62, "PR AUC": 0.10},
                {"Model": "EBM (calibrated)", "ROC AUC": 0.62, "PR AUC": 0.10},
                {"Model": BENCHMARK_MODEL_NAMES[1], "ROC AUC": 0.57, "PR AUC": 0.06},
            ]
        )

        filtered = _non_calibrated_results(results_df)

        assert "EBM" in filtered["Model"].astype(str).tolist()
        assert "EBM (calibrated)" not in filtered["Model"].astype(str).tolist()

    def test_non_calibrated_comparisons_keeps_ebm(self):
        comparisons_df = pd.DataFrame(
            [
                {
                    "candidate_model": "Logistic Regression",
                    "reference_model": BENCHMARK_MODEL_NAMES[0],
                    "auc_improvement": 0.02,
                    "auc_improvement_lo": 0.00,
                    "auc_improvement_hi": 0.04,
                    "auc_delong_p_value": 0.05,
                },
                {
                    "candidate_model": "EBM",
                    "reference_model": BENCHMARK_MODEL_NAMES[0],
                    "auc_improvement": 0.03,
                    "auc_improvement_lo": 0.01,
                    "auc_improvement_hi": 0.05,
                    "auc_delong_p_value": 0.03,
                },
                {
                    "candidate_model": "EBM (calibrated)",
                    "reference_model": BENCHMARK_MODEL_NAMES[0],
                    "auc_improvement": 0.03,
                    "auc_improvement_lo": 0.01,
                    "auc_improvement_hi": 0.05,
                    "auc_delong_p_value": 0.03,
                },
            ]
        )

        filtered = _non_calibrated_comparisons(comparisons_df)

        assert "EBM" in filtered["candidate_model"].astype(str).tolist()
        assert "EBM (calibrated)" not in filtered["candidate_model"].astype(str).tolist()

    def test_create_holdout_curves_chart_supports_ebm(self, tmp_path):
        results_df = pd.DataFrame(
            [
                {"Model": BENCHMARK_MODEL_NAMES[0], "ROC AUC": 0.58, "PR AUC": 0.07},
                {"Model": "EBM", "ROC AUC": 0.62, "PR AUC": 0.10},
                {"Model": BENCHMARK_MODEL_NAMES[1], "ROC AUC": 0.57, "PR AUC": 0.06},
            ]
        )
        holdout_scores_df = pd.DataFrame(
            {
                TARGET: [0, 1, 0, 1],
                "score__risk_score_rf_benchmark": [0.20, 0.60, 0.30, 0.70],
                "score__ebm": [0.10, 0.80, 0.25, 0.85],
                "score__score_rf_benchmark": [0.15, 0.55, 0.35, 0.65],
            }
        )

        output_path = tmp_path / "holdout_curves.png"
        create_holdout_curves_chart(results_df, holdout_scores_df, output_path, selected_model="EBM")

        assert output_path.exists()

    def test_create_kpi_chart_defaults_to_best_candidate_when_selected_model_missing(self, tmp_path, monkeypatch):
        captured = {}

        def fake_save_figure(fig, path):
            captured["figure_text"] = "\n".join(text.get_text() for text in fig.texts)
            path.write_text("stub")
            return path

        monkeypatch.setattr(stakeholder_charts_module, "_save_figure", fake_save_figure)

        results_df = pd.DataFrame(
            [
                {"Model": BENCHMARK_MODEL_NAMES[0], "ROC AUC": 0.58, "PR AUC": 0.07, "N": 4},
                {"Model": "EBM", "ROC AUC": 0.62, "PR AUC": 0.10, "N": 4},
                {"Model": "Logistic Regression", "ROC AUC": 0.61, "PR AUC": 0.09, "N": 4},
            ]
        )
        population_summary_df = pd.DataFrame(
            [
                {"split": "post_split", "status_name": "Booked", "n_rows": 4},
                {"split": "post_split", "status_name": "Rejected", "n_rows": 2},
                {"split": "post_split", "status_name": "Canceled", "n_rows": 1},
            ]
        )
        benchmark_comparisons_df = pd.DataFrame(
            [
                {"candidate_model": "EBM", "reference_model": BENCHMARK_MODEL_NAMES[0], "auc_improvement": 0.04, "n_pos": 1},
                {"candidate_model": "Logistic Regression", "reference_model": BENCHMARK_MODEL_NAMES[0], "auc_improvement": 0.03, "n_pos": 2},
            ]
        )

        output_path = tmp_path / "kpi_chart.png"
        create_kpi_chart(results_df, population_summary_df, benchmark_comparisons_df, output_path, selected_model=None)

        assert output_path.exists()
        assert "Selected model: EBM." in captured["figure_text"]
