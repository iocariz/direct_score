from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_constants import (
    BENCHMARK_MODEL_NAMES,
    EXPERIMENTAL_STACKING_NAME,
)
from training_reporting import (
    delong_auc_test,
    paired_bootstrap_benchmark_comparisons,
    paired_bootstrap_metric_delta,
    split_leaderboard_results,
)


class TestPairedBootstrapMetricDelta:
    def test_identical_scores_have_zero_improvement(self):
        y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
        scores = np.linspace(0.05, 0.95, len(y_true))

        auc_stats = paired_bootstrap_metric_delta(
            y_true, scores, scores.copy(), "AUC", n_bootstrap=200,
        )
        brier_stats = paired_bootstrap_metric_delta(
            y_true, scores, scores.copy(), "Brier", n_bootstrap=200,
        )

        assert auc_stats["improvement"] == pytest.approx(0.0)
        assert auc_stats["improvement_lo"] == pytest.approx(0.0)
        assert auc_stats["improvement_hi"] == pytest.approx(0.0)
        assert auc_stats["p_value"] == pytest.approx(1.0)
        assert brier_stats["improvement"] == pytest.approx(0.0)
        assert brier_stats["p_value"] == pytest.approx(1.0)

    def test_clearly_better_scores_have_positive_improvement(self):
        y_true = np.array([0] * 60 + [1] * 40)
        candidate_scores = np.concatenate([
            np.linspace(0.01, 0.20, 60),
            np.linspace(0.80, 0.99, 40),
        ])
        reference_scores = np.concatenate([
            np.linspace(0.30, 0.55, 60),
            np.linspace(0.45, 0.70, 40),
        ])

        auc_stats = paired_bootstrap_metric_delta(
            y_true, candidate_scores, reference_scores, "AUC", n_bootstrap=200,
        )
        pr_stats = paired_bootstrap_metric_delta(
            y_true, candidate_scores, reference_scores, "PR_AUC", n_bootstrap=200,
        )
        brier_stats = paired_bootstrap_metric_delta(
            y_true, candidate_scores, reference_scores, "Brier", n_bootstrap=200,
        )

        assert auc_stats["improvement"] > 0
        assert auc_stats["improvement_lo"] > 0
        assert pr_stats["improvement"] > 0
        assert pr_stats["improvement_lo"] > 0
        assert brier_stats["improvement"] > 0
        assert brier_stats["improvement_lo"] > 0


class TestPairedBootstrapBenchmarkComparisons:
    def test_returns_expected_schema(self):
        y_true = np.array([0, 0, 1, 1] * 25)
        score_arrays = {
            "Logistic Regression": np.concatenate([
                np.linspace(0.10, 0.30, 50),
                np.linspace(0.70, 0.90, 50),
            ]),
            BENCHMARK_MODEL_NAMES[0]: -np.linspace(300, 700, 100),
            BENCHMARK_MODEL_NAMES[1]: -np.linspace(200, 600, 100),
        }

        comparisons_df = paired_bootstrap_benchmark_comparisons(
            y_true,
            score_arrays,
            candidate_model_names=["Logistic Regression"],
            n_bootstrap=50,
        )

        expected_columns = {
            "candidate_model", "reference_model", "n", "n_pos", "n_neg",
            "candidate_auc", "reference_auc", "auc_improvement", "auc_improvement_lo", "auc_improvement_hi", "auc_p_value",
            "auc_delong_se", "auc_delong_z", "auc_delong_p_value",
            "candidate_pr_auc", "reference_pr_auc", "pr_auc_improvement", "pr_auc_improvement_lo", "pr_auc_improvement_hi", "pr_auc_p_value",
            "candidate_brier", "reference_brier", "brier_improvement", "brier_improvement_lo", "brier_improvement_hi", "brier_p_value",
        }

        assert expected_columns.issubset(comparisons_df.columns)
        assert len(comparisons_df) == 2
        assert set(comparisons_df["reference_model"]) == set(BENCHMARK_MODEL_NAMES)
        assert comparisons_df["brier_improvement"].isna().all()


class TestDelongAucTest:
    def test_identical_predictors_return_null_result(self):
        y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
        scores = np.linspace(0.05, 0.95, len(y_true))

        stats = delong_auc_test(y_true, scores, scores.copy())

        assert stats["auc_improvement"] == pytest.approx(0.0)
        assert stats["p_value"] == pytest.approx(1.0)
        assert stats["z_score"] == pytest.approx(0.0)

    def test_clearly_distinct_predictors_return_small_p_value(self):
        y_true = np.array([0] * 60 + [1] * 40)
        candidate_scores = np.concatenate([
            np.linspace(0.01, 0.20, 60),
            np.linspace(0.80, 0.99, 40),
        ])
        reference_scores = np.concatenate([
            np.linspace(0.20, 0.65, 60),
            np.linspace(0.35, 0.80, 40),
        ])

        stats = delong_auc_test(y_true, candidate_scores, reference_scores)

        assert stats["candidate_auc"] > stats["reference_auc"]
        assert stats["auc_improvement"] > 0
        assert stats["p_value"] < 0.05


class TestSplitLeaderboardResults:
    def test_excludes_experimental_rows_from_primary_leaderboard(self):
        results_df = pd.DataFrame(
            {"PR AUC": [0.09, 0.08, 0.07, 0.06]},
            index=[
                BENCHMARK_MODEL_NAMES[0],
                "Logistic Regression",
                "Logistic Regression (calibrated)",
                EXPERIMENTAL_STACKING_NAME,
            ],
        )

        official_results_df, experimental_results_df = split_leaderboard_results(results_df, reject_inference=False)

        assert EXPERIMENTAL_STACKING_NAME not in official_results_df.index
        assert list(experimental_results_df.index) == [EXPERIMENTAL_STACKING_NAME]

    def test_reject_inference_pushes_trained_models_to_experimental(self):
        results_df = pd.DataFrame(
            {"PR AUC": [0.09, 0.08, 0.07]},
            index=[
                BENCHMARK_MODEL_NAMES[0],
                BENCHMARK_MODEL_NAMES[1],
                "Logistic Regression",
            ],
        )

        official_results_df, experimental_results_df = split_leaderboard_results(results_df, reject_inference=True)

        assert list(official_results_df.index) == BENCHMARK_MODEL_NAMES
        assert list(experimental_results_df.index) == ["Logistic Regression"]
