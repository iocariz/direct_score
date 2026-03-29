"""Tests for reporting functions: lift table, threshold analysis, bootstrap CI, Holm-Bonferroni."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_reporting import (
    _holm_bonferroni,
    bootstrap_confidence_intervals,
    create_lift_table,
    create_threshold_analysis,
    paired_bootstrap_benchmark_comparisons,
)
from training_constants import BENCHMARK_MODEL_NAMES


class TestCreateLiftTable:
    @pytest.fixture()
    def lift_data(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 80), rng.uniform(0.4, 0.9, 20)])
        return y, scores

    def test_returns_expected_columns(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test")
        expected = {"model", "decile", "n_accounts", "n_defaults", "default_rate",
                    "cum_default_rate", "lift", "cum_lift", "capture_rate"}
        assert expected.issubset(df.columns)

    def test_has_10_deciles(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test")
        assert len(df) == 10

    def test_capture_rate_reaches_one(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test")
        assert df["capture_rate"].iloc[-1] == pytest.approx(1.0)

    def test_first_decile_has_highest_lift(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test")
        assert df["lift"].iloc[0] >= df["lift"].iloc[-1]

    def test_cumulative_defaults_sum_to_total(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test")
        assert df["n_defaults"].sum() == int(y.sum())

    def test_custom_decile_count(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "test", n_deciles=5)
        assert len(df) == 5

    def test_model_name_propagated(self, lift_data):
        y, scores = lift_data
        df = create_lift_table(y, scores, "MyModel")
        assert (df["model"] == "MyModel").all()


class TestCreateThresholdAnalysis:
    @pytest.fixture()
    def threshold_data(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 80), rng.uniform(0.4, 0.9, 20)])
        return y, scores

    def test_returns_expected_columns(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test")
        expected = {"model", "reject_pct", "score_threshold", "n_rejected", "n_total",
                    "true_positives", "false_positives", "false_negatives", "true_negatives",
                    "precision", "recall", "fpr", "capture_rate"}
        assert expected.issubset(df.columns)

    def test_default_thresholds(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test")
        assert list(df["reject_pct"]) == [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    def test_custom_thresholds(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test", thresholds_pct=[10.0, 50.0])
        assert len(df) == 2

    def test_confusion_matrix_sums_to_n(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test")
        for _, row in df.iterrows():
            assert row["true_positives"] + row["false_positives"] + row["false_negatives"] + row["true_negatives"] == row["n_total"]

    def test_higher_rejection_captures_more_defaults(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test")
        captures = df["capture_rate"].values
        assert all(captures[i] <= captures[i + 1] for i in range(len(captures) - 1))

    def test_precision_recall_in_range(self, threshold_data):
        y, scores = threshold_data
        df = create_threshold_analysis(y, scores, "test")
        assert (df["precision"] >= 0).all() and (df["precision"] <= 1).all()
        assert (df["recall"] >= 0).all() and (df["recall"] <= 1).all()


class TestHolmBonferroni:
    def test_single_p_value_unchanged(self):
        adjusted = _holm_bonferroni(np.array([0.03]))
        assert adjusted[0] == pytest.approx(0.03)

    def test_adjusts_multiple_p_values_upward(self):
        p_values = np.array([0.01, 0.04, 0.05])
        adjusted = _holm_bonferroni(p_values)
        assert (adjusted >= p_values).all()

    def test_preserves_order_of_significance(self):
        p_values = np.array([0.01, 0.03, 0.05])
        adjusted = _holm_bonferroni(p_values)
        assert adjusted[0] <= adjusted[1] <= adjusted[2]

    def test_caps_at_one(self):
        p_values = np.array([0.5, 0.6, 0.8])
        adjusted = _holm_bonferroni(p_values)
        assert (adjusted <= 1.0).all()

    def test_handles_nan(self):
        p_values = np.array([0.01, np.nan, 0.05])
        adjusted = _holm_bonferroni(p_values)
        assert not np.isnan(adjusted[0])
        assert np.isnan(adjusted[1])
        assert not np.isnan(adjusted[2])

    def test_empty_array(self):
        adjusted = _holm_bonferroni(np.array([]))
        assert len(adjusted) == 0


class TestBootstrapCIPointEstimate:
    def test_uses_observed_statistic_not_median(self):
        """B1 fix: CI point estimate must be the observed test-set statistic."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 80), rng.uniform(0.4, 0.9, 20)])

        observed_auc = roc_auc_score(y, scores)
        ci = bootstrap_confidence_intervals(y, {"test": scores}, n_bootstrap=200)

        assert ci.loc["test", "AUC"] == pytest.approx(observed_auc)

    def test_ci_bounds_bracket_point_estimate(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 80), rng.uniform(0.4, 0.9, 20)])
        ci = bootstrap_confidence_intervals(y, {"test": scores}, n_bootstrap=500)

        assert ci.loc["test", "AUC_lo"] <= ci.loc["test", "AUC"]
        assert ci.loc["test", "AUC_hi"] >= ci.loc["test", "AUC"]

    def test_block_bootstrap_accepts_dates(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 80), rng.uniform(0.4, 0.9, 20)])
        dates = pd.date_range("2024-07-01", periods=100, freq="D")
        ci = bootstrap_confidence_intervals(y, {"test": scores}, n_bootstrap=100, dates=dates)

        assert "AUC_lo" in ci.columns
        assert np.isfinite(ci.loc["test", "AUC"])


class TestBenchmarkComparisonsAdjustedPValues:
    def test_adjusted_columns_present(self):
        y_true = np.array([0, 0, 1, 1] * 25)
        score_arrays = {
            "LR": np.concatenate([np.linspace(0.1, 0.3, 50), np.linspace(0.7, 0.9, 50)]),
            BENCHMARK_MODEL_NAMES[0]: -np.linspace(300, 700, 100),
            BENCHMARK_MODEL_NAMES[1]: -np.linspace(200, 600, 100),
        }
        df = paired_bootstrap_benchmark_comparisons(y_true, score_arrays, ["LR"], n_bootstrap=50)
        assert "auc_p_adjusted" in df.columns
        assert "pr_auc_p_adjusted" in df.columns
        assert "brier_p_adjusted" in df.columns

    def test_effect_size_column_present(self):
        y_true = np.array([0, 0, 1, 1] * 25)
        score_arrays = {
            "LR": np.concatenate([np.linspace(0.1, 0.3, 50), np.linspace(0.7, 0.9, 50)]),
            BENCHMARK_MODEL_NAMES[0]: -np.linspace(300, 700, 100),
        }
        df = paired_bootstrap_benchmark_comparisons(
            y_true, score_arrays, ["LR"],
            reference_model_names=[BENCHMARK_MODEL_NAMES[0]], n_bootstrap=50,
        )
        assert "auc_improvement_pct_of_max" in df.columns
