"""Tests for reporting functions: lift table, threshold analysis, bootstrap CI, Holm-Bonferroni."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_reporting import (
    _bca_bounds,
    _block_jackknife_metrics,
    _holm_bonferroni,
    bootstrap_confidence_intervals,
    calibration_diagnostics,
    compute_calibration_diagnostics_report,
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


class TestBcaBootstrap:
    def test_bca_falls_back_to_percentile_with_no_jackknife(self):
        """With <2 jackknife values BCa cannot compute acceleration; expect percentile bounds."""
        boot = list(np.linspace(0.6, 0.9, 200))
        lo, hi = _bca_bounds(boot, np.array([0.7]), observed=0.75, alpha=0.025)
        # Percentile fallback for a uniformly-spaced sample: ~2.5th and 97.5th
        # percentile of [0.6, 0.9] are roughly 0.6075 and 0.8925.
        assert 0.60 <= lo <= 0.62
        assert 0.88 <= hi <= 0.90

    def test_bca_returns_nan_when_no_bootstraps(self):
        lo, hi = _bca_bounds([], np.array([0.5, 0.6, 0.7]), observed=0.6, alpha=0.025)
        assert np.isnan(lo) and np.isnan(hi)

    def test_bca_brackets_observed_when_distribution_is_skewed(self):
        """Skewed bootstrap distribution: BCa lower bound should be tighter than the symmetric percentile bound."""
        # Construct a deliberately right-skewed bootstrap distribution where
        # the observed value is to the LEFT of the median — this is the
        # regime where BCa correctly tightens the lower bound.
        rng = np.random.RandomState(0)
        # Lognormal samples are right-skewed.
        boot = list(0.5 + rng.lognormal(mean=-2.5, sigma=0.4, size=2000))
        observed = float(np.percentile(boot, 30))  # observed below the median
        # Jackknife mimics a similar skew with smaller magnitude.
        jack = 0.5 + rng.lognormal(mean=-2.5, sigma=0.4, size=80)

        lo_bca, hi_bca = _bca_bounds(boot, jack, observed=observed, alpha=0.025)
        # Percentile baseline:
        lo_pct = float(np.percentile(boot, 2.5))
        hi_pct = float(np.percentile(boot, 97.5))

        # The bracket must always include the observed value.
        assert lo_bca <= observed <= hi_bca
        # BCa correction must produce a different bracket from raw percentile;
        # otherwise the BCa code is a no-op.
        assert (lo_bca, hi_bca) != (lo_pct, hi_pct)

    def test_bca_observed_is_within_ci_for_real_metric(self):
        """End-to-end: the observed AUC must lie inside the BCa CI."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.RandomState(42)
        y = np.array([0] * 200 + [1] * 50)
        scores = np.concatenate([rng.uniform(0.1, 0.5, 200), rng.uniform(0.4, 0.9, 50)])
        observed = roc_auc_score(y, scores)

        ci = bootstrap_confidence_intervals(y, {"m": scores}, n_bootstrap=500)
        lo = float(ci.loc["m", "AUC_lo"])
        hi = float(ci.loc["m", "AUC_hi"])
        assert lo <= observed <= hi
        # CI must be non-trivial.
        assert hi - lo > 0.01

    def test_block_jackknife_skips_blocks_that_remove_a_class(self):
        """Removing a block that contains all positives must not crash _block_jackknife_metrics."""
        y = np.array([0, 0, 0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
        # Block 1 contains both positives — leaving it out yields all-zero y,
        # so AUC/PR are undefined. The helper must skip it gracefully.
        blocks = [np.array([0, 1]), np.array([4, 5])]
        auc_jack, pr_jack, _ = _block_jackknife_metrics(y, s, blocks, is_prob=True)
        # Only the first block is valid.
        assert len(auc_jack) == 1
        assert len(pr_jack) == 1


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


class TestCalibrationDiagnostics:
    """ECE / MCE / per-bin reliability table — standard regulator metrics."""

    def test_perfectly_calibrated_predictions_have_near_zero_ece(self):
        """ECE measures the gap between predicted and observed default rate.
        When the predictions equal the actual base rate, ECE → 0."""
        rng = np.random.RandomState(0)
        n = 5000
        # Uniform "predictions" that match the observed rate by construction.
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        ece, mce, bins_df = calibration_diagnostics(y, p, n_bins=10)
        assert ece < 0.05, f"ECE should be near zero for calibrated preds, got {ece}"
        assert 0 <= mce < 0.10
        assert not bins_df.empty
        assert (bins_df["n"] >= 1).all()

    def test_systematically_overconfident_predictions_have_high_ece(self):
        """A score that always predicts 0.9 with a 30% base rate should have ECE around 0.6."""
        n = 1000
        p = np.full(n, 0.9)
        rng = np.random.RandomState(1)
        y = (rng.uniform(0, 1, n) < 0.30).astype(int)
        ece, mce, _ = calibration_diagnostics(y, p, n_bins=10)
        # observed ~0.30, predicted 0.90 → gap ~0.60
        assert ece > 0.50
        assert mce >= ece

    def test_returns_empty_for_empty_input(self):
        ece, mce, bins_df = calibration_diagnostics(np.array([]), np.array([]))
        assert np.isnan(ece) and np.isnan(mce)
        assert bins_df.empty

    def test_compute_report_skips_non_probability_scores(self):
        """Benchmark scores stored as -raw_score are not in [0,1] — they must be
        excluded from ECE (which is only meaningful on probabilities)."""
        y = np.array([0, 1, 0, 1] * 50)
        scores = {
            "LR (prob)": np.linspace(0.05, 0.95, 200),
            "raw_benchmark": np.linspace(-100, 100, 200),  # not a probability
        }
        summary, details = compute_calibration_diagnostics_report(y, scores, n_bins=5)
        assert "LR (prob)" in summary["model"].values
        assert "raw_benchmark" not in summary["model"].values
        # The detail frame must not have raw_benchmark either.
        if not details.empty:
            assert "raw_benchmark" not in details["model"].values

    def test_per_bin_table_has_predicted_and_observed(self):
        """The per-bin table is the artifact reliability diagrams plot from."""
        rng = np.random.RandomState(2)
        n = 500
        p = rng.uniform(0, 1, n)
        y = (rng.uniform(0, 1, n) < p).astype(int)
        _, _, bins_df = calibration_diagnostics(y, p, n_bins=5)
        for col in ["bin", "n", "mean_predicted", "observed_default_rate", "gap", "weight"]:
            assert col in bins_df.columns
        # predicted and observed should both be in [0, 1]; gap is their difference.
        assert (bins_df["mean_predicted"].between(0, 1)).all()
        assert (bins_df["observed_default_rate"].between(0, 1)).all()
        assert np.allclose(bins_df["gap"], bins_df["mean_predicted"] - bins_df["observed_default_rate"])
