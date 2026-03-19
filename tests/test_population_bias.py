"""Tests for population bias analysis: KS test, selection bias, adverse impact."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_reporting import (
    compute_adverse_impact_analysis,
    compute_population_ks_test,
    compute_selection_bias_correlation,
    sanitize_output_name,
)


def _make_applicant_scores(n_booked=200, n_rejected=300, seed=42):
    rng = np.random.RandomState(seed)
    n = n_booked + n_rejected
    status = np.array(["Booked"] * n_booked + ["Rejected"] * n_rejected)
    risk_score = rng.uniform(10, 100, n)
    # Model scores: correlated with risk_score but with noise
    model_score = (100 - risk_score) / 100 + rng.normal(0, 0.05, n)
    model_score = np.clip(model_score, 0, 1)
    df = pd.DataFrame({
        "status_name": status,
        "risk_score_rf": risk_score,
        f"score__{sanitize_output_name('LightGBM')}": model_score,
        f"score__{sanitize_output_name('Logistic Regression')}": model_score + rng.normal(0, 0.02, n),
    })
    return df


class TestComputePopulationKSTest:
    def test_returns_expected_columns(self):
        df = _make_applicant_scores()
        result = compute_population_ks_test(df, model_names=["LightGBM"])
        expected = {"model", "n_booked", "n_non_booked", "booked_mean_score",
                    "non_booked_mean_score", "ks_statistic", "ks_p_value"}
        assert expected.issubset(result.columns)

    def test_detects_separation(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "status_name": ["Booked"] * 100 + ["Rejected"] * 100,
            "score__test_model": np.concatenate([
                rng.uniform(0.1, 0.3, 100),
                rng.uniform(0.7, 0.9, 100),
            ]),
        })
        result = compute_population_ks_test(df, model_names=["test_model"])
        assert result.iloc[0]["ks_statistic"] > 0.5
        assert result.iloc[0]["ks_p_value"] < 0.01

    def test_no_separation_for_identical_distributions(self):
        rng = np.random.RandomState(42)
        scores = rng.uniform(0.3, 0.7, 200)
        df = pd.DataFrame({
            "status_name": ["Booked"] * 100 + ["Rejected"] * 100,
            "score__model": scores,
        })
        result = compute_population_ks_test(df, model_names=["model"])
        assert result.iloc[0]["ks_p_value"] > 0.05

    def test_empty_when_no_applicant_data(self):
        assert compute_population_ks_test(None).empty
        assert compute_population_ks_test(pd.DataFrame()).empty

    def test_empty_when_single_status(self):
        df = pd.DataFrame({
            "status_name": ["Booked"] * 50,
            "score__model": np.random.uniform(0, 1, 50),
        })
        assert compute_population_ks_test(df, model_names=["model"]).empty


class TestComputeSelectionBiasCorrelation:
    def test_returns_expected_columns(self):
        df = _make_applicant_scores()
        result = compute_selection_bias_correlation(df, model_names=["LightGBM"])
        expected = {"model", "n_valid", "pearson_corr", "spearman_corr", "selection_bias_flag"}
        assert expected.issubset(result.columns)

    def test_detects_high_correlation(self):
        rng = np.random.RandomState(42)
        risk = rng.uniform(10, 100, 200)
        df = pd.DataFrame({
            "risk_score_rf": risk,
            "score__model": (100 - risk) / 100,  # perfect inverse correlation
        })
        result = compute_selection_bias_correlation(df, model_names=["model"])
        assert result.iloc[0]["selection_bias_flag"] == "HIGH"
        assert result.iloc[0]["pearson_corr"] > 0.90

    def test_low_correlation_when_independent(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "risk_score_rf": rng.uniform(10, 100, 200),
            "score__model": rng.uniform(0, 1, 200),
        })
        result = compute_selection_bias_correlation(df, model_names=["model"])
        assert result.iloc[0]["selection_bias_flag"] == "LOW"

    def test_empty_when_no_risk_score(self):
        df = pd.DataFrame({"score__model": [0.5, 0.6, 0.7]})
        assert compute_selection_bias_correlation(df).empty


class TestComputeAdverseImpactAnalysis:
    @pytest.fixture()
    def ai_data(self):
        rng = np.random.RandomState(42)
        n = 500
        age = rng.uniform(18, 70, n)
        y = (rng.uniform(0, 1, n) < 0.08).astype(int)
        scores = rng.uniform(0.01, 0.30, n)
        return y, scores, age

    def test_returns_expected_columns(self, ai_data):
        y, scores, age = ai_data
        df = compute_adverse_impact_analysis(y, scores, age, "test")
        expected = {"model", "age_band", "n", "n_defaults", "observed_default_rate",
                    "mean_predicted_pd", "approval_rate_at_10pct_reject",
                    "adverse_impact_ratio", "air_flag"}
        assert expected.issubset(df.columns)

    def test_air_bounded_zero_to_one(self, ai_data):
        y, scores, age = ai_data
        df = compute_adverse_impact_analysis(y, scores, age, "test")
        assert (df["adverse_impact_ratio"] >= 0).all()
        assert (df["adverse_impact_ratio"] <= 1.0 + 1e-9).all()

    def test_best_band_has_air_one(self, ai_data):
        y, scores, age = ai_data
        df = compute_adverse_impact_analysis(y, scores, age, "test")
        assert df["adverse_impact_ratio"].max() == pytest.approx(1.0)

    def test_flags_disparate_impact(self):
        # Young applicants get systematically higher risk scores
        rng = np.random.RandomState(42)
        n = 400
        age = np.concatenate([rng.uniform(18, 25, n // 2), rng.uniform(40, 60, n // 2)])
        y = np.zeros(n, dtype=int)
        scores = np.concatenate([
            rng.uniform(0.20, 0.50, n // 2),  # young: high PD
            rng.uniform(0.01, 0.10, n // 2),   # older: low PD
        ])
        df = compute_adverse_impact_analysis(y, scores, age, "test")
        young_band = df.loc[df["age_band"] == "18-25"]
        if not young_band.empty:
            assert young_band.iloc[0]["air_flag"] == "FAIL"

    def test_custom_age_bins(self, ai_data):
        y, scores, age = ai_data
        df = compute_adverse_impact_analysis(y, scores, age, "test", age_bins=[(18, 40), (40, 70)])
        assert len(df) == 2

    def test_empty_when_no_valid_data(self):
        df = compute_adverse_impact_analysis(
            np.array([]), np.array([]), np.array([]), "test"
        )
        assert df.empty
