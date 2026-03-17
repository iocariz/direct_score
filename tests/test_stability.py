"""Tests for PSI, CSI, WoE/IV, and stability analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import compute_csi, compute_psi, compute_woe_iv


class TestComputePSI:
    def test_identical_distributions_return_zero(self):
        scores = np.linspace(0.0, 1.0, 500)
        psi = compute_psi(scores, scores.copy())
        assert psi == pytest.approx(0.0, abs=1e-8)

    def test_similar_distributions_return_low_psi(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0.5, 0.1, 1000)
        test = rng.normal(0.5, 0.1, 1000)
        psi = compute_psi(train, test)
        assert psi < 0.10

    def test_shifted_distribution_returns_higher_psi(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0.5, 0.1, 1000)
        test = rng.normal(0.8, 0.1, 1000)
        psi = compute_psi(train, test)
        assert psi > 0.25

    def test_psi_is_nonnegative(self):
        rng = np.random.RandomState(42)
        for _ in range(5):
            train = rng.uniform(0, 1, 300)
            test = rng.uniform(0, 1, 300)
            assert compute_psi(train, test) >= 0.0

    def test_custom_n_bins(self):
        scores = np.linspace(0, 1, 500)
        psi_5 = compute_psi(scores, scores, n_bins=5)
        psi_20 = compute_psi(scores, scores, n_bins=20)
        assert psi_5 == pytest.approx(0.0, abs=1e-8)
        assert psi_20 == pytest.approx(0.0, abs=1e-8)


class TestComputeCSI:
    @pytest.fixture()
    def stable_data(self):
        rng = np.random.RandomState(42)
        n = 200
        train = pd.DataFrame({
            "num1": rng.normal(50, 10, n),
            "num2": rng.uniform(0, 100, n),
            "cat1": rng.choice(["A", "B", "C"], n),
        })
        test = pd.DataFrame({
            "num1": rng.normal(50, 10, n),
            "num2": rng.uniform(0, 100, n),
            "cat1": rng.choice(["A", "B", "C"], n),
        })
        return train, test

    def test_returns_dataframe_with_expected_columns(self, stable_data):
        train, test = stable_data
        csi_df = compute_csi(train, test, ["num1", "num2"], ["cat1"])
        assert {"feature", "type", "csi", "n_bins"}.issubset(csi_df.columns)

    def test_stable_features_have_low_csi(self, stable_data):
        train, test = stable_data
        csi_df = compute_csi(train, test, ["num1", "num2"], ["cat1"])
        assert (csi_df["csi"] < 0.25).all()

    def test_drifted_numerical_has_higher_csi(self):
        rng = np.random.RandomState(42)
        n = 200
        train = pd.DataFrame({"x": rng.normal(0, 1, n), "cat": rng.choice(["A", "B"], n)})
        test = pd.DataFrame({"x": rng.normal(5, 1, n), "cat": rng.choice(["A", "B"], n)})
        csi_df = compute_csi(train, test, ["x"], ["cat"])
        x_csi = csi_df.loc[csi_df["feature"] == "x", "csi"].iloc[0]
        assert x_csi > 0.25

    def test_skips_features_with_few_samples(self):
        train = pd.DataFrame({"x": [1.0, 2.0]})
        test = pd.DataFrame({"x": [1.0, 2.0]})
        csi_df = compute_csi(train, test, ["x"], [])
        # Empty result — either empty DF or DF with no rows for "x"
        assert csi_df.empty or "x" not in csi_df["feature"].values

    def test_sorted_by_csi_descending(self, stable_data):
        train, test = stable_data
        csi_df = compute_csi(train, test, ["num1", "num2"], ["cat1"])
        if len(csi_df) > 1:
            assert (csi_df["csi"].diff().dropna() <= 0).all()


class TestComputeWoeIV:
    @pytest.fixture()
    def woe_data(self):
        rng = np.random.RandomState(42)
        n = 500
        X = pd.DataFrame({
            "income": rng.uniform(20, 100, n),
            "age": rng.uniform(18, 65, n),
            "segment": rng.choice(["A", "B", "C"], n),
        })
        y = pd.Series((rng.uniform(0, 1, n) < 0.10).astype(int))
        return X, y

    def test_returns_two_dataframes(self, woe_data):
        X, y = woe_data
        woe_df, iv_df = compute_woe_iv(X, y, ["income", "age"], ["segment"])
        assert isinstance(woe_df, pd.DataFrame)
        assert isinstance(iv_df, pd.DataFrame)

    def test_woe_has_expected_columns(self, woe_data):
        X, y = woe_data
        woe_df, _ = compute_woe_iv(X, y, ["income", "age"], ["segment"])
        expected = {"feature", "bin", "type", "n_total", "n_good", "n_bad", "event_rate", "woe", "iv"}
        assert expected.issubset(woe_df.columns)

    def test_iv_aggregates_per_feature(self, woe_data):
        X, y = woe_data
        woe_df, iv_df = compute_woe_iv(X, y, ["income", "age"], ["segment"])
        assert set(iv_df["feature"]) == set(woe_df["feature"].unique())

    def test_iv_is_nonnegative(self, woe_data):
        X, y = woe_data
        _, iv_df = compute_woe_iv(X, y, ["income", "age"], ["segment"])
        assert (iv_df["iv"] >= 0).all()

    def test_iv_sorted_descending(self, woe_data):
        X, y = woe_data
        _, iv_df = compute_woe_iv(X, y, ["income", "age"], ["segment"])
        if len(iv_df) > 1:
            assert (iv_df["iv"].diff().dropna() <= 0).all()

    def test_empty_when_no_bad_cases(self):
        X = pd.DataFrame({"x": [1, 2, 3], "cat": ["A", "B", "A"]})
        y = pd.Series([0, 0, 0])
        woe_df, iv_df = compute_woe_iv(X, y, ["x"], ["cat"])
        assert woe_df.empty
        assert iv_df.empty

    def test_handles_missing_values(self):
        rng = np.random.RandomState(42)
        n = 300
        X = pd.DataFrame({
            "x": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(0, 100, n)),
        })
        y = pd.Series((rng.uniform(0, 1, n) < 0.15).astype(int))
        woe_df, iv_df = compute_woe_iv(X, y, ["x"], [])
        assert not woe_df.empty
