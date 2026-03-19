"""Tests for utility / helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from training_constants import MIN_VALID
from training_features import _loo_target_encode, _safe_auc
from training_reporting import _ks_statistic


# ── _loo_target_encode ────────────────────────────────────────────────────────

class TestLooTargetEncode:
    def test_basic(self):
        groups = np.array(["A", "A", "A", "B", "B"])
        y = np.array([1, 0, 0, 1, 1])
        result = _loo_target_encode(groups, y)
        # For group A: LOO means are (0+0)/2, (1+0)/2, (1+0)/2
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.5)
        # For group B: LOO means are 1/1, 1/1
        assert result[3] == pytest.approx(1.0)
        assert result[4] == pytest.approx(1.0)

    def test_single_member_group_uses_global_mean(self):
        groups = np.array(["A", "A", "C"])
        y = np.array([1, 0, 1])
        result = _loo_target_encode(groups, y)
        # Group C has 1 member -> LOO = NaN -> filled with global mean = 2/3
        assert result[2] == pytest.approx(y.mean())

    def test_output_shape(self):
        n = 50
        groups = np.random.choice(["X", "Y", "Z"], n)
        y = np.random.randint(0, 2, n)
        result = _loo_target_encode(groups, y)
        assert result.shape == (n,)


# ── _ks_statistic ─────────────────────────────────────────────────────────────

class TestKsStatistic:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        ks = _ks_statistic(y_true, y_score)
        assert ks == pytest.approx(1.0)

    def test_no_separation(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        ks = _ks_statistic(y_true, y_score)
        assert ks == pytest.approx(0.0)

    def test_range(self):
        rng = np.random.RandomState(123)
        y_true = rng.randint(0, 2, 200)
        y_score = rng.random(200)
        ks = _ks_statistic(y_true, y_score)
        assert 0.0 <= ks <= 1.0


# ── _safe_auc ──────────────────────────────────────────────────────────────────

class TestSafeAuc:
    def test_returns_nan_when_too_few_valid(self):
        y = np.array([0, 1])
        s = np.array([0.3, 0.7])
        auc, n = _safe_auc(y, s)
        assert np.isnan(auc)
        assert n == 2

    def test_returns_nan_when_too_few_positives(self):
        n = MIN_VALID + 100
        y = np.zeros(n)
        y[0] = 1  # only 1 positive (< 10 threshold)
        s = np.random.random(n)
        auc, _ = _safe_auc(y, s)
        assert np.isnan(auc)

    def test_valid_auc(self):
        n = MIN_VALID + 100
        rng = np.random.RandomState(0)
        y = rng.choice([0, 1], size=n, p=[0.9, 0.1])
        s = y.astype(float) + rng.normal(0, 0.3, n)
        auc, count = _safe_auc(y, s)
        assert 0.5 < auc <= 1.0
        assert count == n

    def test_handles_nan_scores(self):
        n = MIN_VALID + 100
        rng = np.random.RandomState(1)
        y = rng.choice([0, 1], size=n, p=[0.9, 0.1])
        s = rng.random(n)
        s[:50] = np.nan
        auc, count = _safe_auc(y, s)
        assert count == n - 50
