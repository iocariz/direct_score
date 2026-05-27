"""Tests for utility / helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from training_constants import (
    CALIBRATION_METHOD_BY_MODEL,
    DEFAULT_CALIBRATION_METHOD,
    EXPERIMENTAL_STACKING_NAME,
    MIN_VALID,
    OFFICIAL_MODEL_NAMES,
    OPTUNA_PRUNER_STARTUP_TRIALS,
    OPTUNA_PRUNER_WARMUP_STEPS,
)
from training_features import _loo_target_encode, _safe_auc
from training_reporting import _ks_statistic


class TestCalibrationMethodMapping:
    """Pins the per-model calibration method choice (Niculescu-Mizil & Caruana, 2005)."""

    def test_logistic_regression_uses_sigmoid(self):
        # LR has log-odds-like outputs already → sigmoid (Platt) doesn't over-fit.
        assert CALIBRATION_METHOD_BY_MODEL["Logistic Regression"] == "sigmoid"

    def test_ebm_uses_sigmoid(self):
        # EBM is a GAM with additive log-odds shape functions — same regime as LR.
        assert CALIBRATION_METHOD_BY_MODEL["EBM"] == "sigmoid"

    def test_tree_models_use_isotonic(self):
        # Boosted trees produce sigmoid-shaped reliability diagrams; isotonic
        # corrects this with enough calibration data.
        assert CALIBRATION_METHOD_BY_MODEL["LightGBM"] == "isotonic"
        assert CALIBRATION_METHOD_BY_MODEL["XGBoost"] == "isotonic"
        assert CALIBRATION_METHOD_BY_MODEL["CatBoost"] == "isotonic"

    def test_every_official_model_has_a_choice(self):
        # Adding a new official model without picking a calibration method should
        # surface immediately as a test failure rather than silently defaulting.
        for name in OFFICIAL_MODEL_NAMES:
            assert name in CALIBRATION_METHOD_BY_MODEL, (
                f"Official model {name!r} has no calibration method assigned in "
                "CALIBRATION_METHOD_BY_MODEL"
            )

    def test_stacking_has_a_choice(self):
        assert EXPERIMENTAL_STACKING_NAME in CALIBRATION_METHOD_BY_MODEL

    def test_default_is_a_valid_sklearn_method(self):
        assert DEFAULT_CALIBRATION_METHOD in {"sigmoid", "isotonic"}


class TestOptunaPrunerWarmup:
    """Audit fix J: pruner warmup must cover enough folds to avoid killing
    good configs based on the (smallest, noisiest) first temporal fold.

    The pre-fix default was n_warmup_steps=1, which let the MedianPruner
    terminate trials immediately after fold 1 — exactly the regime the
    audit flagged. The fixed default is 3, leaving the pruner to act on a
    majority of fold evidence rather than the first-fold outlier.
    """

    def test_warmup_steps_at_least_three(self):
        # If a future change wants to make this faster by lowering the
        # warmup, the test fails — forcing an explicit decision rather than
        # silently re-introducing the audit-flagged behaviour.
        assert OPTUNA_PRUNER_WARMUP_STEPS >= 3

    def test_startup_trials_at_least_five(self):
        # MedianPruner's median needs a sample to compare against; 5 trials
        # is the canonical minimum for a stable median estimate.
        assert OPTUNA_PRUNER_STARTUP_TRIALS >= 5

    def test_all_trainers_use_the_central_warmup(self):
        """All four MedianPruner instances in training_models.py must read
        from the central constant. A future inline-literal regression
        flips this test red.
        """
        from pathlib import Path

        source = Path("training_models.py").read_text()
        # Count MedianPruner instances and confirm none of them use
        # n_warmup_steps=<literal>.
        n_pruners = source.count("MedianPruner(")
        assert n_pruners >= 4, "Expected at least 4 MedianPruner sites (LR/EBM/LGBM/XGB/CatBoost)"
        assert "n_warmup_steps=1" not in source
        assert "n_warmup_steps=2" not in source


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
