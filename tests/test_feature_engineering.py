"""Tests for feature engineering, preprocessing, and selection functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training_features as training_features_module
from training_constants import MONOTONE_MAP
from training_features import (
    GROUP_STAT_PAIRS,
    add_frequency_encoding,
    add_group_stats,
    add_modeling_features,
    build_monotone_constraints,
    build_preprocessors,
    prune_correlated,
    reduce_cardinality,
    run_rfecv,
)


_RNG = np.random.RandomState(99)


class _FixedTemporalCV:
    def __init__(self, folds):
        self._folds = [(np.asarray(train_idx), np.asarray(val_idx)) for train_idx, val_idx in folds]

    def split(self, X=None, y=None, groups=None):
        for train_idx, val_idx in self._folds:
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds)


def _make_fixed_temporal_cv(block_size=20, n_blocks=6):
    folds = []
    for boundary in range(2, n_blocks):
        train_end = boundary * block_size
        val_end = (boundary + 1) * block_size
        folds.append((np.arange(train_end), np.arange(train_end, val_end)))
    return _FixedTemporalCV(folds)


def _make_train_test(n_train=200, n_test=80):
    """Synthetic train/test pair with numerical and categorical columns."""
    def _make(n):
        return pd.DataFrame({
            "num1": _RNG.uniform(0, 100, n),
            "num2": _RNG.normal(50, 10, n),
            "cat1": _RNG.choice(["A", "B", "C", "D", "E"], n),
            "cat2": _RNG.choice(["X", "Y"], n),
        })
    return _make(n_train), _make(n_test)


class TestAddFrequencyEncoding:
    def test_adds_freq_columns(self):
        train, test = _make_train_test()
        train_out, test_out, added = add_frequency_encoding(train, test, ["cat1", "cat2"])
        assert added == ["FREQ_cat1", "FREQ_cat2"]
        assert "FREQ_cat1" in train_out.columns
        assert "FREQ_cat1" in test_out.columns

    def test_frequencies_sum_to_one(self):
        train, test = _make_train_test()
        train_out, _, _ = add_frequency_encoding(train, test, ["cat1"])
        unique_freqs = train_out.groupby("cat1")["FREQ_cat1"].first()
        assert unique_freqs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_unseen_categories_get_zero(self):
        train = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
        test = pd.DataFrame({"cat": ["A", "C"]})
        _, test_out, _ = add_frequency_encoding(train, test, ["cat"])
        assert test_out.loc[test_out["cat"] == "C", "FREQ_cat"].iloc[0] == 0.0

    def test_does_not_modify_originals(self):
        train, test = _make_train_test()
        orig_cols = set(train.columns)
        add_frequency_encoding(train, test, ["cat1"])
        assert set(train.columns) == orig_cols

    def test_preserves_row_count(self):
        train, test = _make_train_test()
        train_out, test_out, _ = add_frequency_encoding(train, test, ["cat1"])
        assert len(train_out) == len(train)
        assert len(test_out) == len(test)


class TestAddGroupStats:
    def test_creates_group_stat_features(self):
        n = 200
        train = pd.DataFrame({
            "INCOME_T1": _RNG.uniform(20, 100, n),
            "CSP": _RNG.choice(["A", "B", "C"], n),
        })
        test = train.iloc[:50].copy()
        _, _, added = add_group_stats(train, test, ["INCOME_T1"], ["CSP"])
        assert "INCOME_T1_VS_CSP" in added

    def test_skips_missing_pair_columns(self):
        train = pd.DataFrame({"x": [1, 2, 3], "cat": ["A", "B", "A"]})
        test = train.copy()
        _, _, added = add_group_stats(train, test, ["x"], ["cat"])
        # GROUP_STAT_PAIRS doesn't include ("x", "cat"), so nothing added
        assert len(added) == 0

    def test_handles_zero_group_median(self):
        train = pd.DataFrame({
            "INCOME_T1": [0.0, 0.0, 0.0, 10.0, 20.0, 30.0],
            "CSP": ["A", "A", "A", "B", "B", "B"],
        })
        test = train.copy()
        train_out, _, added = add_group_stats(train, test, ["INCOME_T1"], ["CSP"])
        if "INCOME_T1_VS_CSP" in added:
            assert train_out["INCOME_T1_VS_CSP"].isna().sum() >= 0  # no crash

    def test_preserves_row_count(self):
        n = 100
        train = pd.DataFrame({
            "INCOME_T1": _RNG.uniform(20, 100, n),
            "CSP": _RNG.choice(["A", "B"], n),
        })
        test = train.iloc[:30].copy()
        train_out, test_out, _ = add_group_stats(train, test, ["INCOME_T1"], ["CSP"])
        assert len(train_out) == len(train)
        assert len(test_out) == 30


class TestPruneCorrelated:
    def test_drops_highly_correlated(self):
        n = 200
        x = _RNG.uniform(0, 100, n)
        df = pd.DataFrame({"a": x, "b": x + _RNG.normal(0, 0.01, n), "c": _RNG.uniform(0, 100, n)})
        to_drop = prune_correlated(df, ["a", "b", "c"], threshold=0.95)
        assert len(to_drop) >= 1
        assert "a" in to_drop or "b" in to_drop

    def test_keeps_uncorrelated(self):
        n = 200
        df = pd.DataFrame({
            "a": _RNG.normal(0, 1, n),
            "b": _RNG.normal(10, 5, n),
            "c": _RNG.uniform(0, 1, n),
        })
        to_drop = prune_correlated(df, ["a", "b", "c"], threshold=0.95)
        assert len(to_drop) == 0

    def test_returns_empty_with_single_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert prune_correlated(df, ["a"]) == []

    def test_returns_empty_with_no_columns(self):
        df = pd.DataFrame()
        assert prune_correlated(df, []) == []


class TestBuildPreprocessors:
    def test_returns_three_objects(self):
        preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(
            ["num1", "num2"], ["cat1"]
        )
        assert hasattr(preprocessor, "fit_transform")
        assert hasattr(lgbm_preprocessor, "fit_transform")
        assert lgbm_cat_indices == [2]

    def test_cat_indices_correct_offset(self):
        _, _, cat_idx = build_preprocessors(["a", "b", "c"], ["x", "y"])
        assert cat_idx == [3, 4]

    def test_empty_categoricals(self):
        preprocessor, lgbm_preprocessor, cat_idx = build_preprocessors(["a", "b"], [])
        assert cat_idx == []

    def test_can_fit_transform(self):
        preprocessor, lgbm_preprocessor, _ = build_preprocessors(["num1"], ["cat1"])
        n = 50
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "num1": rng.uniform(0, 100, n),
            "cat1": rng.choice(["A", "B"], n),
        })
        y = pd.Series(rng.choice([0, 1], n, p=[0.8, 0.2]))
        result = preprocessor.fit_transform(df, y)
        assert result.shape[0] == n
        assert result.shape[1] == 2


class TestBuildMonotoneConstraints:
    def test_length_matches_feature_count(self):
        num = ["INCOME_T1", "AGE_T1", "TOTAL_AMT"]
        cat = ["CSP", "CPRO"]
        constraints = build_monotone_constraints(num, cat)
        assert len(constraints) == len(num) + len(cat)

    def test_categorical_features_unconstrained(self):
        num = ["INCOME_T1"]
        cat = ["CSP", "CPRO", "CMAT"]
        constraints = build_monotone_constraints(num, cat)
        assert constraints[-3:] == [0, 0, 0]

    def test_known_constraints_applied(self):
        num = ["INCOME_T1", "INSTALLMENT_TO_INCOME"]
        constraints = build_monotone_constraints(num, [])
        assert constraints[0] == MONOTONE_MAP["INCOME_T1"]
        assert constraints[1] == MONOTONE_MAP["INSTALLMENT_TO_INCOME"]

    def test_unknown_features_unconstrained(self):
        constraints = build_monotone_constraints(["UNKNOWN_FEATURE"], [])
        assert constraints == [0]


class TestRunRfecv:
    def test_keeps_stable_numeric_signal(self, monkeypatch):
        rng = np.random.RandomState(7)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_THRESHOLD", 0.75)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_MIN_FEATURES", 1)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_C_VALUES", (0.1, 0.3, 1.0))

        n_blocks = 6
        block_size = 20
        n_rows = n_blocks * block_size
        y = np.array(([0] * 10 + [1] * 10) * n_blocks)
        X = pd.DataFrame({
            "num_signal": y + rng.normal(0.0, 0.05, n_rows),
            "num_noise_1": rng.normal(0.0, 1.0, n_rows),
            "num_noise_2": rng.normal(0.0, 1.0, n_rows),
        })

        feature_cols = ["num_signal", "num_noise_1", "num_noise_2"]
        selected_features, selected_num_cols, selected_cat_cols = run_rfecv(
            X,
            pd.Series(y),
            num_cols=feature_cols.copy(),
            cat_cols=[],
            feature_cols=feature_cols.copy(),
            cv=_make_fixed_temporal_cv(block_size=block_size, n_blocks=n_blocks),
        )

        assert "num_signal" in selected_features
        assert "num_signal" in selected_num_cols
        assert selected_cat_cols == []

    def test_keeps_stable_categorical_signal(self, monkeypatch):
        rng = np.random.RandomState(11)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_THRESHOLD", 0.75)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_MIN_FEATURES", 1)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_C_VALUES", (0.1, 0.3, 1.0))

        n_blocks = 6
        block_size = 20
        n_rows = n_blocks * block_size
        y = np.array(([0] * 10 + [1] * 10) * n_blocks)
        X = pd.DataFrame({
            "num_noise": rng.normal(0.0, 1.0, n_rows),
            "cat_signal": np.where(y == 1, "good", "bad"),
            "cat_noise": rng.choice(["A", "B", "C"], size=n_rows),
        })

        feature_cols = ["num_noise", "cat_signal", "cat_noise"]
        selected_features, selected_num_cols, selected_cat_cols = run_rfecv(
            X,
            pd.Series(y),
            num_cols=["num_noise"],
            cat_cols=["cat_signal", "cat_noise"],
            feature_cols=feature_cols.copy(),
            cv=_make_fixed_temporal_cv(block_size=block_size, n_blocks=n_blocks),
        )

        assert "cat_signal" in selected_features
        assert "cat_signal" in selected_cat_cols
        assert set(selected_num_cols) | set(selected_cat_cols) == set(selected_features)

    def test_backfills_minimum_feature_count_when_threshold_is_too_strict(self, monkeypatch):
        rng = np.random.RandomState(19)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_THRESHOLD", 1.1)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_MIN_FEATURES", 3)
        monkeypatch.setattr(training_features_module, "STABILITY_SELECTION_C_VALUES", (0.1, 0.3))

        n_blocks = 6
        block_size = 20
        n_rows = n_blocks * block_size
        y = np.array(([0] * 10 + [1] * 10) * n_blocks)
        X = pd.DataFrame({
            "num_a": y + rng.normal(0.0, 0.20, n_rows),
            "num_b": rng.normal(0.0, 1.0, n_rows),
            "num_c": rng.normal(0.0, 1.0, n_rows),
            "num_d": rng.normal(0.0, 1.0, n_rows),
        })

        feature_cols = ["num_a", "num_b", "num_c", "num_d"]
        selected_features, selected_num_cols, selected_cat_cols = run_rfecv(
            X,
            pd.Series(y),
            num_cols=feature_cols.copy(),
            cat_cols=[],
            feature_cols=feature_cols.copy(),
            cv=_make_fixed_temporal_cv(block_size=block_size, n_blocks=n_blocks),
        )

        assert len(selected_features) == 3
        assert len(selected_num_cols) == 3
        assert selected_cat_cols == []
        assert set(selected_features).issubset(feature_cols)


class TestAddModelingFeatures:
    def test_returns_seven_tuple(self):
        n = 100
        train = pd.DataFrame({
            "INCOME_T1": _RNG.uniform(20, 100, n),
            "AGE_T1": _RNG.uniform(18, 65, n),
            "CSP": _RNG.choice(["A", "B", "C"], n),
        })
        test = train.iloc[:30].copy()
        result = add_modeling_features(train, test, ["INCOME_T1", "AGE_T1", "CSP"], ["INCOME_T1", "AGE_T1"], ["CSP"])
        assert len(result) == 7

    def test_new_features_added_to_lists(self):
        n = 100
        train = pd.DataFrame({
            "INCOME_T1": _RNG.uniform(20, 100, n),
            "AGE_T1": _RNG.uniform(18, 65, n),
            "CSP": _RNG.choice(["A", "B", "C"], n),
        })
        test = train.iloc[:30].copy()
        _, _, all_features, num_cols, cat_cols, freq_cols, group_cols = add_modeling_features(
            train, test, ["INCOME_T1", "AGE_T1", "CSP"], ["INCOME_T1", "AGE_T1"], ["CSP"],
        )
        assert "FREQ_CSP" in freq_cols
        assert "FREQ_CSP" in num_cols
        assert "FREQ_CSP" in all_features
        # cat_cols should remain unchanged
        assert cat_cols == ["CSP"]
