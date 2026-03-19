"""Tests for temporal split, evaluation metrics, and data loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import (
    TemporalExpandingCV,
    build_rolling_oot_windows,
    load_data,
    make_temporal_cv,
    resolve_temporal_feature_discovery_cutoff,
    summarize_population,
    temporal_calibration_split,
    temporal_feature_discovery_split,
    temporal_split,
)
from training_constants import (
    MATURITY_CUTOFF,
    SPLIT_DATE,
    TARGET,
)
from training_features import select_features
from training_reporting import evaluate


class TestTemporalSplit:
    def test_train_before_split_date(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        X_train, y_train, X_test, y_test, _, _, _ = temporal_split(engineered_df, feature_cols)

        # Recover dates from the original df by index alignment
        train_dates = engineered_df.loc[X_train.index, "mis_Date"]
        test_dates = engineered_df.loc[X_test.index, "mis_Date"]

        assert (train_dates < SPLIT_DATE).all(), "Train set must be before split date"
        assert (test_dates >= SPLIT_DATE).all(), "Test set must be on or after split date"

    def test_no_immature_in_model_data(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        X_train, y_train, X_test, y_test, _, _, _ = temporal_split(engineered_df, feature_cols)

        assert not y_train.isna().any(), "Train target must have no NaN"
        assert not y_test.isna().any(), "Test target must have no NaN"

    def test_target_is_binary(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        _, y_train, _, y_test, _, _, _ = temporal_split(engineered_df, feature_cols)

        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_no_overlap(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        X_train, _, X_test, _, _, _, _ = temporal_split(engineered_df, feature_cols)

        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, "Train and test indices must not overlap"


class TestTemporalExpandingCV:
    def test_fold_dates_are_strictly_ordered(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01",
        ])
        cv = TemporalExpandingCV(dates, n_splits=3)

        for train_idx, val_idx in cv.split():
            train_dates = dates[train_idx]
            val_dates = dates[val_idx]
            assert train_dates.max() < val_dates.min()
            assert set(train_dates) & set(val_dates) == set()

    def test_raises_when_distinct_date_blocks_are_insufficient(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
        ])
        with pytest.raises(ValueError, match="distinct date blocks"):
            TemporalExpandingCV(dates, n_splits=3)

    def test_training_folds_expand_monotonically_without_splitting_date_cohorts(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01", "2024-06-01",
        ])
        cv = TemporalExpandingCV(dates, n_splits=4)

        previous_train_idx: set[int] = set()
        previous_train_dates = pd.Index([])
        for train_idx, val_idx in cv.split():
            train_idx_set = set(train_idx.tolist())
            val_idx_set = set(val_idx.tolist())
            train_dates = pd.Index(dates[train_idx])
            val_dates = pd.Index(dates[val_idx])

            assert previous_train_idx.issubset(train_idx_set)
            assert previous_train_dates.isin(train_dates).all()
            assert train_idx_set.isdisjoint(val_idx_set)
            assert set(train_dates.unique()) & set(val_dates.unique()) == set()

            for date_value in train_dates.unique():
                expected_idx = set(np.flatnonzero(dates == date_value).tolist())
                assert expected_idx.issubset(train_idx_set)
                assert expected_idx.isdisjoint(val_idx_set)

            for date_value in val_dates.unique():
                expected_idx = set(np.flatnonzero(dates == date_value).tolist())
                assert expected_idx.issubset(val_idx_set)
                assert expected_idx.isdisjoint(train_idx_set)

            previous_train_idx = train_idx_set
            previous_train_dates = train_dates.unique()


class TestMakeTemporalCV:
    def test_reduces_fold_count_to_available_date_blocks(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
        ])

        cv = make_temporal_cv(dates, max_splits=5)

        assert cv.get_n_splits() == 3
        fold_boundaries = cv.fold_boundaries_
        assert len(fold_boundaries) == 3
        assert fold_boundaries[-1]["val_end"] == pd.Timestamp("2024-04-01")

    def test_fails_clearly_when_too_few_time_blocks_exist(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
        ])

        with pytest.raises(ValueError, match="at least 3 distinct date blocks"):
            make_temporal_cv(dates, max_splits=5)


class TestTemporalCalibrationSplit:
    def test_uses_latest_dates_for_calibration(self):
        X = pd.DataFrame({"x": np.arange(12)})
        y = pd.Series([0, 1] * 6, name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01",
        ])

        X_fit, X_calib, y_fit, y_calib, dates_fit, dates_calib = temporal_calibration_split(
            X, y, dates, calibration_fraction=0.25,
        )

        assert len(X_fit) == len(y_fit) == len(dates_fit)
        assert len(X_calib) == len(y_calib) == len(dates_calib)
        assert pd.Timestamp(dates_fit.max()) < pd.Timestamp(dates_calib.min())
        assert set(pd.to_datetime(dates_calib)) == {
            pd.Timestamp("2024-05-01"),
            pd.Timestamp("2024-06-01"),
        }

    def test_uses_booked_rows_to_choose_calibration_window(self):
        X = pd.DataFrame({"x": np.arange(8)})
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
        ])
        sample_weight = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5])

        _, _, _, _, w_fit, w_calib, dates_fit, dates_calib = temporal_calibration_split(
            X, y, dates, calibration_fraction=0.20, sample_weight=sample_weight,
        )

        booked_calib_dates = pd.to_datetime(dates_calib[w_calib == 1.0])
        assert pd.Timestamp(dates_fit.max()) < booked_calib_dates.min()
        assert set(booked_calib_dates) == {pd.Timestamp("2024-03-01")}

    def test_calibration_set_contains_all_rows_from_latest_selected_date_blocks(self):
        X = pd.DataFrame({"x": np.arange(10)})
        y = pd.Series([0, 1] * 5, name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
        ])

        X_fit, X_calib, y_fit, y_calib, dates_fit, dates_calib = temporal_calibration_split(
            X,
            y,
            dates,
            calibration_fraction=0.30,
        )

        assert len(X_fit) == len(y_fit) == len(dates_fit)
        assert len(X_calib) == len(y_calib) == len(dates_calib)
        assert pd.Timestamp(dates_fit.max()) < pd.Timestamp(dates_calib.min())
        assert set(pd.to_datetime(dates_calib)) == {
            pd.Timestamp("2024-04-01"),
            pd.Timestamp("2024-05-01"),
        }
        assert X_calib.index.tolist() == [6, 7, 8, 9]


class TestTemporalFeatureDiscoverySplit:
    def test_uses_earliest_dates_for_discovery(self):
        X = pd.DataFrame({"x": np.arange(12)})
        y = pd.Series([0, 1] * 6, name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01",
        ])

        X_discovery, X_estimation, y_discovery, y_estimation, dates_discovery, dates_estimation = temporal_feature_discovery_split(
            X, y, dates, discovery_fraction=0.50,
        )

        assert len(X_discovery) == len(y_discovery) == len(dates_discovery)
        assert set(pd.to_datetime(dates_discovery)) == {
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-02-01"),
            pd.Timestamp("2024-03-01"),
        }

    def test_explicit_discovery_end_matches_fraction_cutoff(self):
        X = pd.DataFrame({"x": np.arange(12)})
        y = pd.Series([0, 1] * 6, name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01",
        ])

        discovery_end = resolve_temporal_feature_discovery_cutoff(dates, discovery_fraction=0.50)
        split_from_fraction = temporal_feature_discovery_split(
            X,
            y,
            dates,
            discovery_fraction=0.50,
        )
        split_from_explicit_end = temporal_feature_discovery_split(
            X,
            y,
            dates,
            discovery_end=discovery_end,
        )

        assert discovery_end == pd.Timestamp("2024-03-01")
        assert split_from_fraction[0].index.tolist() == split_from_explicit_end[0].index.tolist()
        assert split_from_fraction[1].index.tolist() == split_from_explicit_end[1].index.tolist()
        assert pd.to_datetime(split_from_fraction[4]).tolist() == pd.to_datetime(split_from_explicit_end[4]).tolist()
        assert pd.to_datetime(split_from_fraction[5]).tolist() == pd.to_datetime(split_from_explicit_end[5]).tolist()

    def test_requires_fraction_between_zero_and_one(self):
        X = pd.DataFrame({"x": np.arange(4)})
        y = pd.Series([0, 1, 0, 1], name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
        ])

        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            temporal_feature_discovery_split(X, y, dates, discovery_fraction=1.0)


class TestBuildRollingOOTWindows:
    def test_windows_are_strictly_ordered_and_expanding(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01",
        ])

        windows = build_rolling_oot_windows(dates, max_windows=3, min_train_date_blocks=2)

        assert len(windows) == 3
        previous_train_size = 0
        for window in windows:
            train_dates = dates[window["train_idx"]]
            validation_dates = dates[window["validation_idx"]]
            assert len(pd.Index(train_dates).unique()) >= 2
            assert train_dates.max() < validation_dates.min()
            assert set(train_dates) & set(validation_dates) == set()
            assert len(window["train_idx"]) > previous_train_size
            previous_train_size = len(window["train_idx"])

    def test_requires_enough_date_blocks(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01",
        ])

        with pytest.raises(ValueError, match="distinct date blocks"):
            build_rolling_oot_windows(dates, max_windows=2, min_train_date_blocks=2)


class TestSummarizePopulation:
    def test_returns_population_metadata_for_weighted_sample(self):
        y = pd.Series([0, 1, 0, 1], name=TARGET)
        dates = pd.to_datetime([
            "2024-01-01",
            "2024-02-01",
            "2024-02-01",
            "2024-03-01",
        ])
        sample_weight = np.array([1.0, 1.0, 0.5, 0.5])

        summary = summarize_population(
            y, dates, "development rows", sample_weight=sample_weight,
        )

        assert summary["sample_definition"] == "development rows"
        assert summary["n_rows"] == 4
        assert summary["n_pos"] == 2
        assert summary["target_rate"] == pytest.approx(0.5)
        assert summary["date_start"] == pd.Timestamp("2024-01-01").date()
        assert summary["date_end"] == pd.Timestamp("2024-03-01").date()
        assert summary["n_booked_rows"] == 2
        assert summary["n_pseudo_labeled_rows"] == 2
        assert summary["n_booked_pos"] == 1

    def test_raises_when_lengths_do_not_match(self):
        y = pd.Series([0, 1], name=TARGET)
        dates = pd.to_datetime(["2024-01-01"])

        with pytest.raises(ValueError, match="same length"):
            summarize_population(y, dates, "bad sample")


class TestEvaluate:
    def test_metrics_keys(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate("test_model", y_true, y_score)
        assert "ROC AUC" in result
        assert "Gini" in result
        assert "KS" in result
        assert "PR AUC" in result
        assert "Brier" in result
        assert "N" in result

    def test_gini_equals_2auc_minus_1(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = evaluate("test", y_true, y_score)
        assert result["Gini"] == pytest.approx(2 * result["ROC AUC"] - 1)

    def test_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = evaluate("perfect", y_true, y_score)
        assert result["ROC AUC"] == pytest.approx(1.0)
        assert result["Gini"] == pytest.approx(1.0)
        assert result["KS"] == pytest.approx(1.0)
        assert result["Brier"] == pytest.approx(0.0)

    def test_nan_scores_filtered(self):
        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, np.nan, 0.3, 0.9])
        result = evaluate("with_nan", y_true, y_score)
        assert result["N"] == 3

    def test_non_probability_skips_brier(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([-10.0, -5.0, 5.0, 10.0])
        result = evaluate("non_prob", y_true, y_score, is_probability=False)
        assert np.isnan(result["Brier"])


class TestLoadData:
    def test_only_booked(self, raw_df_with_rejects, tmp_path):
        # Write synthetic data to parquet, then load
        path = tmp_path / "test_data.parquet"
        raw_df_with_rejects.to_parquet(path)
        df = load_data(str(path))
        assert (df["status_name"] == "Booked").all()

    def test_shape_reduced(self, raw_df_with_rejects, tmp_path):
        path = tmp_path / "test_data.parquet"
        raw_df_with_rejects.to_parquet(path)
        df = load_data(str(path))
        assert len(df) < len(raw_df_with_rejects)
