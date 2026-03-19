"""Tests for reject inference (score-band parceling)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import (
    augment_training_data,
    compute_score_band_bad_rates,
    create_reject_pseudo_labels,
    load_data_with_rejects,
)
from training_constants import (
    REJECT_MAX_RATIO,
    REJECT_MULTIPLIER,
    REJECT_SAMPLE_WEIGHT,
    REJECT_SCORE_COL,
    SPLIT_DATE,
    TARGET,
)
from training_features import engineer_features


class TestLoadDataWithRejects:
    def test_returns_both_dataframes(self, raw_df_with_rejects, tmp_path):
        path = tmp_path / "data.parquet"
        raw_df_with_rejects.to_parquet(path)
        booked, rejected = load_data_with_rejects(str(path))
        assert (booked["status_name"] == "Booked").all()
        assert rejected["status_name"].isin(["Rejected", "Canceled"]).all()

    def test_no_overlap(self, raw_df_with_rejects, tmp_path):
        path = tmp_path / "data.parquet"
        raw_df_with_rejects.to_parquet(path)
        booked, rejected = load_data_with_rejects(str(path))
        assert len(booked) + len(rejected) == len(raw_df_with_rejects)


class TestComputeScoreBandBadRates:
    def test_returns_correct_bins(self, booked_df):
        band_stats, bin_edges = compute_score_band_bad_rates(booked_df)
        assert len(bin_edges) == 11  # 10 bins + 1
        assert bin_edges[0] == -np.inf
        assert bin_edges[-1] == np.inf

    def test_bad_rates_between_0_and_1(self, booked_df):
        band_stats, _ = compute_score_band_bad_rates(booked_df)
        assert (band_stats["bad_rate"] >= 0).all()
        assert (band_stats["bad_rate"] <= 1).all()

    def test_counts_match(self, booked_df):
        band_stats, _ = compute_score_band_bad_rates(booked_df)
        assert (band_stats["n_bad"] <= band_stats["n_booked"]).all()

    def test_uses_only_pre_split_booked_rows(self, booked_df):
        band_stats, _ = compute_score_band_bad_rates(booked_df)
        expected_n = booked_df[
            (booked_df["mis_Date"] < SPLIT_DATE)
            & booked_df[TARGET].notna()
            & booked_df[REJECT_SCORE_COL].notna()
        ].shape[0]
        assert int(band_stats["n_booked"].sum()) == expected_n

    def test_post_split_rows_do_not_change_bad_rate_estimates(self):
        pre_split_scores = np.linspace(10, 100, 10)
        post_split_scores = np.linspace(10, 100, 10)
        df = pd.DataFrame({
            "mis_Date": pd.to_datetime(["2024-06-01"] * 10 + ["2024-08-01"] * 10),
            TARGET: [0] * 9 + [1] + [1] * 10,
            REJECT_SCORE_COL: np.concatenate([pre_split_scores, post_split_scores]),
            "status_name": ["Booked"] * 20,
        })

        band_stats, _ = compute_score_band_bad_rates(df)
        observed_bad_rate = band_stats["n_bad"].sum() / band_stats["n_booked"].sum()
        assert observed_bad_rate == pytest.approx(0.10)


class TestCreateRejectPseudoLabels:
    @pytest.fixture()
    def reject_inputs(self, raw_df_with_rejects):
        booked = raw_df_with_rejects[raw_df_with_rejects["status_name"] == "Booked"].copy()
        rejected = raw_df_with_rejects[raw_df_with_rejects["status_name"] != "Booked"].copy()
        booked = engineer_features(booked)
        rejected = engineer_features(rejected)
        band_stats, bin_edges = compute_score_band_bad_rates(booked)
        return rejected, band_stats, bin_edges

    def test_only_training_period(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=100,
        )
        if not result.empty:
            assert (result["mis_Date"] < SPLIT_DATE).all()

    def test_target_is_binary(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=100,
        )
        if not result.empty:
            assert set(result[TARGET].unique()).issubset({0, 1})

    def test_pseudo_bad_rate_capped(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=100,
        )
        if not result.empty:
            assert (result["pseudo_bad_rate"] <= 0.50).all()

    def test_down_sampling(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        n_booked = 10  # very small -> forces aggressive down-sampling
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=n_booked,
        )
        assert len(result) <= int(n_booked * REJECT_MAX_RATIO) or len(result) == 0

    def test_missing_scores_are_excluded(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        rejected = rejected.copy()
        rejected.loc[rejected.index[:3], REJECT_SCORE_COL] = np.nan

        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=100,
        )

        if not result.empty:
            assert result[REJECT_SCORE_COL].notna().all()

    def test_requires_pre_split_band_stats(self, reject_inputs):
        rejected, band_stats, bin_edges = reject_inputs
        band_stats = band_stats.copy()
        band_stats.attrs.clear()

        with pytest.raises(ValueError, match="pre-split booked rows only"):
            create_reject_pseudo_labels(
                rejected, band_stats, bin_edges, n_booked_train=100,
            )


class TestAugmentTrainingData:
    def test_shapes(self):
        X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        y_train = pd.Series([0, 1, 0], name=TARGET)
        reject_labeled = pd.DataFrame({
            "f1": [7.0, 8.0], "f2": [9.0, 10.0],
            TARGET: [1, 0], "_reject_inference_weight": [0.5, 0.5],
        })

        X_aug, y_aug, w = augment_training_data(X_train, y_train, reject_labeled, ["f1", "f2"])
        assert len(X_aug) == 5
        assert len(y_aug) == 5
        assert len(w) == 5

    def test_weights(self):
        X_train = pd.DataFrame({"f1": [1.0, 2.0]})
        y_train = pd.Series([0, 1], name=TARGET)
        reject_labeled = pd.DataFrame({
            "f1": [3.0],
            TARGET: [1],
            "_reject_inference_weight": [REJECT_SAMPLE_WEIGHT],
        })

        _, _, w = augment_training_data(X_train, y_train, reject_labeled, ["f1"])
        assert w[0] == 1.0  # booked
        assert w[1] == 1.0  # booked
        assert w[2] == pytest.approx(REJECT_SAMPLE_WEIGHT)  # reject

    def test_target_values_preserved(self):
        X_train = pd.DataFrame({"f1": [1.0]})
        y_train = pd.Series([0], name=TARGET)
        reject_labeled = pd.DataFrame({
            "f1": [2.0], TARGET: [1],
            "_reject_inference_weight": [0.5],
        })

        _, y_aug, _ = augment_training_data(X_train, y_train, reject_labeled, ["f1"])
        assert y_aug.iloc[0] == 0
        assert y_aug.iloc[1] == 1
