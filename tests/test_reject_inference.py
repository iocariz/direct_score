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
        """Down-sampling caps the number of *applicants* (post-fix I), not
        the number of rows — each surviving applicant produces up to two
        rows after the deterministic (bad, good) expansion.
        """
        rejected, band_stats, bin_edges = reject_inputs
        n_booked = 10  # very small -> forces aggressive down-sampling
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=n_booked,
        )
        applicant_cap = int(n_booked * REJECT_MAX_RATIO)
        if result.empty:
            return
        # Applicant count (deduped on identity) must respect the cap.
        n_applicants = result.groupby(["authorization_id", "mis_Date"]).ngroups
        assert n_applicants <= applicant_cap
        # Row count is bounded above by 2 × applicants (each emits a pair).
        assert len(result) <= 2 * applicant_cap

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


class TestDeterministicRejectExpansion:
    """Audit fix I: pseudo-labels are deterministic 2x row expansion, not Bernoulli draws.

    Each rejected applicant produces a (bad, good) row pair whose total weight
    equals REJECT_SAMPLE_WEIGHT and whose expected gradient contribution
    matches a Bernoulli(p) draw exactly — but with no Monte Carlo noise.
    """

    @pytest.fixture()
    def reject_inputs(self, raw_df_with_rejects):
        booked = raw_df_with_rejects[raw_df_with_rejects["status_name"] == "Booked"].copy()
        rejected = raw_df_with_rejects[raw_df_with_rejects["status_name"] != "Booked"].copy()
        booked = engineer_features(booked)
        rejected = engineer_features(rejected)
        band_stats, bin_edges = compute_score_band_bad_rates(booked)
        return rejected, band_stats, bin_edges

    def test_each_applicant_yields_at_most_two_rows(self, reject_inputs):
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=200,
        )
        if result.empty:
            pytest.skip("synthetic fixture produced no valid rejects post-filter")
        # The expansion writes two rows per applicant, then drops zero-weight rows.
        # Group by (authorization_id, mis_Date) — should be at most 2 per group.
        per_applicant_counts = result.groupby(["authorization_id", "mis_Date"]).size()
        assert (per_applicant_counts <= 2).all()
        # And each surviving applicant must have at least one row.
        assert (per_applicant_counts >= 1).all()
        # Per-row weight column must be present and positive.
        assert REJECT_ROW_WEIGHT_COL in result.columns
        assert (result[REJECT_ROW_WEIGHT_COL] > 0).all()

    def test_pair_weights_sum_to_reject_sample_weight(self, reject_inputs):
        from training_constants import REJECT_SAMPLE_WEIGHT
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=200,
        )
        if result.empty:
            pytest.skip("synthetic fixture produced no valid rejects post-filter")
        # For applicants whose p is in (0, 1), both halves of the pair survive
        # and the pair weights must sum to REJECT_SAMPLE_WEIGHT.
        pair_weights = result.groupby(["authorization_id", "mis_Date"])[REJECT_ROW_WEIGHT_COL].sum()
        # All surviving pair-sums equal REJECT_SAMPLE_WEIGHT exactly (modulo
        # boundary cases where p = 0 or p = 1, in which only one half survives
        # and the surviving half carries the full REJECT_SAMPLE_WEIGHT).
        assert np.allclose(pair_weights, REJECT_SAMPLE_WEIGHT)

    def test_per_row_weight_proportional_to_pseudo_bad_rate(self, reject_inputs):
        from training_constants import REJECT_SAMPLE_WEIGHT
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        rejected, band_stats, bin_edges = reject_inputs
        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=200,
        )
        if result.empty:
            pytest.skip("synthetic fixture produced no valid rejects post-filter")
        # bad rows: weight ≈ p · REJECT_SAMPLE_WEIGHT
        bad_rows = result.loc[result[TARGET] == 1]
        if not bad_rows.empty:
            assert np.allclose(
                bad_rows[REJECT_ROW_WEIGHT_COL].values,
                bad_rows["pseudo_bad_rate"].values * REJECT_SAMPLE_WEIGHT,
            )
        # good rows: weight ≈ (1 - p) · REJECT_SAMPLE_WEIGHT
        good_rows = result.loc[result[TARGET] == 0]
        if not good_rows.empty:
            assert np.allclose(
                good_rows[REJECT_ROW_WEIGHT_COL].values,
                (1.0 - good_rows["pseudo_bad_rate"].values) * REJECT_SAMPLE_WEIGHT,
            )

    def test_two_runs_produce_identical_output(self, reject_inputs):
        """Determinism: two back-to-back invocations must produce byte-identical frames.

        The previous Bernoulli implementation also passed this test thanks to
        RANDOM_STATE seeding, but flipping the seed by 1 would scramble the
        labels. The deterministic expansion has zero seed sensitivity for the
        actual pseudo-labels — the only RNG dependence is the down-sampling
        step that picks WHICH applicants survive. Hence we test back-to-back
        identity (the strong determinism guarantee).
        """
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        rejected, band_stats, bin_edges = reject_inputs
        a = create_reject_pseudo_labels(rejected, band_stats, bin_edges, n_booked_train=200)
        b = create_reject_pseudo_labels(rejected, band_stats, bin_edges, n_booked_train=200)
        if a.empty:
            pytest.skip("synthetic fixture produced no valid rejects post-filter")
        # Compare the columns we care about (TARGET + weight + bad-rate).
        for col in [TARGET, REJECT_ROW_WEIGHT_COL, "pseudo_bad_rate"]:
            assert np.allclose(a[col].values, b[col].values), f"{col} differs between runs"

    def test_zero_pseudo_bad_rate_drops_bad_row(self):
        """A band with zero observed bad rate emits only the good row."""
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        rejected = pd.DataFrame({
            "authorization_id": [1.0, 2.0],
            "mis_Date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            REJECT_SCORE_COL: [50.0, 60.0],
            "extra": [1, 2],
        })
        band_stats = pd.DataFrame({
            "score_band": [0],
            "n_booked": [100],
            "n_bad": [0],
            "bad_rate": [0.0],  # zero observed bads in this band
        })
        band_stats.attrs["pre_split_only"] = True
        band_stats.attrs["source_max_date"] = pd.Timestamp("2024-05-01")
        band_stats.attrs["source_min_date"] = pd.Timestamp("2023-01-01")
        bin_edges = np.array([-np.inf, np.inf])

        result = create_reject_pseudo_labels(
            rejected, band_stats, bin_edges, n_booked_train=100,
        )
        # Both applicants survive, but only their good half (TARGET=0) — the
        # bad half had weight zero and was dropped.
        assert len(result) == 2
        assert (result[TARGET] == 0).all()
        assert (result[REJECT_ROW_WEIGHT_COL] > 0).all()


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

    def test_uses_per_row_weight_column_when_present(self):
        """augment_training_data must honour the deterministic per-row weight
        produced by create_reject_pseudo_labels rather than the uniform
        REJECT_SAMPLE_WEIGHT default."""
        from training_reject_inference import REJECT_ROW_WEIGHT_COL

        X_train = pd.DataFrame({"f1": [1.0, 2.0]})
        y_train = pd.Series([0, 1], name=TARGET)
        # Build a frame as create_reject_pseudo_labels would: the bad and good
        # halves of one applicant (p = 0.4, REJECT_SAMPLE_WEIGHT = 0.5):
        reject_labeled = pd.DataFrame({
            "f1": [3.0, 3.0],
            TARGET: [1, 0],
            REJECT_ROW_WEIGHT_COL: [0.4 * 0.5, 0.6 * 0.5],
        })

        _, _, w = augment_training_data(X_train, y_train, reject_labeled, ["f1"])
        assert w[0] == 1.0  # booked
        assert w[1] == 1.0  # booked
        assert w[2] == pytest.approx(0.20)  # bad row: 0.4 * 0.5
        assert w[3] == pytest.approx(0.30)  # good row: 0.6 * 0.5
        # Total reject contribution sums to REJECT_SAMPLE_WEIGHT (0.5).
        assert (w[2] + w[3]) == pytest.approx(0.50)
