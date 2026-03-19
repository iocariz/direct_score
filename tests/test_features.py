"""Tests for feature engineering, selection, and interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training_features as training_features_module
from training_constants import (
    DROP_COLS,
    MISS_CANDIDATES,
    RAW_CAT,
    RAW_NUM,
    TARGET,
)
from training_features import (
    _safe_auc,
    _temporal_numeric_auc,
    add_interactions,
    build_feature_provenance,
    engineer_features,
    reduce_cardinality,
    search_interactions,
    select_features,
)


class TestEngineerFeatures:
    def test_adds_expected_columns(self, booked_df):
        result = engineer_features(booked_df)
        expected = [
            "HAS_CODEBTOR", "HOUSEHOLD_INCOME", "CODEBTOR_INCOME_SHARE",
            "TOTAL_PRODUCTS", "INSTALLMENT_TO_INCOME", "TOTAL_AMT_TO_INCOME",
            "INSTALLMENT_TO_HOUSEHOLD", "TOTAL_AMT_TO_HOUSEHOLD", "AMT_PER_MONTH",
            "BOOK_RATIO_LOAN", "BOOK_RATIO_CARD", "LOG_INCOME_T1",
            "PRODTYPE3_X_HOUSE", "PRODTYPE3_X_CUSTTYPE", "CUSTTYPE_X_HOUSE",
        ]
        for col in expected:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_missing_flags_created(self, booked_df):
        result = engineer_features(booked_df)
        for col in MISS_CANDIDATES:
            miss_rate = booked_df[col].isna().mean()
            if miss_rate > 0.01:
                assert f"MISS_{col}" in result.columns

    def test_does_not_drop_rows(self, booked_df):
        result = engineer_features(booked_df)
        assert len(result) == len(booked_df)


class TestSelectFeatures:
    def test_scrplust1_excluded(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        assert "SCRPLUST1" not in feature_cols

    def test_target_excluded(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        assert TARGET not in feature_cols

    def test_drop_cols_excluded(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        for col in DROP_COLS:
            if col in engineered_df.columns:
                assert col not in feature_cols, f"{col} should be excluded"

    def test_product_type_1_excluded(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        assert "product_type_1" not in feature_cols

    def test_acct_booked_h0_excluded(self, engineered_df):
        feature_cols, _, _ = select_features(engineered_df)
        assert "acct_booked_H0" not in feature_cols

    def test_num_cat_partition(self, engineered_df):
        feature_cols, num_cols, cat_cols = select_features(engineered_df)
        assert set(num_cols) | set(cat_cols) == set(feature_cols)
        assert set(num_cols) & set(cat_cols) == set()


class TestReduceCardinality:
    def test_caps_categories(self):
        n = 200
        rng = np.random.RandomState(0)
        # Create a column with 30 distinct categories
        levels = [f"cat_{i}" for i in range(30)]
        X_train = pd.DataFrame({"high_card": rng.choice(levels, n)})
        X_test = pd.DataFrame({"high_card": rng.choice(levels, 50)})

        X_train_out, X_test_out, maps = reduce_cardinality(X_train, X_test, ["high_card"])
        assert X_train_out["high_card"].nunique() <= 21  # 20 + "Other"
        assert "Other" in X_train_out["high_card"].values

    def test_preserves_low_cardinality(self):
        X_train = pd.DataFrame({"low_card": ["A", "B", "C"] * 10})
        X_test = pd.DataFrame({"low_card": ["A", "B"] * 3})

        X_train_out, X_test_out, _ = reduce_cardinality(X_train, X_test, ["low_card"])
        assert "Other" not in X_train_out["low_card"].values

    def test_test_unseen_categories_become_other(self):
        X_train = pd.DataFrame({"col": ["A", "B"] * 50})
        X_test = pd.DataFrame({"col": ["A", "B", "UNSEEN"]})

        X_train_out, X_test_out, _ = reduce_cardinality(X_train, X_test, ["col"])
        assert "Other" in X_test_out["col"].values


class TestAddInteractions:
    def test_adds_ratio(self):
        df = pd.DataFrame({"A": [10.0, 20.0], "B": [2.0, 4.0]})
        interactions = pd.DataFrame([{
            "name": "A/B", "type": "ratio", "feat_a": "A", "feat_b": "B",
            "auc": 0.6, "lift": 0.05,
        }])
        result = add_interactions(df, interactions)
        assert "A_DIV_B" in result.columns
        assert result["A_DIV_B"].iloc[0] == pytest.approx(5.0)

    def test_adds_product(self):
        df = pd.DataFrame({"A": [3.0, 4.0], "B": [2.0, 5.0]})
        interactions = pd.DataFrame([{
            "name": "A*B", "type": "product", "feat_a": "A", "feat_b": "B",
            "auc": 0.6, "lift": 0.05,
        }])
        result = add_interactions(df, interactions)
        assert "A_X_B" in result.columns
        assert result["A_X_B"].iloc[0] == pytest.approx(6.0)

    def test_adds_cat_concat(self):
        df = pd.DataFrame({"A": ["x", "y"], "B": ["1", "2"]})
        interactions = pd.DataFrame([{
            "name": "A_x_B", "type": "cat_concat", "feat_a": "A", "feat_b": "B",
            "auc": 0.6, "lift": 0.05,
        }])
        result = add_interactions(df, interactions)
        assert "A_x_B" in result.columns
        assert result["A_x_B"].iloc[0] == "x_1"

    def test_skips_existing_column(self):
        df = pd.DataFrame({"A": [1.0], "B": [2.0], "A_DIV_B": [99.0]})
        interactions = pd.DataFrame([{
            "name": "A/B", "type": "ratio", "feat_a": "A", "feat_b": "B",
            "auc": 0.6, "lift": 0.05,
        }])
        result = add_interactions(df, interactions)
        assert result["A_DIV_B"].iloc[0] == 99.0  # unchanged


class TestTemporalInteractionScoring:
    def test_temporal_numeric_auc_penalizes_late_regime_shift(self, monkeypatch):
        monkeypatch.setattr(training_features_module, "MIN_VALID", 2)

        y = np.array(
            [0] * 8 + [1] * 8
            + [0] * 8 + [1] * 8
            + [0] * 5 + [1] * 5
            + [0] * 5 + [1] * 5
        )
        scores = np.array(
            [0.10] * 8 + [0.90] * 8
            + [0.10] * 8 + [0.90] * 8
            + [0.70] * 5 + [0.30] * 5
            + [0.70] * 5 + [0.30] * 5
        )
        dates = pd.to_datetime(
            ["2024-01-01"] * 16
            + ["2024-02-01"] * 16
            + ["2024-03-01"] * 10
            + ["2024-04-01"] * 10
        )

        pooled_auc, pooled_n = _safe_auc(y, scores)
        temporal_auc, temporal_n = _temporal_numeric_auc(scores, y, dates)

        assert pooled_n == 52
        assert temporal_n == 20
        assert pooled_auc > temporal_auc
        assert pooled_auc > 0.70
        assert temporal_auc < 0.50


class TestSearchInteractions:
    def test_return_diagnostics_uses_temporal_scoring_and_reports_gating(self, monkeypatch):
        monkeypatch.setattr(training_features_module, "RAW_NUM", ["num_a", "num_b", "num_c"])
        monkeypatch.setattr(training_features_module, "RAW_CAT", ["cat_a", "cat_b", "cat_c"])
        monkeypatch.setattr(training_features_module, "INTERACTION_SEARCH_TOP_K_NUM", 2)
        monkeypatch.setattr(training_features_module, "INTERACTION_SEARCH_TOP_K_CAT", 2)
        monkeypatch.setattr(training_features_module, "MIN_VALID", 2)
        monkeypatch.setattr(training_features_module, "MIN_LIFT", 0.0)

        dates = pd.to_datetime(
            ["2024-01-01"] * 20
            + ["2024-02-01"] * 20
            + ["2024-03-01"] * 20
            + ["2024-04-01"] * 20
        )
        y = np.array(([0] * 10 + [1] * 10) * 4)
        late_mask = np.repeat([False, False, True, True], 20)

        df = pd.DataFrame({
            "mis_Date": dates,
            TARGET: y,
            "num_a": np.where(late_mask, np.where(y == 1, 1.0, 9.0), np.where(y == 1, 9.0, 1.0)),
            "num_b": np.where(y == 1, 2.0, 5.0),
            "num_c": np.tile(np.linspace(1.0, 2.0, 20), 4),
            "cat_a": np.where(late_mask, np.where(y == 1, "bad", "good"), np.where(y == 1, "good", "bad")),
            "cat_b": np.where(np.arange(len(y)) % 2 == 0, "x", "y"),
            "cat_c": np.where(np.arange(len(y)) % 3 == 0, "m", "n"),
        })

        result = search_interactions(df, end_before_date="2024-05-01", return_diagnostics=True)
        summary = result.interaction_search_summary_df.iloc[0]

        assert summary["numeric_scoring_strategy"] == "temporal_validation"
        assert summary["categorical_scoring_strategy"] == "temporal_target_encode"
        assert int(summary["raw_num_features"]) == 3
        assert int(summary["raw_cat_features"]) == 3
        assert int(summary["screened_num_features"]) == 2
        assert int(summary["screened_cat_features"]) == 2
        assert int(summary["screened_num_pairs"]) == 1
        assert int(summary["screened_cat_pairs"]) == 1
        assert set(result.selected_interactions["name"]) == set(
            result.interaction_leaderboard_df.loc[result.interaction_leaderboard_df["selected"], "name"]
        )

    def test_return_diagnostics_falls_back_without_enough_date_blocks(self, monkeypatch):
        monkeypatch.setattr(training_features_module, "RAW_NUM", ["num_a", "num_b"])
        monkeypatch.setattr(training_features_module, "RAW_CAT", ["cat_a", "cat_b"])
        monkeypatch.setattr(training_features_module, "INTERACTION_SEARCH_TOP_K_NUM", 2)
        monkeypatch.setattr(training_features_module, "INTERACTION_SEARCH_TOP_K_CAT", 2)
        monkeypatch.setattr(training_features_module, "MIN_VALID", 2)
        monkeypatch.setattr(training_features_module, "MIN_LIFT", 0.0)

        dates = pd.to_datetime(["2024-01-01"] * 20 + ["2024-02-01"] * 20)
        y = np.array(([0] * 10 + [1] * 10) * 2)

        df = pd.DataFrame({
            "mis_Date": dates,
            TARGET: y,
            "num_a": np.where(y == 1, 9.0, 1.0),
            "num_b": np.where(y == 1, 2.0, 5.0),
            "cat_a": np.where(y == 1, "good", "bad"),
            "cat_b": np.where(np.arange(len(y)) % 2 == 0, "x", "y"),
        })

        result = search_interactions(df, end_before_date="2024-03-01", return_diagnostics=True)
        summary = result.interaction_search_summary_df.iloc[0]

        assert summary["numeric_scoring_strategy"] == "fallback_pooled_auc"
        assert summary["categorical_scoring_strategy"] == "fallback_cv_target_encode"
        assert int(summary["screened_num_pairs"]) == 1
        assert int(summary["screened_cat_pairs"]) == 1
        assert "selected" in result.interaction_leaderboard_df.columns


class TestBuildFeatureProvenance:
    def test_tracks_provenance_and_rfecv_flags(self):
        interactions = pd.DataFrame([{
            "name": "RAW_A/RAW_B",
            "type": "ratio",
            "feat_a": "RAW_A",
            "feat_b": "RAW_B",
            "auc": 0.61,
            "lift": 0.03,
        }])

        provenance_df = build_feature_provenance(
            raw_feature_cols=["RAW_A", "RAW_B"],
            engineered_feature_cols=["ENG_C"],
            interactions=interactions,
            freq_cols=["FREQ_CAT_A"],
            group_cols=["RAW_A_VS_CAT_A"],
            feature_space_num_cols=["RAW_A", "RAW_B", "ENG_C", "RAW_A_DIV_RAW_B", "FREQ_CAT_A", "RAW_A_VS_CAT_A"],
            feature_space_cat_cols=["CAT_A"],
            rfecv_candidate_cols=["RAW_A", "ENG_C", "RAW_A_DIV_RAW_B", "FREQ_CAT_A"],
            rfecv_kept_cols=["RAW_A", "RAW_A_DIV_RAW_B"],
        )

        ratio_row = provenance_df.loc[provenance_df["feature"] == "RAW_A_DIV_RAW_B"].iloc[0]
        engineered_row = provenance_df.loc[provenance_df["feature"] == "ENG_C"].iloc[0]
        frequency_row = provenance_df.loc[provenance_df["feature"] == "FREQ_CAT_A"].iloc[0]

        assert ratio_row["provenance"] == "interaction"
        assert ratio_row["interaction_type"] == "ratio"
        assert bool(ratio_row["rfecv_kept"]) is True
        assert engineered_row["provenance"] == "engineered"
        assert bool(engineered_row["rfecv_candidate"]) is True
        assert frequency_row["provenance"] == "frequency"
        assert bool(frequency_row["rfecv_kept"]) is False
