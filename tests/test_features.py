"""Tests for feature engineering, selection, and interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import training_features as training_features_module
from training_constants import (
    DROP_COLS,
    MATURITY_CUTOFF,
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
    enforce_matured_target,
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
            "CODEBTOR_X_INST_TO_INC", "CODEBTOR_X_AMT_TO_INC", "CODEBTOR_X_AMT_PER_MONTH",
            "BOOK_RATIO_LOAN", "BOOK_RATIO_CARD",
            "HAS_CARDS", "HAS_LOANS",
            "LOG_INCOME_T1", "LOG_TOTAL_AMT", "LOG_MAX_CREDIT",
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

    def test_scrplust1_not_in_raw_num_or_raw_cat(self):
        """Closes interaction-search exclusion gap.

        select_features uses DROP_COLS, but the interaction search iterates over
        RAW_NUM and RAW_CAT. If SCRPLUST1 is added to either, it would silently
        enter the interaction grid even though DROP_COLS excludes the raw column.
        """
        assert "SCRPLUST1" not in RAW_NUM
        assert "SCRPLUST1" not in RAW_CAT

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

    def test_binned_num_cat_reuses_stored_bin_edges(self):
        df = pd.DataFrame({
            "A": [1.0, 2.0, 100.0],
            "B": ["x", "x", "x"],
        })
        bin_edges = (-np.inf, 1.5, 2.5, np.inf)
        interactions = pd.DataFrame([{
            "name": "BIN_A_x_B", "type": "binned_num_cat", "feat_a": "A", "feat_b": "B",
            "auc": 0.6, "lift": 0.05, "bin_edges": bin_edges,
        }])

        result = add_interactions(df, interactions)

        expected_bins = pd.cut(df["A"], bins=np.array(bin_edges), include_lowest=True).astype(str)
        expected = expected_bins + "_" + df["B"]
        assert list(result["BIN_A_x_B"]) == list(expected)


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

    def test_records_selector_source_when_split_kept_sets_provided(self):
        """build_feature_provenance must surface which selector kept each feature."""
        interactions = pd.DataFrame(columns=["name", "type", "feat_a", "feat_b"])

        provenance_df = build_feature_provenance(
            raw_feature_cols=["A", "B", "C", "D"],
            engineered_feature_cols=[],
            interactions=interactions,
            freq_cols=[],
            group_cols=[],
            feature_space_num_cols=["A", "B", "C", "D"],
            feature_space_cat_cols=[],
            rfecv_candidate_cols=["A", "B", "C", "D"],
            rfecv_kept_cols=["A", "B", "C"],
            stability_kept_cols=["A", "B"],
            tree_aware_kept_cols=["B", "C"],
        )

        rows = provenance_df.set_index("feature")
        # A — only the LR stability selector kept it.
        assert rows.loc["A", "kept_by"] == "stability_only"
        assert bool(rows.loc["A", "stability_kept"]) is True
        assert bool(rows.loc["A", "tree_aware_kept"]) is False
        # B — both selectors agreed.
        assert rows.loc["B", "kept_by"] == "both"
        # C — only the tree-aware pass kept it; this is exactly the case the
        # methodology audit said the LR-only contract was missing.
        assert rows.loc["C", "kept_by"] == "tree_aware_only"
        # D — neither selector kept it.
        assert rows.loc["D", "kept_by"] == "dropped"


class TestRunTreeAwareFeaturePass:
    def test_drops_pure_noise_features(self):
        """A feature that is pure noise should fall below the importance threshold."""
        from training_features import run_tree_aware_feature_pass
        from training_temporal import make_temporal_cv

        rng = np.random.RandomState(0)
        n = 600
        # signal + noise: y depends only on signal, noise columns are random.
        signal = rng.normal(size=n)
        prob = 1 / (1 + np.exp(-1.8 * signal))
        y = pd.Series((rng.uniform(size=n) < prob).astype(int))
        # Construct three months of dates to allow a 2-fold temporal CV.
        dates = pd.to_datetime(
            ["2024-01-01"] * (n // 3) + ["2024-02-01"] * (n // 3) + ["2024-03-01"] * (n - 2 * (n // 3))
        )
        X = pd.DataFrame({
            "signal": signal,
            "noise_1": rng.normal(size=n),
            "noise_2": rng.normal(size=n),
            "noise_3": rng.normal(size=n),
        })
        cv = make_temporal_cv(dates, max_splits=2)

        kept = run_tree_aware_feature_pass(
            X,
            y,
            num_cols=["signal", "noise_1", "noise_2", "noise_3"],
            cat_cols=[],
            feature_cols=["signal", "noise_1", "noise_2", "noise_3"],
            cv=cv,
        )

        assert "signal" in kept, "signal must be kept by the tree-aware pass"
        # The noise features should mostly be dropped — at minimum the signal
        # is kept and the kept set is a strict subset of the candidates.
        assert len(kept) < 4

    def test_returns_empty_when_no_folds_are_valid(self):
        """If every fold has only one class, the pass returns [] gracefully."""
        from training_features import run_tree_aware_feature_pass
        from training_temporal import make_temporal_cv

        n = 60
        y = pd.Series([0] * n)  # Single-class — no fold can fit.
        dates = pd.to_datetime(["2024-01-01"] * 20 + ["2024-02-01"] * 20 + ["2024-03-01"] * 20)
        X = pd.DataFrame({"x": np.arange(n, dtype=float)})
        cv = make_temporal_cv(dates, max_splits=2)

        result = run_tree_aware_feature_pass(
            X, y, num_cols=["x"], cat_cols=[], feature_cols=["x"], cv=cv,
        )
        assert result == []


class TestEnforceMaturedTarget:
    """Maturity invariant: basel_bad requires 12 months on book.

    Filtering by mis_Date <= MATURITY_CUTOFF and dropping NaN targets used to
    be done with implicit dropna(). enforce_matured_target makes the contract
    explicit and surfaces upstream-data violations loudly.
    """

    def _frame(self, rows):
        return pd.DataFrame(rows)

    def test_keeps_mature_rows_with_observed_target(self):
        cutoff = pd.Timestamp(MATURITY_CUTOFF)
        df = self._frame([
            {"mis_Date": cutoff - pd.Timedelta(days=400), TARGET: 0.0, "x": 1.0},
            {"mis_Date": cutoff - pd.Timedelta(days=200), TARGET: 1.0, "x": 2.0},
        ])
        result = enforce_matured_target(df)
        assert len(result) == 2
        assert list(result["x"]) == [1.0, 2.0]

    def test_drops_immature_rows_with_null_target(self):
        cutoff = pd.Timestamp(MATURITY_CUTOFF)
        df = self._frame([
            {"mis_Date": cutoff - pd.Timedelta(days=200), TARGET: 0.0, "x": 1.0},
            # immature row, target left NaN by upstream — must be silently dropped
            {"mis_Date": cutoff + pd.Timedelta(days=30), TARGET: float("nan"), "x": 2.0},
        ])
        result = enforce_matured_target(df)
        assert len(result) == 1
        assert result.iloc[0]["x"] == 1.0

    def test_drops_pre_cutoff_rows_that_are_still_unobserved(self):
        cutoff = pd.Timestamp(MATURITY_CUTOFF)
        df = self._frame([
            {"mis_Date": cutoff - pd.Timedelta(days=400), TARGET: 0.0, "x": 1.0},
            {"mis_Date": cutoff - pd.Timedelta(days=200), TARGET: float("nan"), "x": 2.0},
        ])
        result = enforce_matured_target(df)
        assert len(result) == 1
        assert result.iloc[0]["x"] == 1.0

    def test_warns_and_drops_when_immature_row_has_observed_target(self):
        """Audit fix F: when upstream populates immature targets (e.g. right-
        censoring zeros), the helper emits a regulator-visible warning and
        the static cutoff filter still drops the offending rows.

        Originally this case raised — but real upstream feeds frequently fill
        0 for 'no default observed yet' on rows that haven't matured, which
        would block every legitimate training run. The warning preserves the
        regulator-visible signal while letting the cutoff handle the safety.
        """
        from loguru import logger as _logger

        cutoff = pd.Timestamp(MATURITY_CUTOFF)
        df = self._frame([
            {"mis_Date": cutoff - pd.Timedelta(days=200), TARGET: 0.0, "x": 1.0},
            # adversarial: immature row with non-null target (e.g. censored 0)
            {"mis_Date": cutoff + pd.Timedelta(days=30), TARGET: 0.0, "x": 2.0},
        ])

        messages: list[str] = []
        handler_id = _logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = enforce_matured_target(df)
        finally:
            _logger.remove(handler_id)

        # The static cutoff filter drops the offending row.
        assert len(result) == 1
        assert result.iloc[0]["x"] == 1.0
        # And a regulator-visible warning fires.
        assert any("Maturity contract drift" in m for m in messages)
        assert any("right-censoring" in m for m in messages)

    def test_returns_a_copy(self):
        cutoff = pd.Timestamp(MATURITY_CUTOFF)
        df = self._frame([
            {"mis_Date": cutoff - pd.Timedelta(days=200), TARGET: 0.0, "x": 1.0},
        ])
        result = enforce_matured_target(df)
        # Mutating the result must not touch the input.
        result.loc[result.index[0], "x"] = 999.0
        assert df.iloc[0]["x"] == 1.0


class TestCheckCalibrationHoldoutSize:
    """Audit fix G: warn when the calibration holdout is too small for stable isotonic.

    Niculescu-Mizil & Caruana (2005) and follow-ups recommend ~100 positives as
    the lower bound for monotonic-fit calibrators on imbalanced targets.
    Below that the helper logs a regulator-visible warning but does NOT fail —
    legitimate small-portfolio runs are allowed to proceed.
    """

    @staticmethod
    def _capture_warnings():
        """Return (messages_list, handler_id) — caller must remove the handler."""
        from loguru import logger as _logger
        messages: list[str] = []
        handler_id = _logger.add(lambda message: messages.append(str(message)), level="WARNING")
        return messages, handler_id

    def test_returns_true_when_above_minimum(self):
        from training_features import check_calibration_holdout_size
        from loguru import logger as _logger

        messages, handler_id = self._capture_warnings()
        try:
            y = pd.Series([0] * 1000 + [1] * 150)
            assert check_calibration_holdout_size(y) is True
        finally:
            _logger.remove(handler_id)

        # No warning should have been emitted.
        assert not any("Calibration holdout has only" in m for m in messages)

    def test_returns_false_and_warns_when_below_minimum(self):
        from training_features import check_calibration_holdout_size
        from loguru import logger as _logger

        messages, handler_id = self._capture_warnings()
        try:
            y = pd.Series([0] * 1000 + [1] * 50)  # 50 < 100
            assert check_calibration_holdout_size(y) is False
        finally:
            _logger.remove(handler_id)

        # Warning text must mention both the actual count and the threshold.
        assert any("Calibration holdout has only 50 positives" in m for m in messages)
        assert any("100 recommended" in m for m in messages)

    def test_threshold_is_overridable(self):
        from training_features import check_calibration_holdout_size
        from loguru import logger as _logger

        messages, handler_id = self._capture_warnings()
        try:
            y = pd.Series([0] * 100 + [1] * 30)
            # 30 positives is below the default 100 but at-or-above an
            # explicit 30, so no warning should fire.
            assert check_calibration_holdout_size(y, minimum_positives=30) is True
        finally:
            _logger.remove(handler_id)
        assert not any("Calibration holdout has only" in m for m in messages)
