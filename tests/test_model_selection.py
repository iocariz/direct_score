"""Tests for overfitting diagnostics and model selection framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_reporting import compute_overfit_report, select_best_model


class TestComputeOverfitReport:
    @pytest.fixture()
    def overfit_data(self):
        rng = np.random.RandomState(42)
        y_train = np.array([0] * 160 + [1] * 40)
        y_test = np.array([0] * 80 + [1] * 20)
        # Well-generalizing model
        good_train = np.concatenate([rng.uniform(0.1, 0.4, 160), rng.uniform(0.5, 0.9, 40)])
        good_test = np.concatenate([rng.uniform(0.1, 0.4, 80), rng.uniform(0.5, 0.9, 20)])
        # Overfit model (perfect on train, worse on test)
        overfit_train = np.concatenate([rng.uniform(0.0, 0.05, 160), rng.uniform(0.95, 1.0, 40)])
        overfit_test = np.concatenate([rng.uniform(0.2, 0.6, 80), rng.uniform(0.4, 0.8, 20)])
        train_scores = {"Good": good_train, "Overfit": overfit_train}
        test_scores = {"Good": good_test, "Overfit": overfit_test}
        return y_train, y_test, train_scores, test_scores

    def test_returns_expected_columns(self, overfit_data):
        y_tr, y_te, tr_scores, te_scores = overfit_data
        df = compute_overfit_report(y_tr, y_te, tr_scores, te_scores)
        expected = {
            "model", "train_auc", "test_auc", "auc_delta",
            "train_pr_auc", "test_pr_auc", "pr_auc_delta",
            "train_brier", "test_brier", "brier_delta",
            "train_ks", "test_ks", "ks_delta",
            "train_n", "test_n", "overfit_flag",
        }
        assert expected.issubset(df.columns)

    def test_flags_overfit_model(self, overfit_data):
        y_tr, y_te, tr_scores, te_scores = overfit_data
        df = compute_overfit_report(y_tr, y_te, tr_scores, te_scores)
        overfit_row = df.loc[df["model"] == "Overfit"].iloc[0]
        assert overfit_row["overfit_flag"] == "YES"
        assert overfit_row["auc_delta"] > 0.03

    def test_good_model_not_flagged(self, overfit_data):
        y_tr, y_te, tr_scores, te_scores = overfit_data
        df = compute_overfit_report(y_tr, y_te, tr_scores, te_scores)
        good_row = df.loc[df["model"] == "Good"].iloc[0]
        assert good_row["auc_delta"] < 0.03

    def test_train_n_matches_input(self, overfit_data):
        y_tr, y_te, tr_scores, te_scores = overfit_data
        df = compute_overfit_report(y_tr, y_te, tr_scores, te_scores)
        assert (df["train_n"] == len(y_tr)).all()
        assert (df["test_n"] == len(y_te)).all()

    def test_respects_model_names_filter(self, overfit_data):
        y_tr, y_te, tr_scores, te_scores = overfit_data
        df = compute_overfit_report(y_tr, y_te, tr_scores, te_scores, model_names=["Good"])
        assert len(df) == 1
        assert df.iloc[0]["model"] == "Good"


class TestSelectBestModel:
    @pytest.fixture()
    def selection_data(self):
        results_df = pd.DataFrame(
            {
                "ROC AUC": [0.82, 0.85, 0.80, 0.83],
                "PR AUC": [0.15, 0.18, 0.13, 0.16],
                "KS": [0.50, 0.55, 0.45, 0.52],
                "Brier": [0.06, 0.055, 0.065, 0.058],
                "N": [1000, 1000, 1000, 1000],
            },
            index=["Logistic Regression", "LightGBM", "XGBoost", "CatBoost"],
        )
        overfit_df = pd.DataFrame({
            "model": ["Logistic Regression", "LightGBM", "XGBoost", "CatBoost"],
            "auc_delta": [0.005, 0.02, 0.04, 0.015],
        })
        rolling_oot = pd.DataFrame({
            "Model": ["Logistic Regression", "LightGBM", "XGBoost", "CatBoost"],
            "mean_PR_AUC": [0.14, 0.17, 0.12, 0.155],
            "n_folds": [4, 4, 4, 4],
        })
        return results_df, overfit_df, rolling_oot

    def test_returns_all_candidates(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        assert len(df) == 4

    def test_exactly_one_recommended(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        assert df["recommended"].sum() == 1

    def test_recommended_has_highest_score(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        assert df.iloc[0]["recommended"] == True
        assert df.iloc[0]["weighted_score"] >= df.iloc[-1]["weighted_score"]

    def test_penalizes_overfit_model(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        xgb = df.loc[df["model"] == "XGBoost"].iloc[0]
        # XGBoost has auc_delta=0.04 > 0.03, should have reduced generalization
        assert xgb["generalization_score"] < 100.0

    def test_generalization_penalty_uses_pr_auc_delta_when_larger(self):
        """A model with low AUC delta but high PR-AUC delta must still be penalized.

        Pre-item-D, only auc_delta drove the generalization penalty, so a
        model overfitting on PR AUC (which is what matters for an imbalanced
        target) would silently score a perfect 100. Post-fix, the penalty
        ramps based on max(auc_delta, pr_auc_delta).
        """
        results_df = pd.DataFrame(
            {
                "ROC AUC": [0.85, 0.85],
                "PR AUC": [0.18, 0.18],
                "KS": [0.55, 0.55],
                "Brier": [0.055, 0.055],
                "N": [1000, 1000],
            },
            index=["Stable", "Sneaky"],
        )
        # Both models have negligible AUC delta. Only Sneaky overfits on PR AUC.
        overfit_df = pd.DataFrame({
            "model": ["Stable", "Sneaky"],
            "auc_delta": [0.005, 0.005],
            "pr_auc_delta": [0.005, 0.080],  # Sneaky's PR AUC delta is well above 0.01
        })
        rolling_oot = pd.DataFrame({
            "Model": ["Stable", "Sneaky"],
            "mean_PR_AUC": [0.17, 0.17],
            "n_folds": [4, 4],
        })

        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )

        stable = df.loc[df["model"] == "Stable"].iloc[0]
        sneaky = df.loc[df["model"] == "Sneaky"].iloc[0]
        assert stable["generalization_score"] == pytest.approx(100.0)
        # Sneaky's pr_auc_delta=0.08 → penalty = (0.08-0.01)/0.09*100 ≈ 77.8
        # → generalization ≈ 22.2, which is well below 100.
        assert sneaky["generalization_score"] < 50.0

    def test_generalization_falls_back_when_pr_auc_delta_missing(self):
        """Old artifacts without pr_auc_delta column must still produce a sensible score."""
        results_df = pd.DataFrame(
            {"ROC AUC": [0.85], "PR AUC": [0.18], "KS": [0.55], "Brier": [0.055], "N": [1000]},
            index=["Solo"],
        )
        # Note: no pr_auc_delta column. The selector must fall back to AUC delta only.
        overfit_df = pd.DataFrame({"model": ["Solo"], "auc_delta": [0.005]})

        df = select_best_model(
            results_df, overfit_df=overfit_df, candidate_names=list(results_df.index),
        )
        assert df.loc[df["model"] == "Solo", "generalization_score"].iloc[0] == pytest.approx(100.0)

    # ── Audit fix K: weighted_score quality floor ────────────────────────────
    # An absolute floor prevents recommending a 'least bad' model when every
    # candidate is genuinely poor.

    def test_no_recommendation_when_all_candidates_below_floor(self):
        """All weak candidates → no recommended model."""
        # Construct a 2-candidate scenario where both are weak. With no
        # benchmark_comparisons_df the lift_score defaults to 50 and with no
        # rolling_oot_summary_df stability defaults to 50 — the only way to
        # push the winner below 50 is to make discrimination = 0 for both.
        # Identical PR AUC means min-max scaling sets discrimination = 50 for
        # both, giving weighted ~ 50. Tweak via overfit to push under floor.
        results_df = pd.DataFrame(
            {"ROC AUC": [0.55, 0.55], "PR AUC": [0.06, 0.06],
             "KS": [0.10, 0.10], "Brier": [0.20, 0.20], "N": [1000, 1000]},
            index=["A", "B"],
        )
        # Both heavily overfit (delta well above the 0.10 ramp ceiling), so
        # generalization = 0 for both — pulling the weighted score below 50.
        overfit_df = pd.DataFrame({
            "model": ["A", "B"],
            "auc_delta": [0.20, 0.20],
            "pr_auc_delta": [0.20, 0.20],
        })

        df = select_best_model(
            results_df, overfit_df=overfit_df, candidate_names=["A", "B"],
        )

        # Every candidate must be flagged as below the floor.
        assert (~df["meets_quality_floor"]).all()
        # And no model should be recommended.
        assert df["recommended"].sum() == 0

    def test_recommendation_made_when_floor_is_cleared(self):
        """When the winner clears the floor, the floor doesn't change behaviour."""
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82, 0.85], "PR AUC": [0.15, 0.20],
             "KS": [0.50, 0.55], "Brier": [0.06, 0.05], "N": [1000, 1000]},
            index=["LR", "LightGBM"],
        )
        df = select_best_model(results_df, candidate_names=["LR", "LightGBM"])
        # The winner must clear the floor. (LightGBM is best on every dimension
        # so its weighted_score should be high.)
        recommended = df.loc[df["recommended"]].iloc[0]
        assert recommended["weighted_score"] >= 50.0
        assert bool(recommended["meets_quality_floor"]) is True
        assert df["recommended"].sum() == 1

    def test_quality_floor_parameter_is_overridable(self):
        """A caller can pass a custom quality_floor (e.g. for a less mature portfolio)."""
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82], "PR AUC": [0.15], "KS": [0.50], "Brier": [0.06], "N": [1000]},
            index=["Solo"],
        )
        # With one candidate, every per-dimension min-max scaling falls back
        # to the 50 sentinel, so the weighted_score is exactly 50.
        df = select_best_model(
            results_df, candidate_names=["Solo"], quality_floor=80.0,
        )
        # 50 < 80 → no recommendation under the higher floor.
        assert df["recommended"].sum() == 0
        assert (~df["meets_quality_floor"]).all()

    def test_meets_quality_floor_column_always_present(self):
        """Column must exist for every candidate row, even on the happy path."""
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82, 0.85], "PR AUC": [0.15, 0.18],
             "KS": [0.50, 0.55], "Brier": [0.06, 0.055], "N": [1000, 1000]},
            index=["LR", "LightGBM"],
        )
        df = select_best_model(results_df, candidate_names=["LR", "LightGBM"])
        assert "meets_quality_floor" in df.columns
        assert df["meets_quality_floor"].dtype == bool

    def test_scores_in_valid_range(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        for col in ["discrimination_score", "calibration_score", "stability_score",
                     "generalization_score", "lift_score", "weighted_score"]:
            assert (df[col] >= 0).all() and (df[col] <= 100).all()

    def test_works_without_optional_inputs(self):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.80, 0.85], "PR AUC": [0.12, 0.15],
             "KS": [0.45, 0.50], "Brier": [0.07, 0.06], "N": [100, 100]},
            index=["Logistic Regression", "LightGBM"],
        )
        df = select_best_model(results_df, candidate_names=["Logistic Regression", "LightGBM"])
        assert len(df) == 2
        assert df["recommended"].sum() == 1

    def test_lightgbm_best_when_best_on_all_criteria(self, selection_data):
        results_df, overfit_df, rolling_oot = selection_data
        df = select_best_model(
            results_df, overfit_df=overfit_df, rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
        )
        # LightGBM has best PR AUC, best Brier, best OOT, low overfit → should win
        recommended = df.loc[df["recommended"]].iloc[0]
        assert recommended["model"] == "LightGBM"

    def test_prefers_stronger_benchmark_evidence_within_selection_band(self):
        results_df = pd.DataFrame(
            {
                "ROC AUC": [0.85, 0.85, 0.80],
                "PR AUC": [0.18, 0.18, 0.13],
                "KS": [0.55, 0.55, 0.45],
                "Brier": [0.055, 0.055, 0.065],
                "N": [1000, 1000, 1000],
            },
            index=["Logistic Regression", "LightGBM", "XGBoost"],
        )
        overfit_df = pd.DataFrame({
            "model": ["Logistic Regression", "LightGBM", "XGBoost"],
            "auc_delta": [0.015, 0.015, 0.040],
        })
        rolling_oot = pd.DataFrame({
            "Model": ["Logistic Regression", "LightGBM", "XGBoost"],
            "mean_PR_AUC": [0.17, 0.17, 0.12],
            "n_folds": [4, 4, 4],
        })
        benchmark_comparisons_df = pd.DataFrame({
            "candidate_model": ["Logistic Regression", "LightGBM", "XGBoost"],
            "reference_model": ["risk_score_rf (benchmark)"] * 3,
            "auc_improvement": [0.020, 0.020, 0.005],
            "auc_improvement_lo": [-0.005, 0.005, -0.010],
            "auc_p_adjusted": [0.200, 0.040, 0.900],
        })

        df = select_best_model(
            results_df,
            overfit_df=overfit_df,
            rolling_oot_summary_df=rolling_oot,
            candidate_names=list(results_df.index),
            benchmark_comparisons_df=benchmark_comparisons_df,
        )

        recommended = df.loc[df["recommended"]].iloc[0]
        assert recommended["model"] == "LightGBM"
        assert bool(df.loc[df["model"] == "Logistic Regression", "selection_band"].iloc[0]) is True
        assert bool(df.loc[df["model"] == "LightGBM", "selection_band"].iloc[0]) is True
