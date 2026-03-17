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
