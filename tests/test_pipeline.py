"""Integration tests for the training pipeline.

These tests use synthetic data and minimal Optuna trials to verify
that the full pipeline runs end-to-end without errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import training as training_module
import numpy as np
import pandas as pd
import pytest

from training import (
    build_applicant_score_frame,
    build_population_summary_df,
    run_rolling_out_of_time_validation,
    temporal_split,
    train_catboost,
    train_logistic_regression,
    train_lgbm,
    train_stacking,
    train_xgboost,
)
from training_constants import (
    BENCHMARK_MODEL_NAMES,
    EXPERIMENTAL_STACKING_NAME,
    POPULATION_MODE_UNDERWRITING,
    RANDOM_STATE,
    SPLIT_DATE,
    TARGET,
)
from training_features import (
    build_preprocessors,
    engineer_features,
    reduce_cardinality,
    select_features,
)
from training_reporting import (
    build_holdout_score_frame,
    evaluate,
    evaluate_all,
    extract_feature_importance,
    plot_score_distributions,
    save_artifacts,
)
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


class _DummyStudy:
    def __init__(self, best_params: dict | None = None):
        self.best_value = 0.10
        self.best_trial = SimpleNamespace(number=0)
        self.best_params = {"C": 1.0} if best_params is None else best_params


class _IdentityCalibrator:
    def __init__(self, estimator, method="sigmoid"):
        self.estimator = estimator
        self.method = method

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


def _fit_quick_pipeline(X_train, y_train, preprocessor, sample_weight=None) -> Pipeline:
    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def _patch_fast_main_dependencies(monkeypatch, structured_raw_df_with_rejects):
    booked_df = structured_raw_df_with_rejects.loc[
        structured_raw_df_with_rejects["status_name"] == "Booked"
    ].copy()
    rejected_df = structured_raw_df_with_rejects.loc[
        structured_raw_df_with_rejects["status_name"].isin(["Rejected", "Canceled"])
    ].copy()

    monkeypatch.setattr(
        training_module,
        "load_data",
        lambda _data_path: booked_df.copy(),
    )
    monkeypatch.setattr(
        training_module,
        "load_data_with_rejects",
        lambda _data_path: (booked_df.copy(), rejected_df.copy()),
    )
    monkeypatch.setattr(
        training_module,
        "search_interactions",
        lambda *args, **kwargs: pd.DataFrame(columns=["name", "type", "feat_a", "feat_b", "auc", "lift"]),
    )
    monkeypatch.setattr(
        training_module,
        "run_rfecv",
        lambda X_train, y_train, num_cols, cat_cols, feature_cols, cv=None: (feature_cols, num_cols, cat_cols),
    )
    monkeypatch.setattr(training_module, "CalibratedClassifierCV", _IdentityCalibrator)

    monkeypatch.setattr(
        training_module,
        "train_logistic_regression",
        lambda X_train, y_train, preprocessor, cv, n_trials, sample_weight=None, num_cols=None, cat_cols=None: (
            _fit_quick_pipeline(X_train, y_train, preprocessor, sample_weight=sample_weight),
            _DummyStudy({"C": 1.0, "smooth": 10.0}),
        ),
    )
    monkeypatch.setattr(
        training_module,
        "train_lgbm",
        lambda X_train, y_train, lgbm_preprocessor, lgbm_cat_indices, pos_weight, cv, n_trials, sample_weight=None, monotone_constraints=None: (
            _fit_quick_pipeline(X_train, y_train, lgbm_preprocessor, sample_weight=sample_weight),
            _DummyStudy({"num_leaves": 31}),
            10,
        ),
    )
    monkeypatch.setattr(
        training_module,
        "train_xgboost",
        lambda X_train, y_train, preprocessor, pos_weight, cv, n_trials, sample_weight=None, monotone_constraints=None: (
            _fit_quick_pipeline(X_train, y_train, preprocessor, sample_weight=sample_weight),
            _DummyStudy({"max_depth": 4}),
            10,
        ),
    )
    monkeypatch.setattr(
        training_module,
        "train_catboost",
        lambda X_train, y_train, lgbm_preprocessor, pos_weight, cv, n_trials, sample_weight=None, monotone_constraints=None: (
            _fit_quick_pipeline(X_train, y_train, lgbm_preprocessor, sample_weight=sample_weight),
            _DummyStudy({"depth": 4}),
            10,
        ),
    )
    monkeypatch.setattr(
        training_module,
        "train_stacking",
        lambda X_train, y_train, base_models, cv, sample_weight=None: next(iter(base_models.values())),
    )

    def _fake_rolling_oot(*args, **kwargs):
        base_models = args[-1]
        results_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "Model": model_name,
                    "train_start": "2024-01-01",
                    "train_end": "2024-04-01",
                    "calibration_start": "2024-05-01",
                    "calibration_end": "2024-05-01",
                    "validation_start": "2024-06-01",
                    "validation_end": "2024-07-01",
                    "n_fit": 20,
                    "n_calibration": 4,
                    "n_validation": 8,
                    "n_validation_pos": 2,
                    "ROC AUC": 0.61,
                    "Gini": 0.22,
                    "KS": 0.11,
                    "PR AUC": 0.10,
                    "N": 8,
                    "Brier": 0.19,
                    "is_calibrated": False,
                }
                for model_name in base_models
            ]
        )
        summary_df = pd.DataFrame(
            [
                {
                    "Model": model_name,
                    "n_folds": 1,
                    "mean_ROC_AUC": 0.61,
                    "std_ROC_AUC": 0.0,
                    "mean_PR_AUC": 0.10,
                    "std_PR_AUC": 0.0,
                    "mean_Brier": 0.19,
                    "std_Brier": 0.0,
                    "validation_start_min": "2024-06-01",
                    "validation_end_max": "2024-07-01",
                }
                for model_name in list(base_models) + BENCHMARK_MODEL_NAMES
            ]
        )
        return results_df, summary_df

    monkeypatch.setattr(training_module, "run_rolling_out_of_time_validation", _fake_rolling_oot)

    def _fake_bootstrap_ci(y_true, score_arrays, *args, **kwargs):
        records = []
        for model_name in score_arrays:
            is_benchmark = model_name in BENCHMARK_MODEL_NAMES
            records.append(
                {
                    "Model": model_name,
                    "AUC": 0.60,
                    "AUC_lo": 0.55,
                    "AUC_hi": 0.65,
                    "PR_AUC": 0.09,
                    "PR_AUC_lo": 0.07,
                    "PR_AUC_hi": 0.11,
                    "Brier": np.nan if is_benchmark else 0.20,
                    "Brier_lo": np.nan if is_benchmark else 0.18,
                    "Brier_hi": np.nan if is_benchmark else 0.22,
                }
            )
        return pd.DataFrame(records).set_index("Model")

    monkeypatch.setattr(training_module, "bootstrap_confidence_intervals", _fake_bootstrap_ci)

    def _fake_benchmark_comparisons(y_true, score_arrays, candidate_model_names, reference_model_names=None, *args, **kwargs):
        if not candidate_model_names:
            return pd.DataFrame()
        references = reference_model_names or BENCHMARK_MODEL_NAMES
        records = []
        for candidate_name in candidate_model_names:
            for reference_name in references:
                records.append(
                    {
                        "candidate_model": candidate_name,
                        "reference_model": reference_name,
                        "n": len(y_true),
                        "n_pos": int(np.sum(y_true)),
                        "n_neg": int(len(y_true) - np.sum(y_true)),
                        "candidate_auc": 0.61,
                        "reference_auc": 0.58,
                        "auc_improvement": 0.03,
                        "auc_improvement_lo": 0.01,
                        "auc_improvement_hi": 0.05,
                        "auc_p_value": 0.03,
                        "auc_delong_se": 0.01,
                        "auc_delong_z": 2.0,
                        "auc_delong_p_value": 0.04,
                        "candidate_pr_auc": 0.10,
                        "reference_pr_auc": 0.08,
                        "pr_auc_improvement": 0.02,
                        "pr_auc_improvement_lo": 0.00,
                        "pr_auc_improvement_hi": 0.03,
                        "pr_auc_p_value": 0.05,
                        "candidate_brier": 0.20,
                        "reference_brier": np.nan,
                        "brier_improvement": np.nan,
                        "brier_improvement_lo": np.nan,
                        "brier_improvement_hi": np.nan,
                        "brier_p_value": np.nan,
                    }
                )
        return pd.DataFrame(records)

    monkeypatch.setattr(training_module, "paired_bootstrap_benchmark_comparisons", _fake_benchmark_comparisons)
    monkeypatch.setattr(
        training_module,
        "run_phase3_ablations",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "component": "calibration",
                    "variant": "sigmoid_calibrated",
                    "model": "Logistic Regression",
                    "n_features": 10,
                    "n_num": 8,
                    "n_cat": 2,
                    "n_train": 40,
                    "n_calibration": 8,
                    "n_test": 12,
                    "uses_rfecv": True,
                    "uses_calibration": True,
                    "uses_reject_inference": False,
                    "ROC AUC": 0.61,
                    "PR AUC": 0.10,
                    "Brier": 0.19,
                }
            ]
        ),
    )
    monkeypatch.setattr(training_module, "compute_shap_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        training_module,
        "compute_woe_iv",
        lambda *args, **kwargs: (
            pd.DataFrame([{"feature": "INCOME_T1", "bin": "all", "woe": 0.0, "iv": 0.10}]),
            pd.DataFrame([{"feature": "INCOME_T1", "iv": 0.10}]),
        ),
    )
    monkeypatch.setattr(training_module, "run_stability_analysis", lambda *args, **kwargs: None)


@pytest.fixture()
def pipeline_data(train_test_data):
    """Prepare data ready for model training (cardinality reduced, preprocessors built)."""
    X_train, y_train, X_test, y_test, num_cols, cat_cols, feature_cols, train_dates = train_test_data

    if len(X_train) < 20 or y_train.sum() < 2:
        pytest.skip("Not enough data for model training")

    X_train, X_test, _ = reduce_cardinality(X_train, X_test, cat_cols)
    preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(num_cols, cat_cols)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_STATE)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "preprocessor": preprocessor,
        "lgbm_preprocessor": lgbm_preprocessor,
        "lgbm_cat_indices": lgbm_cat_indices,
        "pos_weight": pos_weight,
        "cv": cv,
        "train_dates": pd.to_datetime(np.asarray(train_dates)),
    }


@pytest.fixture()
def temporal_pipeline_data(structured_raw_df):
    df = engineer_features(structured_raw_df)
    feature_cols, num_cols, cat_cols = select_features(df)
    X_train, y_train, X_test, y_test, _, _, train_dates = temporal_split(df, feature_cols)
    X_train, X_test, _ = reduce_cardinality(X_train, X_test, cat_cols)
    preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(num_cols, cat_cols)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    temporal_cv = training_module.make_temporal_cv(train_dates, max_splits=3)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "feature_cols": feature_cols,
        "preprocessor": preprocessor,
        "lgbm_preprocessor": lgbm_preprocessor,
        "lgbm_cat_indices": lgbm_cat_indices,
        "pos_weight": pos_weight,
        "temporal_cv": temporal_cv,
        "train_dates": pd.to_datetime(np.asarray(train_dates)),
    }


class TestTrainLogisticRegression:
    def test_returns_fitted_model(self, pipeline_data):
        d = pipeline_data
        model, study = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        proba = model.predict_proba(d["X_test"])
        assert proba.shape == (len(d["X_test"]), 2)
        assert study.best_value > 0

    def test_with_sample_weight(self, pipeline_data):
        d = pipeline_data
        w = np.ones(len(d["X_train"]))
        w[::3] = 0.5  # every 3rd sample at half weight
        model, study = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"],
            n_trials=2, sample_weight=w,
        )
        proba = model.predict_proba(d["X_test"])
        assert proba.shape[1] == 2


class TestTrainLgbm:
    def test_returns_fitted_model(self, pipeline_data):
        d = pipeline_data
        model, study, best_n = train_lgbm(
            d["X_train"], d["y_train"],
            d["lgbm_preprocessor"], d["lgbm_cat_indices"],
            d["pos_weight"], d["cv"], n_trials=2,
        )
        proba = model.predict_proba(d["X_test"])
        assert proba.shape == (len(d["X_test"]), 2)
        assert best_n > 0

    def test_with_sample_weight(self, pipeline_data):
        d = pipeline_data
        w = np.ones(len(d["X_train"]))
        model, _, _ = train_lgbm(
            d["X_train"], d["y_train"],
            d["lgbm_preprocessor"], d["lgbm_cat_indices"],
            d["pos_weight"], d["cv"], n_trials=2, sample_weight=w,
        )
        assert model.predict_proba(d["X_test"]).shape[1] == 2


class TestTrainXgboost:
    def test_returns_fitted_model(self, pipeline_data):
        d = pipeline_data
        model, study, best_n = train_xgboost(
            d["X_train"], d["y_train"], d["preprocessor"],
            d["pos_weight"], d["cv"], n_trials=2,
        )
        proba = model.predict_proba(d["X_test"])
        assert proba.shape == (len(d["X_test"]), 2)
        assert best_n > 0


class TestTrainStacking:
    def test_returns_fitted_temporal_model(self, temporal_pipeline_data):
        d = temporal_pipeline_data
        base_models = {
            "Logistic Regression": _fit_quick_pipeline(d["X_train"], d["y_train"], d["preprocessor"]),
            "LightGBM": _fit_quick_pipeline(d["X_train"], d["y_train"], d["lgbm_preprocessor"]),
        }

        stack = train_stacking(
            d["X_train"],
            d["y_train"],
            base_models,
            d["temporal_cv"],
        )
        proba = stack.predict_proba(d["X_test"])
        assert proba.shape == (len(d["X_test"]), 2)
        assert set(stack.named_estimators_) == {"Logistic Regression", "LightGBM"}
        assert len(stack.fold_validation_positions_) > 0

    def test_uses_temporal_oof_predictions_without_future_leakage(self, temporal_pipeline_data):
        d = temporal_pipeline_data
        base_models = {
            "Logistic Regression": _fit_quick_pipeline(d["X_train"], d["y_train"], d["preprocessor"]),
        }

        stack = train_stacking(
            d["X_train"],
            d["y_train"],
            base_models,
            d["temporal_cv"],
        )

        validation_positions = np.concatenate(stack.fold_validation_positions_)
        assert len(validation_positions) == len(np.unique(validation_positions))
        assert np.array_equal(np.sort(validation_positions), np.sort(stack.meta_training_positions_))

        train_dates = d["train_dates"]
        for train_idx, val_idx in zip(stack.fold_training_positions_, stack.fold_validation_positions_, strict=True):
            assert pd.Timestamp(train_dates[train_idx].max()) < pd.Timestamp(train_dates[val_idx].min())


class TestEvaluateAll:
    def test_returns_results_and_scores(self, pipeline_data):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        models = {"LR": model}
        bench_risk = pd.Series(np.random.random(len(d["X_test"])), index=d["X_test"].index)
        bench_score = pd.Series(np.random.random(len(d["X_test"])), index=d["X_test"].index)

        results_df, score_arrays = evaluate_all(
            d["X_test"], d["y_test"], models, bench_risk, bench_score,
        )
        assert "ROC AUC" in results_df.columns
        assert "Gini" in results_df.columns
        assert "KS" in results_df.columns
        assert "LR" in score_arrays


class TestRollingOOTValidation:
    def test_returns_fold_results_and_summary(self):
        dates = pd.to_datetime([
            "2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01",
            "2024-02-01", "2024-02-01", "2024-02-01", "2024-02-01",
            "2024-03-01", "2024-03-01", "2024-03-01", "2024-03-01",
            "2024-04-01", "2024-04-01", "2024-04-01", "2024-04-01",
            "2024-05-01", "2024-05-01", "2024-05-01", "2024-05-01",
            "2024-06-01", "2024-06-01", "2024-06-01", "2024-06-01",
            "2024-07-01", "2024-07-01", "2024-07-01", "2024-07-01",
            "2024-08-01", "2024-08-01", "2024-08-01", "2024-08-01",
        ])
        X_base = pd.DataFrame(
            {
                "num_a": np.linspace(0.0, 3.1, len(dates)),
                "num_b": np.tile([1.0, 2.0, 3.0, 4.0], len(dates) // 4),
                "cat_a": np.tile(["A", "B", "A", "B"], len(dates) // 4),
            }
        )
        y = pd.Series(np.tile([0, 1, 0, 1], len(dates) // 4), name=TARGET)
        bench_risk = pd.Series(np.linspace(600, 500, len(dates)), index=X_base.index)
        bench_score = pd.Series(np.linspace(300, 450, len(dates)), index=X_base.index)
        preprocessor, _, _ = build_preprocessors(["num_a", "num_b"], ["cat_a"])
        base_models = {
            "Logistic Regression": Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                )),
            ])
        }

        results_df, summary_df = run_rolling_out_of_time_validation(
            X_base,
            y,
            dates,
            bench_risk,
            bench_score,
            ["num_a", "num_b", "cat_a"],
            ["num_a", "num_b"],
            ["cat_a"],
            ["num_a", "num_b", "cat_a"],
            ["num_a", "num_b"],
            ["cat_a"],
            base_models,
            max_windows=3,
        )

        assert not results_df.empty
        assert not summary_df.empty
        assert results_df["fold"].nunique() == 3
        assert "Logistic Regression" in summary_df["Model"].values
        assert "risk_score_rf (benchmark)" in summary_df["Model"].values
        assert {"validation_start", "validation_end", "n_fit", "n_calibration", "n_validation"} <= set(results_df.columns)


class TestExtractFeatureImportance:
    def test_extracts_lr_coefficients(self, pipeline_data):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        feat_imp = extract_feature_importance(
            {"Logistic Regression": model}, d["num_cols"], d["cat_cols"],
        )
        assert not feat_imp.empty
        assert "coefficient" in feat_imp["type"].values

    def test_extracts_lgbm_importance(self, pipeline_data):
        d = pipeline_data
        model, _, _ = train_lgbm(
            d["X_train"], d["y_train"],
            d["lgbm_preprocessor"], d["lgbm_cat_indices"],
            d["pos_weight"], d["cv"], n_trials=2,
        )
        feat_imp = extract_feature_importance(
            {"LightGBM": model}, d["num_cols"], d["cat_cols"],
        )
        assert not feat_imp.empty
        assert "split_importance" in feat_imp["type"].values

    def test_skips_calibrated_and_stacking(self, pipeline_data):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        feat_imp = extract_feature_importance(
            {"LR (calibrated)": model, "Stacking": model},
            d["num_cols"], d["cat_cols"],
        )
        assert feat_imp.empty


class TestArtifacts:
    def test_save_artifacts(self, pipeline_data, tmp_path):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        models = {"LR": model}
        results_df = pd.DataFrame([evaluate("LR", d["y_test"].values, model.predict_proba(d["X_test"])[:, 1])]).set_index("Model")
        feat_imp = extract_feature_importance(models, d["num_cols"], d["cat_cols"])

        save_artifacts(models, results_df, feat_imp, tmp_path)
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "feature_importance.csv").exists()
        assert (tmp_path / "models" / "lr.joblib").exists()

    def test_plot_score_distributions(self, pipeline_data, tmp_path):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        scores = {"Logistic Regression": model.predict_proba(d["X_test"])[:, 1]}
        out = tmp_path / "dist.png"
        plot_score_distributions(d["y_test"].values, scores, out, "Test")
        assert out.exists()

    def test_save_artifacts_writes_optional_comparison_outputs(self, pipeline_data, tmp_path):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        models = {"LR": model}
        results_df = pd.DataFrame(
            [evaluate("LR", d["y_test"].values, model.predict_proba(d["X_test"])[:, 1])]
        ).set_index("Model")
        feat_imp = extract_feature_importance(models, d["num_cols"], d["cat_cols"])
        experimental_results_df = results_df.rename(index={"LR": "Stacking (experimental)"})
        benchmark_comparisons_df = pd.DataFrame(
            [
                {
                    "candidate_model": "LR",
                    "reference_model": "risk_score_rf (benchmark)",
                    "n": len(d["y_test"]),
                    "candidate_auc": 0.6,
                    "reference_auc": 0.55,
                    "auc_improvement": 0.05,
                    "auc_improvement_lo": 0.01,
                    "auc_improvement_hi": 0.08,
                    "auc_p_value": 0.03,
                    "candidate_pr_auc": 0.10,
                    "reference_pr_auc": 0.08,
                    "pr_auc_improvement": 0.02,
                    "pr_auc_improvement_lo": 0.00,
                    "pr_auc_improvement_hi": 0.04,
                    "pr_auc_p_value": 0.05,
                    "candidate_brier": 0.2,
                    "reference_brier": np.nan,
                    "brier_improvement": np.nan,
                    "brier_improvement_lo": np.nan,
                    "brier_improvement_hi": np.nan,
                    "brier_p_value": np.nan,
                }
            ]
        )
        feature_provenance_df = pd.DataFrame(
            [
                {
                    "feature": "A",
                    "provenance": "raw",
                    "data_type": "numerical",
                    "rfecv_candidate": True,
                    "rfecv_kept": True,
                    "interaction_type": np.nan,
                    "feat_a": np.nan,
                    "feat_b": np.nan,
                }
            ]
        )
        interaction_leaderboard_df = pd.DataFrame(
            [
                {
                    "name": "A/B",
                    "type": "ratio",
                    "auc": 0.61,
                    "lift": 0.03,
                    "feat_a": "A",
                    "feat_b": "B",
                    "feat_a_power": 0.05,
                    "feat_b_power": 0.02,
                    "parent_power": 0.05,
                    "power": 0.08,
                    "selected": True,
                    "scoring_strategy": "temporal_validation",
                }
            ]
        )
        feature_discovery_boundary_df = pd.DataFrame(
            [
                {
                    "feature_discovery_fraction": 0.5,
                    "feature_discovery_end": "2024-03-01",
                    "interaction_search_cutoff": "2024-04-01",
                    "discovery_seed_rows": 100,
                    "discovery_seed_positives": 12,
                    "estimation_seed_rows": 100,
                    "estimation_seed_positives": 13,
                    "numeric_scoring_strategy": "temporal_validation",
                    "categorical_scoring_strategy": "temporal_target_encode",
                    "screened_num_pairs": 10,
                    "screened_cat_pairs": 6,
                    "selected_interactions": 1,
                }
            ]
        )
        ablation_results_df = pd.DataFrame(
            [
                {
                    "component": "calibration",
                    "variant": "sigmoid_calibrated",
                    "model": "Logistic Regression",
                    "n_features": 10,
                    "n_num": 8,
                    "n_cat": 2,
                    "n_train": 100,
                    "n_calibration": 20,
                    "n_test": len(d["y_test"]),
                    "uses_rfecv": True,
                    "uses_calibration": True,
                    "uses_reject_inference": False,
                    "ROC AUC": 0.61,
                    "PR AUC": 0.09,
                    "Brier": 0.18,
                }
            ]
        )
        rolling_oot_results_df = pd.DataFrame(
            [
                {
                    "fold": 1,
                    "Model": "Logistic Regression",
                    "train_start": "2024-01-01",
                    "train_end": "2024-03-01",
                    "calibration_start": "2024-04-01",
                    "calibration_end": "2024-04-01",
                    "validation_start": "2024-05-01",
                    "validation_end": "2024-06-01",
                    "n_fit": 100,
                    "n_calibration": 20,
                    "n_validation": len(d["y_test"]),
                    "n_validation_pos": int(d["y_test"].sum()),
                    "ROC AUC": 0.62,
                    "Gini": 0.24,
                    "KS": 0.11,
                    "PR AUC": 0.10,
                    "N": len(d["y_test"]),
                    "Brier": 0.18,
                    "is_calibrated": False,
                }
            ]
        )
        rolling_oot_summary_df = pd.DataFrame(
            [
                {
                    "Model": "Logistic Regression",
                    "n_folds": 3,
                    "mean_ROC_AUC": 0.62,
                    "std_ROC_AUC": 0.01,
                    "mean_PR_AUC": 0.10,
                    "std_PR_AUC": 0.01,
                    "mean_Brier": 0.18,
                    "std_Brier": 0.01,
                    "validation_start_min": "2024-05-01",
                    "validation_end_max": "2024-07-01",
                }
            ]
        )

        save_artifacts(
            models,
            results_df,
            feat_imp,
            tmp_path,
            experimental_results_df=experimental_results_df,
            benchmark_comparisons_df=benchmark_comparisons_df,
            feature_provenance_df=feature_provenance_df,
            interaction_leaderboard_df=interaction_leaderboard_df,
            feature_discovery_boundary_df=feature_discovery_boundary_df,
            ablation_results_df=ablation_results_df,
            rolling_oot_results_df=rolling_oot_results_df,
            rolling_oot_summary_df=rolling_oot_summary_df,
        )

        assert (tmp_path / "results_experimental.csv").exists()
        assert (tmp_path / "benchmark_comparisons.csv").exists()
        assert (tmp_path / "feature_provenance.csv").exists()
        assert (tmp_path / "interaction_leaderboard.csv").exists()
        assert (tmp_path / "feature_discovery_boundary.csv").exists()
        assert (tmp_path / "ablation_results.csv").exists()
        assert (tmp_path / "rolling_oot_results.csv").exists()
        assert (tmp_path / "rolling_oot_summary.csv").exists()


class TestMainSmoke:
    def test_official_underwriting_run_writes_only_official_outputs(
        self,
        monkeypatch,
        structured_raw_df_with_rejects,
        tmp_path,
    ):
        _patch_fast_main_dependencies(monkeypatch, structured_raw_df_with_rejects)

        training_module.main(
            data_path="unused.parquet",
            optuna_trials=1,
            output_dir=str(tmp_path),
            reject_inference=False,
            enable_experimental_stacking=False,
            population_mode=POPULATION_MODE_UNDERWRITING,
        )

        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "benchmark_comparisons.csv").exists()
        assert (tmp_path / "confidence_intervals.csv").exists()
        assert (tmp_path / "population_summary.csv").exists()
        assert (tmp_path / "applicant_scores_post_split.csv").exists()
        assert (tmp_path / "holdout_test_scores.csv").exists()
        assert not (tmp_path / "results_experimental.csv").exists()
        assert not (tmp_path / "benchmark_comparisons_experimental.csv").exists()

        results_df = pd.read_csv(tmp_path / "results.csv")
        holdout_scores_df = pd.read_csv(tmp_path / "holdout_test_scores.csv")
        assert EXPERIMENTAL_STACKING_NAME not in results_df["Model"].values
        assert "score__logistic_regression" in holdout_scores_df.columns
        assert (holdout_scores_df["evaluation_population"] == "booked_proxy").all()

    def test_experimental_stacking_run_separates_official_and_experimental_outputs(
        self,
        monkeypatch,
        structured_raw_df_with_rejects,
        tmp_path,
    ):
        _patch_fast_main_dependencies(monkeypatch, structured_raw_df_with_rejects)

        training_module.main(
            data_path="unused.parquet",
            optuna_trials=1,
            output_dir=str(tmp_path),
            reject_inference=False,
            enable_experimental_stacking=True,
            population_mode=POPULATION_MODE_UNDERWRITING,
        )

        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "results_experimental.csv").exists()
        assert (tmp_path / "benchmark_comparisons.csv").exists()
        assert (tmp_path / "benchmark_comparisons_experimental.csv").exists()

        official_results_df = pd.read_csv(tmp_path / "results.csv")
        experimental_results_df = pd.read_csv(tmp_path / "results_experimental.csv")
        assert EXPERIMENTAL_STACKING_NAME not in official_results_df["Model"].values
        assert EXPERIMENTAL_STACKING_NAME in experimental_results_df["Model"].values


class TestUnderwritingOutputs:
    def test_build_holdout_score_frame_persists_scores_and_metadata(self):
        holdout_scores_df = build_holdout_score_frame(
            pd.Series([0, 1, 0], name=TARGET),
            {
                "Logistic Regression": np.array([0.10, 0.80, 0.25]),
                "score_RF (benchmark)": np.array([-400.0, -250.0, -375.0]),
            },
            population_mode=POPULATION_MODE_UNDERWRITING,
            evaluation_population="booked_proxy",
        )

        assert list(holdout_scores_df[TARGET]) == [0.0, 1.0, 0.0]
        assert "score__logistic_regression" in holdout_scores_df.columns
        assert "score__score_rf_benchmark" in holdout_scores_df.columns
        assert (holdout_scores_df["population_mode"] == POPULATION_MODE_UNDERWRITING).all()
        assert (holdout_scores_df["evaluation_population"] == "booked_proxy").all()

    def test_build_population_summary_df_tracks_underwriting_groups(self, raw_df_with_rejects):
        booked_df = raw_df_with_rejects[raw_df_with_rejects["status_name"] == "Booked"].copy()
        rejected_df = raw_df_with_rejects[raw_df_with_rejects["status_name"].isin(["Rejected", "Canceled"])].copy()

        summary_df = build_population_summary_df(
            booked_df,
            rejected_df,
            population_mode=POPULATION_MODE_UNDERWRITING,
        )

        assert not summary_df.empty
        assert (summary_df["population_mode"] == POPULATION_MODE_UNDERWRITING).all()
        assert {"booked", "decisioned_non_booked"} <= set(summary_df["population_group"])
        assert set(summary_df["split"]) <= {"pre_split", "post_split"}

        expected_post_split = int(
            (booked_df["mis_Date"] >= pd.Timestamp(SPLIT_DATE)).sum()
            + (rejected_df["mis_Date"] >= pd.Timestamp(SPLIT_DATE)).sum()
        )
        observed_post_split = int(summary_df.loc[summary_df["split"] == "post_split", "n_rows"].sum())
        assert observed_post_split == expected_post_split

    def test_build_applicant_score_frame_scores_post_split_decisioned_rows(self, raw_df_with_rejects):
        booked_df = raw_df_with_rejects[raw_df_with_rejects["status_name"] == "Booked"].copy()
        rejected_df = raw_df_with_rejects[raw_df_with_rejects["status_name"].isin(["Rejected", "Canceled"])].copy()

        pre_split_booked_idx = booked_df.index[
            (booked_df["mis_Date"] < pd.Timestamp(SPLIT_DATE))
            & booked_df[TARGET].notna()
        ]
        if len(pre_split_booked_idx) < 4:
            pytest.skip("Not enough booked pre-split rows for underwriting scoring test")

        midpoint = len(pre_split_booked_idx) // 2
        booked_df.loc[pre_split_booked_idx[:midpoint], TARGET] = 0
        booked_df.loc[pre_split_booked_idx[midpoint:], TARGET] = 1

        booked_df = engineer_features(booked_df)
        rejected_df = engineer_features(rejected_df)
        base_feature_cols, base_num_cols, base_cat_cols = select_features(booked_df)
        X_train_base, y_train, _, _, _, _, _ = temporal_split(booked_df, base_feature_cols)

        preprocessor, _, _ = build_preprocessors(base_num_cols, base_cat_cols)
        preprocessor.set_params(cat__encoder__cv=2)
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
                solver="lbfgs",
            )),
        ])
        model.fit(X_train_base, y_train)

        applicant_scores_df = build_applicant_score_frame(
            booked_df,
            rejected_df,
            X_train_base,
            base_feature_cols,
            base_num_cols,
            base_cat_cols,
            base_feature_cols,
            {"Logistic Regression": model},
        )

        expected_rows = int(
            (
                pd.concat([booked_df, rejected_df], axis=0)
                .loc[lambda df_: df_["mis_Date"] >= pd.Timestamp(SPLIT_DATE)]
                .shape[0]
            )
        )

        assert not applicant_scores_df.empty
        assert len(applicant_scores_df) == expected_rows
        assert (applicant_scores_df["mis_Date"] >= pd.Timestamp(SPLIT_DATE)).all()
        assert applicant_scores_df["status_name"].isin(["Booked", "Rejected", "Canceled"]).all()
        assert "AGE_T1" in applicant_scores_df.columns
        assert applicant_scores_df["AGE_T1"].notna().all()
        assert "score__logistic_regression" in applicant_scores_df.columns
        assert applicant_scores_df["score__logistic_regression"].notna().all()
        assert (
            applicant_scores_df.loc[applicant_scores_df[TARGET].notna(), "target_source"] == "observed_booked"
        ).all()
        assert (
            applicant_scores_df.loc[applicant_scores_df[TARGET].isna(), "target_source"] == "unobserved_application"
        ).all()

    def test_save_artifacts_writes_underwriting_output_files(self, pipeline_data, tmp_path):
        d = pipeline_data
        model, _ = train_logistic_regression(
            d["X_train"], d["y_train"], d["preprocessor"], d["cv"], n_trials=2,
        )
        models = {"LR": model}
        results_df = pd.DataFrame(
            [evaluate("LR", d["y_test"].values, model.predict_proba(d["X_test"])[:, 1])]
        ).set_index("Model")
        feat_imp = extract_feature_importance(models, d["num_cols"], d["cat_cols"])
        population_summary_df = pd.DataFrame(
            [
                {
                    "population_mode": POPULATION_MODE_UNDERWRITING,
                    "split": "post_split",
                    "status_name": "Rejected",
                    "population_group": "decisioned_non_booked",
                    "n_rows": 12,
                    "n_with_observed_target": 0,
                    "n_bad_observed": 0,
                    "date_start": "2024-07-01",
                    "date_end": "2024-08-01",
                }
            ]
        )
        applicant_scores_df = pd.DataFrame(
            [
                {
                    "authorization_id": 1.0,
                    "mis_Date": "2024-08-01",
                    "status_name": "Rejected",
                    TARGET: np.nan,
                    "has_observed_target": False,
                    "target_source": "unobserved_application",
                    "risk_score_rf": 42.0,
                    "score_RF": 350.0,
                    "score__lr": 0.12,
                }
            ]
        )
        holdout_scores_df = pd.DataFrame(
            [
                {
                    TARGET: 0.0,
                    "score__lr": 0.12,
                    "population_mode": POPULATION_MODE_UNDERWRITING,
                    "evaluation_population": "booked_proxy",
                }
            ]
        )

        save_artifacts(
            models,
            results_df,
            feat_imp,
            tmp_path,
            population_summary_df=population_summary_df,
            applicant_scores_df=applicant_scores_df,
            holdout_scores_df=holdout_scores_df,
        )

        assert (tmp_path / "population_summary.csv").exists()
        assert (tmp_path / "applicant_scores_post_split.csv").exists()
        assert (tmp_path / "holdout_test_scores.csv").exists()
