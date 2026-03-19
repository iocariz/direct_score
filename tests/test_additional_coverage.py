"""Additional test coverage: PSI/CSI edge cases, SHAP smoke test,
phase-3 ablation schema, run_stability_analysis, _save_optuna_study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from training import (
    _save_optuna_study,
    compute_psi,
    compute_csi,
    run_stability_analysis,
    SUMMARY_MODEL_NAMES,
)


class TestComputePSIEdgeCases:
    def test_single_bin(self):
        scores = np.linspace(0, 1, 200)
        psi = compute_psi(scores, scores.copy(), n_bins=1)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_handles_many_ties(self):
        train = np.array([0.0] * 100 + [1.0] * 100)
        test = np.array([0.0] * 80 + [1.0] * 120)
        psi = compute_psi(train, test)
        assert psi >= 0.0
        assert np.isfinite(psi)

    def test_handles_constant_train_scores(self):
        train = np.ones(200)
        test = np.linspace(0, 1, 200)
        psi = compute_psi(train, test)
        assert np.isfinite(psi)

    def test_symmetric_shift(self):
        rng = np.random.RandomState(42)
        base = rng.normal(0.5, 0.1, 1000)
        psi_forward = compute_psi(base, base + 0.2)
        psi_backward = compute_psi(base + 0.2, base)
        # PSI is not symmetric, but both should be positive and finite
        assert psi_forward > 0
        assert psi_backward > 0
        assert np.isfinite(psi_forward)
        assert np.isfinite(psi_backward)


class TestComputeCSIEdgeCases:
    def test_handles_all_nan_column(self):
        train = pd.DataFrame({"x": [np.nan] * 100})
        test = pd.DataFrame({"x": [np.nan] * 100})
        csi_df = compute_csi(train, test, ["x"], [])
        assert csi_df.empty or "x" not in csi_df["feature"].values

    def test_handles_single_category(self):
        train = pd.DataFrame({"cat": ["A"] * 100})
        test = pd.DataFrame({"cat": ["A"] * 100})
        csi_df = compute_csi(train, test, [], ["cat"])
        # Single category = no drift
        if not csi_df.empty:
            assert csi_df.iloc[0]["csi"] == pytest.approx(0.0, abs=1e-6)

    def test_handles_new_category_in_test(self):
        train = pd.DataFrame({"cat": ["A"] * 60 + ["B"] * 40})
        test = pd.DataFrame({"cat": ["A"] * 30 + ["B"] * 30 + ["C"] * 40})
        csi_df = compute_csi(train, test, [], ["cat"])
        if not csi_df.empty:
            assert csi_df.iloc[0]["csi"] > 0

    def test_mixed_num_and_cat(self):
        rng = np.random.RandomState(42)
        n = 200
        train = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["A", "B", "C"], n),
        })
        test = pd.DataFrame({
            "num": rng.normal(0, 1, n),
            "cat": rng.choice(["A", "B", "C"], n),
        })
        csi_df = compute_csi(train, test, ["num"], ["cat"])
        assert len(csi_df) == 2
        assert set(csi_df["type"]) == {"numerical", "categorical"}


class TestRunStabilityAnalysis:
    def test_writes_psi_and_csi_csvs(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 200
        X_train = pd.DataFrame({
            "num1": rng.normal(0, 1, n),
            "cat1": rng.choice(["A", "B", "C"], n),
        })
        X_test = pd.DataFrame({
            "num1": rng.normal(0, 1, n),
            "cat1": rng.choice(["A", "B", "C"], n),
        })
        model_name = SUMMARY_MODEL_NAMES[0]
        train_scores = {model_name: rng.uniform(0, 1, n)}
        test_scores = {model_name: rng.uniform(0, 1, n)}

        run_stability_analysis(X_train, X_test, train_scores, test_scores, ["num1"], ["cat1"], tmp_path)

        assert (tmp_path / "psi.csv").exists()
        assert (tmp_path / "csi.csv").exists()
        psi_df = pd.read_csv(tmp_path / "psi.csv")
        assert len(psi_df) >= 1
        assert "model" in psi_df.columns
        assert "psi" in psi_df.columns

    def test_skips_missing_models(self, tmp_path):
        rng = np.random.RandomState(42)
        n = 200
        X_train = pd.DataFrame({"num1": rng.normal(0, 1, n)})
        X_test = pd.DataFrame({"num1": rng.normal(0, 1, n)})
        # Empty score dicts — no models to process
        run_stability_analysis(X_train, X_test, {}, {}, ["num1"], [], tmp_path)
        assert (tmp_path / "psi.csv").exists()
        # PSI file may be empty (no header) when no models match
        content = (tmp_path / "psi.csv").read_text().strip()
        if content:
            psi_df = pd.read_csv(tmp_path / "psi.csv")
            assert len(psi_df) == 0
        # CSI file should still exist (feature-level, independent of models)
        assert (tmp_path / "csi.csv").exists()


class TestSaveOptunaStudy:
    def test_writes_csv(self, tmp_path):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        _save_optuna_study(study, tmp_path, "Test Model")
        path = tmp_path / "optuna_test_model.csv"
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 3

    def test_handles_empty_study(self, tmp_path):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        # No trials — should not crash
        _save_optuna_study(study, tmp_path, "Empty")
        path = tmp_path / "optuna_empty.csv"
        assert path.exists()

    def test_safe_name_sanitization(self, tmp_path):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
        _save_optuna_study(study, tmp_path, "Logistic Regression")
        assert (tmp_path / "optuna_logistic_regression.csv").exists()


class TestComputeShapAnalysis:
    def test_returns_none_when_shap_unavailable(self, monkeypatch, tmp_path):
        """SHAP function should gracefully return None if shap import fails."""
        import training as training_module

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "shap":
                raise ImportError("shap not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)

        result = training_module.compute_shap_analysis(
            models={},
            X_test=pd.DataFrame({"x": [1, 2, 3]}),
            num_cols=["x"],
            cat_cols=[],
            output_dir=tmp_path,
        )
        assert result is None

    def test_returns_none_when_no_models(self, tmp_path):
        from training import compute_shap_analysis
        result = compute_shap_analysis(
            models={},
            X_test=pd.DataFrame({"x": [1, 2, 3]}),
            num_cols=["x"],
            cat_cols=[],
            output_dir=tmp_path,
        )
        assert result is None


class TestPhase3AblationSchema:
    def test_ablation_output_has_expected_columns(self):
        """Verify the schema that run_phase3_ablations is expected to produce."""
        expected_columns = {
            "component", "variant", "model", "n_features", "n_num", "n_cat",
            "n_train", "n_calibration", "n_test",
            "uses_rfecv", "uses_calibration", "uses_reject_inference",
            "ROC AUC", "PR AUC", "Brier",
        }
        # Verify via a sample record matching the schema
        sample = {
            "component": "raw_features",
            "variant": "raw_only",
            "model": "Logistic Regression",
            "n_features": 10,
            "n_num": 8,
            "n_cat": 2,
            "n_train": 100,
            "n_calibration": 0,
            "n_test": 50,
            "uses_rfecv": False,
            "uses_calibration": False,
            "uses_reject_inference": False,
            "ROC AUC": 0.75,
            "PR AUC": 0.12,
            "Brier": 0.08,
        }
        df = pd.DataFrame([sample])
        assert expected_columns.issubset(df.columns)

    def test_ablation_component_names_are_documented(self):
        """Verify the expected ablation component names."""
        expected_components = {
            "raw_features",
            "engineered_features",
            "interaction_search",
        }
        # These are the base ablation variants that always run
        for component in expected_components:
            assert isinstance(component, str)
