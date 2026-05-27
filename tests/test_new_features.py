"""Tests for concept drift, post-hoc ensemble, scoring API, and EnsembleModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import EnsembleModel
from training_reporting import compute_concept_drift_report, train_post_hoc_ensemble


class TestComputeConceptDriftReport:
    @pytest.fixture()
    def stable_oot(self):
        return pd.DataFrame({
            "Model": ["LR", "LR", "LR", "LR", "LGBM", "LGBM", "LGBM", "LGBM"],
            "fold": [1, 2, 3, 4, 1, 2, 3, 4],
            "PR AUC": [0.15, 0.14, 0.15, 0.14, 0.18, 0.17, 0.18, 0.17],
            "ROC AUC": [0.82, 0.81, 0.82, 0.81, 0.85, 0.84, 0.85, 0.84],
            "Brier": [0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05],
            "is_calibrated": [False] * 8,
        })

    @pytest.fixture()
    def drifting_oot(self):
        return pd.DataFrame({
            "Model": ["LR", "LR", "LR", "LR"],
            "fold": [1, 2, 3, 4],
            "PR AUC": [0.18, 0.15, 0.13, 0.10],
            "ROC AUC": [0.85, 0.82, 0.80, 0.77],
            "Brier": [0.05, 0.06, 0.07, 0.08],
            "is_calibrated": [False] * 4,
        })

    def test_returns_expected_columns(self, stable_oot):
        df = compute_concept_drift_report(stable_oot)
        expected = {"model", "n_folds", "pr_auc_first", "pr_auc_last",
                    "pr_auc_slope_per_fold", "concept_drift_flag"}
        assert expected.issubset(df.columns)

    def test_stable_model_not_flagged(self, stable_oot):
        df = compute_concept_drift_report(stable_oot)
        assert (df["concept_drift_flag"] == "NO").all()

    def test_drifting_model_flagged(self, drifting_oot):
        df = compute_concept_drift_report(drifting_oot)
        assert df.iloc[0]["concept_drift_flag"] == "YES"
        assert df.iloc[0]["pr_auc_slope_per_fold"] < 0

    def test_empty_when_no_data(self):
        assert compute_concept_drift_report(None).empty
        assert compute_concept_drift_report(pd.DataFrame()).empty

    def test_filters_to_uncalibrated(self):
        df = pd.DataFrame({
            "Model": ["LR", "LR", "LR (calibrated)", "LR (calibrated)"],
            "fold": [1, 2, 1, 2],
            "PR AUC": [0.15, 0.14, 0.16, 0.15],
            "ROC AUC": [0.82, 0.81, 0.83, 0.82],
            "is_calibrated": [False, False, True, True],
        })
        result = compute_concept_drift_report(df)
        assert len(result) == 1
        assert result.iloc[0]["model"] == "LR"


class TestTrainPostHocEnsemble:
    def test_finds_best_weights(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        lr_scores = np.concatenate([rng.uniform(0.1, 0.4, 80), rng.uniform(0.5, 0.8, 20)])
        lgbm_scores = np.concatenate([rng.uniform(0.05, 0.3, 80), rng.uniform(0.6, 0.95, 20)])
        scores = {"Logistic Regression": lr_scores, "LightGBM": lgbm_scores}

        result = train_post_hoc_ensemble(y, scores)
        assert "lr_weight" in result
        assert "tree_weight" in result
        assert "pr_auc" in result
        assert result["lr_weight"] + result["tree_weight"] == pytest.approx(1.0)

    def test_returns_empty_when_no_lr(self):
        result = train_post_hoc_ensemble(
            np.array([0, 1]),
            {"LightGBM": np.array([0.3, 0.7])},
        )
        assert result == {}

    def test_selects_best_tree(self):
        rng = np.random.RandomState(42)
        y = np.array([0] * 80 + [1] * 20)
        lr = np.concatenate([rng.uniform(0.1, 0.4, 80), rng.uniform(0.5, 0.8, 20)])
        good_tree = np.concatenate([rng.uniform(0.0, 0.2, 80), rng.uniform(0.8, 1.0, 20)])
        bad_tree = rng.uniform(0.3, 0.7, 100)
        scores = {"Logistic Regression": lr, "LightGBM": good_tree, "XGBoost": bad_tree}

        result = train_post_hoc_ensemble(y, scores)
        assert result["tree_name"] == "LightGBM"


class TestEnsembleModel:
    def test_predict_proba_blends(self):
        class _MockModel:
            def predict_proba(self, X):
                return np.array([[0.6, 0.4]] * len(X))

        class _MockModel2:
            def predict_proba(self, X):
                return np.array([[0.2, 0.8]] * len(X))

        ens = EnsembleModel(_MockModel(), _MockModel2(), 0.5, 0.5)
        result = ens.predict_proba(np.zeros((3, 2)))
        expected = np.array([[0.4, 0.6]] * 3)
        np.testing.assert_allclose(result, expected)

    def test_named_steps_is_none(self):
        ens = EnsembleModel(None, None, 0.5, 0.5)
        assert ens.named_steps is None


class TestScoringService:
    def test_score_applicant_returns_result(self, tmp_path):
        from scoring import ScoringService, ScoringResult

        # Create a minimal mock model
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression()),
        ])
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 1, 0, 1])
        model.fit(X, y)

        service = ScoringService(
            model=model,
            model_name="test",
            feature_cols=["a", "b"],
            model_version="abc123",
        )
        result = service.score_applicant({"a": 5.0, "b": 6.0})

        assert isinstance(result, ScoringResult)
        assert 0.0 <= result.predicted_pd <= 1.0
        assert result.risk_tier in {"LOW", "MODERATE", "ELEVATED", "HIGH", "VERY_HIGH", "UNKNOWN"}
        assert result.model_name == "test"

    def test_score_batch(self):
        from scoring import ScoringService
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        model.fit(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]))

        service = ScoringService(model=model, model_name="test", feature_cols=["a", "b"])
        result = service.score_batch(pd.DataFrame({"a": [1, 5], "b": [2, 6]}))

        assert len(result) == 2
        assert "predicted_pd" in result.columns
        assert "risk_tier" in result.columns

    def test_input_validation_warns_on_out_of_range(self):
        from scoring import ScoringService
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        model.fit(np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 0, 1]))

        service = ScoringService(
            model=model, model_name="test", feature_cols=["a", "b"],
            training_stats={
                "a": {"type": "numerical", "min": 1.0, "max": 5.0, "missing_pct": 0.0, "n_unique": 3},
                "b": {"type": "numerical", "min": 2.0, "max": 6.0, "missing_pct": 0.0, "n_unique": 3},
            },
        )
        result = service.score_applicant({"a": 100.0, "b": 3.0})
        assert len(result.warnings) >= 1
        assert "above training max" in result.warnings[0]

    def test_to_dict(self):
        from scoring import ScoringResult
        result = ScoringResult(predicted_pd=0.05, risk_tier="LOW", model_name="test", model_version="v1")
        d = result.to_dict()
        assert d["predicted_pd"] == 0.05
        assert d["risk_tier"] == "LOW"

    def test_from_output_dir_loads_recommended_model(self, tmp_path):
        import joblib
        from sklearn.linear_model import LogisticRegression
        from scoring import ScoringService

        model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model, models_dir / "logistic_regression.joblib")

        pd.DataFrame([{"model": "Logistic Regression", "recommended": True}]).to_csv(
            tmp_path / "model_selection.csv", index=False
        )
        pd.DataFrame([{"feature": "a"}]).to_csv(tmp_path / "variable_dictionary.csv", index=False)
        pd.DataFrame(
            [{"feature": "a", "type": "numerical", "min": 0.0, "max": 3.0, "missing_pct": 0.0, "n_unique": 4}]
        ).to_csv(tmp_path / "data_quality_development_fit.csv", index=False)

        service = ScoringService.from_output_dir(tmp_path)
        result = service.score_applicant({"a": 2.0})
        assert service.model_name == "Logistic Regression"
        assert 0.0 <= result.predicted_pd <= 1.0

    def test_from_output_dir_rejects_path_traversal_in_model_name(self, tmp_path):
        """A model_selection.csv that names ../../etc/passwd must not escape models_dir."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from scoring import ScoringService

        model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model, models_dir / "logistic_regression.joblib")
        pd.DataFrame([{"feature": "a"}]).to_csv(tmp_path / "variable_dictionary.csv", index=False)

        # Adversarial model name with traversal sequence.
        pd.DataFrame([{"model": "../../etc/passwd", "recommended": True}]).to_csv(
            tmp_path / "model_selection.csv", index=False
        )
        # Sanitization reduces this to a basename, so the load fails as "not found"
        # rather than escaping into the filesystem. Either FileNotFoundError or
        # ValueError is acceptable; what matters is that no file outside models_dir
        # is read.
        with pytest.raises((FileNotFoundError, ValueError)):
            ScoringService.from_output_dir(tmp_path)

    def test_from_output_dir_verifies_checksum(self, tmp_path):
        """Tampering with a .joblib after checksums.json is written must be detected."""
        import json
        import joblib
        from sklearn.linear_model import LogisticRegression
        from scoring import ScoringService

        model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_path = models_dir / "logistic_regression.joblib"
        joblib.dump(model, model_path)

        # Compute the correct checksum, then deliberately corrupt the file.
        import hashlib
        good = hashlib.sha256(model_path.read_bytes()).hexdigest()
        (models_dir / "checksums.json").write_text(json.dumps({"logistic_regression.joblib": good}))
        # Append bytes to invalidate the checksum.
        with model_path.open("ab") as f:
            f.write(b"\x00")

        pd.DataFrame([{"model": "Logistic Regression", "recommended": True}]).to_csv(
            tmp_path / "model_selection.csv", index=False
        )
        pd.DataFrame([{"feature": "a"}]).to_csv(tmp_path / "variable_dictionary.csv", index=False)

        with pytest.raises(ValueError, match="Checksum mismatch"):
            ScoringService.from_output_dir(tmp_path)

    def test_score_applicant_rejects_raw_input(self):
        """A caller who forgot to run feature engineering must get a loud error."""
        from scoring import ScoringService
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
        model.fit(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 0, 1, 1]))

        # Model expects 10 engineered features, but caller passes only 1 raw key.
        feature_cols = [
            "INCOME_T1", "FREQ_CSP", "INSTALLMENT_TO_INCOME", "HAS_CODEBTOR",
            "LOG_INCOME_T1", "INCOME_T1_VS_CSP", "BOOK_RATIO_LOAN",
            "TOTAL_AMT_TO_HOUSEHOLD", "MISS_HOUSE_YEARS", "PRODTYPE3_X_HOUSE",
        ]
        service = ScoringService(model=model, model_name="test", feature_cols=feature_cols)

        with pytest.raises(ValueError, match="engineering pipeline"):
            service.score_applicant({"INCOME_T1": 45000})

    def test_from_output_dir_loads_risk_tiers(self, tmp_path):
        import json
        import joblib
        from sklearn.linear_model import LogisticRegression
        from scoring import ScoringService

        model = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        joblib.dump(model, models_dir / "logistic_regression.joblib")
        pd.DataFrame([{"model": "Logistic Regression", "recommended": True}]).to_csv(
            tmp_path / "model_selection.csv", index=False
        )
        pd.DataFrame([{"feature": "a"}]).to_csv(tmp_path / "variable_dictionary.csv", index=False)

        custom_tiers = [[0.0, 0.9, "LOW"], [0.9, 1.01, "HIGH"]]
        (tmp_path / "risk_tiers.json").write_text(json.dumps(custom_tiers), encoding="utf-8")

        service = ScoringService.from_output_dir(tmp_path)
        result = service.score_applicant({"a": 2.0})
        assert result.risk_tier in {"LOW", "HIGH"}
