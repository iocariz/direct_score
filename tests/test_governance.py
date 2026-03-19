"""Tests for model governance artifacts: model card, variable dictionary, data quality."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_governance import (
    generate_data_quality_report,
    generate_model_card,
    generate_variable_dictionary,
)
from training_constants import MONOTONE_MAP, RAW_CAT, RAW_NUM


class TestGenerateModelCard:
    def test_creates_file(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82], "PR AUC": [0.15], "KS": [0.50], "Brier": [0.06], "N": [1000]},
            index=["Logistic Regression"],
        )
        path = generate_model_card(results_df, None, None, None, None, None, tmp_path)
        assert path.exists()
        assert path.name == "model_card.txt"

    def test_contains_key_sections(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82], "PR AUC": [0.15], "KS": [0.50], "Brier": [0.06], "N": [1000]},
            index=["Logistic Regression"],
        )
        path = generate_model_card(results_df, None, None, None, None, None, tmp_path)
        content = path.read_text()
        assert "MODEL OVERVIEW" in content
        assert "PERFORMANCE SUMMARY" in content
        assert "KNOWN LIMITATIONS" in content
        assert "TECHNICAL CONFIGURATION" in content

    def test_includes_recommendation_when_provided(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.85], "PR AUC": [0.18], "KS": [0.55], "Brier": [0.055], "N": [1000]},
            index=["LightGBM"],
        )
        selection_df = pd.DataFrame({
            "model": ["LightGBM"],
            "weighted_score": [78.5],
            "test_auc": [0.85],
            "test_pr_auc": [0.18],
            "test_brier_raw": [0.055], "test_brier_calibrated": [0.050],
            "recommended": [True],
        })
        path = generate_model_card(results_df, selection_df, None, None, None, None, tmp_path)
        content = path.read_text()
        assert "LightGBM" in content
        assert "78.5" in content

    def test_includes_overfit_assessment(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82], "PR AUC": [0.15], "KS": [0.50], "Brier": [0.06], "N": [1000]},
            index=["Logistic Regression"],
        )
        overfit_df = pd.DataFrame({
            "model": ["Logistic Regression"],
            "train_auc": [0.84], "test_auc": [0.82], "auc_delta": [0.02],
            "overfit_flag": ["NO"],
        })
        path = generate_model_card(results_df, None, overfit_df, None, None, None, tmp_path)
        content = path.read_text()
        assert "OVERFITTING ASSESSMENT" in content
        assert "No models show significant overfitting" in content

    def test_includes_executive_summary_for_recommended_model(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.85], "PR AUC": [0.18], "KS": [0.55], "Brier": [0.055], "N": [1000]},
            index=["LightGBM"],
        )
        selection_df = pd.DataFrame({
            "model": ["LightGBM"],
            "weighted_score": [78.5],
            "test_auc": [0.85],
            "test_pr_auc": [0.18],
            "test_brier_raw": [0.055], "test_brier_calibrated": [0.050],
            "recommended": [True],
        })

        path = generate_model_card(results_df, selection_df, None, None, None, None, tmp_path)
        content = path.read_text()

        assert "EXECUTIVE SUMMARY" in content
        assert "Recommended production candidate: LightGBM" in content
        assert "Headline findings:" in content

    def test_feature_inventory_uses_generic_selection_wording(self, tmp_path):
        results_df = pd.DataFrame(
            {"ROC AUC": [0.82], "PR AUC": [0.15], "KS": [0.50], "Brier": [0.06], "N": [1000]},
            index=["Logistic Regression"],
        )
        feature_provenance_df = pd.DataFrame(
            [
                {"feature": "INCOME_T1", "provenance": "raw", "data_type": "numerical", "rfecv_candidate": True, "rfecv_kept": True, "interaction_type": np.nan},
                {"feature": "CSP", "provenance": "raw", "data_type": "categorical", "rfecv_candidate": True, "rfecv_kept": False, "interaction_type": np.nan},
                {"feature": "INCOME_T1_DIV_TOTAL_LOAN_NBR", "provenance": "interaction", "data_type": "numerical", "rfecv_candidate": True, "rfecv_kept": True, "interaction_type": "ratio"},
            ]
        )

        path = generate_model_card(results_df, None, None, None, None, feature_provenance_df, tmp_path)
        content = path.read_text()

        assert "Selected modeling features" in content
        assert "temporal elastic-net stability selection" in content
        assert "RFECV-selected features" not in content


class TestGenerateVariableDictionary:
    def test_creates_csv(self, tmp_path):
        path = generate_variable_dictionary(
            ["INCOME_T1", "AGE_T1", "CSP"],
            ["INCOME_T1", "AGE_T1"],
            ["CSP"],
            None, None, tmp_path,
        )
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 3

    def test_includes_monotone_constraints(self, tmp_path):
        path = generate_variable_dictionary(
            ["INCOME_T1", "INSTALLMENT_TO_INCOME"],
            ["INCOME_T1", "INSTALLMENT_TO_INCOME"],
            [], None, None, tmp_path,
        )
        df = pd.read_csv(path)
        income_row = df.loc[df["feature"] == "INCOME_T1"].iloc[0]
        assert income_row["monotone_constraint"] == "decreasing"
        iti_row = df.loc[df["feature"] == "INSTALLMENT_TO_INCOME"].iloc[0]
        assert iti_row["monotone_constraint"] == "increasing"

    def test_includes_iv_when_provided(self, tmp_path):
        iv_df = pd.DataFrame({"feature": ["INCOME_T1", "CSP"], "iv": [0.35, 0.12]})
        path = generate_variable_dictionary(
            ["INCOME_T1", "CSP"], ["INCOME_T1"], ["CSP"], None, iv_df, tmp_path,
        )
        df = pd.read_csv(path)
        income_iv = df.loc[df["feature"] == "INCOME_T1", "information_value"].iloc[0]
        assert income_iv == pytest.approx(0.35)

    def test_column_schema(self, tmp_path):
        path = generate_variable_dictionary(["x"], ["x"], [], None, None, tmp_path)
        df = pd.read_csv(path)
        expected = {"feature", "type", "source", "monotone_constraint",
                    "information_value", "in_raw_num", "in_raw_cat", "in_monotone_map"}
        assert expected.issubset(df.columns)


class TestGenerateDataQualityReport:
    def test_creates_csv(self, tmp_path):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "num1": rng.uniform(0, 100, 200),
            "cat1": rng.choice(["A", "B", "C"], 200),
        })
        path = generate_data_quality_report(X, ["num1"], ["cat1"], tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_detects_missing_values(self, tmp_path):
        X = pd.DataFrame({
            "num1": [1.0, np.nan, 3.0, np.nan, 5.0],
        })
        path = generate_data_quality_report(X, ["num1"], [], tmp_path)
        df = pd.read_csv(path)
        assert df.iloc[0]["missing_count"] == 2
        assert df.iloc[0]["missing_pct"] == pytest.approx(0.4)

    def test_counts_outliers(self, tmp_path):
        X = pd.DataFrame({
            "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
        })
        path = generate_data_quality_report(X, ["num1"], [], tmp_path)
        df = pd.read_csv(path)
        assert df.iloc[0]["n_outliers_iqr"] >= 1

    def test_reports_cardinality_for_categoricals(self, tmp_path):
        X = pd.DataFrame({"cat1": ["A", "B", "C", "A", "B"]})
        path = generate_data_quality_report(X, [], ["cat1"], tmp_path)
        df = pd.read_csv(path)
        assert df.iloc[0]["n_unique"] == 3

    def test_custom_label(self, tmp_path):
        X = pd.DataFrame({"x": [1, 2, 3]})
        path = generate_data_quality_report(X, ["x"], [], tmp_path, label="test_set")
        assert "test_set" in path.name
