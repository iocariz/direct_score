from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types

import pandas as pd


_LAST_SAVED_DOCUMENT = None


class _FakeFont:
    def __init__(self):
        self.name = None
        self.size = None
        self.bold = False
        self.color = types.SimpleNamespace(rgb=None)


class _FakeRun:
    def __init__(self, text: str = ""):
        self.text = text
        self.bold = False
        self.font = _FakeFont()


class _FakeParagraph:
    def __init__(self, text: str = ""):
        self.alignment = None
        self.runs = []
        self.text = ""
        if text:
            self.add_run(text)

    def add_run(self, text: str = ""):
        run = _FakeRun(str(text))
        self.runs.append(run)
        self.text += str(text)
        return run


class _FakeTcPr:
    def makeelement(self, *_args, **_kwargs):
        return object()

    def append(self, _child):
        return None


class _FakeElement:
    def get_or_add_tcPr(self):
        return _FakeTcPr()


class _FakeCell:
    def __init__(self):
        self._element = _FakeElement()
        self._text = ""
        self.paragraphs = [_FakeParagraph()]

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = str(value)
        self.paragraphs = [_FakeParagraph(self._text)]


class _FakeRow:
    def __init__(self, cols: int):
        self.cells = [_FakeCell() for _ in range(cols)]


class _FakeTable:
    def __init__(self, rows: int, cols: int):
        self.rows = [_FakeRow(cols) for _ in range(rows)]
        self.alignment = None


class _FakeStyle:
    def __init__(self):
        self.font = _FakeFont()


class _FakeDocument:
    def __init__(self):
        self.styles = {"Normal": _FakeStyle()}
        self.paragraphs = []
        self.tables = []

    def add_heading(self, text: str = "", level: int = 0):
        paragraph = _FakeParagraph(str(text))
        self.paragraphs.append(paragraph)
        return paragraph

    def add_paragraph(self, text: str = "", style=None):
        paragraph = _FakeParagraph(str(text))
        self.paragraphs.append(paragraph)
        return paragraph

    def add_table(self, rows: int, cols: int, style=None):
        table = _FakeTable(rows, cols)
        self.tables.append(table)
        return table

    def add_picture(self, path: str, width=None):
        paragraph = _FakeParagraph(str(path))
        self.paragraphs.append(paragraph)
        return paragraph

    def save(self, path):
        global _LAST_SAVED_DOCUMENT
        _LAST_SAVED_DOCUMENT = self
        Path(path).write_text("fake-docx")


class _FakeRGBColor:
    def __init__(self, *args):
        self.args = args


def _install_fake_docx(monkeypatch):
    fake_docx = types.ModuleType("docx")
    fake_shared = types.ModuleType("docx.shared")
    fake_enum = types.ModuleType("docx.enum")
    fake_enum_text = types.ModuleType("docx.enum.text")
    fake_enum_table = types.ModuleType("docx.enum.table")
    fake_oxml = types.ModuleType("docx.oxml")
    fake_oxml_ns = types.ModuleType("docx.oxml.ns")

    fake_docx.Document = _FakeDocument
    fake_shared.Inches = lambda value: value
    fake_shared.Pt = lambda value: value
    fake_shared.Cm = lambda value: value
    fake_shared.RGBColor = _FakeRGBColor
    fake_enum_text.WD_ALIGN_PARAGRAPH = types.SimpleNamespace(CENTER="center")
    fake_enum_table.WD_TABLE_ALIGNMENT = types.SimpleNamespace(CENTER="center")
    fake_oxml_ns.qn = lambda value: value

    monkeypatch.setitem(sys.modules, "docx", fake_docx)
    monkeypatch.setitem(sys.modules, "docx.shared", fake_shared)
    monkeypatch.setitem(sys.modules, "docx.enum", fake_enum)
    monkeypatch.setitem(sys.modules, "docx.enum.text", fake_enum_text)
    monkeypatch.setitem(sys.modules, "docx.enum.table", fake_enum_table)
    monkeypatch.setitem(sys.modules, "docx.oxml", fake_oxml)
    monkeypatch.setitem(sys.modules, "docx.oxml.ns", fake_oxml_ns)


class TestGenerateReport:
    def test_generate_report_handles_ebm_as_recommended_model(self, tmp_path, monkeypatch):
        _install_fake_docx(monkeypatch)
        sys.modules.pop("generate_report", None)
        generate_report_module = importlib.import_module("generate_report")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        pd.DataFrame(
            [
                {"Model": "EBM", "ROC AUC": 0.64, "Gini": 0.28, "KS": 0.19, "PR AUC": 0.11, "Brier": 0.18, "N": 120},
                {"Model": "Logistic Regression", "ROC AUC": 0.63, "Gini": 0.26, "KS": 0.18, "PR AUC": 0.10, "Brier": 0.19, "N": 120},
                {"Model": "LightGBM", "ROC AUC": 0.62, "Gini": 0.24, "KS": 0.17, "PR AUC": 0.09, "Brier": 0.20, "N": 120},
                {"Model": "risk_score_rf (benchmark)", "ROC AUC": 0.60, "Gini": 0.20, "KS": 0.16, "PR AUC": 0.08, "Brier": None, "N": 120},
                {"Model": "score_RF (benchmark)", "ROC AUC": 0.57, "Gini": 0.14, "KS": 0.13, "PR AUC": 0.06, "Brier": None, "N": 120},
            ]
        ).to_csv(output_dir / "results.csv", index=False)

        pd.DataFrame(
            [
                {
                    "model": "EBM",
                    "discrimination_score": 90.0,
                    "calibration_score": 82.0,
                    "stability_score": 79.0,
                    "generalization_score": 88.0,
                    "lift_score": 74.0,
                    "weighted_score": 84.0,
                    "recommended": True,
                },
                {
                    "model": "Logistic Regression",
                    "discrimination_score": 86.0,
                    "calibration_score": 80.0,
                    "stability_score": 78.0,
                    "generalization_score": 81.0,
                    "lift_score": 70.0,
                    "weighted_score": 80.0,
                    "recommended": False,
                },
            ]
        ).to_csv(output_dir / "model_selection.csv", index=False)

        pd.DataFrame(
            [
                {
                    "model": "EBM",
                    "train_auc": 0.69,
                    "test_auc": 0.64,
                    "auc_delta": 0.05,
                    "train_pr_auc": 0.15,
                    "test_pr_auc": 0.11,
                    "pr_auc_delta": 0.04,
                    "overfit_flag": "YES",
                },
                {
                    "model": "Logistic Regression",
                    "train_auc": 0.66,
                    "test_auc": 0.63,
                    "auc_delta": 0.03,
                    "train_pr_auc": 0.12,
                    "test_pr_auc": 0.10,
                    "pr_auc_delta": 0.02,
                    "overfit_flag": "NO",
                },
                {
                    "model": "LightGBM",
                    "train_auc": 0.82,
                    "test_auc": 0.62,
                    "auc_delta": 0.20,
                    "train_pr_auc": 0.25,
                    "test_pr_auc": 0.09,
                    "pr_auc_delta": 0.16,
                    "overfit_flag": "YES",
                },
            ]
        ).to_csv(output_dir / "overfit_report.csv", index=False)

        pd.DataFrame(
            [
                {
                    "candidate_model": "EBM",
                    "reference_model": "score_RF (benchmark)",
                    "auc_improvement": 0.07,
                    "auc_improvement_lo": 0.03,
                    "auc_improvement_hi": 0.10,
                    "auc_delong_p_value": 0.01,
                    "pr_auc_improvement": 0.03,
                    "n_pos": 12,
                },
                {
                    "candidate_model": "EBM",
                    "reference_model": "risk_score_rf (benchmark)",
                    "auc_improvement": 0.04,
                    "auc_improvement_lo": 0.00,
                    "auc_improvement_hi": 0.07,
                    "auc_delong_p_value": 0.08,
                    "pr_auc_improvement": -0.01,
                    "n_pos": 12,
                },
            ]
        ).to_csv(output_dir / "benchmark_comparisons.csv", index=False)

        pd.DataFrame(
            [
                {"split": "pre_split", "status_name": "Booked", "n_rows": 300, "n_bad_observed": 18, "date_start": "2023-09-01", "date_end": "2024-06-01"},
                {"split": "post_split", "status_name": "Booked", "n_rows": 120, "n_bad_observed": 12, "date_start": "2024-07-01", "date_end": "2025-01-01"},
                {"split": "post_split", "status_name": "Rejected", "n_rows": 80, "n_bad_observed": 0, "date_start": "2024-07-01", "date_end": "2025-01-01"},
                {"split": "post_split", "status_name": "Canceled", "n_rows": 40, "n_bad_observed": 0, "date_start": "2024-07-01", "date_end": "2025-01-01"},
            ]
        ).to_csv(output_dir / "population_summary.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "decile": 1, "n_accounts": 12, "n_defaults": 4, "default_rate": 0.33, "cum_default_rate": 0.33, "lift": 3.3, "cum_lift": 3.3, "capture_rate": 0.33},
                {"model": "EBM", "decile": 2, "n_accounts": 12, "n_defaults": 3, "default_rate": 0.25, "cum_default_rate": 0.29, "lift": 2.5, "cum_lift": 2.9, "capture_rate": 0.58},
                {"model": "EBM", "decile": 10, "n_accounts": 12, "n_defaults": 0, "default_rate": 0.00, "cum_default_rate": 0.10, "lift": 0.0, "cum_lift": 1.0, "capture_rate": 1.0},
            ]
        ).to_csv(output_dir / "lift_table.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "reject_pct": 10.0, "n_rejected": 12, "precision": 0.25, "recall": 0.42, "capture_rate": 0.42},
                {"model": "EBM", "reject_pct": 20.0, "n_rejected": 24, "precision": 0.21, "recall": 0.58, "capture_rate": 0.58},
            ]
        ).to_csv(output_dir / "threshold_analysis.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "psi": 0.03},
                {"model": "Logistic Regression", "psi": 0.02},
            ]
        ).to_csv(output_dir / "psi.csv", index=False)

        pd.DataFrame(
            [
                {"Model": "EBM", "n_folds": 3, "mean_ROC_AUC": 0.63, "std_ROC_AUC": 0.01, "mean_PR_AUC": 0.10, "std_PR_AUC": 0.01},
            ]
        ).to_csv(output_dir / "rolling_oot_summary.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "pr_auc_first": 0.10, "pr_auc_last": 0.09, "pr_auc_slope_per_fold": -0.005, "concept_drift_flag": "NO"},
            ]
        ).to_csv(output_dir / "concept_drift.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "pearson_corr": 0.12, "spearman_corr": 0.10, "selection_bias_flag": "LOW"},
            ]
        ).to_csv(output_dir / "selection_bias_correlation.csv", index=False)

        pd.DataFrame(
            [
                {"model": "EBM", "age_band": "18-25", "n": 30, "observed_default_rate": 0.08, "mean_predicted_pd": 0.12, "approval_rate_at_10pct_reject": 0.84, "adverse_impact_ratio": 0.84, "air_flag": "PASS"},
                {"model": "EBM", "age_band": "45-55", "n": 40, "observed_default_rate": 0.03, "mean_predicted_pd": 0.07, "approval_rate_at_10pct_reject": 1.00, "adverse_impact_ratio": 1.00, "air_flag": "PASS"},
            ]
        ).to_csv(output_dir / "adverse_impact_age.csv", index=False)

        pd.DataFrame(
            [
                {"feature": "INSTALLMENT_TO_INCOME", "iv": 0.18},
                {"feature": "AGE_T1", "iv": 0.09},
            ]
        ).to_csv(output_dir / "iv_summary.csv", index=False)

        report_path = tmp_path / "report.docx"
        generate_report_module.generate_report(output_dir=str(output_dir), report_path=str(report_path))

        assert report_path.exists()

        doc = _LAST_SAVED_DOCUMENT
        paragraph_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        table_text = "\n".join(
            cell.text
            for table in doc.tables
            for row in table.rows
            for cell in row.cells
        )
        full_text = f"{paragraph_text}\n{table_text}"

        assert "The recommended production model is EBM" in full_text
        assert "Five candidate architectures are trained" in full_text
        assert "Sigmoid (Platt) calibration is retained for Logistic Regression and EBM" in full_text
        assert "Four candidate architectures" not in full_text
