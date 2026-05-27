"""Build a Markdown summary of the latest training run.

Audience: developers, PR descriptions, Slack/email updates. Renders as
plain Markdown so it diff-reviews cleanly in git and pastes into any
chat tool.

Usage:
    uv run python scripts/build_run_summary.py
    uv run python scripts/build_run_summary.py --output run_summary.md
    # Pipe to clipboard for a PR description:
    uv run python scripts/build_run_summary.py --stdout | pbcopy
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val, digits: int = 4, signed: bool = False) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    sign = "+" if signed else ""
    return f"{float(val):{sign}.{digits}f}"


def _table_md(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def build_summary(output_dir: Path) -> str:
    results = _read_csv(output_dir / "results.csv")
    if results is not None:
        results = results.set_index(results.columns[0])
    sel = _read_csv(output_dir / "model_selection.csv")
    bench = _read_csv(output_dir / "benchmark_comparisons.csv")
    cal = _read_csv(output_dir / "calibration_diagnostics.csv")
    psi = _read_csv(output_dir / "psi.csv")
    rolling = _read_csv(output_dir / "rolling_oot_summary.csv")
    air = _read_csv(output_dir / "adverse_impact_age.csv")
    checksums = _read_json(output_dir / "models" / "checksums.json")

    rec = "Logistic Regression"
    score = float("nan")
    deploy_ok = False
    if sel is not None and not sel.empty:
        winners = sel.loc[sel.get("recommended", False).fillna(False).astype(bool)]
        if not winners.empty:
            row = winners.iloc[0]
            rec = str(row["model"])
            score = float(row["weighted_score"])
            deploy_ok = bool(row.get("meets_quality_floor", True))

    lines: list[str] = []
    verdict = "✅ Deploy" if deploy_ok else "🛑 Do not deploy"
    lines.append(f"# basel_bad PD model — run summary")
    lines.append("")
    lines.append(f"_{date.today().isoformat()}_")
    lines.append("")
    lines.append(f"**Verdict:** {verdict}  •  **Recommended:** `{rec} (calibrated)`  •  **Weighted score:** {score:.1f}/100")
    lines.append("")

    # Headline metrics
    rec_res = results.loc[rec] if (results is not None and rec in results.index) else None
    rec_cal = f"{rec} (calibrated)"
    rec_cal_res = results.loc[rec_cal] if (results is not None and rec_cal in results.index) else None
    if rec_res is not None:
        lines.append("## Headline metrics (test, booked-proxy)")
        lines.append("")
        rows = [
            ["ROC AUC", _fmt(rec_res.get("ROC AUC"), 4)],
            ["Gini", _fmt(rec_res.get("Gini"), 4)],
            ["KS", _fmt(rec_res.get("KS"), 4)],
            ["PR AUC", _fmt(rec_res.get("PR AUC"), 4)],
            ["N", f"{int(rec_res.get('N', 0)):,}"],
        ]
        if rec_cal_res is not None and not pd.isna(rec_cal_res.get("Brier")):
            rows.append(["Calibrated Brier", _fmt(rec_cal_res.get("Brier"), 4)])
        if cal is not None and not cal.empty:
            cal_row = cal.loc[cal["model"] == rec_cal]
            if not cal_row.empty:
                rows.append(["ECE", _fmt(cal_row.iloc[0]["ece"], 4)])
                rows.append(["Max Calibration Error", _fmt(cal_row.iloc[0]["mce"], 4)])
        lines.append(_table_md(rows, ["Metric", "Value"]))
        lines.append("")

    # Benchmark comparison
    if bench is not None and not bench.empty:
        cand = bench.loc[bench["candidate_model"] == rec]
        if not cand.empty:
            lines.append("## Benchmark comparison")
            lines.append("")
            rows = []
            for _, row in cand.iterrows():
                ref = str(row["reference_model"])
                lo = float(row["auc_improvement_lo"])
                imp = float(row["auc_improvement"])
                outcome = "win" if (imp > 0 and lo > 0) else ("tie" if imp >= 0 else "loss")
                rows.append([
                    ref,
                    _fmt(imp, 4, signed=True),
                    f"[{_fmt(lo, 4, signed=True)}, {_fmt(row['auc_improvement_hi'], 4, signed=True)}]",
                    _fmt(row["auc_p_adjusted"], 3),
                    outcome,
                ])
            lines.append(_table_md(rows, ["Reference", "ΔAUC", "95% CI", "adj. p", "Outcome"]))
            lines.append("")

    # Selection table
    if sel is not None and not sel.empty:
        lines.append("## Candidate ranking")
        lines.append("")
        cols = [c for c in ["model", "weighted_score", "discrimination_score", "stability_score",
                            "calibration_score", "generalization_score", "lift_score",
                            "meets_quality_floor", "recommended"] if c in sel.columns]
        rows = []
        for _, row in sel.iterrows():
            rows.append([
                str(row.get("model", "")),
                _fmt(row.get("weighted_score"), 1),
                _fmt(row.get("discrimination_score"), 1),
                _fmt(row.get("stability_score"), 1),
                _fmt(row.get("calibration_score"), 1),
                _fmt(row.get("generalization_score"), 1),
                _fmt(row.get("lift_score"), 1),
                "✅" if bool(row.get("meets_quality_floor", False)) else "❌",
                "🏆" if bool(row.get("recommended", False)) else "",
            ])
        lines.append(_table_md(rows, ["Model", "Weighted", "Disc.", "Stab.", "Calib.", "Gener.", "Lift", "Floor", ""]))
        lines.append("")

    # Watch-list
    watch: list[str] = []
    if psi is not None and not psi.empty:
        high = psi.loc[psi["psi"] >= 0.25]
        if not high.empty:
            watch.append(f"PSI **HIGH DRIFT** on: {', '.join(high['model'].astype(str).head(5))}")
        moderate = psi.loc[(psi["psi"] >= 0.10) & (psi["psi"] < 0.25)]
        if not moderate.empty:
            watch.append(f"PSI moderate drift on: {', '.join(moderate['model'].astype(str).head(5))}")
    if air is not None and not air.empty and "ratio_status" in air.columns:
        fails = air.loc[air["ratio_status"] == "FAIL"]
        if not fails.empty:
            failing_bands = sorted(set(map(str, fails.get("age_band", pd.Series(dtype=str)).tolist())))
            n_models = fails["model"].nunique() if "model" in fails.columns else 0
            watch.append(f"Adverse-impact ratio < 0.80 in age band(s) {failing_bands} for {n_models} model(s)")
    if bench is not None and not bench.empty:
        rec_vs_rf = bench.loc[(bench["candidate_model"] == rec) & (bench["reference_model"] == "risk_score_rf (benchmark)")]
        if not rec_vs_rf.empty and float(rec_vs_rf.iloc[0]["pr_auc_improvement"]) < 0:
            gap = float(rec_vs_rf.iloc[0]["pr_auc_improvement"])
            watch.append(f"Loses to `risk_score_rf` benchmark on PR AUC by {abs(gap):.4f} — feature-ceiling issue, see methodology report §13")

    if watch:
        lines.append("## Watch-list")
        lines.append("")
        for item in watch:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("## Watch-list")
        lines.append("")
        lines.append("- _No issues flagged in this run._")
        lines.append("")

    # Rolling OOT
    if rolling is not None and not rolling.empty:
        lines.append("## Rolling out-of-time performance")
        lines.append("")
        focus = ["Logistic Regression (calibrated)", "EBM (calibrated)", "LightGBM (calibrated)",
                 "XGBoost (calibrated)", "CatBoost (calibrated)",
                 "risk_score_rf (benchmark)", "score_RF (benchmark)"]
        rows = []
        for m in focus:
            r = rolling.loc[rolling["Model"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([
                m,
                _fmt(r.get("mean_PR_AUC"), 4),
                _fmt(r.get("std_PR_AUC"), 4),
                _fmt(r.get("mean_ROC_AUC"), 4),
                str(int(r.get("n_folds", 0))),
            ])
        lines.append(_table_md(rows, ["Model", "Mean PR AUC", "Std PR AUC", "Mean ROC AUC", "Folds"]))
        lines.append("")

    # Artifact integrity
    lines.append("## Artifact integrity")
    lines.append("")
    if checksums:
        n_models = len(checksums) if isinstance(checksums, dict) else 0
        lines.append(f"- {n_models} model artifacts have SHA-256 checksums in `output/models/checksums.json`")
    else:
        lines.append("- _No `checksums.json` found — verify `output/models/` was produced by a recent training run._")
    lines.append("- ScoringService refuses to load any model whose checksum does not match.")
    lines.append("")

    # Pointers
    lines.append("## Reference artifacts")
    lines.append("")
    lines.append("- Methodology + results: `methodology_report.docx`")
    lines.append("- Credit-committee brief: `executive_brief.docx`")
    lines.append("- Operational quick-status report: `report.docx`")
    lines.append("- Model card: `output/model_card.txt`")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output", help="Pipeline output directory")
    parser.add_argument("--summary", default="run_summary.md", help="Path to write the markdown summary")
    parser.add_argument("--stdout", action="store_true", help="Also write to stdout")
    args = parser.parse_args(argv)

    text = build_summary(Path(args.output_dir))
    Path(args.summary).write_text(text, encoding="utf-8")
    print(f"Saved run summary: {args.summary}")
    if args.stdout:
        sys.stdout.write("\n" + text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
