# basel_bad PD model — run summary

_2026-04-30_

**Verdict:** ✅ Deploy  •  **Recommended:** `Logistic Regression (calibrated)`  •  **Weighted score:** 95.0/100

## Headline metrics (test, booked-proxy)

| Metric | Value |
|---|---|
| ROC AUC | 0.6673 |
| Gini | 0.3347 |
| KS | 0.2617 |
| PR AUC | 0.0648 |
| N | 6,835 |
| Calibrated Brier | 0.0366 |

## Benchmark comparison

| Reference | ΔAUC | 95% CI | adj. p | Outcome |
|---|---|---|---|---|
| risk_score_rf (benchmark) | -0.0029 | [-0.0495, +0.0392] | 1.000 | loss |
| score_RF (benchmark) | +0.0497 | [+0.0210, +0.0780] | 0.044 | win |

## Candidate ranking

| Model | Weighted | Disc. | Stab. | Calib. | Gener. | Lift | Floor |  |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 95.0 | 100.0 | 74.9 | 100.0 | 100.0 | 100.0 | ✅ | 🏆 |
| EBM | 79.7 | 68.5 | 100.0 | 59.8 | 100.0 | 78.6 | ✅ |  |
| XGBoost | 78.9 | 81.0 | 67.0 | 67.9 | 100.0 | 79.7 | ✅ |  |
| LightGBM | 56.7 | 70.1 | 37.1 | 0.0 | 100.0 | 64.7 | ✅ |  |
| CatBoost | 24.4 | 0.0 | 0.0 | 7.6 | 100.0 | 54.7 | ❌ |  |

## Watch-list

- PSI **HIGH DRIFT** on: CatBoost
- PSI moderate drift on: XGBoost
- Loses to `risk_score_rf` benchmark on PR AUC by 0.0255 — feature-ceiling issue, see methodology report §13

## Rolling out-of-time performance

| Model | Mean PR AUC | Std PR AUC | Mean ROC AUC | Folds |
|---|---|---|---|---|
| Logistic Regression (calibrated) | 0.0609 | 0.0077 | 0.6393 | 4 |
| EBM (calibrated) | 0.0641 | 0.0103 | 0.6487 | 4 |
| LightGBM (calibrated) | 0.0529 | 0.0071 | 0.6185 | 4 |
| XGBoost (calibrated) | 0.0519 | 0.0059 | 0.6151 | 4 |
| CatBoost (calibrated) | 0.0471 | 0.0017 | 0.5863 | 4 |
| risk_score_rf (benchmark) | 0.0841 | 0.0093 | 0.6587 | 4 |
| score_RF (benchmark) | 0.0544 | 0.0051 | 0.6264 | 4 |

## Artifact integrity

- 11 model artifacts have SHA-256 checksums in `output/models/checksums.json`
- ScoringService refuses to load any model whose checksum does not match.

## Reference artifacts

- Methodology + results: `methodology_report.docx`
- Credit-committee brief: `executive_brief.docx`
- Operational quick-status report: `report.docx`
- Model card: `output/model_card.txt`
