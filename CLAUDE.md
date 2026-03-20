# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

Binary classification pipeline predicting `basel_bad` (12-month default flag) for consumer credit. Entrypoint: `uv run main.py`.

## Running

```bash
uv sync                    # install deps
uv run main.py             # run training pipeline (default: underwriting mode)
uv run pytest              # run 239 tests
uv run stakeholder_charts.py --output-dir output  # regenerate charts
```

## Key Rules

- **SCRPLUST1 must be excluded** from all modelling — direct features and interactions. It is already in `DROP_COLS`.
- **`basel_bad` target needs 12 months** on book to mature. Only use data with `mis_Date <= 2025-01`.
- **Temporal split only** — never use random train/test split. Train: `mis_Date < 2024-07-01`, test: `mis_Date >= 2024-07-01`.
- **Only booked accounts** (`status_name == 'Booked'`) for supervised training — rejected/canceled have no observed outcome.
- **No StratifiedKFold(shuffle=True)** anywhere in the pipeline — all CV is temporal.

## Architecture

- `training.py` — Pipeline orchestration: load → feature engineering → temporal stability selection → calibration holdout → model training (LR, LGBM, XGB, CatBoost, ensemble) → isotonic/sigmoid calibration → evaluation → diagnostics → governance
- `training_features.py` — Feature discovery: interaction search (ratio, product, binned num x cat), frequency encoding, group stats, stability selection
- `training_reporting.py` — Evaluation, bootstrap CIs, model selection, overfitting diagnostics, population bias analysis, lift/threshold tables
- `training_constants.py` — Shared constants (thresholds, feature lists, hyperparameter bounds)
- `stakeholder_charts.py` — Stakeholder chart pack with selected-model highlighting
- `model_governance.py` — Model card, variable dictionary, data quality reports
- `scoring.py` — Production scoring API (`ScoringService.from_output_dir()`)
- `data/demand_direct.parquet` — Input data (not committed)

## Known Limitations

- Stacking is experimental (non-temporal OOF predictions)
- Tree models still overfit significantly vs Logistic Regression despite tightened bounds
- See `todo_list.md` for the full improvement backlog

## Dependencies

Managed via `uv` and `pyproject.toml`. Key: scikit-learn, lightgbm, xgboost, catboost, optuna, pandas, loguru, shap, scipy.
