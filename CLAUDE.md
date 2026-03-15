# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

Binary classification pipeline predicting `basel_bad` (12-month default flag) for consumer credit. Training script in `training.py`, developed from `notebooks/3-model.ipynb`.

## Running

```bash
uv sync                    # install deps
uv run python training.py  # run training pipeline
```

## Key Rules

- **SCRPLUST1 must be excluded** from all modelling — direct features and interactions. It is already in `DROP_COLS`.
- **`basel_bad` target needs 12 months** on book to mature. Only use data with `mis_Date <= 2025-01`.
- **Temporal split only** — never use random train/test split. Train: `mis_Date < 2024-07-01`, test: `mis_Date >= 2024-07-01`.
- **Only booked accounts** (`status_name == 'Booked'`) — rejected/canceled have no observed outcome.

## Architecture

- `training.py` — Full pipeline: load → feature engineering → interaction screening (LOO target encoding) → RFECV (OrdinalEncoder, no target leakage) → calibration holdout split → model training (LR, LightGBM, XGBoost, Stacking) with early stopping → isotonic calibration → evaluation
- `notebooks/3-model.ipynb` — Development notebook. When modifying the pipeline, update `training.py` (the source of truth)
- `data/demand_direct.parquet` — Input data (not committed)
- `product_type_1` and `acct_booked_H0` are single-value columns — excluded via `DROP_COLS`

## Known Limitations

- Stacking LGBM uses TargetEncoder (not OrdinalEncoder + native categoricals) because `StackingClassifier` doesn't support per-estimator fit_params. Documented in code.
- See `todo_list.md` for remaining improvement backlog (SHAP, PSI/CSI, monotonicity constraints, etc.).

## Dependencies

Managed via `uv` and `pyproject.toml`. Key: scikit-learn, lightgbm, xgboost, optuna, pandas, loguru.
