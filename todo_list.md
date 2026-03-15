# Training Pipeline — Review Findings & TODO

## Critical Bugs

- [x] **LR grid search ignores `l1_ratio` — missing `penalty='elasticnet'`**
  Fixed: added `penalty="elasticnet"` to LR in both `train_logistic_regression` and `train_stacking`. Also added `l1_ratio=0.5` to the grid.

- [x] **Target leakage in RFECV**
  Fixed: `run_rfecv` now builds its own OrdinalEncoder-based preprocessor internally (no TargetEncoder, so no target leakage when pre-fitting on all data).

- [x] **Target leakage in categorical interaction screening**
  Fixed: replaced in-sample `groupby().transform("mean")` with leave-one-out target encoding (`_loo_target_encode`) for both base AUC computation and categorical pair screening.

- [x] **Stacking LightGBM uses wrong preprocessor**
  Documented as known limitation: StackingClassifier does not support per-estimator fit_params, so stacking LGBM uses TargetEncoder (generic preprocessor) instead of OrdinalEncoder + native categoricals. The meta-learner compensates.

## Methodological Issues

- [x] **No early stopping for tree models**
  Fixed: Optuna objectives now use manual CV with early stopping (n_estimators ceiling=2000, early_stopping_rounds=50). Median best_iteration across folds is tracked and used for the final model fit.

- [x] **Poor probability calibration**
  Fixed: added `CalibratedClassifierCV(cv="prefit", method="isotonic")` calibration step using a 15% held-out calibration set for LR, LightGBM, and XGBoost. Both calibrated and uncalibrated versions are evaluated.

- [x] **Interaction sample naming is misleading**
  Fixed: removed misleading `pos`/`neg`/`sample` variables; code now uses `df_search` and `y_search` directly.

- [x] **`product_type_1` and `acct_booked_H0` not in interaction search**
  Fixed: added both to `DROP_COLS` — they are single-value columns with no discriminative power.

- [ ] **No monotonicity constraints for credit risk features**
  Features like INCOME, AGE, TENOR have business-expected monotonic relationships with default probability. LightGBM and XGBoost both support `monotone_constraints`. Not using them risks non-intuitive model behavior and regulatory challenge.

- [x] **CV folds mismatch between notebook and script**
  Script uses `n_splits=5` consistently. Notebook was the development version; `training.py` is the source of truth.

## Additional Features & Improvements

- [ ] **Add SHAP explainability**
  Compute SHAP values for the best tree model. Critical for regulatory explainability (IRB/IFRS9) and model governance. Plot summary, dependence, and force plots.

- [ ] **Add PSI / CSI stability metrics**
  Population Stability Index (score drift) and Characteristic Stability Index (feature drift) between train/test. Standard in credit risk model monitoring.

- [ ] **Add KS statistic and Gini coefficient**
  KS (max separation of cumulative distributions) and Gini (2 * AUC - 1) are standard credit risk discrimination metrics. Add to evaluation output.

- [ ] **Add confusion matrix at business thresholds**
  Compute precision/recall/F1 at relevant approval-rate cutoffs (e.g., top 5%, 10%, 20% risk). Show lift tables and capture rates.

- [ ] **Add WoE / IV analysis**
  Weight of Evidence binning and Information Value for individual feature assessment. Standard regulatory requirement for scorecard development.

- [ ] **Model persistence**
  Save trained models (`joblib.dump`) and preprocessing artifacts for deployment and reproducibility. Currently nothing is saved.

- [ ] **Experiment tracking**
  Integrate MLflow or similar to log hyperparameters, metrics, and artifacts across runs.

- [ ] **Reject inference**
  Only booked accounts are modeled. Rejected/canceled accounts (95.7% of data) are discarded. Consider reject inference techniques (e.g., parceling, augmentation, bivariate) to reduce selection bias.

- [ ] **Score distribution plots**
  Add score distribution histograms split by target class for train and test sets to visually assess discrimination and stability.

- [ ] **Feature importance export**
  Save feature importance rankings (split-based and gain-based for trees, coefficients for LR) to a CSV/JSON artifact for model documentation.

- [ ] **Add stratified bootstrap confidence intervals**
  Report confidence intervals for AUC, PR AUC, and Brier on the test set via bootstrap resampling. Point estimates alone are insufficient for model comparison with this sample size.
