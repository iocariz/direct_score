# Direct Score -- Basel Bad Classification

Binary classification pipeline predicting `basel_bad` (12-month default flag) for direct consumer-credit applications.

## Problem

The repo now supports two explicit population modes:

- **`booked_monitoring`**: booked-account monitoring and benchmark comparison on booked rows only
- **`underwriting`**: applicant-stage scoring for all decisioned applications (`Booked`, `Rejected`, `Canceled`)

The current default run is `population_mode=underwriting`. In that mode, the pipeline loads the full decisioned population, trains on booked rows with observed 12-month outcomes, saves applicant-stage scores for all post-split decisioned rows, and reports headline evaluation metrics on post-split booked matured rows only. Those metrics are tagged `evaluation_population=booked_proxy` in the saved artifacts.

**Key constraints:**
- The target requires a 12-month observation window to mature, so only booked rows provide a defensible observed default outcome
- Underwriting-mode evaluation is still an accepted-population proxy; rejected and canceled applications do not have counterfactual repayment outcomes
- Reject inference remains experimental and is excluded from the official summary tables
- Stacking remains experimental until it is rebuilt with temporally valid out-of-fold predictions
- `SCRPLUST1` is excluded from all modelling
- `INCOME_T2` and all derivatives are excluded due to extreme temporal drift
- Booked matured rows remain highly imbalanced at roughly 3.8% default rate

## Data

### Current underwriting-mode run

| Population | Period | Rows | Notes |
|------------|--------|------|-------|
| Booked development rows with observed target | 2023-01 to 2024-06 | 50,782 | Used for feature discovery, estimation, and temporal calibration |
| Held-out booked proxy test | 2024-07 to 2025-01 | 16,109 | 611 defaults, 3.80% default rate |
| Post-split decisioned scoring population | 2024-07 to 2026-02 | 1,180,746 | 50,735 booked, 980,972 rejected, 149,039 canceled |

Source: `data/demand_direct.parquet` (~2.4M raw rows). `population_summary.csv` records 1,205,939 pre-split and 1,180,746 post-split decisioned applications across the three underwriting statuses.

---

## Methodology

### 1. Population design and temporal samples

- `SPLIT_DATE=2024-07-01` separates development from the held-out period
- `MATURITY_CUTOFF=2025-01-01` limits official evaluation to booked rows with mature outcomes
- In `underwriting` mode the pipeline loads all decisioned applications, but supervised fitting still uses booked rows only
- The pre-split booked sample is split again into an earlier feature-discovery window and a later estimation window so the feature set is frozen before final fitting
- The latest pre-split booked block is reserved for temporal calibration
- Post-split booked matured rows drive the official metrics, while the full post-split decisioned population is saved as an applicant-stage score frame

### 2. Feature engineering

Engineered features are built on top of the raw numerical and categorical inputs:

- **Counts:** total products and portfolio-composition flags
- **Affordability / leverage ratios:** installment-to-income, amount-to-income, amount-per-month, and related capacity ratios
- **Log transforms:** income, loan amount, and bureau-capacity style variables where monotonic compression helps linear models
- **Categorical interactions:** targeted crosses such as product, housing, and customer-type combinations
- **Missingness indicators:** binary flags for fields with meaningful missing rates

### 3. Dedicated feature discovery workflow

Feature discovery is fully separated from final estimation:

- Pairwise interaction search runs only on the earlier pre-split discovery sample
- Numerical pairs are screened via ratios and products
- Categorical pairs are screened with cross-validated target encoding to reduce in-sample optimism
- Additional post-screening enrichments include frequency encoding and group-relative statistics computed from training data only
- Correlation pruning removes near-duplicate signals before the final selector
- Final feature elimination uses temporal stability selection across expanding folds
- Numericals are imputed and scaled while categoricals are target-encoded within each fold before elastic-net selection
- The helper still uses the historical name `run_rfecv(...)`, but the current method is no longer classical sklearn RFECV

### 4. Preprocessing and models

| Model | Preprocessing | Tuning | Notes |
|-------|---------------|--------|-------|
| Logistic Regression | Median imputation + scaling + target encoding | Optuna on `C` | Recommended model in the current weighted selection |
| LightGBM | Median imputation + ordinal encoding | Optuna | Strong raw hold-out candidate, but materially weaker generalization than Logistic Regression |
| XGBoost | Median imputation + scaling + target encoding | Optuna | Competitive discrimination, but unstable in the current generalization checks |
| CatBoost | Median imputation + ordinal encoding | Optuna | Best raw calibration score in model selection, but weakest held-out generalization in the current run |
| Stacking | Mixed base learners | None | Experimental only until temporal OOF stacking is implemented |

Cardinality reduction keeps the top categories per feature and groups the tail into `"Other"`.

### 5. Calibration

Probability calibration uses sigmoid (Platt) scaling on the latest booked pre-split holdout. This preserves ranking metrics while materially improving Brier score calibration for the model probabilities.

### 6. Benchmark comparison and validation

- Hold-out evaluation reports ROC AUC, Gini, KS, PR AUC, and Brier
- `confidence_intervals.csv` stores marginal bootstrap intervals
- `benchmark_comparisons.csv` stores paired candidate-vs-benchmark comparisons against `risk_score_rf` and `score_RF`
- `rolling_oot_results.csv` and `rolling_oot_summary.csv` store 4-window rolling out-of-time validation summaries on the booked proxy population

### 7. Underwriting-specific outputs

Underwriting mode adds two applicant-population artifacts:

- `population_summary.csv`: pre/post split row counts by status and population group
- `applicant_scores_post_split.csv`: benchmark scores plus model scores for all post-split decisioned applications

Official score tables, benchmark comparisons, rolling OOT summaries, and ablations are tagged with `population_mode` and `evaluation_population` so booked-proxy metrics are explicitly labeled.

### 8. Experimental modes

- **Reject inference:** score-anchored parceling using `risk_score_rf`; opt-in and excluded from the official benchmark tables
- **Stacking:** still excluded from the official path because the current implementation relies on non-temporal folds for meta-learner training

---

## Results

### Official underwriting-mode benchmark run

The latest saved artifacts were generated with `population_mode=underwriting`. All headline metrics below are evaluated on the post-split booked matured holdout and explicitly tagged `evaluation_population=booked_proxy`.

Held-out booked proxy test: 16,109 booked rows from 2024-07 to 2025-01, with 611 observed defaults.

| Model | ROC AUC | Gini | KS | PR AUC | Brier |
|-------|---------|------|-----|--------|-------|
| **Logistic Regression** | **0.6668** | **0.3336** | **0.2598** | **0.0645** | 0.2235 |
| risk_score_rf (benchmark) | 0.6613 | 0.3225 | 0.2428 | 0.0879 | -- |
| LightGBM | 0.6533 | 0.3066 | 0.2267 | 0.0638 | 0.1090 |
| XGBoost | 0.6528 | 0.3056 | 0.2310 | 0.0629 | 0.1025 |
| CatBoost | 0.6310 | 0.2619 | 0.2015 | 0.0586 | 0.0631 |
| score_RF (benchmark) | 0.6185 | 0.2371 | 0.1842 | 0.0538 | -- |

Sigmoid calibration does not change ranking metrics, and it brings the calibrated model variants into a tight `0.0361` to `0.0363` Brier range in the saved `results.csv`.

### Weighted model selection

`model_selection.csv` ranks only the official candidate models on discrimination, calibration, stability, generalization, and benchmark lift. The current recommendation is **Logistic Regression**.

| Model | Disc. | Calib. | Stab. | Gen. | Lift | Overall | Recommended |
|-------|------:|-------:|------:|-----:|-----:|--------:|-------------|
| Logistic Regression | 100.0 | 0.0 | 100.0 | 73.7 | 100.0 | 81.0 | Yes |
| LightGBM | 89.0 | 71.4 | 80.7 | 0.0 | 82.9 | 70.5 | No |
| XGBoost | 73.3 | 75.4 | 11.6 | 0.0 | 82.2 | 51.6 | No |
| CatBoost | 0.0 | 100.0 | 0.0 | 0.0 | 54.4 | 23.2 | No |

The tree models recover stronger raw calibration scores than Logistic Regression, but the recommendation favors Logistic Regression because it combines the best booked-proxy discrimination, the best stability profile, the cleanest generalization, and the strongest lift versus `score_RF`.

### Paired benchmark comparisons

`benchmark_comparisons.csv` now stores paired model-vs-benchmark deltas rather than relying on overlap of marginal confidence intervals.

- **Logistic Regression vs `score_RF`**: AUC improvement `+0.0482` with 95% bootstrap interval `[+0.0308, +0.0672]`; DeLong p-value `0.000001`. PR AUC improvement is `+0.0107` with interval `[+0.0034, +0.0181]`.
- **Logistic Regression vs `risk_score_rf`**: AUC improvement `+0.0054` with interval `[-0.0236, +0.0349]`; DeLong p-value `0.720479`. PR AUC is materially lower at `-0.0234`, so the current candidate does not beat the stronger benchmark on event concentration.
- **LightGBM and XGBoost** also beat `score_RF` on ROC AUC, but both remain behind Logistic Regression on the weighted recommendation once stability and generalization are included.

### Rolling out-of-time summary

Rolling OOT validation covers 4 forward windows on the booked proxy population:

| Model | Windows | Mean ROC AUC | Mean PR AUC |
|-------|---------|--------------|-------------|
| **risk_score_rf (benchmark)** | 4 | **0.6587** | **0.0841** |
| Logistic Regression | 4 | 0.6384 | 0.0590 |
| LightGBM | 4 | 0.6212 | 0.0579 |
| XGBoost | 4 | 0.6051 | 0.0536 |
| CatBoost | 4 | 0.5968 | 0.0528 |
| score_RF (benchmark) | 4 | 0.6264 | 0.0544 |

The rolling view is directionally consistent with the hold-out test: `risk_score_rf` remains the strongest benchmark, while **Logistic Regression** is the most stable in-house candidate across both the single hold-out sample and the rolling OOT windows.

### Overfitting diagnostics

`overfit_report.csv` flags all four official models at the current threshold, but the magnitude differs sharply:

| Model | Train AUC | Test AUC | AUC Δ | Train PR AUC | Test PR AUC | PR Δ |
|-------|----------:|---------:|------:|-------------:|------------:|-----:|
| Logistic Regression | 0.7005 | 0.6668 | +0.0337 | 0.0727 | 0.0645 | +0.0082 |
| LightGBM | 0.9907 | 0.6533 | +0.3374 | 0.7654 | 0.0638 | +0.7015 |
| XGBoost | 0.9927 | 0.6528 | +0.3399 | 0.8151 | 0.0629 | +0.7522 |
| CatBoost | 1.0000 | 0.6310 | +0.3690 | 1.0000 | 0.0586 | +0.9414 |

This explains why the multi-criteria selection penalizes the tree models heavily even when some held-out point estimates remain competitive.

### Bootstrap confidence intervals

`confidence_intervals.csv` stores marginal bootstrap intervals for the current run:

| Model | AUC [95% CI] | PR AUC [95% CI] |
|-------|-------------|-----------------|
| Logistic Regression | 0.667 [0.647, 0.687] | 0.064 [0.059, 0.072] |
| risk_score_rf (benchmark) | 0.661 [0.636, 0.684] | 0.088 [0.075, 0.103] |
| LightGBM | 0.653 [0.632, 0.672] | 0.064 [0.058, 0.072] |
| XGBoost | 0.653 [0.633, 0.675] | 0.063 [0.058, 0.072] |
| CatBoost | 0.631 [0.609, 0.653] | 0.059 [0.053, 0.067] |
| score_RF (benchmark) | 0.619 [0.599, 0.640] | 0.054 [0.050, 0.061] |

### Underwriting population coverage

The underwriting-specific artifacts extend the pipeline beyond the booked holdout:

- `population_summary.csv` records the full decisioned population before and after the split, broken out into `booked` and `decisioned_non_booked`
- `applicant_scores_post_split.csv` contains scores for **1,180,746** post-split decisioned applications, including booked, rejected, and canceled statuses

## Limitations

- **Booked-proxy evaluation only:** even in underwriting mode, the measurable target exists only for booked accounts, so the headline metrics do not fully identify applicant-stage grant/decline performance
- **Selection bias is still unresolved:** reject inference is available only as an experimental option and is excluded from the official benchmark tables
- **Stacking is still not temporally clean:** the current stacking implementation depends on non-temporal folds and therefore remains experimental
- **Tree ensembles remain overfit in the current setup:** all official candidates breach the overfit threshold, but the tree models do so by a much wider margin than Logistic Regression
- **Benchmark ceiling remains external:** Logistic Regression slightly edges `risk_score_rf` on held-out ROC AUC, but `risk_score_rf` still leads on PR AUC and on the rolling OOT benchmark view
- **Feature selection currently favors stable numeric surrogates:** raw categorical variables often give way to frequency-encoded or grouped numeric derivatives, which helps stability but can reduce direct interpretability of the original categories

## Next Steps

- Implement applicant-stage policy evaluation: approval-rate curves, bad-rate curves, and expected-loss / profit simulations
- Rebuild or remove stacking from any official path unless temporal out-of-fold predictions are used end to end
- Explore better selection-bias correction for underwriting evaluation, rather than treating reject inference as the only experimental option
- Add grouped or quota-based feature selection if retaining more raw categorical variables becomes a business or governance requirement
- Add richer underwriting outputs such as score deciles, decline-threshold analysis, and benchmark-vs-model routing scenarios
- Continue closing the gap to `risk_score_rf`, especially on PR AUC and benchmark event concentration where the current feature set still trails

---

## SHAP Explainability

SHAP is generated for the recommended model in `model_selection.csv`. In the latest run that selected model is **Logistic Regression**, so the saved SHAP artifacts reflect the recommendation rather than a hard-coded tree fallback.

### Top Features by Importance

| Rank | Feature | mean |SHAP| | Family |
|------|---------|--------------|--------|
| 1 | TENOR | 0.4203 | Raw numerical |
| 2 | LOG_TOTAL_AMT | 0.4198 | Log transform |
| 3 | LOG_MAX_CREDIT | 0.2877 | Log transform |
| 4 | MAX_CREDIT_TJ_AV | 0.2731 | Raw numerical |
| 5 | AGE_T1 | 0.2525 | Raw numerical |
| 6 | FREQ_product_type_3 | 0.2172 | Frequency encoding |
| 7 | FREQ_PRODTYPE3_X_HOUSE | 0.2043 | Frequency encoding |
| 8 | FREQ_ESTCLI1 | 0.2036 | Frequency encoding |
| 9 | INSTALLMENT_TO_HOUSEHOLD | 0.1920 | Ratio |
| 10 | FREQ_FAMILY_SITUATION_x_product_type_3 | 0.1804 | Frequency encoding |

The current ranking is dominated by credit-capacity scale, tenor, age, and several frequency-encoded surrogates for high-cardinality categorical segments. This is consistent with the current feature-selection workflow, which often keeps stable numeric derivatives of categorical information.

### SHAP Beeswarm Plot

![SHAP Summary](output/plots/shap_summary.png)

The beeswarm and bar plots should now be read as explanations of the **recommended Logistic Regression model**. In this run, the largest absolute contributions come from loan term, transformed amount and bureau-capacity variables, and frequency-encoded segment features.

### SHAP Dependence Plots

![SHAP Dependence](output/plots/shap_dependence.png)

The dependence plots are generated for the top-ranked features in the selected model. Use them to inspect how the signed SHAP contribution changes over the feature range for the current recommendation, not for a fixed LightGBM fallback.

---

## Feature Assessment (WoE / IV)

Information Value measures univariate predictive power on the development-fit sample:

| IV Range | Interpretation | Count |
|----------|---------------|-------|
| < 0.02 | Useless | 18 |
| 0.02 -- 0.10 | Weak | 41 |
| 0.10 -- 0.30 | Medium | 16 |
| 0.30 -- 0.50 | Strong | 0 |
| > 0.50 | Suspicious | 0 |

**Top 10 by Information Value:**

| Feature | IV | Strength |
|---------|-----|----------|
| AGE_T1_DIV_MAX_CREDIT_TJ_AV | 0.224 | Medium |
| FREQ_HOUSE_TYPE_x_FAMILY_SITUATION | 0.190 | Medium |
| LEFT_TO_LIVE_DIV_MAX_CREDIT_TJ_AV | 0.162 | Medium |
| MAX_CREDIT_TJ_AV | 0.154 | Medium |
| FREQ_ESTCLI2_x_FAMILY_SITUATION | 0.140 | Medium |
| FREQ_HOUSE_TYPE_x_ESTCLI2 | 0.130 | Medium |
| LOG_MAX_CREDIT | 0.128 | Medium |
| HOUSEHOLD_INCOME | 0.127 | Medium |
| INCOME_T1_X_AGE_T1 | 0.125 | Medium |
| LEFT_TO_LIVE_X_INSTALLMENT_AMT | 0.124 | Medium |

The current IV ranking is led by bureau-capacity ratios, affordability interactions, and frequency-encoded categorical surrogates rather than raw categorical crosses. No feature looks suspiciously predictive on its own.

---

## Stability Analysis (PSI / CSI)

### Score Stability (PSI)

| Model | PSI | Interpretation |
|-------|-----|---------------|
| Logistic Regression | 0.0036 | Stable |
| XGBoost | 0.0088 | Stable |
| CatBoost | 0.0093 | Stable |
| LightGBM | 0.0096 | Stable |

All PSI values are far below 0.10. Score distributions are very stable between the development-fit booked sample and the held-out booked proxy test.

### Feature Stability (CSI)

`csi.csv` contains 64 analyzed features in the current frozen set. None exceed CSI 0.10, so there are **0 high-drift** and **0 moderate-drift** features in the current run.

| Feature | CSI | Note |
|---------|-----|------|
| FREQ_CPRO_x_CMAT | 0.0697 | Highest CSI in the current run, still low |
| FREQ_CMAT_x_product_type_2 | 0.0656 | Low drift |
| FREQ_CPRO | 0.0346 | Low drift |
| FREQ_PRODTYPE3_X_HOUSE | 0.0328 | Low drift |
| LOG_TOTAL_AMT | 0.0318 | Low drift |
| TOTAL_AMT | 0.0318 | Low drift |

The current stability picture is much cleaner than earlier iterations: score PSI is low across all official models and the analyzed feature set shows no meaningful CSI alarms.

---

## Score Distributions

![Test Score Distributions](output/plots/score_dist_test.png)

The saved score-distribution plots remain useful as a qualitative diagnostic, but they should be interpreted together with `overfit_report.csv` and `model_selection.csv`. In the current run, the tree models show far larger train-to-test performance drops than Logistic Regression, which is one of the main reasons Logistic Regression remains the recommended model despite competitive raw hold-out AUC from LightGBM and XGBoost.

---

## Usage

```bash
# Install dependencies
uv sync

# Run the default underwriting-mode pipeline
uv run main.py

# Explicit booked-only monitoring mode
uv run main.py --population-mode booked_monitoring

# Explicit underwriting mode with a custom output directory
uv run main.py --population-mode underwriting --optuna-trials 100 --output-dir output_v2

# Experimental: enable reject inference
uv run main.py --population-mode underwriting --reject-inference

# Experimental: enable stacking
uv run main.py --population-mode underwriting --enable-experimental-stacking

# Regenerate stakeholder charts from the latest output directory
uv run stakeholder_charts.py --output-dir output
```

### Output Artifacts

```
output/
  results.csv                          # Leaderboard for the current run
  confidence_intervals.csv             # Bootstrap 95% CIs
  benchmark_comparisons.csv            # Paired candidate vs benchmark deltas
  model_selection.csv                  # Weighted model recommendation scorecard
  overfit_report.csv                   # Train-vs-test generalization diagnostics
  lift_table.csv                       # Decile-style lift summary by model
  threshold_analysis.csv               # Precision / recall / capture at operating cutoffs
  feature_importance.csv               # Split importance (trees) / coefficients (LR)
  feature_provenance.csv               # Feature lineage + stability-selection status
  interaction_leaderboard.csv          # Interaction search diagnostics
  feature_discovery_boundary.csv       # Discovery / estimation temporal cutoff
  ablation_results.csv                 # Phase 3 ablation summary
  rolling_oot_results.csv              # Fold-level rolling OOT metrics
  rolling_oot_summary.csv              # Model-level rolling OOT summary
  population_summary.csv               # Pre/post split underwriting population counts
  applicant_scores_post_split.csv      # Scores for post-split decisioned applications
  holdout_test_scores.csv              # Per-row scores on the booked proxy holdout
  shap_values.csv                      # Per-observation SHAP values
  shap_importance.csv                  # Feature ranking by mean |SHAP|
  iv_summary.csv                       # Information Value per feature
  woe_detail.csv                       # Weight of Evidence per bin
  psi.csv                              # Population Stability Index per model
  csi.csv                              # Characteristic Stability Index per feature
  model_card.txt                       # Governance summary for the current run
  variable_dictionary.csv              # Feature dictionary with provenance and IV
  data_quality_development_fit.csv     # Missingness / uniqueness checks on fit sample
  data_quality_test.csv                # Missingness / uniqueness checks on test sample
  models/                              # Serialized model pipelines (joblib)
  plots/
    score_dist_test.png                # Test set score distributions by class
    score_dist_train.png               # Train set score distributions by class
    shap_summary.png                   # SHAP beeswarm plot
    shap_importance.png                # SHAP bar importance
    shap_dependence.png                # Top-6 SHAP dependence plots
    stakeholder_*.png                  # Stakeholder chart pack outputs
```

## Pipeline Phases

| Phase | Description |
|------|-------------|
| 1 | Load the chosen population mode and build `population_summary.csv` |
| 2 | Engineer raw, ratio, log, interaction, and missingness features |
| 3 | Split the booked pre-test sample into feature-discovery and estimation windows |
| 4 | Run interaction search, enrichment, correlation pruning, and temporal stability selection |
| 5 | Freeze the feature set, split development vs calibration, and train the official candidate models |
| 6 | Calibrate official models on the latest booked pre-split holdout |
| 7 | Run rolling OOT validation, booked-proxy hold-out evaluation, and applicant-population scoring |
| 8 | Produce bootstrap comparisons, ablations, SHAP, WoE / IV, PSI / CSI, lift, threshold, and overfit diagnostics |
| 9 | Select the recommended model with the weighted multi-criteria scorecard |
| 10 | Save governance artifacts, stakeholder inputs, plots, and serialized models |

## Project Structure

```
direct_score/
  main.py               # Thin CLI entrypoint used by `uv run main.py`
  training.py           # Pipeline orchestration
  training_features.py  # Feature discovery, interaction search, and selection
  training_reporting.py # Evaluation, model selection, and artifact helpers
  stakeholder_charts.py # Stakeholder-facing chart pack generator
  model_governance.py   # Model card and variable dictionary generation
  pyproject.toml        # Dependencies (uv)
  CLAUDE.md             # Development instructions
  todo_list.md          # Known issues and improvement backlog
  data/
    demand_direct.parquet
  notebooks/
    1-eda.ipynb        # Exploratory data analysis
    3-model.ipynb      # Model development notebook
  output/              # Generated artifacts (not committed)
  tests/               # Unit tests
```
