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
- Correlation pruning removes near-duplicate signals before RFECV
- RFECV uses `TemporalExpandingCV` with average precision scoring

### 4. Preprocessing and models

| Model | Preprocessing | Tuning | Notes |
|-------|---------------|--------|-------|
| Logistic Regression | Median imputation + scaling + target encoding | Optuna on `C` | Strongest current candidate on the held-out booked proxy |
| LightGBM | Median imputation + ordinal encoding | Optuna | Native tree learner used for RFECV and one of the official candidates |
| XGBoost | Median imputation + scaling + target encoding | Optuna | Strongest candidate on mean rolling OOT ROC AUC |
| CatBoost | Median imputation + ordinal encoding | Optuna | Strongest candidate on mean rolling OOT PR AUC |
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
| **risk_score_rf (benchmark)** | **0.6613** | **0.3225** | **0.2428** | **0.0879** | -- |
| **Logistic Regression** | **0.6552** | **0.3105** | 0.2412 | **0.0641** | 0.2248 |
| XGBoost | 0.6403 | 0.2806 | 0.2200 | 0.0595 | 0.2233 |
| CatBoost | 0.6347 | 0.2695 | 0.2123 | 0.0577 | 0.2208 |
| LightGBM | 0.6319 | 0.2638 | 0.2134 | 0.0569 | 0.0364 |
| score_RF (benchmark) | 0.6185 | 0.2371 | 0.1842 | 0.0538 | -- |

Sigmoid calibration does not change ranking metrics, and it brings the calibrated model variants into a tight `0.0361` to `0.0363` Brier range in the saved `results.csv`.

### Paired benchmark comparisons

`benchmark_comparisons.csv` now stores paired model-vs-benchmark deltas rather than relying on overlap of marginal confidence intervals.

- **Logistic Regression vs `score_RF`**: AUC improvement `+0.0367` with 95% bootstrap interval `[+0.0190, +0.0555]`; DeLong p-value `0.000219`. PR AUC improvement is `+0.0104` with interval `[+0.0031, +0.0177]`.
- **XGBoost vs `score_RF`**: AUC improvement `+0.0218` with interval `[+0.0034, +0.0413]`; DeLong p-value `0.025961`. PR AUC lift is positive but less decisive.
- **No candidate model beats `risk_score_rf`** on the current booked-proxy evaluation. The closest candidate is Logistic Regression at `-0.0060` AUC relative to `risk_score_rf`, with interval `[-0.0347, +0.0232]`.

### Rolling out-of-time summary

Rolling OOT validation covers 4 forward windows on the booked proxy population:

| Model | Windows | Mean ROC AUC | Mean PR AUC |
|-------|---------|--------------|-------------|
| **risk_score_rf (benchmark)** | 4 | **0.6587** | **0.0841** |
| XGBoost | 4 | **0.6380** | 0.0606 |
| CatBoost | 4 | 0.6344 | **0.0639** |
| Logistic Regression | 4 | 0.6339 | 0.0576 |
| LightGBM | 4 | 0.6064 | 0.0508 |
| score_RF (benchmark) | 4 | 0.6264 | 0.0544 |

The ranking is directionally consistent with the held-out test: `risk_score_rf` remains strongest, Logistic Regression is best on the single held-out booked test, and XGBoost / CatBoost remain competitive once the view is widened to rolling OOT windows.

### Bootstrap confidence intervals

`confidence_intervals.csv` stores marginal bootstrap intervals for the current run:

| Model | AUC [95% CI] | PR AUC [95% CI] |
|-------|-------------|-----------------|
| risk_score_rf (benchmark) | 0.661 [0.636, 0.684] | 0.088 [0.075, 0.103] |
| Logistic Regression | 0.655 [0.634, 0.676] | 0.065 [0.058, 0.072] |
| XGBoost | 0.640 [0.619, 0.661] | 0.060 [0.055, 0.066] |
| CatBoost | 0.634 [0.612, 0.657] | 0.058 [0.052, 0.065] |
| LightGBM | 0.631 [0.610, 0.653] | 0.057 [0.052, 0.063] |
| score_RF (benchmark) | 0.619 [0.599, 0.640] | 0.054 [0.050, 0.061] |

### Underwriting population coverage

The underwriting-specific artifacts extend the pipeline beyond the booked holdout:

- `population_summary.csv` records the full decisioned population before and after the split, broken out into `booked` and `decisioned_non_booked`
- `applicant_scores_post_split.csv` contains scores for **1,180,746** post-split decisioned applications, including booked, rejected, and canceled statuses

## Limitations

- **Booked-proxy evaluation only:** even in underwriting mode, the measurable target exists only for booked accounts, so the headline metrics do not fully identify applicant-stage grant/decline performance
- **Selection bias is still unresolved:** reject inference is available only as an experimental option and is excluded from the official benchmark tables
- **Stacking is still not temporally clean:** the current stacking implementation depends on non-temporal folds and therefore remains experimental
- **Benchmark ceiling remains external:** `risk_score_rf` still leads on ROC AUC and PR AUC, likely because it reflects bureau or production information not fully reproduced by the current feature set
- **Feature drift still exists:** overall score PSI is stable, but several interaction features continue to show elevated CSI

## Next Steps

- Implement applicant-stage policy evaluation: approval-rate curves, bad-rate curves, and expected-loss / profit simulations
- Rebuild or remove stacking from any official path unless temporal out-of-fold predictions are used end to end
- Explore better selection-bias correction for underwriting evaluation, rather than treating reject inference as the only experimental option
- Add richer underwriting outputs such as score deciles, decline-threshold analysis, and benchmark-vs-model routing scenarios
- Continue closing the gap to `risk_score_rf`, especially where the benchmark benefits from external bureau-style information not present in the current training set

---

## SHAP Explainability

SHAP (TreeExplainer) analysis on the LightGBM model.

### Top Features by Importance

| Rank | Feature | mean |SHAP| | Type |
|------|---------|--------------|------|
| 1 | MAX_CREDIT_TJ_AV | 0.0190 | Bureau: max credit available |
| 2 | HOUSE_TYPE_x_ESTCLI1 | 0.0183 | Interaction: housing x client status |
| 3 | TOTAL_AMT_X_LEFT_TO_LIVE | 0.0134 | Interaction: loan amount x remaining life |
| 4 | FAMILY_SITUATION_x_ESTCLI2 | 0.0127 | Interaction: family x client status |
| 5 | PRODTYPE3_X_HOUSE | 0.0115 | Interaction: product type x housing |
| 6 | TENOR_DIV_MAX_CREDIT_TJ_AV | 0.0112 | Ratio: loan tenor / bureau capacity |
| 7 | FREQ_CPRO_x_CMAT | 0.0097 | Frequency: profession x material code (NEW) |
| 8 | product_type_3_x_ESTCLI1 | 0.0084 | Interaction: product x client status |
| 9 | LOG_INCOME_T1 | 0.0070 | Log income (NEW) |
| 10 | TOTAL_AMT_X_AGE_T1 | 0.0061 | Interaction: loan amount x age |

New features (frequency encoding, log transforms, group stats) contribute meaningfully: `FREQ_CPRO_x_CMAT` ranks #7, `LOG_INCOME_T1` ranks #9, and `AGE_T1_VS_HOUSE_TYPE` (group stat) also appears in the top 20.

### SHAP Beeswarm Plot

![SHAP Summary](output/plots/shap_summary.png)

**Reading the beeswarm:** Each dot is one test observation. Horizontal position = SHAP value (positive = pushes toward default, negative = away). Colour = feature value (pink = high, blue = low).

**Key patterns:**

- **MAX_CREDIT_TJ_AV:** Clear negative relationship -- high values (pink) cluster at negative SHAP (lower risk). Higher bureau credit capacity signals established creditworthiness. The steep gradient below ~1,000 shows that borrowers with minimal bureau history are at greatest risk.

- **HOUSE_TYPE_x_ESTCLI1:** Discrete clusters from categorical combinations. Certain housing/client-status pairs carry 2--3x the risk impact of others, showing strong segment-level risk differentiation.

- **TOTAL_AMT_X_LEFT_TO_LIVE:** Higher values (larger loans taken by older borrowers with less remaining working life) push toward higher default risk -- captures overextension in pre-retirement borrowers.

- **TENOR_DIV_MAX_CREDIT_TJ_AV:** High values (long tenors relative to bureau capacity) push toward higher risk. This bureau-utilization proxy captures borrowers whose loan commitment is disproportionate to their credit footprint.

- **LOG_INCOME_T1:** Higher income reduces risk, with a clear monotonic pattern. The log transform allows this feature to contribute meaningfully in the LightGBM model by compressing the heavy tail.

- **FREQ_CPRO_x_CMAT:** Applicants in rare profession/material-code combinations (low frequency) face higher risk -- captures niche segments that have less credit data and higher uncertainty.

### SHAP Dependence Plots

![SHAP Dependence](output/plots/shap_dependence.png)

The MAX_CREDIT_TJ_AV dependence (top-left) shows a sharp non-linear threshold: risk drops steeply between 0 and ~2,000, then flattens. The TENOR_DIV_MAX_CREDIT_TJ_AV plot (bottom-right) shows a clear positive relationship with an inflection around 1,000 -- borrowers whose tenor exceeds their bureau capacity are penalised.

---

## Feature Assessment (WoE / IV)

Information Value measures univariate predictive power:

| IV Range | Interpretation | Count |
|----------|---------------|-------|
| < 0.02 | Useless | 0 |
| 0.02 -- 0.10 | Weak | 14 |
| 0.10 -- 0.30 | Medium | 27 |
| 0.30 -- 0.50 | Strong | 0 |
| > 0.50 | Suspicious | 0 |

**Top 10 by Information Value:**

| Feature | IV | Strength |
|---------|-----|----------|
| PRODTYPE3_X_HOUSE | 0.191 | Medium |
| HOUSE_TYPE_x_ESTCLI1 | 0.181 | Medium |
| FAMILY_SITUATION_x_ESTCLI2 | 0.176 | Medium |
| product_type_3_x_ESTCLI1 | 0.164 | Medium |
| TENOR_DIV_MAX_CREDIT_TJ_AV | 0.139 | Medium |
| CODRAMA_DIV_MAX_CREDIT_TJ_AV | 0.138 | Medium |
| AGE_T1_DIV_MAX_CREDIT_TJ_AV | 0.138 | Medium |
| HOUSE_TYPE_x_CMAT | 0.136 | Medium |
| LEFT_TO_LIVE_DIV_MAX_CREDIT_TJ_AV | 0.132 | Medium |
| FREQ_product_type_3_x_ESTCLI2 | 0.128 | Medium |

No features reach "Suspicious" IV (>0.50). Categorical interactions dominate the top tier, followed by bureau-normalised ratios. Frequency-encoded features (`FREQ_product_type_3_x_ESTCLI2`, `FREQ_CPRO_x_CMAT`) deliver Medium IV, validating their inclusion.

---

## Stability Analysis (PSI / CSI)

### Score Stability (PSI)

| Model | PSI | Interpretation |
|-------|-----|---------------|
| Logistic Regression | 0.020 | Stable |
| LightGBM | 0.027 | Stable |
| CatBoost | 0.039 | Stable |
| XGBoost | 0.041 | Stable |
| Stacking | 0.038 | Stable |

All PSI < 0.10. Score distributions are stable between train and test periods.

### Feature Stability (CSI)

6 features exceed CSI 0.25 (high drift by the CSI threshold):

| Feature | CSI | Note |
|---------|-----|------|
| PRODTYPE3_X_HOUSE | 0.77 | Product mix shift |
| FREQ_product_type_3_x_CPRO | 0.60 | Derived from above |
| product_type_3_x_ESTCLI1 | 0.58 | Product mix shift |
| HOUSE_TYPE_x_ESTCLI1 | 0.49 | Housing/client shift |
| FAMILY_SITUATION_x_ESTCLI2 | 0.44 | Family/client shift |
| HOUSE_TYPE_x_CMAT | 0.42 | Housing/material shift |

These are all categorical interactions involving `product_type_3` or `HOUSE_TYPE`. The underlying category distributions shifted between train and test periods, likely reflecting changes in the product portfolio or applicant mix. Despite this, overall model PSI remains low -- the models are robust to these shifts because multiple redundant features compensate.

Dropping INCOME_T2 (and all derivatives) eliminated the 5 most extreme drifters from the previous run (CSI 1.0--1.6), substantially improving the stability profile.

---

## Score Distributions

![Test Score Distributions](output/plots/score_dist_test.png)

- **CatBoost:** Widest score spread (0.05--0.70) with the clearest class separation. The "Bad" distribution has a distinct rightward tail.
- **XGBoost:** Narrow range (0.38--0.50) with limited class overlap differentiation.
- **LightGBM:** Very narrow range (0.06--0.08), still underfitting despite widened search space.
- **Logistic Regression:** Widest single-model range but heavy class overlap.

---

## Usage

```bash
# Install dependencies
uv sync

# Run the default underwriting-mode pipeline
uv run python training.py

# Explicit booked-only monitoring mode
uv run python training.py --population-mode booked_monitoring

# Explicit underwriting mode with a custom output directory
uv run python training.py --population-mode underwriting --optuna-trials 100 --output-dir output_v2

# Experimental: enable reject inference
uv run python training.py --population-mode underwriting --reject-inference

# Experimental: enable stacking
uv run python training.py --population-mode underwriting --enable-experimental-stacking
```

### Output Artifacts

```
output/
  results.csv                          # Leaderboard for the current run
  results_experimental.csv             # Experimental-only leaderboard (optional)
  confidence_intervals.csv             # Bootstrap 95% CIs
  benchmark_comparisons.csv            # Paired candidate vs benchmark deltas
  benchmark_comparisons_experimental.csv  # Experimental paired comparisons (optional)
  feature_importance.csv               # Split importance (trees) / coefficients (LR)
  feature_provenance.csv               # Feature lineage + RFECV status
  ablation_results.csv                 # Phase 3 ablation summary
  rolling_oot_results.csv              # Fold-level rolling OOT metrics
  rolling_oot_summary.csv              # Model-level rolling OOT summary
  population_summary.csv               # Pre/post split underwriting population counts
  applicant_scores_post_split.csv      # Scores for post-split decisioned applications
  shap_values.csv                      # Per-observation SHAP values
  shap_importance.csv                  # Feature ranking by mean |SHAP|
  iv_summary.csv                       # Information Value per feature
  woe_detail.csv                       # Weight of Evidence per bin
  psi.csv                              # Population Stability Index per model
  csi.csv                              # Characteristic Stability Index per feature
  models/                              # Serialized model pipelines (joblib)
  plots/
    score_dist_test.png                # Test set score distributions by class
    score_dist_train.png               # Train set score distributions by class
    shap_summary.png                   # SHAP beeswarm plot
    shap_importance.png                # SHAP bar importance
    shap_dependence.png                # Top-6 SHAP dependence plots
```

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load booked-only or full decisioned population depending on `population_mode` |
| 2 | Build `population_summary.csv` and log the sample definition |
| 3 | Feature engineering (ratios, portfolio, log transforms, interactions, missing flags) |
| 4 | Split the booked development sample into feature-discovery and estimation windows |
| 5 | Interaction screening on the earlier discovery window only |
| 6 | Cardinality reduction, enhanced features, and correlation pruning |
| 7 | RFECV feature elimination with temporal CV |
| 8 | Logistic Regression (Optuna, temporal CV) |
| 9 | LightGBM (Optuna, temporal CV) |
| 10 | XGBoost (Optuna, temporal CV) |
| 11 | CatBoost (Optuna, temporal CV) |
| 12 | Stacking ensemble (experimental; disabled by default) |
| 13 | Temporal calibration on the latest pre-split booked block |
| 14 | Hold-out evaluation + paired benchmark comparisons |
| 15 | Rolling out-of-time validation |
| 16 | Underwriting applicant score-frame generation |
| 17 | SHAP, WoE / IV, and PSI / CSI diagnostics |
| 18 | Save official, experimental, and underwriting-specific artifacts |

## Project Structure

```
direct_score/
  training.py          # Reproducible training pipeline (source of truth)
  pyproject.toml       # Dependencies (uv)
  CLAUDE.md            # Development instructions
  todo_list.md         # Known issues and improvement backlog
  data/
    demand_direct.parquet
  notebooks/
    1-eda.ipynb        # Exploratory data analysis
    3-model.ipynb      # Model development notebook
  output/              # Generated artifacts (not committed)
  tests/               # Unit tests
```
