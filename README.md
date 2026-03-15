# Direct Score — Basel Bad Classification

Binary classification pipeline predicting `basel_bad` (12-month default flag) for consumer credit applications.

## Problem

Predict whether a booked credit application will default within 12 months (Basel II definition). The model replaces / complements existing risk scores (`risk_score_rf`, `score_RF`) used in production.

**Key constraints:**
- Target requires 12-month observation window to mature — accounts booked after 2025-01 have no observable outcome
- Only booked accounts are modeled (rejected/canceled excluded to avoid selection bias from unobserved outcomes)
- SCRPLUST1 is excluded from all modelling (direct features and interactions)
- Severe class imbalance: ~2.6% default rate (2,662 defaults out of 102,115 booked accounts)

## Data

| Split | Period | Rows | Default Rate |
|-------|--------|------|-------------|
| Train | 2023-01 to 2024-06 | 50,782 | 3.81% |
| Test | 2024-07 to 2025-01 | 16,109 | — |
| Excluded (immature) | 2025-02+ | — | N/A |

Source: `data/demand_direct.parquet` (2,386,685 rows raw; 102,115 after booked-only filter).

## Methodology

### 1. Feature Engineering

Three tiers of engineered features on top of 16 numerical and 12 categorical raw inputs:

- **Tier 1 — Ratios & flags:** income ratio (T2/T1), income change, product counts
- **Tier 2 — Affordability:** installment-to-income, total-amount-to-income, amount per month
- **Tier 3 — Categorical interactions:** product type x house type, product type x customer type, etc.
- **Missing indicators:** binary flags for columns with >1% missingness (MAX_CREDIT_TJ_AV, ESTCLI1, ESTCLI2)

### 2. Automated Interaction Screening

Brute-force screening of all pairwise combinations on training data only:
- Numerical pairs: ratio (A/B) and product (A*B) — 120 pairs screened
- Categorical pairs: concatenation — 66 pairs screened
- Kept interactions with >= 1% AUC lift over the best parent feature
- Result: 63 interactions added (22 numerical, 41 categorical)

### 3. Feature Selection

Recursive Feature Elimination with Cross-Validation (RFECV) using a lightweight LightGBM estimator:
- Reduced from 108 candidate features to ~47 optimal features
- Scoring: Average Precision (PR AUC)
- CV: StratifiedKFold, 5 splits

### 4. Preprocessing

| Feature Type | Imputation | Encoding |
|-------------|-----------|---------|
| Numerical | Median | StandardScaler |
| Categorical | Constant (`"missing"`) | TargetEncoder (smooth=auto, cv=5) |

Cardinality reduction: top-20 categories kept per feature; rest grouped as `"Other"`.

LightGBM uses a separate preprocessor with OrdinalEncoder (native categorical handling).

### 5. Models

| Model | Tuning | Notes |
|-------|--------|-------|
| Logistic Regression | GridSearchCV (C, l1_ratio) | `class_weight='balanced'`, saga solver |
| LightGBM | Optuna (50 trials) | `scale_pos_weight`, native categoricals |
| XGBoost | Optuna (50 trials) | `scale_pos_weight`, TargetEncoder preprocessing |
| Stacking Ensemble | — | LR + LightGBM + XGBoost base; LR meta-learner |

All models optimized on Average Precision (PR AUC) via StratifiedKFold CV.

### 6. Temporal Validation

Train/test split is strictly temporal (`mis_Date < 2024-07-01` for train), not random. This prevents look-ahead bias and simulates real deployment conditions.

## Results

Test set evaluation (accounts booked 2024-07 to 2025-01):

| Model | ROC AUC | PR AUC | Brier |
|-------|---------|--------|-------|
| **Stacking** | **0.6628** | **0.0669** | **0.0361** |
| risk_score_rf (benchmark) | 0.6613 | 0.0879 | — |
| XGBoost | 0.6612 | 0.0658 | 0.1948 |
| LightGBM | 0.6555 | 0.0655 | 0.1262 |
| Logistic Regression | 0.6550 | 0.0652 | 0.2279 |
| score_RF (benchmark) | 0.6185 | 0.0538 | — |

**Key observations:**
- The stacking ensemble achieves the highest ROC AUC (0.6628), marginally above the existing `risk_score_rf` benchmark (0.6613)
- The benchmark `risk_score_rf` has higher PR AUC (0.0879 vs 0.0669), suggesting it captures more of the positive class at low thresholds
- Calibration is poor for individual models (Brier 0.12–0.23); stacking's meta-learner provides better calibration (0.0361)
- All models substantially outperform `score_RF` on both ROC AUC and PR AUC

## Usage

```bash
# Install dependencies
uv sync

# Run training with defaults
uv run python training.py

# Custom options
uv run python training.py --data-path data/demand_direct.parquet --optuna-trials 100
```

## Project Structure

```
├── training.py          # Reproducible training pipeline
├── main.py              # Package entry point
├── pyproject.toml       # Dependencies (uv)
├── data/
│   └── demand_direct.parquet
├── notebooks/
│   ├── 1-eda.ipynb      # Exploratory data analysis
│   └── 3-model.ipynb    # Model development notebook (training.py source)
└── todo_list.md         # Known issues and improvement backlog
```
