TARGET = "basel_bad"
MATURITY_CUTOFF = "2025-01-01"
SPLIT_DATE = "2024-07-01"
MODEL_SELECTION_DATE = "2024-10-01"
RANDOM_STATE = 42
MAX_CATEGORIES = 20
MIN_LIFT = 0.01
MIN_VALID = 5_000
N_ESTIMATORS_CEILING = 1000
EARLY_STOPPING_ROUNDS = 30
N_BOOTSTRAP = 1000
# Optuna MedianPruner warmup: minimum number of CV folds an Optuna trial must
# complete before the pruner is allowed to terminate it. Set to 3 because the
# first temporal fold is the smallest (earliest data block) and therefore the
# noisiest single estimate; pruning after fold 1 (the previous default) was
# killing configs that recovered on later folds. With max 5 temporal folds,
# warmup=3 means the pruner only acts on majority of fold evidence.
OPTUNA_PRUNER_WARMUP_STEPS = 3
OPTUNA_PRUNER_STARTUP_TRIALS = 5
# Minimum weighted_score (0-100) a candidate must clear to be recommended for
# production. weighted_score is min-max-scaled per dimension across the
# candidate set, which means one model always wins by being least-bad even if
# every candidate is poor in absolute terms. The floor adds an absolute
# quality gate: if no candidate clears it, no model is recommended and the
# pipeline emits a warning instead of silently endorsing a weak model.
# 50 = "the candidate is roughly in the middle of every criterion's
# achievable range" — a reasonable lower bar for a regulated PD model.
MODEL_SELECTION_QUALITY_FLOOR = 50.0
CALIBRATION_FRACTION = 0.15
# Minimum positive (basel_bad=1) count in the calibration holdout below which
# isotonic regression is unstable. The literature (Niculescu-Mizil & Caruana
# 2005, Foster & Stine 2004) recommends >= 100 positives for monotonic-fit
# calibrators on imbalanced targets. Below this threshold, sigmoid (Platt)
# calibration is more reliable than isotonic. We emit a warning rather than
# failing so portfolios that genuinely have low positive counts can still
# train — the warning is the regulator-visible signal.
CALIBRATION_MIN_POSITIVES = 100
FEATURE_DISCOVERY_FRACTION = 0.50
INTERACTION_SEARCH_TOP_K_NUM = 12
INTERACTION_SEARCH_TOP_K_CAT = 10
# Temporal stability selection (Meinshausen & Bühlmann 2010, adapted for
# time-series CV).
#
# THRESHOLD: a feature is kept if it is selected (non-zero coefficient) in at
# least 50% of temporal folds across the (C, l1_ratio) elastic-net grid. 50%
# is the literature-default cutoff and balances false-keep and false-drop
# rates symmetrically. Above 60% the selector becomes too conservative on
# small portfolios; below 40% it admits too many noisy features.
STABILITY_SELECTION_THRESHOLD = 0.50
# MIN_FEATURES: floor on the number of features the selector must keep, even
# if the THRESHOLD criterion would otherwise drop more. Calibrated for this
# portfolio's feature space (RAW_NUM has 14 numerics, RAW_CAT has 12
# categoricals, plus engineered + interaction + freq + group features →
# typically 60-100 candidates post-engineering, of which ~15-30 survive).
# A floor of 5 means:
#   - Below this the linear meta-model has too few degrees of freedom and
#     becomes unidentifiable when paired with the L2-regularised LR
#     downstream (effectively underfit).
#   - 5 represents ~5-10% of the candidate set — well below the typical
#     stability-selected count, so this floor only binds in degenerate
#     scenarios (e.g. portfolio with very few positives or near-collinear
#     candidates that all get dropped together).
#   - The post-fix-B feature contract is the UNION of stability and
#     tree-aware selectors, so even if stability returns the bare floor of
#     5, the tree-aware pass typically adds another ~10-15 features.
# Worth revisiting if the candidate set ever drops below 30 (e.g. a
# stripped-down portfolio with no interaction search).
STABILITY_SELECTION_MIN_FEATURES = 5
# C and l1_ratio grid for the elastic-net inside stability selection. C
# spans 2.5 orders of magnitude on a log scale (0.03 to 3.0) so the
# selector samples both heavy and light regularisation; l1_ratio mixes
# Lasso-like (0.8) with closer-to-Ridge (0.2) sparsity patterns to avoid
# pure-Lasso instability on correlated features.
STABILITY_SELECTION_C_VALUES = (0.03, 0.1, 0.3, 1.0, 3.0)
STABILITY_SELECTION_L1_RATIOS = (0.2, 0.5, 0.8)
# Tree-aware feature pass (LGBM permutation importance, temporal CV).
# A candidate feature is kept if its mean permutation importance across folds
# is at least this fraction of the most-important feature's mean importance.
# Final feature contract = LR-stability-selected ∪ tree-aware-selected.
TREE_AWARE_IMPORTANCE_RATIO = 0.05
TREE_AWARE_PERMUTATION_REPEATS = 5
ROLLING_OOT_MAX_WINDOWS = 4
POPULATION_MODE_BOOKED_MONITORING = "booked_monitoring"
POPULATION_MODE_UNDERWRITING = "underwriting"
UNDERWRITING_DECISION_STATUSES = ("Booked", "Rejected", "Canceled")
OFFICIAL_MODEL_NAMES = [
    "Logistic Regression",
    "EBM",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]
EXPERIMENTAL_STACKING_NAME = "Stacking (experimental)"

# Per-model calibration method.
#
# Choice follows Niculescu-Mizil & Caruana (2005), "Predicting Good Probabilities
# With Supervised Learning" (ICML 2005):
#   - Models that produce log-odds-like outputs (logistic regression, GAMs/EBM)
#     are already monotonically calibrated and benefit from sigmoid (Platt)
#     scaling: a 2-parameter fit that does not over-fit small calibration sets.
#   - Boosted trees and bagged trees produce sigmoid-shaped reliability
#     diagrams (over-confident at the extremes) and benefit from isotonic
#     regression — a non-parametric monotonic fit that corrects this shape
#     with enough calibration data.
#
# This is a literature-driven default, not a data-driven choice — there is no
# goodness-of-fit test on the calibration set that flips the method, so the
# calibration holdout is never used to select between methods (which would
# double-dip the same data).
CALIBRATION_METHOD_BY_MODEL: dict[str, str] = {
    "Logistic Regression": "sigmoid",
    "EBM": "sigmoid",
    "LightGBM": "isotonic",
    "XGBoost": "isotonic",
    "CatBoost": "isotonic",
    EXPERIMENTAL_STACKING_NAME: "isotonic",
}
DEFAULT_CALIBRATION_METHOD = "isotonic"
SUMMARY_MODEL_NAMES = OFFICIAL_MODEL_NAMES + [EXPERIMENTAL_STACKING_NAME]
BENCHMARK_MODEL_NAMES = [
    "risk_score_rf (benchmark)",
    "score_RF (benchmark)",
]

REJECT_SCORE_COL = "risk_score_rf"
REJECT_N_BINS = 10
REJECT_MULTIPLIER = 1.5
REJECT_MAX_RATIO = 1.0
REJECT_SAMPLE_WEIGHT = 0.5

DROP_COLS = [
    TARGET,
    "authorization_id",
    "mis_Date",
    "rf_business_name",
    "rf_ext_business_name",
    "a_business_name",
    "ext_business_name",
    "SCRPLUST1",
    "reject_reason",
    "status_name",
    "risk_score_rf",
    "score_RF",
    "product_type_1",
    "acct_booked_H0",
    "INCOME_T2",
]

RAW_NUM = [
    "CODRAMA", "TOTAL_AMT", "INSTALLMENT_AMT",
    "TOTAL_CARD_NBR", "TOTAL_LOAN_NBR", "BOOK_CARD_NBR", "BOOK_LOAN_NBR",
    "AGE_T1", "LEFT_TO_LIVE", "HOUSE_YEARS", "TENOR",
    "MAX_CREDIT_TJ_AV", "INCOME_T1", "INCIT1_L12", "flag_risk3",
]

RAW_CAT = [
    "CUSTOMER_TYPE", "FAMILY_SITUATION", "HOUSE_TYPE",
    "product_type_2", "product_type_3", "CSP", "CPRO", "CMAT",
    "ESTCLI1", "ESTCLI2", "CSECTOR", "FLAG_COTIT",
]

MISS_CANDIDATES = ["MAX_CREDIT_TJ_AV", "INCIT1_L12", "HOUSE_YEARS", "ESTCLI1", "ESTCLI2"]

DEFAULT_THRESHOLD_PCTS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# ── Diagnostic thresholds ──────────────────────────────────────────────────────
# PSI/CSI population-stability index cut-offs (industry-standard).
PSI_MODERATE_THRESHOLD = 0.10
PSI_HIGH_DRIFT_THRESHOLD = 0.25
# Train-vs-test AUC / PR-AUC delta beyond which a model is flagged as overfit.
OVERFIT_DELTA_THRESHOLD = 0.03
# Negative slope in rolling-OOT PR AUC that signals concept drift
# (first-vs-last delta must be at least this negative to flag).
CONCEPT_DRIFT_DELTA_THRESHOLD = -0.02

# ── Risk tier thresholds (PD bands) ────────────────────────────────────────────
# Source of truth for risk_tiers.json. Both training (writer) and scoring
# (consumer) read from this constant so the band cuts stay in lockstep.
DEFAULT_TIER_THRESHOLDS: list[tuple[float, float, str]] = [
    (0.00, 0.03, "LOW"),
    (0.03, 0.06, "MODERATE"),
    (0.06, 0.10, "ELEVATED"),
    (0.10, 0.20, "HIGH"),
    (0.20, 1.01, "VERY_HIGH"),
]

MONOTONE_MAP = {
    "INCOME_T1": -1, "LOG_INCOME_T1": -1, "HOUSEHOLD_INCOME": -1,
    "MAX_CREDIT_TJ_AV": -1, "LOG_MAX_CREDIT": -1,
    "HAS_CODEBTOR": -1, "CODEBTOR_INCOME_SHARE": -1,
    "BOOK_RATIO_LOAN": -1, "BOOK_RATIO_CARD": -1,
    "AGE_T1": -1, "LEFT_TO_LIVE": -1, "HOUSE_YEARS": -1,
    "INSTALLMENT_TO_INCOME": 1, "TOTAL_AMT_TO_INCOME": 1,
    "INSTALLMENT_TO_HOUSEHOLD": 1, "TOTAL_AMT_TO_HOUSEHOLD": 1,
    "AMT_PER_MONTH": 1,
    "INSTALLMENT_AMT_DIV_MAX_CREDIT_TJ_AV": 1,
    "TENOR_DIV_MAX_CREDIT_TJ_AV": 1,
}
