TARGET = "basel_bad"
MATURITY_CUTOFF = "2025-01-01"
SPLIT_DATE = "2024-07-01"
RANDOM_STATE = 42
MAX_CATEGORIES = 20
MIN_LIFT = 0.01
MIN_VALID = 5_000
N_ESTIMATORS_CEILING = 2000
EARLY_STOPPING_ROUNDS = 50
N_BOOTSTRAP = 1000
CALIBRATION_FRACTION = 0.15
FEATURE_DISCOVERY_FRACTION = 0.50
ROLLING_OOT_MAX_WINDOWS = 4
POPULATION_MODE_BOOKED_MONITORING = "booked_monitoring"
POPULATION_MODE_UNDERWRITING = "underwriting"
UNDERWRITING_DECISION_STATUSES = ("Booked", "Rejected", "Canceled")
OFFICIAL_MODEL_NAMES = [
    "Logistic Regression",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]
EXPERIMENTAL_STACKING_NAME = "Stacking (experimental)"
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
