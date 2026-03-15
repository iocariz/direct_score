"""Training pipeline for basel_bad binary classification.

Converts the notebook 3-model.ipynb into a reproducible training script.

Usage:
    uv run python training.py
    uv run python training.py --data-path data/demand_direct.parquet
    uv run python training.py --optuna-trials 100
"""

from __future__ import annotations

import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET = "basel_bad"
MATURITY_CUTOFF = "2025-01-01"
SPLIT_DATE = "2024-07-01"
RANDOM_STATE = 42
MAX_CATEGORIES = 20
MIN_LIFT = 0.01
MIN_VALID = 5_000
N_ESTIMATORS_CEILING = 2000
EARLY_STOPPING_ROUNDS = 50

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
    "product_type_1",    # single-value column, no discriminative power
    "acct_booked_H0",    # single-value column, no discriminative power
]

RAW_NUM = [
    "CODRAMA", "TOTAL_AMT", "INSTALLMENT_AMT",
    "TOTAL_CARD_NBR", "TOTAL_LOAN_NBR", "BOOK_CARD_NBR", "BOOK_LOAN_NBR",
    "AGE_T1", "LEFT_TO_LIVE", "HOUSE_YEARS", "TENOR",
    "MAX_CREDIT_TJ_AV", "INCOME_T1", "INCOME_T2", "INCIT1_L12", "flag_risk3",
]

RAW_CAT = [
    "CUSTOMER_TYPE", "FAMILY_SITUATION", "HOUSE_TYPE",
    "product_type_2", "product_type_3", "CSP", "CPRO", "CMAT",
    "ESTCLI1", "ESTCLI2", "CSECTOR", "FLAG_COTIT",
]

MISS_CANDIDATES = ["MAX_CREDIT_TJ_AV", "INCIT1_L12", "HOUSE_YEARS", "ESTCLI1", "ESTCLI2"]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_path: str) -> pd.DataFrame:
    logger.info("Loading data from {}", data_path)
    df = pq.read_table(data_path).to_pandas()
    logger.info("Raw shape: {}", df.shape)

    df = df[df["status_name"] == "Booked"].copy()
    logger.info("Booked-only shape: {}", df.shape)
    logger.info(
        "Target distribution:\n{}",
        df[TARGET].value_counts(dropna=False).to_string(),
    )
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering features...")

    # Tier 1: Strong signal
    df["INCOME_RATIO_T2_T1"] = df["INCOME_T2"] / df["INCOME_T1"].replace(0, np.nan)
    df["INCOME_CHANGE"] = df["INCOME_T2"] - df["INCOME_T1"]
    df["HAS_INCOME_T2"] = df["INCOME_T2"].notna().astype(int)
    df["TOTAL_PRODUCTS"] = df["TOTAL_CARD_NBR"] + df["TOTAL_LOAN_NBR"]

    # Tier 2: Moderate signal
    df["INSTALLMENT_TO_INCOME"] = df["INSTALLMENT_AMT"] / df["INCOME_T1"].replace(0, np.nan)
    df["TOTAL_AMT_TO_INCOME"] = df["TOTAL_AMT"] / df["INCOME_T1"].replace(0, np.nan)
    df["TOTAL_INCOME"] = df["INCOME_T1"] + df["INCOME_T2"].fillna(0)
    df["TOTAL_AMT_TO_TOTAL_INCOME"] = df["TOTAL_AMT"] / df["TOTAL_INCOME"].replace(0, np.nan)
    df["AMT_PER_MONTH"] = df["TOTAL_AMT"] / df["TENOR"].replace(0, np.nan)

    # Tier 3: Categorical interactions
    df["PRODTYPE3_X_HOUSE"] = df["product_type_3"] + "_" + df["HOUSE_TYPE"]
    df["PRODTYPE3_X_CUSTTYPE"] = df["product_type_3"] + "_" + df["CUSTOMER_TYPE"]
    df["CUSTTYPE_X_HOUSE"] = df["CUSTOMER_TYPE"] + "_" + df["HOUSE_TYPE"]

    # Missing value indicators
    for col in MISS_CANDIDATES:
        miss_rate = df[col].isna().mean()
        if miss_rate > 0.01:
            flag_name = f"MISS_{col}"
            df[flag_name] = df[col].isna().astype(int)
            logger.debug("  {}: {:.1%} missing", flag_name, miss_rate)

    logger.info("Feature engineering complete. Shape: {}", df.shape)
    return df


# ── Interaction search ─────────────────────────────────────────────────────────

def _loo_target_encode(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out target encoding to avoid in-sample leakage."""
    df_temp = pd.DataFrame({"group": groups, "y": y})
    group_sum = df_temp.groupby("group")["y"].transform("sum")
    group_count = df_temp.groupby("group")["y"].transform("count")
    loo_mean = (group_sum - df_temp["y"]) / (group_count - 1)
    loo_mean = loo_mean.fillna(y.mean())
    return loo_mean.values


def _safe_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, int]:
    valid = np.isfinite(s)
    n = valid.sum()
    if n < MIN_VALID or y[valid].sum() < 10:
        return np.nan, n
    return roc_auc_score(y[valid], s[valid]), n


def search_interactions(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Screening feature interactions (training data only)...")

    df_search = df[df["mis_Date"] < SPLIT_DATE].dropna(subset=[TARGET]).copy()
    df_search[TARGET] = df_search[TARGET].astype(int)
    y_search = df_search[TARGET].values

    logger.info(
        "Search set: {:,} rows ({:,} pos, {:,} neg)",
        len(df_search), y_search.sum(), (y_search == 0).sum(),
    )

    # Precompute base AUCs
    base_auc = {}
    for col in RAW_NUM:
        auc, _ = _safe_auc(y_search, df_search[col].values)
        if not np.isnan(auc):
            base_auc[col] = abs(auc - 0.5)
    for col in RAW_CAT:
        te = _loo_target_encode(df_search[col].astype(str).values, y_search)
        auc, _ = _safe_auc(y_search, te)
        if not np.isnan(auc):
            base_auc[col] = abs(auc - 0.5)

    logger.debug("Base AUCs computed for {} features", len(base_auc))

    # Screen numerical pairs
    results = []
    for a, b in combinations(RAW_NUM, 2):
        va = df_search[a].values.astype(float)
        vb = df_search[b].values.astype(float)
        parent_power = max(base_auc.get(a, 0), base_auc.get(b, 0))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = va / vb
        auc_r, n_r = _safe_auc(y_search, ratio)
        if not np.isnan(auc_r):
            power = abs(auc_r - 0.5)
            if power - parent_power >= MIN_LIFT:
                results.append({
                    "name": f"{a}/{b}", "type": "ratio",
                    "auc": auc_r, "lift": power - parent_power,
                    "feat_a": a, "feat_b": b,
                })

        product = va * vb
        auc_p, n_p = _safe_auc(y_search, product)
        if not np.isnan(auc_p):
            power = abs(auc_p - 0.5)
            if power - parent_power >= MIN_LIFT:
                results.append({
                    "name": f"{a}*{b}", "type": "product",
                    "auc": auc_p, "lift": power - parent_power,
                    "feat_a": a, "feat_b": b,
                })

    # Screen categorical pairs (LOO target encoding to avoid leakage)
    for a, b in combinations(RAW_CAT, 2):
        combo = (df_search[a].astype(str) + "_" + df_search[b].astype(str)).values
        te = _loo_target_encode(combo, y_search)
        auc_c, n_c = _safe_auc(y_search, te)
        if not np.isnan(auc_c):
            parent_power = max(base_auc.get(a, 0), base_auc.get(b, 0))
            power = abs(auc_c - 0.5)
            if power - parent_power >= MIN_LIFT:
                results.append({
                    "name": f"{a}_x_{b}", "type": "cat_concat",
                    "auc": auc_c, "lift": power - parent_power,
                    "feat_a": a, "feat_b": b,
                })

    all_results = pd.DataFrame(results)
    if not all_results.empty:
        all_results = all_results.sort_values("lift", ascending=False)
    logger.info("Found {} interactions with >= {:.0%} lift", len(all_results), MIN_LIFT)
    return all_results


def add_interactions(df: pd.DataFrame, all_results: pd.DataFrame) -> pd.DataFrame:
    existing = set(df.columns)
    added = []

    for _, row in all_results.iterrows():
        col_name = row["name"].replace("/", "_DIV_").replace("*", "_X_")
        if col_name in existing:
            continue

        a, b = row["feat_a"], row["feat_b"]
        if row["type"] == "ratio":
            with np.errstate(divide="ignore", invalid="ignore"):
                df[col_name] = df[a] / df[b].replace(0, np.nan)
        elif row["type"] == "product":
            df[col_name] = df[a] * df[b]
        elif row["type"] == "cat_concat":
            df[col_name] = df[a].astype(str) + "_" + df[b].astype(str)
        added.append(col_name)
        existing.add(col_name)

    logger.info("Auto-added {} interaction features. Shape: {}", len(added), df.shape)
    return df


# ── Feature selection & split ──────────────────────────────────────────────────

def select_features(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object" or df[c].dtype.name == "str"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    logger.info("Features — {} numerical, {} categorical, {} total", len(num_cols), len(cat_cols), len(feature_cols))
    return feature_cols, num_cols, cat_cols


def temporal_split(df: pd.DataFrame, feature_cols: list[str]):
    df_model = df[df["mis_Date"] <= MATURITY_CUTOFF].dropna(subset=[TARGET]).copy()
    df_model[TARGET] = df_model[TARGET].astype(int)

    train_mask = df_model["mis_Date"] < SPLIT_DATE
    test_mask = df_model["mis_Date"] >= SPLIT_DATE

    X_train = df_model.loc[train_mask, feature_cols]
    y_train = df_model.loc[train_mask, TARGET]
    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, TARGET]

    bench_risk_score_rf = df_model.loc[test_mask, "risk_score_rf"]
    bench_score_RF = df_model.loc[test_mask, "score_RF"]

    logger.info("Train: {}  ({:.4f} target rate)", X_train.shape, y_train.mean())
    logger.info("Test:  {}  ({:.4f} target rate)", X_test.shape, y_test.mean())
    return X_train, y_train, X_test, y_test, bench_risk_score_rf, bench_score_RF


def reduce_cardinality(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    logger.info("Reducing cardinality (max {} categories)...", MAX_CATEGORIES)
    X_train = X_train.copy()
    X_test = X_test.copy()
    cardinality_maps = {}

    for col in cat_cols:
        top_cats = X_train[col].value_counts().nlargest(MAX_CATEGORIES).index
        cardinality_maps[col] = set(top_cats)
        n_before = X_train[col].nunique()
        X_train[col] = X_train[col].where(X_train[col].isin(top_cats), "Other")
        X_test[col] = X_test[col].where(X_test[col].isin(top_cats), "Other")
        n_after = X_train[col].nunique()
        if n_before != n_after:
            logger.debug("  {}: {} -> {} categories", col, n_before, n_after)

    logger.info("Cardinality reduction complete.")
    return X_train, X_test, cardinality_maps


# ── Preprocessors ──────────────────────────────────────────────────────────────

def build_preprocessors(num_cols: list[str], cat_cols: list[str]):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", TargetEncoder(smooth="auto", cv=5, random_state=RANDOM_STATE)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols),
    ])

    lgbm_num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    lgbm_cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    lgbm_preprocessor = ColumnTransformer([
        ("num", lgbm_num_transformer, num_cols),
        ("cat", lgbm_cat_transformer, cat_cols),
    ])
    lgbm_cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    return preprocessor, lgbm_preprocessor, lgbm_cat_indices


# ── RFECV ──────────────────────────────────────────────────────────────────────

def run_rfecv(
    X_train: pd.DataFrame, y_train: pd.Series,
    num_cols: list[str], cat_cols: list[str],
    feature_cols: list[str],
):
    logger.info("Running RFECV...")
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    rfe_estimator = LGBMClassifier(
        n_estimators=100, num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )

    all_feature_names = num_cols + cat_cols

    # Use OrdinalEncoder-based preprocessing to avoid target leakage.
    # TargetEncoder would leak validation-fold targets when pre-fitted on all data.
    rfe_preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    X_rfe = rfe_preprocessor.fit_transform(X_train)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        rfecv = RFECV(
            estimator=rfe_estimator, step=1,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            scoring="average_precision", min_features_to_select=5,
            n_jobs=-1, verbose=0,
        )
        rfecv.fit(X_rfe, y_train)

    selected_mask = rfecv.support_
    selected_set = {f for f, s in zip(all_feature_names, selected_mask) if s}
    eliminated = [f for f, s in zip(all_feature_names, selected_mask) if not s]

    logger.info("RFECV optimal: {} / {} features", rfecv.n_features_, len(all_feature_names))
    if eliminated:
        logger.info("Eliminated ({}): {}", len(eliminated), eliminated)

    num_cols = [c for c in num_cols if c in selected_set]
    cat_cols = [c for c in cat_cols if c in selected_set]
    feature_cols = [c for c in feature_cols if c in selected_set]

    return feature_cols, num_cols, cat_cols


# ── Model training ─────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, preprocessor, cv):
    logger.info("Training Logistic Regression (GridSearchCV)...")
    lr_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            penalty="elasticnet",
            class_weight="balanced", max_iter=5000,
            random_state=RANDOM_STATE, solver="saga",
        )),
    ])
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__l1_ratio": [0.0, 0.5, 1.0],
    }
    lr_search = GridSearchCV(
        lr_pipe, lr_param_grid,
        scoring="average_precision",
        cv=cv, n_jobs=-1, verbose=0,
    )
    lr_search.fit(X_train, y_train)
    logger.info("LR best params: {}", lr_search.best_params_)
    logger.info("LR best CV PR AUC: {:.4f}", lr_search.best_score_)
    return lr_search


def train_lgbm(X_train, y_train, lgbm_preprocessor, lgbm_cat_indices, pos_weight, cv, n_trials: int):
    logger.info("Training LightGBM (Optuna, {} trials, early stopping)...", n_trials)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 80),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]

            pre = clone(lgbm_preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = LGBMClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                scale_pos_weight=pos_weight, max_depth=-1,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
                **params,
            )
            clf.fit(
                X_tr_t, y_f_tr,
                eval_set=[(X_va_t, y_f_va)],
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    log_evaluation(-1),
                ],
                categorical_feature=lgbm_cat_indices,
            )
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(clf.best_iteration_)

        trial.set_user_attr("best_n_estimators", int(np.median(fold_best_iters)))
        return np.mean(fold_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lgbm")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n_estimators = study.best_trial.user_attrs["best_n_estimators"]
    logger.info("LightGBM best trial: {}", study.best_trial.number)
    logger.info("LightGBM best CV PR AUC: {:.4f}", study.best_value)
    logger.info("LightGBM best n_estimators (early stopping): {}", best_n_estimators)
    logger.info("LightGBM best params: {}", study.best_params)

    lgbm_model = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=pos_weight, max_depth=-1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
            **study.best_params,
        )),
    ])
    lgbm_model.fit(X_train, y_train, classifier__categorical_feature=lgbm_cat_indices)
    return lgbm_model, study, best_n_estimators


def train_xgboost(X_train, y_train, preprocessor, pos_weight, cv, n_trials: int):
    logger.info("Training XGBoost (Optuna, {} trials, early stopping)...", n_trials)

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]

            pre = clone(preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr, y=y_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = XGBClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
                eval_metric="aucpr", **params,
            )
            clf.fit(X_tr_t, y_f_tr, eval_set=[(X_va_t, y_f_va)], verbose=False)
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(clf.best_iteration + 1)

        trial.set_user_attr("best_n_estimators", int(np.median(fold_best_iters)))
        return np.mean(fold_scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="xgb")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n_estimators = study.best_trial.user_attrs["best_n_estimators"]
    logger.info("XGBoost best trial: {}", study.best_trial.number)
    logger.info("XGBoost best CV PR AUC: {:.4f}", study.best_value)
    logger.info("XGBoost best n_estimators (early stopping): {}", best_n_estimators)
    logger.info("XGBoost best params: {}", study.best_params)

    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="aucpr", **study.best_params,
        )),
    ])
    xgb_model.fit(X_train, y_train)
    return xgb_model, study, best_n_estimators


def train_stacking(
    X_train, y_train,
    preprocessor, lr_search,
    lgbm_best_n_estimators, lgbm_study,
    xgb_best_n_estimators, xgb_study,
    pos_weight, cv,
):
    logger.info("Training stacking ensemble...")
    # Note: stacking LGBM uses TargetEncoder (generic preprocessor) rather than
    # OrdinalEncoder + native categoricals, because StackingClassifier does not
    # support per-estimator fit_params. Hyperparameters from Optuna are used as
    # reasonable starting values; the meta-learner compensates for any mismatch.
    stack_estimators = [
        ("lr", Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                penalty="elasticnet",
                class_weight="balanced", max_iter=5000,
                random_state=RANDOM_STATE, solver="saga",
                C=lr_search.best_params_["classifier__C"],
                l1_ratio=lr_search.best_params_["classifier__l1_ratio"],
            )),
        ])),
        ("lgbm", Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(
                n_estimators=lgbm_best_n_estimators,
                scale_pos_weight=pos_weight, max_depth=-1,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
                **lgbm_study.best_params,
            )),
        ])),
        ("xgb", Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=xgb_best_n_estimators,
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
                eval_metric="aucpr", **xgb_study.best_params,
            )),
        ])),
    ]

    stack_model = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=cv, stack_method="predict_proba",
        n_jobs=-1, passthrough=False,
    )
    stack_model.fit(X_train, y_train)
    return stack_model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true: np.ndarray, y_score: np.ndarray, is_probability: bool = True) -> dict:
    mask = ~np.isnan(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    result = {
        "Model": name,
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "N": mask.sum(),
    }
    result["Brier"] = brier_score_loss(y_true, np.clip(y_score, 0, 1)) if is_probability else np.nan
    return result


def evaluate_all(
    X_test, y_test, models: dict,
    bench_risk_score_rf, bench_score_RF,
) -> pd.DataFrame:
    logger.info("Evaluating on test set...")
    results = []

    for name, mdl in models.items():
        y_proba = mdl.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test.values, y_proba))

    # Benchmark scores: higher = safer, negate for ranking
    for name, scores in [
        ("risk_score_rf (benchmark)", bench_risk_score_rf),
        ("score_RF (benchmark)", bench_score_RF),
    ]:
        results.append(evaluate(name, y_test.values, -scores.values, is_probability=False))

    results_df = pd.DataFrame(results).set_index("Model")
    results_df = results_df.sort_values("PR AUC", ascending=False)
    logger.info("Results:\n{}", results_df.to_string(float_format="%.4f"))
    return results_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main(data_path: str = "data/demand_direct.parquet", optuna_trials: int = 50):
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # 1. Load data
    df = load_data(data_path)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Interaction search & add
    interactions = search_interactions(df)
    df = add_interactions(df, interactions)

    # 4. Feature selection
    feature_cols, num_cols, cat_cols = select_features(df)

    # 5. Train/test split
    X_train, y_train, X_test, y_test, bench_risk, bench_score = temporal_split(df, feature_cols)

    # 6. Cardinality reduction
    X_train, X_test, _ = reduce_cardinality(X_train, X_test, cat_cols)

    # 7. RFECV (uses OrdinalEncoder internally to avoid target leakage)
    feature_cols, num_cols, cat_cols = run_rfecv(
        X_train, y_train, num_cols, cat_cols, feature_cols,
    )
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # 8. Build preprocessors with selected features
    preprocessor, lgbm_preprocessor, lgbm_cat_indices = build_preprocessors(num_cols, cat_cols)

    # 9. Hold out calibration set (15%) before model training
    X_fit, X_calib, y_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE,
    )
    logger.info("Fit set: {}  Calibration set: {}", X_fit.shape, X_calib.shape)

    # 10. Model training (on X_fit only)
    pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    logger.info("Class imbalance ratio: {:.0f}:1", pos_weight)

    lr_search = train_logistic_regression(X_fit, y_fit, preprocessor, cv)
    lr_model = lr_search.best_estimator_

    lgbm_model, lgbm_study, lgbm_best_n = train_lgbm(
        X_fit, y_fit, lgbm_preprocessor, lgbm_cat_indices, pos_weight, cv, optuna_trials,
    )

    xgb_model, xgb_study, xgb_best_n = train_xgboost(
        X_fit, y_fit, preprocessor, pos_weight, cv, optuna_trials,
    )

    stack_model = train_stacking(
        X_fit, y_fit, preprocessor, lr_search,
        lgbm_best_n, lgbm_study, xgb_best_n, xgb_study,
        pos_weight, cv,
    )

    # 11. Calibrate individual models on held-out calibration set
    logger.info("Calibrating models (isotonic, prefit)...")
    models = {
        "Logistic Regression": lr_model,
        "LightGBM": lgbm_model,
        "XGBoost": xgb_model,
        "Stacking": stack_model,
    }
    for name in ["Logistic Regression", "LightGBM", "XGBoost"]:
        cal = CalibratedClassifierCV(models[name], cv="prefit", method="isotonic")
        cal.fit(X_calib, y_calib)
        models[f"{name} (calibrated)"] = cal

    # 12. Evaluation
    results_df = evaluate_all(X_test, y_test, models, bench_risk, bench_score)

    logger.info("=" * 60)
    logger.info("Training pipeline complete")
    logger.info("=" * 60)

    return models, results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train basel_bad classifier")
    parser.add_argument("--data-path", default="data/demand_direct.parquet", help="Path to parquet data file")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials per model")
    args = parser.parse_args()

    main(data_path=args.data_path, optuna_trials=args.optuna_trials)
