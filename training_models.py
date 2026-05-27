"""Single-model training functions and shared model utilities.

One Optuna-driven train function per algorithm: Logistic Regression, EBM,
LightGBM, XGBoost, CatBoost. Every train function returns a fitted sklearn
Pipeline plus the Optuna study (for hyperparameter logging). Boosters also
return the conservatively-selected n_estimators.

Cross-validation is always temporal — see :class:`training_temporal.TemporalExpandingCV`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from loguru import logger
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from xgboost import XGBClassifier

from training_constants import (
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS_CEILING,
    OPTUNA_PRUNER_STARTUP_TRIALS,
    OPTUNA_PRUNER_WARMUP_STEPS,
    RANDOM_STATE,
)
from training_features import build_preprocessors


class EnsembleModel:
    """Lightweight wrapper that blends two pipelines' predict_proba outputs."""

    def __init__(
        self,
        model_a,
        model_b,
        weight_a: float,
        weight_b: float,
        name: str = "Ensemble",
    ) -> None:
        self._a, self._b = model_a, model_b
        self._wa, self._wb = weight_a, weight_b
        self.name = name

    def predict_proba(self, X):
        pa = self._a.predict_proba(X)
        pb = self._b.predict_proba(X)
        return self._wa * pa + self._wb * pb

    @property
    def named_steps(self):
        return None  # Not a Pipeline — skip feature importance extraction


def save_optuna_study(study: optuna.Study, output_dir: Path, model_name: str) -> None:
    """Save Optuna trial history as CSV for hyperparameter sensitivity analysis."""
    try:
        trials_df = study.trials_dataframe()
        safe_name = model_name.lower().replace(" ", "_")
        path = output_dir / f"optuna_{safe_name}.csv"
        trials_df.to_csv(path, index=False, float_format="%.6f")
        logger.info("Saved Optuna study for {}: {} ({} trials)", model_name, path, len(trials_df))
    except Exception as exc:
        logger.warning("Could not save Optuna study for {}: {}", model_name, exc)


def safe_stratified_n_splits(y, max_splits: int = 5) -> int:
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError("Stratified cross-validation requires at least 2 examples in each class")
    return int(min(max_splits, len(y), class_counts.min()))


def normalize_estimator_count(value, fallback: int = 1) -> int:
    if value is None or pd.isna(value):
        return fallback
    return max(int(value), fallback)


def select_conservative_boosting_rounds(
    best_iterations: list[int],
    fallback: int = N_ESTIMATORS_CEILING,
    quantile: float = 0.25,
) -> int:
    if not best_iterations:
        return normalize_estimator_count(fallback)
    normalized = np.asarray(
        [normalize_estimator_count(value, fallback=fallback) for value in best_iterations],
        dtype=float,
    )
    conservative_value = np.floor(np.quantile(normalized, quantile))
    return normalize_estimator_count(conservative_value, fallback=int(normalized.min()))


def normalize_xgboost_monotone_constraints(monotone_constraints):
    if monotone_constraints is None or isinstance(monotone_constraints, (tuple, str, dict)):
        return monotone_constraints
    if isinstance(monotone_constraints, np.ndarray):
        return tuple(monotone_constraints.tolist())
    if isinstance(monotone_constraints, list):
        return tuple(monotone_constraints)
    return monotone_constraints


def _lgbm_prauc_eval(y_true, y_raw):
    """Custom LightGBM eval: PR AUC on probabilities (for early stopping)."""
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_raw, -500, 500)))
    return "prauc", float(average_precision_score(y_true, y_prob)), True


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv: Any,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
) -> tuple[Pipeline, optuna.Study]:
    logger.info("Optuna: {} trials x {} folds (LR L2 + TargetEncoder smooth tuning)",
                n_trials, cv.n_splits)
    folds = list(cv.split(X_train, y_train))

    def objective(trial):
        C = trial.suggest_float("C", 1e-4, 100.0, log=True)
        smooth = trial.suggest_float("smooth", 1.0, 200.0, log=True)

        fold_scores = []
        for train_idx, val_idx in folds:
            if num_cols is not None and cat_cols is not None:
                pre = build_preprocessors(num_cols, cat_cols, target_encoder_smooth=smooth)[0]
            else:
                pre = clone(preprocessor)
            X_tr_t = pre.fit_transform(X_train.iloc[train_idx], y=y_train.iloc[train_idx])
            X_va_t = pre.transform(X_train.iloc[val_idx])
            w_fold = sample_weight[train_idx] if sample_weight is not None else None
            clf = LogisticRegression(
                C=C, class_weight="balanced",
                max_iter=5000, random_state=RANDOM_STATE, solver="lbfgs",
            )
            clf.fit(X_tr_t, y_train.iloc[train_idx], sample_weight=w_fold)
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_train.iloc[val_idx], y_pred))

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", study_name="lr",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE),
    )
    # n_jobs=1: sequential trials. Tree objectives use n_jobs=1 per fold
    # to avoid contention; parallel trials would multiply memory usage.
    # Set n_jobs=-1 for parallel trials if memory allows.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  C={:.4f}, smooth={:.1f} (penalty=l2, solver=lbfgs)", bp["C"], bp["smooth"])

    if num_cols is not None and cat_cols is not None:
        best_preprocessor = build_preprocessors(num_cols, cat_cols, target_encoder_smooth=bp["smooth"])[0]
    else:
        best_preprocessor = preprocessor
    lr_model = Pipeline([
        ("preprocessor", best_preprocessor),
        ("classifier", LogisticRegression(
            C=bp["C"], class_weight="balanced",
            max_iter=5000, random_state=RANDOM_STATE, solver="lbfgs",
        )),
    ])
    lr_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    return lr_model, study


def train_ebm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv: Any,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
) -> tuple[Pipeline, optuna.Study]:
    """Train an Explainable Boosting Machine (GAM) with Optuna hyperparameter search.

    EBMs learn additive shape functions per feature + optional pairwise interactions.
    They sit between LR and full tree ensembles in complexity: more flexible than LR
    (captures non-linear effects per feature) but less prone to overfitting than
    tree ensembles (additive structure prevents high-order interactions).
    """
    logger.info("Optuna: {} trials x {} folds (EBM — Explainable Boosting Machine)",
                n_trials, cv.n_splits)
    folds = list(cv.split(X_train, y_train))

    def objective(trial):
        max_bins = trial.suggest_int("max_bins", 64, 256)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
        max_interaction_bins = trial.suggest_int("max_interaction_bins", 16, 64)
        interactions = trial.suggest_int("interactions", 0, 15)
        inner_bags = trial.suggest_int("inner_bags", 0, 8)
        outer_bags = trial.suggest_int("outer_bags", 4, 16)
        smoothing_rounds = trial.suggest_int("smoothing_rounds", 0, 500)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 50)

        fold_scores = []
        for fold_i, (train_idx, val_idx) in enumerate(folds):
            pre = clone(preprocessor)
            X_tr_t = pre.fit_transform(X_train.iloc[train_idx], y=y_train.iloc[train_idx])
            X_va_t = pre.transform(X_train.iloc[val_idx])
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            ebm = ExplainableBoostingClassifier(
                max_bins=max_bins,
                learning_rate=learning_rate,
                max_interaction_bins=max_interaction_bins,
                interactions=interactions,
                inner_bags=inner_bags,
                outer_bags=outer_bags,
                smoothing_rounds=smoothing_rounds,
                min_samples_leaf=min_samples_leaf,
                random_state=RANDOM_STATE,
            )
            if w_fold is not None:
                ebm.fit(X_tr_t, y_train.iloc[train_idx], sample_weight=w_fold)
            else:
                ebm.fit(X_tr_t, y_train.iloc[train_idx])
            y_pred = ebm.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_train.iloc[val_idx], y_pred))

            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", study_name="ebm",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
        ),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  max_bins={}, lr={:.4f}, interactions={}, outer_bags={}, min_samples_leaf={}",
                bp["max_bins"], bp["learning_rate"], bp["interactions"],
                bp["outer_bags"], bp["min_samples_leaf"])

    ebm_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", ExplainableBoostingClassifier(
            max_bins=bp["max_bins"],
            learning_rate=bp["learning_rate"],
            max_interaction_bins=bp["max_interaction_bins"],
            interactions=bp["interactions"],
            inner_bags=bp["inner_bags"],
            outer_bags=bp["outer_bags"],
            smoothing_rounds=bp["smoothing_rounds"],
            min_samples_leaf=bp["min_samples_leaf"],
            random_state=RANDOM_STATE,
        )),
    ])
    if sample_weight is not None:
        ebm_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        ebm_model.fit(X_train, y_train)
    return ebm_model, study


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_preprocessor: ColumnTransformer,
    lgbm_cat_indices: list[int],
    pos_weight: float,
    cv: Any,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
    monotone_constraints: list[int] | tuple[int, ...] | None = None,
) -> tuple[Pipeline, optuna.Study, int]:
    # When sample_weight is provided (e.g. reject inference), it already
    # rebalances the class distribution.  Combining it with scale_pos_weight
    # would over-amplify the minority class, biasing the model toward
    # predicting more defaults than warranted.
    effective_pos_weight = 1.0 if sample_weight is not None else pos_weight
    if sample_weight is not None and pos_weight != 1.0:
        logger.info("sample_weight provided — disabling scale_pos_weight to avoid double-rebalancing")
    logger.info("Optuna: {} trials x {} folds, early stopping after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)
    lgbm_subsample_freq = 1

    def objective(trial):
        # Depth bounds loosened to [3, 7] per methodology audit. Tighter
        # regularization (min_child_samples up to 500, reg_lambda to 100,
        # min_split_gain to 5) compensates so deeper trees don't overfit.
        # num_leaves capped well below 2^max_depth (=128) to keep trees sparse.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.6, 0.85),
            "subsample_freq": lgbm_subsample_freq,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.85),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "max_bin": trial.suggest_int("max_bin", 63, 127),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-3, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for fold_i, (train_idx, val_idx) in enumerate(tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False)):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(lgbm_preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = LGBMClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                scale_pos_weight=effective_pos_weight,
                monotone_constraints=monotone_constraints,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
                **params,
            )
            clf.fit(
                X_tr_t, y_f_tr,
                sample_weight=w_fold,
                eval_set=[(X_va_t, y_f_va)],
                eval_metric=_lgbm_prauc_eval,
                callbacks=[
                    early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    log_evaluation(-1),
                ],
                categorical_feature=lgbm_cat_indices,
            )
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(normalize_estimator_count(clf.best_iteration_, fallback=N_ESTIMATORS_CEILING))

            # Report intermediate result for pruning
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", study_name="lgbm",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
        ),
    )
    # n_jobs=1: sequential trials. Tree objectives use n_jobs=1 per fold
    # to avoid contention; parallel trials would multiply memory usage.
    # Set n_jobs=-1 for parallel trials if memory allows.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  n_estimators={} (conservative early stop), lr={:.4f}, leaves={}, depth={}, min_child={}",
                best_n_estimators, bp["learning_rate"], bp["num_leaves"], bp["max_depth"], bp["min_child_samples"])
    logger.info("  subsample={:.2f} (freq={}), colsample_tree={:.2f}, colsample_node={:.2f}, max_bin={}, min_split_gain={:.2e}, alpha={:.2e}, lambda={:.2e}",
                bp["subsample"], lgbm_subsample_freq, bp["colsample_bytree"], bp["colsample_bynode"], bp["max_bin"], bp["min_split_gain"], bp["reg_alpha"], bp["reg_lambda"])

    lgbm_model = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", LGBMClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=effective_pos_weight,
            subsample_freq=lgbm_subsample_freq,
            monotone_constraints=monotone_constraints,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
            **study.best_params,
        )),
    ])
    fit_params = {"classifier__categorical_feature": lgbm_cat_indices}
    if sample_weight is not None:
        fit_params["classifier__sample_weight"] = sample_weight
    lgbm_model.fit(X_train, y_train, **fit_params)
    return lgbm_model, study, best_n_estimators


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    pos_weight: float,
    cv: Any,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
    monotone_constraints: list[int] | tuple[int, ...] | None = None,
) -> tuple[Pipeline, optuna.Study, int]:
    # Same guard as LGBM: skip scale_pos_weight when sample_weight handles rebalancing.
    effective_pos_weight = 1.0 if sample_weight is not None else pos_weight
    if sample_weight is not None and pos_weight != 1.0:
        logger.info("sample_weight provided — disabling scale_pos_weight to avoid double-rebalancing")
    logger.info("Optuna: {} trials x {} folds, early stopping after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)
    xgb_monotone_constraints = normalize_xgboost_monotone_constraints(monotone_constraints)

    def objective(trial):
        # Depth bounds loosened to [3, 7] per methodology audit. The previous
        # [2, 4] cap forced near-additive trees. Tighter regularization
        # (min_child_weight to 200, gamma to 20, reg_lambda to 100) keeps
        # the deeper bound from over-fitting.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.85),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for fold_i, (train_idx, val_idx) in enumerate(tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False)):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr, y=y_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = XGBClassifier(
                n_estimators=N_ESTIMATORS_CEILING,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                scale_pos_weight=effective_pos_weight,
                monotone_constraints=xgb_monotone_constraints,
                random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
                eval_metric="aucpr", **params,
            )
            clf.fit(X_tr_t, y_f_tr, sample_weight=w_fold,
                    eval_set=[(X_va_t, y_f_va)], verbose=False)
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(
                normalize_estimator_count(
                    None if clf.best_iteration is None else clf.best_iteration + 1,
                    fallback=N_ESTIMATORS_CEILING,
                )
            )

            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", study_name="xgb",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
        ),
    )
    # n_jobs=1: sequential trials. Tree objectives use n_jobs=1 per fold
    # to avoid contention; parallel trials would multiply memory usage.
    # Set n_jobs=-1 for parallel trials if memory allows.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  n_estimators={} (conservative early stop), lr={:.4f}, depth={}, min_child_w={}",
                best_n_estimators, bp["learning_rate"], bp["max_depth"], bp["min_child_weight"])
    logger.info("  subsample={:.2f}, colsample_tree={:.2f}, colsample_node={:.2f}, gamma={:.2e}, alpha={:.2e}, lambda={:.2e}",
                bp["subsample"], bp["colsample_bytree"], bp["colsample_bynode"], bp["gamma"], bp["reg_alpha"], bp["reg_lambda"])

    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=best_n_estimators,
            scale_pos_weight=effective_pos_weight,
            monotone_constraints=xgb_monotone_constraints,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
            eval_metric="aucpr", **study.best_params,
        )),
    ])
    if sample_weight is not None:
        xgb_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        xgb_model.fit(X_train, y_train)
    return xgb_model, study, best_n_estimators


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lgbm_preprocessor: ColumnTransformer,
    pos_weight: float,
    cv: Any,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
    monotone_constraints: list[int] | tuple[int, ...] | None = None,
) -> tuple[Pipeline, optuna.Study, int]:
    logger.info("Optuna: {} trials x {} folds, early stopping (PRAUC) after {} rounds",
                n_trials, cv.n_splits, EARLY_STOPPING_ROUNDS)

    def objective(trial):
        # Depth bounds loosened to [3, 7] per methodology audit. CatBoost uses
        # symmetric (oblivious) trees, so depth 7 = 128 leaves max — still
        # bounded. Tighter regularization (l2_leaf_reg to 100,
        # min_data_in_leaf to 500) compensates.
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 3, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "random_strength": trial.suggest_float("random_strength", 0.1, 20.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        }

        fold_scores = []
        fold_best_iters = []
        folds = list(cv.split(X_train, y_train))
        for fold_i, (train_idx, val_idx) in enumerate(tqdm(folds, desc=f"  Trial {trial.number} folds", leave=False)):
            X_f_tr = X_train.iloc[train_idx]
            y_f_tr = y_train.iloc[train_idx]
            X_f_va = X_train.iloc[val_idx]
            y_f_va = y_train.iloc[val_idx]
            w_fold = sample_weight[train_idx] if sample_weight is not None else None

            pre = clone(lgbm_preprocessor)
            X_tr_t = pre.fit_transform(X_f_tr)
            X_va_t = pre.transform(X_f_va)

            clf = CatBoostClassifier(
                iterations=N_ESTIMATORS_CEILING,
                auto_class_weights="Balanced",
                monotone_constraints=monotone_constraints,
                random_seed=RANDOM_STATE,
                eval_metric="PRAUC",
                verbose=0,
                **params,
            )
            clf.fit(
                X_tr_t, y_f_tr,
                sample_weight=w_fold,
                eval_set=[(X_va_t, y_f_va)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=0,
            )
            y_pred = clf.predict_proba(X_va_t)[:, 1]
            fold_scores.append(average_precision_score(y_f_va, y_pred))
            fold_best_iters.append(
                normalize_estimator_count(
                    None if clf.best_iteration_ is None else clf.best_iteration_ + 1,
                    fallback=N_ESTIMATORS_CEILING,
                )
            )

            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr(
            "best_n_estimators",
            select_conservative_boosting_rounds(fold_best_iters, fallback=N_ESTIMATORS_CEILING),
        )
        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize", study_name="catboost",
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_PRUNER_WARMUP_STEPS,
        ),
    )
    # n_jobs=1: sequential trials. Tree objectives use n_jobs=1 per fold
    # to avoid contention; parallel trials would multiply memory usage.
    # Set n_jobs=-1 for parallel trials if memory allows.
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best_n_estimators = normalize_estimator_count(
        study.best_trial.user_attrs["best_n_estimators"],
        fallback=N_ESTIMATORS_CEILING,
    )
    bp = study.best_params
    logger.info("Best trial #{}: CV PR AUC {:.4f}", study.best_trial.number, study.best_value)
    logger.info("  iterations={} (conservative early stop), lr={:.4f}, depth={}, min_data_in_leaf={}",
                best_n_estimators, bp["learning_rate"], bp["depth"], bp["min_data_in_leaf"])
    logger.info("  l2_leaf_reg={:.2e}, random_strength={:.2e}, bagging_temp={:.2f}",
                bp["l2_leaf_reg"], bp["random_strength"], bp["bagging_temperature"])

    catboost_model = Pipeline([
        ("preprocessor", lgbm_preprocessor),
        ("classifier", CatBoostClassifier(
            iterations=best_n_estimators,
            auto_class_weights="Balanced",
            monotone_constraints=monotone_constraints,
            random_seed=RANDOM_STATE, verbose=0,
            **study.best_params,
        )),
    ])
    if sample_weight is not None:
        catboost_model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        catboost_model.fit(X_train, y_train)
    return catboost_model, study, best_n_estimators
