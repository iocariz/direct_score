"""Temporal stacking ensemble.

`train_stacking` produces an OOF-trained Logistic Regression meta-learner over
base-model predict_proba outputs. The OOF predictions are produced via the
same temporal cross-validator as the base models — never with random shuffling.
This is experimental: stacking can over-fit when base learners are strongly
correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from training_constants import RANDOM_STATE
from training_models import normalize_xgboost_monotone_constraints
from training_reporting import sanitize_output_name


def build_fresh_pipeline_from_fitted(model: Pipeline) -> Pipeline:
    """Clone a fitted pipeline's structure with a fresh, unfit classifier.

    Used by stacking to retrain the same base-model architecture on each OOF
    fold without leaking the original fit state.
    """
    preprocessor = clone(model.named_steps["preprocessor"])
    classifier = model.named_steps["classifier"]

    if isinstance(classifier, LogisticRegression):
        fresh_classifier = LogisticRegression(**classifier.get_params())
    elif isinstance(classifier, LGBMClassifier):
        fresh_classifier = LGBMClassifier(**classifier.get_params())
    elif isinstance(classifier, XGBClassifier):
        classifier_params = classifier.get_params()
        classifier_params["monotone_constraints"] = normalize_xgboost_monotone_constraints(
            classifier_params.get("monotone_constraints")
        )
        fresh_classifier = XGBClassifier(**classifier_params)
    elif isinstance(classifier, CatBoostClassifier):
        classifier_params = classifier.get_params()
        monotone_constraints = classifier_params.get("monotone_constraints")
        if isinstance(monotone_constraints, np.ndarray):
            classifier_params["monotone_constraints"] = monotone_constraints.tolist()
        fresh_classifier = CatBoostClassifier(**classifier_params)
    else:
        fresh_classifier = clone(classifier)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", fresh_classifier),
    ])


class TemporalStackingClassifier:
    def __init__(
        self,
        named_estimators_: dict[str, Pipeline],
        final_estimator_: LogisticRegression,
        base_model_names_: list[str],
        meta_feature_names_: list[str],
        meta_training_positions_: np.ndarray,
        fold_training_positions_: list[np.ndarray],
        fold_validation_positions_: list[np.ndarray],
    ) -> None:
        self.named_estimators_ = named_estimators_
        self.final_estimator_ = final_estimator_
        self.base_model_names_ = base_model_names_
        self.meta_feature_names_ = meta_feature_names_
        self.meta_training_positions_ = np.asarray(meta_training_positions_, dtype=int)
        self.fold_training_positions_ = [np.asarray(idx, dtype=int) for idx in fold_training_positions_]
        self.fold_validation_positions_ = [np.asarray(idx, dtype=int) for idx in fold_validation_positions_]
        self.classes_ = final_estimator_.classes_

    def _build_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        meta_features = {
            feature_name: self.named_estimators_[model_name].predict_proba(X)[:, 1]
            for model_name, feature_name in zip(self.base_model_names_, self.meta_feature_names_, strict=True)
        }
        return pd.DataFrame(meta_features, index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_estimator_.predict_proba(self._build_meta_features(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_estimator_.predict(self._build_meta_features(X))


def fit_pipeline_from_template(
    model_template: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    fitted_model = build_fresh_pipeline_from_fitted(model_template)
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) >= 2:
        safe_target_encoder_cv = int(min(5, len(y_train), class_counts.min()))
        params = fitted_model.get_params()
        if "preprocessor__cat__encoder__cv" in params:
            fitted_model.set_params(preprocessor__cat__encoder__cv=max(2, safe_target_encoder_cv))

    classifier = fitted_model.named_steps["classifier"]
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["classifier__sample_weight"] = sample_weight
    if isinstance(classifier, LGBMClassifier):
        preprocessor = fitted_model.named_steps["preprocessor"]
        num_cols = list(preprocessor.transformers[0][2])
        cat_cols = list(preprocessor.transformers[1][2])
        fit_kwargs["classifier__categorical_feature"] = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    fitted_model.fit(X_train, y_train, **fit_kwargs)
    return fitted_model


def compute_temporal_oof_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_templates: dict[str, Pipeline],
    cv,
    sample_weight: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = y_train.copy()

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must have the same length as X_train")

    oof_scores: dict[str, np.ndarray] = {}
    for name, model_template in model_templates.items():
        if not isinstance(model_template, Pipeline):
            logger.warning(
                "Skipping temporal OOF scores for {}: unsupported model type {}",
                name,
                model_template.__class__.__name__,
            )
            continue

        model_scores = np.full(len(X_train), np.nan, dtype=float)
        folds_used = 0
        for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
            y_fold_train = y_train.iloc[train_idx]
            class_counts = pd.Series(y_fold_train).value_counts()
            if len(class_counts) < 2 or class_counts.min() < 2:
                logger.warning(
                    "OOF fold {} for {} skipped: insufficient class support ({})",
                    fold_number,
                    name,
                    class_counts.to_dict(),
                )
                continue

            X_fold_train = X_train.iloc[train_idx].copy()
            X_fold_validation = X_train.iloc[val_idx].copy()
            w_fold_train = sample_weight[train_idx] if sample_weight is not None else None
            fold_model = fit_pipeline_from_template(
                model_template,
                X_fold_train,
                y_fold_train,
                sample_weight=w_fold_train,
            )
            model_scores[val_idx] = fold_model.predict_proba(X_fold_validation)[:, 1]
            folds_used += 1

        logger.info(
            "Temporal OOF scores for {}: {:,} / {:,} rows covered across {} folds",
            name,
            int(np.isfinite(model_scores).sum()),
            len(model_scores),
            folds_used,
        )
        oof_scores[name] = model_scores

    return oof_scores


def train_stacking(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_models: dict[str, Pipeline],
    cv,
    sample_weight: np.ndarray | None = None,
) -> TemporalStackingClassifier:
    if not base_models:
        raise ValueError("Temporal stacking requires at least one base model")

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = y_train.copy()

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must have the same length as X_train")

    base_model_names = list(base_models)
    meta_feature_names = [f"stack__{sanitize_output_name(name)}" for name in base_model_names]
    oof_meta_features = np.full((len(X_train), len(base_model_names)), np.nan, dtype=float)
    fold_training_positions: list[np.ndarray] = []
    fold_validation_positions: list[np.ndarray] = []

    logger.info(
        "{} base learners -> LR meta-learner, {} temporal folds",
        len(base_model_names),
        cv.n_splits,
    )

    for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        y_fold_train = y_train.iloc[train_idx]
        class_counts = pd.Series(y_fold_train).value_counts()
        if len(class_counts) < 2 or class_counts.min() < 2:
            logger.warning(
                "Stacking fold {} skipped: fit window has insufficient class support ({})",
                fold_number,
                class_counts.to_dict(),
            )
            continue

        X_fold_train = X_train.iloc[train_idx].copy()
        X_fold_validation = X_train.iloc[val_idx].copy()
        w_fold_train = sample_weight[train_idx] if sample_weight is not None else None

        for model_idx, model_name in enumerate(base_model_names):
            fold_model = fit_pipeline_from_template(
                base_models[model_name],
                X_fold_train,
                y_fold_train,
                sample_weight=w_fold_train,
            )
            oof_meta_features[val_idx, model_idx] = fold_model.predict_proba(X_fold_validation)[:, 1]

        fold_training_positions.append(np.asarray(train_idx, dtype=int))
        fold_validation_positions.append(np.asarray(val_idx, dtype=int))

    meta_training_mask = np.isfinite(oof_meta_features).all(axis=1)
    if not meta_training_mask.any():
        raise ValueError("Temporal stacking produced no out-of-fold predictions")

    y_meta = y_train.iloc[meta_training_mask]
    meta_class_counts = pd.Series(y_meta).value_counts()
    if len(meta_class_counts) < 2:
        raise ValueError("Temporal stacking meta-learner requires at least 2 classes in OOF predictions")

    meta_training_frame = pd.DataFrame(
        oof_meta_features[meta_training_mask],
        columns=meta_feature_names,
        index=X_train.index[meta_training_mask],
    )
    meta_model = LogisticRegression(max_iter=20_000, random_state=RANDOM_STATE)
    meta_fit_kwargs = {}
    if sample_weight is not None:
        meta_fit_kwargs["sample_weight"] = sample_weight[meta_training_mask]
    meta_model.fit(meta_training_frame, y_meta, **meta_fit_kwargs)

    logger.info(
        "Temporal stacking meta-learner fit on {:,} OOF rows across {} folds ({} rows excluded)",
        len(meta_training_frame),
        len(fold_validation_positions),
        len(X_train) - len(meta_training_frame),
    )

    return TemporalStackingClassifier(
        named_estimators_=dict(base_models),
        final_estimator_=meta_model,
        base_model_names_=base_model_names,
        meta_feature_names_=meta_feature_names,
        meta_training_positions_=np.flatnonzero(meta_training_mask),
        fold_training_positions_=fold_training_positions,
        fold_validation_positions_=fold_validation_positions,
    )
