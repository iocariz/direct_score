"""Temporal split utilities — strict no-look-ahead.

All cross-validation, calibration splits, feature-discovery splits, and
rolling out-of-time windows in this codebase MUST come from this module.
StratifiedKFold(shuffle=True) is forbidden anywhere in the pipeline because
it leaks future outcomes into earlier folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from training_constants import (
    CALIBRATION_FRACTION,
    FEATURE_DISCOVERY_FRACTION,
    ROLLING_OOT_MAX_WINDOWS,
)


class TemporalExpandingCV:
    def __init__(self, dates, n_splits: int = 5) -> None:
        dates = pd.to_datetime(pd.Series(dates), errors="raise")
        if dates.isna().any():
            raise ValueError("TemporalExpandingCV dates must not contain missing values")

        unique_dates = pd.Index(np.sort(dates.unique()))
        if len(unique_dates) < n_splits + 1:
            raise ValueError(
                f"TemporalExpandingCV requires at least {n_splits + 1} distinct date blocks, "
                f"got {len(unique_dates)}"
            )

        counts_by_date = dates.value_counts().sort_index()
        cumulative_counts = counts_by_date.cumsum().to_numpy()
        targets = np.linspace(0, len(dates), n_splits + 2)[1:-1]
        boundaries = [0]
        last_boundary = 0
        for fold_idx, target in enumerate(targets, start=1):
            boundary = int(np.searchsorted(cumulative_counts, target, side="left")) + 1
            min_boundary = last_boundary + 1
            max_boundary = len(unique_dates) - (n_splits - fold_idx + 1)
            boundary = min(max(boundary, min_boundary), max_boundary)
            boundaries.append(boundary)
            last_boundary = boundary
        boundaries.append(len(unique_dates))

        self._folds: list[tuple[np.ndarray, np.ndarray]] = []
        self.fold_boundaries_: list[dict] = []
        dates_array = dates.to_numpy()
        for k in range(n_splits):
            train_dates = unique_dates[:boundaries[k + 1]]
            val_dates = unique_dates[boundaries[k + 1]:boundaries[k + 2]]
            if len(train_dates) == 0 or len(val_dates) == 0:
                raise ValueError("TemporalExpandingCV produced an empty training or validation fold")
            train_idx = np.flatnonzero(np.isin(dates_array, train_dates))
            val_idx = np.flatnonzero(np.isin(dates_array, val_dates))
            train_max = pd.Timestamp(train_dates[-1])
            val_min = pd.Timestamp(val_dates[0])
            if not train_max < val_min:
                raise ValueError("TemporalExpandingCV validation dates must be strictly later than training dates")
            self._folds.append((train_idx, val_idx))
            self.fold_boundaries_.append(
                {
                    "fold": k,
                    "train_start": pd.Timestamp(train_dates[0]),
                    "train_end": train_max,
                    "val_start": val_min,
                    "val_end": pd.Timestamp(val_dates[-1]),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                }
            )
        self.n_splits = len(self._folds)

    def split(self, X=None, y=None, groups=None):
        for train_idx, val_idx in self._folds:
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


def temporal_calibration_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates,
    calibration_fraction: float = CALIBRATION_FRACTION,
    sample_weight: np.ndarray | None = None,
):
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"), index=X.index)
    if dates.isna().any():
        raise ValueError("Calibration dates must not contain missing values")
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError("X, y, and dates must have the same length")
    if sample_weight is not None and len(sample_weight) != len(dates):
        raise ValueError("sample_weight must have the same length as X")

    if sample_weight is not None:
        reference_mask = np.asarray(sample_weight) == 1.0
        reference_dates = dates.loc[reference_mask]
    else:
        reference_dates = dates

    if reference_dates.empty:
        raise ValueError("Calibration split requires at least one booked row")

    counts_by_date = reference_dates.value_counts().sort_index()
    if len(counts_by_date) < 2:
        raise ValueError("Calibration split requires at least 2 distinct date blocks")

    target_rows = max(1, int(np.ceil(len(reference_dates) * calibration_fraction)))
    calibration_start = counts_by_date.index[-1]
    cumulative_rows = 0
    for date_value, count in counts_by_date.sort_index(ascending=False).items():
        cumulative_rows += int(count)
        calibration_start = date_value
        if cumulative_rows >= target_rows:
            break

    calibration_mask = dates >= calibration_start
    fit_mask = ~calibration_mask
    if not calibration_mask.any() or not fit_mask.any():
        raise ValueError("Calibration split must produce non-empty fit and calibration sets")

    X_fit = X.loc[fit_mask].copy()
    X_calib = X.loc[calibration_mask].copy()
    y_fit = y.loc[fit_mask].copy()
    y_calib = y.loc[calibration_mask].copy()
    dates_fit = dates.loc[fit_mask].to_numpy()
    dates_calib = dates.loc[calibration_mask].to_numpy()

    if sample_weight is None:
        return X_fit, X_calib, y_fit, y_calib, dates_fit, dates_calib

    sample_weight = np.asarray(sample_weight)
    w_fit = sample_weight[fit_mask.to_numpy()]
    w_calib = sample_weight[calibration_mask.to_numpy()]
    return X_fit, X_calib, y_fit, y_calib, w_fit, w_calib, dates_fit, dates_calib


def resolve_temporal_feature_discovery_cutoff(
    dates,
    discovery_fraction: float = FEATURE_DISCOVERY_FRACTION,
) -> pd.Timestamp:
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"))
    if dates.isna().any():
        raise ValueError("Feature discovery dates must not contain missing values")
    if not 0 < discovery_fraction < 1:
        raise ValueError("discovery_fraction must be strictly between 0 and 1")

    counts_by_date = dates.value_counts().sort_index()
    if len(counts_by_date) < 2:
        raise ValueError("Feature discovery split requires at least 2 distinct date blocks")

    target_rows = max(1, int(np.ceil(len(dates) * discovery_fraction)))
    discovery_end = counts_by_date.index[0]
    cumulative_rows = 0
    for date_value, count in counts_by_date.items():
        cumulative_rows += int(count)
        discovery_end = date_value
        if cumulative_rows >= target_rows:
            break

    discovery_mask = dates <= discovery_end
    estimation_mask = ~discovery_mask
    if not discovery_mask.any() or not estimation_mask.any():
        raise ValueError("Feature discovery split must produce non-empty discovery and estimation sets")
    return pd.Timestamp(discovery_end)


def temporal_feature_discovery_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates,
    discovery_fraction: float = FEATURE_DISCOVERY_FRACTION,
    discovery_end: str | pd.Timestamp | None = None,
):
    dates = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"), index=X.index)
    if dates.isna().any():
        raise ValueError("Feature discovery dates must not contain missing values")
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError("X, y, and dates must have the same length")
    if discovery_end is None:
        discovery_end = resolve_temporal_feature_discovery_cutoff(
            dates,
            discovery_fraction=discovery_fraction,
        )
    else:
        discovery_end = pd.Timestamp(discovery_end)

    discovery_mask = dates <= discovery_end
    estimation_mask = ~discovery_mask
    if not discovery_mask.any() or not estimation_mask.any():
        raise ValueError("Feature discovery split must produce non-empty discovery and estimation sets")

    X_discovery = X.loc[discovery_mask].copy()
    X_estimation = X.loc[estimation_mask].copy()
    y_discovery = y.loc[discovery_mask].copy()
    y_estimation = y.loc[estimation_mask].copy()
    dates_discovery = dates.loc[discovery_mask].to_numpy()
    dates_estimation = dates.loc[estimation_mask].to_numpy()
    return X_discovery, X_estimation, y_discovery, y_estimation, dates_discovery, dates_estimation


def make_temporal_cv(dates, max_splits: int = 5) -> TemporalExpandingCV:
    distinct_dates = pd.Index(np.sort(pd.to_datetime(np.asarray(dates), errors="raise").unique()))
    if len(distinct_dates) < 3:
        raise ValueError("Temporal CV requires at least 3 distinct date blocks")
    n_splits = min(max_splits, len(distinct_dates) - 1)
    if n_splits != max_splits:
        logger.info(
            "Temporal CV: using {} folds from {} distinct date blocks",
            n_splits,
            len(distinct_dates),
        )
    return TemporalExpandingCV(dates, n_splits=n_splits)


def build_rolling_oot_windows(
    dates,
    max_windows: int = ROLLING_OOT_MAX_WINDOWS,
    min_train_date_blocks: int = 2,
) -> list[dict]:
    dates_series = pd.Series(pd.to_datetime(np.asarray(dates), errors="raise"))
    if dates_series.isna().any():
        raise ValueError("Rolling OOT dates must not contain missing values")
    if max_windows < 1:
        raise ValueError("max_windows must be at least 1")
    if min_train_date_blocks < 1:
        raise ValueError("min_train_date_blocks must be at least 1")

    unique_dates = pd.Index(np.sort(dates_series.unique()))
    if len(unique_dates) < min_train_date_blocks + 1:
        raise ValueError(
            "Rolling OOT validation requires at least "
            f"{min_train_date_blocks + 1} distinct date blocks"
        )

    validation_blocks = unique_dates[min_train_date_blocks:]
    n_windows = min(max_windows, len(validation_blocks))
    window_groups = [
        pd.Index(group)
        for group in np.array_split(validation_blocks.to_numpy(), n_windows)
        if len(group) > 0
    ]

    windows = []
    dates_array = dates_series.to_numpy()
    for fold_idx, validation_group in enumerate(window_groups, start=1):
        validation_start = pd.Timestamp(validation_group[0])
        validation_end = pd.Timestamp(validation_group[-1])
        train_dates = unique_dates[unique_dates < validation_start]
        if len(train_dates) < min_train_date_blocks:
            raise ValueError("Rolling OOT validation produced too few training date blocks")

        train_idx = np.flatnonzero(np.isin(dates_array, train_dates))
        validation_idx = np.flatnonzero(np.isin(dates_array, validation_group))
        if len(train_idx) == 0 or len(validation_idx) == 0:
            raise ValueError("Rolling OOT validation produced an empty train or validation window")

        windows.append(
            {
                "fold": fold_idx,
                "train_idx": train_idx,
                "validation_idx": validation_idx,
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[-1]),
                "validation_start": validation_start,
                "validation_end": validation_end,
                "n_train": len(train_idx),
                "n_validation": len(validation_idx),
            }
        )
    return windows
