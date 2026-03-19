"""Shared fixtures for training pipeline tests.

Provides small synthetic DataFrames that mirror the real data schema
so tests run in seconds without loading the 70 MB parquet file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training import temporal_split
from training_constants import (
    DROP_COLS,
    MATURITY_CUTOFF,
    MISS_CANDIDATES,
    RAW_CAT,
    RAW_NUM,
    REJECT_SCORE_COL,
    SPLIT_DATE,
    TARGET,
)
from training_features import engineer_features, select_features

_RNG = np.random.RandomState(42)


def _make_raw_df(n: int = 300, include_rejects: bool = False) -> pd.DataFrame:
    """Generate a synthetic DataFrame matching the real parquet schema."""
    if include_rejects:
        status = _RNG.choice(["Booked", "Rejected", "Canceled"], size=n, p=[0.3, 0.5, 0.2])
    else:
        status = np.full(n, "Booked")

    dates = pd.date_range("2023-06-01", periods=24, freq="MS")
    mis_date = _RNG.choice(dates, size=n)

    # Target: only for booked + matured
    target = np.full(n, np.nan)
    booked_mask = status == "Booked"
    matured_mask = mis_date <= pd.Timestamp(MATURITY_CUTOFF)
    observable = booked_mask & matured_mask
    target[observable] = _RNG.choice([0, 1], size=observable.sum(), p=[0.95, 0.05])

    data: dict = {
        "authorization_id": np.arange(n, dtype=float),
        "mis_Date": mis_date,
        "status_name": status,
        TARGET: target,
        "SCRPLUST1": _RNG.uniform(0, 100, n),
        "reject_reason": _RNG.choice(["09-score", "risk", "budget", "other"], n),
        "risk_score_rf": _RNG.uniform(10, 100, n),
        "score_RF": _RNG.uniform(300, 500, n),
        "product_type_1": np.full(n, "A"),
        "acct_booked_H0": np.full(n, 1.0),
        "INCOME_T2": np.where(_RNG.random(n) < 0.35, np.nan, _RNG.uniform(0, 100, n)),
        "rf_business_name": np.full(n, "BIZ"),
        "rf_ext_business_name": np.full(n, "EXT"),
        "a_business_name": np.full(n, "A"),
        "ext_business_name": np.full(n, "E"),
    }

    # Numerical features
    for col in RAW_NUM:
        vals = _RNG.uniform(0, 100, n).astype(float)
        # Inject missingness for MISS_CANDIDATES
        if col in MISS_CANDIDATES:
            vals[_RNG.random(n) < 0.10] = np.nan
        data[col] = vals

    # Categorical features
    cat_levels = {
        "CUSTOMER_TYPE": ["NEW", "EXISTING"],
        "FAMILY_SITUATION": ["SINGLE", "MARRIED", "OTHER"],
        "HOUSE_TYPE": ["OWN", "RENT", "FAMILY"],
        "product_type_2": ["PL", "CC"],
        "product_type_3": ["CASH", "REVOLVING"],
        "CSP": ["A", "B", "C"],
        "CPRO": ["P1", "P2"],
        "CMAT": ["M1", "M2", "M3"],
        "ESTCLI1": ["E1", "E2", "E3"],
        "ESTCLI2": ["F1", "F2"],
        "CSECTOR": ["S1", "S2"],
        "FLAG_COTIT": ["Y", "N"],
    }
    for col in RAW_CAT:
        levels = cat_levels.get(col, ["X", "Y"])
        vals = _RNG.choice(levels, n)
        if col in MISS_CANDIDATES:
            mask = _RNG.random(n) < 0.10
            vals = np.where(mask, None, vals)
        data[col] = vals

    return pd.DataFrame(data)


def _make_structured_temporal_df(include_rejects: bool = False) -> pd.DataFrame:
    n_months = 18
    rows_per_month = 8
    df = _make_raw_df(n_months * rows_per_month, include_rejects=include_rejects)
    df["authorization_id"] = np.arange(len(df), dtype=float)
    df["mis_Date"] = np.repeat(
        pd.date_range("2023-09-01", periods=n_months, freq="MS").to_numpy(),
        rows_per_month,
    )

    if include_rejects:
        status_pattern = np.array(["Booked", "Booked", "Booked", "Booked", "Rejected", "Rejected", "Canceled", "Rejected"])
    else:
        status_pattern = np.array(["Booked"] * rows_per_month)
    df["status_name"] = np.tile(status_pattern, n_months)

    target = np.full(len(df), np.nan)
    booked_observed_mask = (
        df["status_name"].eq("Booked")
        & (df["mis_Date"] <= pd.Timestamp(MATURITY_CUTOFF))
    )
    booked_targets = np.resize(np.array([0.0, 0.0, 1.0, 0.0]), int(booked_observed_mask.sum()))
    target[np.flatnonzero(booked_observed_mask)] = booked_targets
    df[TARGET] = target
    return df


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Synthetic booked-only DataFrame (300 rows)."""
    return _make_raw_df(300, include_rejects=False)


@pytest.fixture()
def raw_df_with_rejects() -> pd.DataFrame:
    """Synthetic DataFrame with Booked + Rejected + Canceled (300 rows)."""
    return _make_raw_df(300, include_rejects=True)


@pytest.fixture()
def structured_raw_df() -> pd.DataFrame:
    return _make_structured_temporal_df(include_rejects=False)


@pytest.fixture()
def structured_raw_df_with_rejects() -> pd.DataFrame:
    return _make_structured_temporal_df(include_rejects=True)


@pytest.fixture()
def booked_df(raw_df) -> pd.DataFrame:
    """Booked-only filtered DataFrame (like load_data output)."""
    return raw_df[raw_df["status_name"] == "Booked"].copy()


@pytest.fixture()
def engineered_df(booked_df) -> pd.DataFrame:
    """Booked DataFrame after feature engineering."""
    return engineer_features(booked_df)


@pytest.fixture()
def train_test_data(engineered_df):
    """Minimal train/test split arrays for model testing."""
    feature_cols, num_cols, cat_cols = select_features(engineered_df)
    X_train, y_train, X_test, y_test, bench_risk, bench_score, train_dates = temporal_split(
        engineered_df, feature_cols,
    )
    return X_train, y_train, X_test, y_test, num_cols, cat_cols, feature_cols, train_dates
