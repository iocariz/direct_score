"""Production scoring API: load a trained model and score individual applicants.

Usage:
    from scoring import ScoringService
    service = ScoringService.from_output_dir("output")
    result = service.score_applicant({"INCOME_T1": 45000, "AGE_T1": 32, ...})
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ScoringResult:
    """Result of scoring a single applicant."""

    predicted_pd: float
    risk_tier: str
    model_name: str
    model_version: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "predicted_pd": self.predicted_pd,
            "risk_tier": self.risk_tier,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "warnings": self.warnings,
        }


# Default PD thresholds for risk tiering
DEFAULT_TIER_THRESHOLDS = [
    (0.00, 0.03, "LOW"),
    (0.03, 0.06, "MODERATE"),
    (0.06, 0.10, "ELEVATED"),
    (0.10, 0.20, "HIGH"),
    (0.20, 1.01, "VERY_HIGH"),
]


def _compute_model_version(model_path: Path, feature_cols: list[str]) -> str:
    """Deterministic version hash from model file + feature set."""
    h = hashlib.sha256()
    h.update(model_path.name.encode())
    h.update(str(model_path.stat().st_size).encode())
    h.update(",".join(sorted(feature_cols)).encode())
    return h.hexdigest()[:12]


def _assign_risk_tier(
    pd_value: float,
    thresholds: list[tuple[float, float, str]] | None = None,
) -> str:
    if thresholds is None:
        thresholds = DEFAULT_TIER_THRESHOLDS
    for lo, hi, tier in thresholds:
        if lo <= pd_value < hi:
            return tier
    return "UNKNOWN"


class ScoringService:
    """Loads a trained model and scores applicants one at a time."""

    def __init__(
        self,
        model,
        model_name: str,
        feature_cols: list[str],
        model_version: str = "unknown",
        training_stats: dict | None = None,
        tier_thresholds: list[tuple[float, float, str]] | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.model_version = model_version
        self.training_stats = training_stats or {}
        self.tier_thresholds = tier_thresholds

    @classmethod
    def from_output_dir(
        cls,
        output_dir: str | Path,
        model_name: str | None = None,
    ) -> "ScoringService":
        """Load the recommended (or specified) model from pipeline output."""
        output_path = Path(output_dir)
        models_dir = output_path / "models"

        # Determine which model to load
        if model_name is None:
            selection_path = output_path / "model_selection.csv"
            if selection_path.exists():
                sel_df = pd.read_csv(selection_path)
                recommended = sel_df.loc[sel_df["recommended"].fillna(False).astype(bool)]
                if not recommended.empty:
                    model_name = str(recommended.iloc[0]["model"])
            if model_name is None:
                model_name = "Logistic Regression"
        logger.info("Loading model: {}", model_name)

        # Load model
        safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")
        model_path = models_dir / f"{safe_name}.joblib"
        if not model_path.exists():
            # Try calibrated variant
            cal_safe = f"{safe_name}_calibrated"
            cal_path = models_dir / f"{cal_safe}.joblib"
            if cal_path.exists():
                model_path = cal_path
                model_name = f"{model_name} (calibrated)"
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)

        # Load feature list from variable dictionary
        feature_cols = []
        var_dict_path = output_path / "variable_dictionary.csv"
        if var_dict_path.exists():
            var_df = pd.read_csv(var_dict_path)
            feature_cols = var_df["feature"].tolist()

        # Load training stats for input validation
        training_stats = {}
        dq_path = output_path / "data_quality_development_fit.csv"
        if dq_path.exists():
            dq_df = pd.read_csv(dq_path)
            for _, row in dq_df.iterrows():
                feat = row["feature"]
                training_stats[feat] = {
                    "type": row.get("type", "unknown"),
                    "min": row.get("min", None),
                    "max": row.get("max", None),
                    "missing_pct": row.get("missing_pct", 0),
                    "n_unique": row.get("n_unique", None),
                }

        tier_thresholds = None
        tier_path = output_path / "risk_tiers.json"
        if tier_path.exists():
            with tier_path.open("r", encoding="utf-8") as f:
                tier_thresholds = json.load(f)

        version = _compute_model_version(model_path, feature_cols)
        logger.info("Loaded {} (version {}, {} features)", model_name, version, len(feature_cols))

        return cls(
            model=model,
            model_name=model_name,
            feature_cols=feature_cols,
            model_version=version,
            training_stats=training_stats,
            tier_thresholds=tier_thresholds,
        )

    def _validate_input(self, row_df: pd.DataFrame) -> list[str]:
        """Check for out-of-distribution values. Returns warnings."""
        warnings = []
        for feat, stats in self.training_stats.items():
            if feat not in row_df.columns:
                continue
            val = row_df[feat].iloc[0]
            if pd.isna(val):
                if stats.get("missing_pct", 0) < 0.01:
                    warnings.append(f"{feat}: missing (rare in training, {stats['missing_pct']:.1%} missing)")
                continue
            if stats.get("type") == "numerical":
                train_min = stats.get("min")
                train_max = stats.get("max")
                if train_min is not None and not pd.isna(train_min) and float(val) < float(train_min):
                    warnings.append(f"{feat}: {val} below training min ({train_min})")
                if train_max is not None and not pd.isna(train_max) and float(val) > float(train_max):
                    warnings.append(f"{feat}: {val} above training max ({train_max})")
        return warnings

    def score_applicant(self, features: dict) -> ScoringResult:
        """Score a single applicant from a dictionary of raw features.

        Args:
            features: Dictionary mapping feature names to values.

        Returns:
            ScoringResult with predicted PD, risk tier, and any warnings.
        """
        row_df = pd.DataFrame([features])

        # Ensure all required columns exist (fill missing with NaN)
        for col in self.feature_cols:
            if col not in row_df.columns:
                row_df[col] = np.nan

        row_df = row_df[self.feature_cols]
        warnings = self._validate_input(row_df)

        proba = self.model.predict_proba(row_df)[:, 1]
        pd_value = float(proba[0])
        tier = _assign_risk_tier(pd_value, self.tier_thresholds)

        return ScoringResult(
            predicted_pd=pd_value,
            risk_tier=tier,
            model_name=self.model_name,
            model_version=self.model_version,
            warnings=warnings,
        )

    def score_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch of applicants.

        Args:
            features_df: DataFrame where each row is an applicant.

        Returns:
            DataFrame with predicted_pd, risk_tier, and warning_count columns.
        """
        df = features_df.copy()
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = np.nan

        proba = self.model.predict_proba(df[self.feature_cols])[:, 1]
        result = pd.DataFrame({
            "predicted_pd": proba,
            "risk_tier": [_assign_risk_tier(p, self.tier_thresholds) for p in proba],
        })
        return result
