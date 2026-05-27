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

from training_constants import DEFAULT_TIER_THRESHOLDS as _SHARED_DEFAULT_TIER_THRESHOLDS


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


# Default PD thresholds for risk tiering — re-exported from training_constants
# so that training (writer of risk_tiers.json) and scoring (consumer) share a
# single source of truth.
DEFAULT_TIER_THRESHOLDS = list(_SHARED_DEFAULT_TIER_THRESHOLDS)


def _compute_file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file's full contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_model_version(model_path: Path, feature_cols: list[str]) -> str:
    """Deterministic version hash from model file content + feature set.

    Uses the full file SHA-256 (not size) so that any change to the model
    binary surfaces as a different version string.
    """
    h = hashlib.sha256()
    h.update(model_path.name.encode())
    h.update(_compute_file_sha256(model_path).encode())
    h.update(",".join(sorted(feature_cols)).encode())
    return h.hexdigest()[:12]


def _sanitize_model_name(name: str) -> str:
    """Reject path-traversal in a model name and reduce to a basename.

    Mirrors training_reporting.sanitize_output_name, then takes the final
    path component to defend against any traversal sequence that survived
    sanitization.
    """
    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("\x00", "")
    )
    cleaned = cleaned.strip("._-")
    cleaned = cleaned or "model"
    # Final guard: take only the basename in case any separator slipped through.
    return Path(cleaned).name


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
        """Load the recommended (or specified) model from pipeline output.

        Defends against path traversal by:
          - sanitizing model_name to a basename
          - resolving model_path and asserting it is under models_dir.resolve()
          - verifying SHA-256 against checksums.json (when present) before
            calling joblib.load (joblib.load uses pickle and is RCE-prone if
            the artifact is attacker-controlled).
        """
        output_path = Path(output_dir).resolve()
        models_dir = (output_path / "models").resolve()

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

        # Sanitize name and contain path under models_dir
        safe_name = _sanitize_model_name(model_name)
        model_path = (models_dir / f"{safe_name}.joblib").resolve()
        if not model_path.is_relative_to(models_dir):
            raise ValueError(f"Refusing to load model outside models_dir: {model_path}")
        if not model_path.exists():
            cal_safe = _sanitize_model_name(f"{safe_name}_calibrated")
            cal_path = (models_dir / f"{cal_safe}.joblib").resolve()
            if cal_path.is_relative_to(models_dir) and cal_path.exists():
                model_path = cal_path
                model_name = f"{model_name} (calibrated)"
            else:
                raise FileNotFoundError(f"Model not found: {model_path.name} in {models_dir}")

        # Verify integrity against checksums.json if present (training writes it).
        checksums_path = models_dir / "checksums.json"
        if checksums_path.exists():
            with checksums_path.open("r", encoding="utf-8") as f:
                checksums = json.load(f)
            expected = checksums.get(model_path.name)
            if expected is None:
                raise ValueError(
                    f"No checksum entry for {model_path.name} in {checksums_path}; refusing to load."
                )
            actual = _compute_file_sha256(model_path)
            if actual != expected:
                raise ValueError(
                    f"Checksum mismatch for {model_path.name}: expected {expected}, got {actual}."
                )
        else:
            logger.warning(
                "No checksums.json in {} — loading without integrity check.", models_dir
            )

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
        """Check for out-of-distribution values. Returns flag-only warnings.

        Warning strings deliberately omit the actual applicant value to keep
        ScoringResult free of PII (income, age, etc.). Training-set bounds are
        included because they are population-level statistics, not personal data.
        """
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
                    warnings.append(f"{feat}: below training min ({train_min})")
                if train_max is not None and not pd.isna(train_max) and float(val) > float(train_max):
                    warnings.append(f"{feat}: above training max ({train_max})")
        return warnings

    # Maximum fraction of expected feature columns allowed to be missing in
    # caller-supplied input. Above this we assume the caller passed raw features
    # without running the training-time engineering pipeline (engineer_features
    # + interactions + frequency encoding + group stats), which would silently
    # produce garbage scores. The threshold is generous (50%) because the model
    # is robust to a few legitimately-missing values handled by the imputer.
    _MAX_MISSING_FEATURE_FRACTION = 0.5

    def _assert_engineered_features_present(self, df: pd.DataFrame) -> None:
        """Loudly reject input that looks like raw applicant features.

        The trained sklearn Pipeline expects features produced by the training
        pipeline (engineer_features + add_interactions + add_modeling_features).
        If most of those columns are missing, the model would predict from an
        all-NaN matrix and quietly return invalid scores — this is the failure
        mode flagged in the audit. Detect that case and raise instead.
        """
        if not self.feature_cols:
            return
        present_mask = pd.Series(self.feature_cols).isin(df.columns)
        n_present_columns = int(present_mask.sum())
        n_expected = len(self.feature_cols)
        if n_expected == 0:
            return
        missing_fraction = 1.0 - (n_present_columns / n_expected)
        if missing_fraction > self._MAX_MISSING_FEATURE_FRACTION:
            sample_missing = [c for c, p in zip(self.feature_cols, present_mask) if not p][:8]
            raise ValueError(
                "ScoringService received input missing "
                f"{missing_fraction:.0%} of expected feature columns "
                f"({n_expected - n_present_columns}/{n_expected}). "
                "This usually means the caller passed raw applicant features "
                "without running the training-time engineering pipeline. "
                f"First missing columns: {sample_missing}. "
                "Run engineer_features + add_interactions + add_modeling_features "
                "(or use the saved feature_engineering artifacts) before scoring."
            )

    def score_applicant(self, features: dict) -> ScoringResult:
        """Score a single applicant from a dictionary of pre-engineered features.

        The model was trained on engineered features (engineer_features +
        interactions + frequency encoding + group statistics). Callers must
        run the same engineering pipeline before invoking this method. If most
        engineered columns are missing, this method raises rather than scoring
        silently from NaN.

        Args:
            features: Dictionary mapping feature names to values. Must include
                the engineered feature set the model was trained on.

        Returns:
            ScoringResult with predicted PD, risk tier, and any OOD warnings.
        """
        row_df = pd.DataFrame([features])
        self._assert_engineered_features_present(row_df)

        # Fill genuinely-missing columns with NaN — the imputer in the trained
        # pipeline is responsible for handling these.
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
        """Score a batch of pre-engineered applicants.

        See `score_applicant` for the feature-engineering contract. Like
        `score_applicant`, this raises if too many engineered columns are
        missing rather than silently scoring from NaN.

        Args:
            features_df: DataFrame where each row is an applicant.

        Returns:
            DataFrame with predicted_pd and risk_tier columns.
        """
        self._assert_engineered_features_present(features_df)
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
