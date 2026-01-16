"""
Evaluation metrics for regression models.

Provides standardized metrics computation and reporting.
Implements metrics aligned with Friedrich et al. (2019) methodology.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    max_error as sklearn_max_error,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# SQV scaling factor for daily traffic volumes (Friedrich et al., 2019)
SQV_SCALING_FACTOR = 10_000.0


def compute_sqv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaling_factor: float = SQV_SCALING_FACTOR,
) -> float:
    """
    Compute SQV (Schätzqualitätswert / estimation quality value).

    SQV measures prediction quality on a 0-1 scale where:
    - 1.0 = perfect predictions
    - 0.0 = all predictions have error >= scaling_factor
    - SQV > 0.85 is considered good quality (Friedrich et al., 2019)

    Formula: SQV = mean(clip(1 - |error| / scaling_factor, 0, 1))

    Args:
        y_true: True values.
        y_pred: Predicted values.
        scaling_factor: Maximum error for zero quality contribution.
            Default 10,000 for daily traffic volumes.

    Returns:
        SQV value in range [0, 1].
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) == 0:
        return 0.0

    abs_errors = np.abs(y_pred - y_true)
    quality_scores = np.clip(1.0 - abs_errors / scaling_factor, 0.0, 1.0)

    return float(np.mean(quality_scores))


@dataclass(frozen=True)
class RegressionMetrics:
    """
    Standard regression metrics.

    Attributes:
        r2: R² (coefficient of determination)
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        max_error: Maximum absolute error
        sqv: SQV (Schätzqualitätswert) - estimation quality value (0-1)
        n_samples: Number of samples
    """

    r2: float
    rmse: float
    mae: float
    mape: float
    max_error: float
    sqv: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "max_error": self.max_error,
            "sqv": self.sqv,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"R²={self.r2:.4f}, RMSE={self.rmse:.2f}, "
            f"MAE={self.mae:.2f}, MAPE={self.mape:.2%}, "
            f"MaxErr={self.max_error:.0f}, SQV={self.sqv:.4f}"
        )


@dataclass(frozen=True)
class CVMetrics:
    """
    Cross-validation metrics with overfitting detection.

    Attributes:
        r2_cv: R² from cross-validation (average across folds)
        r2_no_cv: R² on full training set (without CV)
        r2_gap: Overfitting indicator (r2_no_cv - r2_cv)
        rmse: Root Mean Squared Error (CV average)
        mae: Mean Absolute Error (CV average)
        mape: Mean Absolute Percentage Error (CV average)
        max_error: Maximum absolute error (CV average)
        sqv: SQV estimation quality value (CV average)
        training_time_s: Training time in seconds
        prediction_time_s: Prediction time in seconds per sample
    """

    r2_cv: float
    r2_no_cv: float
    r2_gap: float
    rmse: float
    mae: float
    mape: float
    max_error: float
    sqv: float
    training_time_s: float
    prediction_time_s: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "r2_cv": self.r2_cv,
            "r2_no_cv": self.r2_no_cv,
            "r2_gap": self.r2_gap,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "max_error": self.max_error,
            "sqv": self.sqv,
            "training_time_s": self.training_time_s,
            "prediction_time_s": self.prediction_time_s,
        }

    @property
    def overfitting_risk(self) -> str:
        """
        Assess overfitting risk based on R² gap.

        Returns:
            'low' if gap < 0.05
            'moderate' if 0.05 <= gap < 0.10
            'high' if gap >= 0.10
        """
        if self.r2_gap < 0.05:
            return "low"
        if self.r2_gap < 0.10:
            return "moderate"
        return "high"

    def __str__(self) -> str:
        """String representation."""
        return (
            f"R²(CV)={self.r2_cv:.4f}, R²(no CV)={self.r2_no_cv:.4f}, "
            f"Gap={self.r2_gap:.3f} ({self.overfitting_risk}), "
            f"SQV={self.sqv:.4f}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sqv_scaling_factor: float = SQV_SCALING_FACTOR,
) -> RegressionMetrics:
    """
    Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        sqv_scaling_factor: Scaling factor for SQV computation.

    Returns:
        RegressionMetrics object.
    """
    # Ensure arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        log.warning("Empty arrays provided for metrics")
        return RegressionMetrics(
            r2=0.0,
            rmse=0.0,
            mae=0.0,
            mape=0.0,
            max_error=0.0,
            sqv=0.0,
            n_samples=0,
        )

    metrics = RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
        max_error=float(sklearn_max_error(y_true, y_pred)),
        sqv=compute_sqv(y_true, y_pred, sqv_scaling_factor),
        n_samples=len(y_true),
    )

    log.debug("Computed metrics", **metrics.to_dict())
    return metrics


def compute_accuracy_bands(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: list[float] | None = None,
) -> dict[str, float]:
    """
    Compute accuracy within percentage bands.

    Args:
        y_true: True values.
        y_pred: Predicted values.
        bands: List of percentage bands (default: [0.1, 0.2, 0.3, 0.5]).

    Returns:
        Dictionary of band -> accuracy ratio.
    """
    if bands is None:
        bands = [0.1, 0.2, 0.3, 0.5]

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) == 0:
        return {f"within_{int(b * 100)}pct": 0.0 for b in bands}

    # Compute relative errors
    rel_error = np.abs(y_pred - y_true) / np.maximum(y_true, 1e-10)

    results = {}
    for band in bands:
        within = np.mean(rel_error <= band)
        results[f"within_{int(band * 100)}pct"] = float(within)

    log.debug("Accuracy bands computed", **results)
    return results


def compute_residual_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute residual statistics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with residual statistics.
    """
    residuals = y_true - y_pred

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
    }
