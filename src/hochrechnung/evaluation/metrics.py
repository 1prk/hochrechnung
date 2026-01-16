"""
Evaluation metrics for regression models.

Provides standardized metrics computation and reporting.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    """
    Standard regression metrics.

    Attributes:
        r2: R² (coefficient of determination)
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        mape: Mean Absolute Percentage Error
        n_samples: Number of samples
    """

    r2: float
    rmse: float
    mae: float
    mape: float
    n_samples: int

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"R²={self.r2:.4f}, RMSE={self.rmse:.2f}, "
            f"MAE={self.mae:.2f}, MAPE={self.mape:.2%}"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RegressionMetrics:
    """
    Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RegressionMetrics object.
    """
    # Ensure arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        log.warning("Empty arrays provided for metrics")
        return RegressionMetrics(r2=0.0, rmse=0.0, mae=0.0, mape=0.0, n_samples=0)

    metrics = RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
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
