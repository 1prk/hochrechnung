"""
Calibration validation with Leave-One-Out Cross-Validation (LOOCV).

Provides metrics for evaluating calibration quality and detecting
overfitting on small calibration datasets.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hochrechnung.calibration.base import CalibratorBase
from hochrechnung.evaluation.metrics import RegressionMetrics, compute_metrics


@dataclass
class LOOCVMetrics:
    """Leave-One-Out Cross-Validation metrics.

    Attributes:
        mae_mean: Mean absolute error (mean across folds).
        mae_std: MAE standard deviation across folds.
        rmse_mean: Root mean squared error (mean across folds).
        rmse_std: RMSE standard deviation across folds.
        per_fold_errors: List of absolute errors for each held-out sample.
    """

    mae_mean: float
    mae_std: float
    rmse_mean: float
    rmse_std: float
    per_fold_errors: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mae_mean": self.mae_mean,
            "mae_std": self.mae_std,
            "rmse_mean": self.rmse_mean,
            "rmse_std": self.rmse_std,
            "n_folds": len(self.per_fold_errors),
        }


@dataclass
class CalibrationResult:
    """Results from calibration with validation.

    Attributes:
        calibrator: The fitted calibrator.
        uncalibrated_metrics: Metrics before calibration.
        calibrated_metrics: Metrics after calibration.
        loocv_metrics: LOOCV results (if computed).
        n_stations: Number of calibration stations used.
        improvement: Dict of improvement metrics (reduction ratios).
    """

    calibrator: CalibratorBase
    uncalibrated_metrics: RegressionMetrics
    calibrated_metrics: RegressionMetrics
    loocv_metrics: LOOCVMetrics | None
    n_stations: int
    improvement: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "n_stations": self.n_stations,
            "uncalibrated": self.uncalibrated_metrics.to_dict(),
            "calibrated": self.calibrated_metrics.to_dict(),
            "loocv": self.loocv_metrics.to_dict() if self.loocv_metrics else None,
            "improvement": self.improvement,
        }


def run_loocv(
    calibrator_class: type[CalibratorBase],
    y_pred: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame | None = None,
    **calibrator_kwargs: Any,
) -> LOOCVMetrics:
    """Run Leave-One-Out Cross-Validation for calibration.

    For each station, fits calibrator on all other stations and predicts
    the held-out station. Returns aggregated error metrics.

    Args:
        calibrator_class: Class of calibrator to instantiate.
        y_pred: Uncalibrated predictions at station locations.
        y_true: Observed DTV values from counting stations.
        meta: Optional metadata for stratified calibrators.
        **calibrator_kwargs: Additional arguments passed to calibrator constructor.

    Returns:
        LOOCVMetrics with per-fold and aggregated errors.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    n = len(y_pred)

    errors = []
    for i in range(n):
        # Create leave-one-out mask
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        # Fit calibrator on n-1 samples
        cal = calibrator_class(**calibrator_kwargs)
        meta_train = meta.iloc[mask] if meta is not None else None

        try:
            cal.fit(y_pred[mask], y_true[mask], meta_train)
        except Exception:
            # If fitting fails (e.g., too few samples), use identity
            errors.append(abs(y_pred[i] - y_true[i]))
            continue

        # Predict held-out sample
        meta_test = meta.iloc[[i]] if meta is not None else None

        try:
            y_cal = cal.predict(y_pred[[i]], meta_test)
            errors.append(abs(y_cal[0] - y_true[i]))
        except Exception:
            # If prediction fails, use uncalibrated error
            errors.append(abs(y_pred[i] - y_true[i]))

    errors = np.array(errors)
    squared_errors = errors**2

    return LOOCVMetrics(
        mae_mean=float(np.mean(errors)),
        mae_std=float(np.std(errors)),
        rmse_mean=float(np.sqrt(np.mean(squared_errors))),
        rmse_std=float(np.std(np.sqrt(squared_errors))),
        per_fold_errors=errors.tolist(),
    )


def calibrate_and_evaluate(
    calibrator: CalibratorBase,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    meta: pd.DataFrame | None = None,
    *,
    run_loocv_validation: bool = True,
) -> CalibrationResult:
    """Fit calibrator and compute evaluation metrics.

    Args:
        calibrator: Calibrator instance to fit.
        y_pred: Uncalibrated predictions at station locations.
        y_true: Observed DTV values from counting stations.
        meta: Optional metadata for stratified calibrators.
        run_loocv_validation: Whether to run LOOCV (default True).

    Returns:
        CalibrationResult with fitted calibrator and metrics.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()

    # Uncalibrated metrics
    uncal_metrics = compute_metrics(y_true, y_pred)

    # Fit and apply calibration
    calibrator.fit(y_pred, y_true, meta)
    y_calibrated = calibrator.predict(y_pred, meta)

    # Calibrated metrics
    cal_metrics = compute_metrics(y_true, y_calibrated)

    # LOOCV if requested and enough samples
    loocv = None
    if run_loocv_validation and len(y_pred) >= 3:
        # Get calibrator kwargs for instantiation
        calibrator_kwargs = _get_calibrator_kwargs(calibrator)
        loocv = run_loocv(
            type(calibrator),
            y_pred,
            y_true,
            meta,
            **calibrator_kwargs,
        )

    # Compute improvement metrics
    improvement = _compute_improvement(uncal_metrics, cal_metrics)

    return CalibrationResult(
        calibrator=calibrator,
        uncalibrated_metrics=uncal_metrics,
        calibrated_metrics=cal_metrics,
        loocv_metrics=loocv,
        n_stations=len(y_pred),
        improvement=improvement,
    )


def _get_calibrator_kwargs(calibrator: CalibratorBase) -> dict[str, Any]:
    """Extract constructor kwargs from calibrator instance.

    This is a heuristic that works for our calibrators. Inspects
    common attributes to reconstruct kwargs.
    """
    kwargs: dict[str, Any] = {}

    # LogLinearCalibrator
    if hasattr(calibrator, "alpha"):
        kwargs["alpha"] = calibrator.alpha

    # StratifiedCalibrator
    if hasattr(calibrator, "stratify_by"):
        kwargs["stratify_by"] = calibrator.stratify_by
    if hasattr(calibrator, "min_n_per_stratum"):
        kwargs["min_n_per_stratum"] = calibrator.min_n_per_stratum
    if hasattr(calibrator, "base_calibrator_class"):
        kwargs["base_calibrator_class"] = calibrator.base_calibrator_class

    # SpatiallyWeightedCalibrator
    if hasattr(calibrator, "bandwidth_m"):
        kwargs["bandwidth_m"] = calibrator.bandwidth_m

    return kwargs


def _compute_improvement(
    uncal: RegressionMetrics,
    cal: RegressionMetrics,
) -> dict[str, float]:
    """Compute improvement metrics between uncalibrated and calibrated.

    Positive values indicate improvement after calibration.
    """
    # Avoid division by zero
    mae_reduction = (
        (uncal.mae - cal.mae) / uncal.mae if uncal.mae > 0 else 0.0
    )
    rmse_reduction = (
        (uncal.rmse - cal.rmse) / uncal.rmse if uncal.rmse > 0 else 0.0
    )

    return {
        "mae_reduction": mae_reduction,
        "rmse_reduction": rmse_reduction,
        "r2_gain": cal.r2 - uncal.r2,
        "sqv_gain": cal.sqv - uncal.sqv,
    }
