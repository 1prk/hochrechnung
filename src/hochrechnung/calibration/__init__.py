"""
Post-prediction calibration system for DTV estimates.

This module provides calibrators to correct systematic bias when
transferring pretrained DTV models to new regions using local
counting station data.

Scientific references:
    - Richter et al. (2025): "GPS data is derivable only from a subset
      of the cycling population"
    - Kaiser et al. (2025): "By incorporating just 10 days of sample
      counts ... we are able to almost halve the error"
    - Camacho-Torregrosa et al. (2021): Defines "Strava Usage Rate (SUR)"
      as an expansion/calibration factor

Available calibrators:
    - GlobalMultiplicativeCalibrator: Simple linear scaling y_cal = factor * y_pred
    - LogLinearCalibrator: Log-space calibration log1p(y_cal) = a * log1p(y_pred) + b
    - StratifiedCalibrator: Per-stratum calibration with global fallback
    - SpatiallyWeightedCalibrator: IDW-based spatial calibration (not implemented)

Example usage:
    >>> from hochrechnung.calibration import (
    ...     LogLinearCalibrator,
    ...     calibrate_and_evaluate,
    ...     save_calibrator,
    ... )
    >>> calibrator = LogLinearCalibrator(alpha=1.0)
    >>> result = calibrate_and_evaluate(calibrator, y_pred, y_true)
    >>> print(f"MAE reduction: {result.improvement['mae_reduction']:.1%}")
    >>> save_calibrator(result.calibrator, output_path, result)
"""

from hochrechnung.calibration.base import CalibrationParams, CalibratorBase
from hochrechnung.calibration.export import (
    export_calibration_data,
    flag_calibration_stations,
)
from hochrechnung.calibration.server import start_calibration_verification_server
from hochrechnung.calibration.loader import (
    CalibrationDataLoader,
    load_verified_calibration_counters,
    save_verified_calibration_counters,
)
from hochrechnung.calibration.loglinear import LogLinearCalibrator
from hochrechnung.calibration.multiplicative import GlobalMultiplicativeCalibrator
from hochrechnung.calibration.persistence import load_calibrator, save_calibrator
from hochrechnung.calibration.spatial import SpatiallyWeightedCalibrator
from hochrechnung.calibration.stratified import StratifiedCalibrator
from hochrechnung.calibration.validation import (
    CalibrationResult,
    LOOCVMetrics,
    calibrate_and_evaluate,
    run_loocv,
)
from hochrechnung.config.settings import CalibratorType

__all__ = [
    # Types
    "CalibratorType",
    "CalibrationParams",
    "CalibrationResult",
    "LOOCVMetrics",
    # Base class
    "CalibratorBase",
    # Calibrators
    "GlobalMultiplicativeCalibrator",
    "LogLinearCalibrator",
    "StratifiedCalibrator",
    "SpatiallyWeightedCalibrator",
    # Functions
    "calibrate_and_evaluate",
    "run_loocv",
    "save_calibrator",
    "load_calibrator",
    # Data loading
    "CalibrationDataLoader",
    "load_verified_calibration_counters",
    "save_verified_calibration_counters",
    # Data export
    "export_calibration_data",
    "flag_calibration_stations",
    # Verification server
    "start_calibration_verification_server",
    # Factory function
    "create_calibrator",
]


def create_calibrator(
    calibrator_type: CalibratorType | str,
    stratify_by: list[str] | None = None,
    min_stations_per_stratum: int = 3,
    alpha: float = 1.0,
) -> CalibratorBase:
    """Factory function to create calibrator by type.

    Args:
        calibrator_type: Type of calibrator to create.
        stratify_by: Columns for stratified calibration (if applicable).
        min_stations_per_stratum: Minimum stations per stratum for StratifiedCalibrator.
        alpha: Ridge regularization for LogLinearCalibrator.

    Returns:
        Instantiated calibrator.

    Raises:
        ValueError: If calibrator_type is unknown.
        NotImplementedError: If SpatiallyWeightedCalibrator is requested.
    """
    # Normalize string to enum
    if isinstance(calibrator_type, str):
        try:
            calibrator_type = CalibratorType(calibrator_type)
        except ValueError:
            raise ValueError(
                f"Unknown calibrator type: {calibrator_type}. "
                f"Valid types: {[t.value for t in CalibratorType]}"
            ) from None

    if calibrator_type == CalibratorType.GLOBAL_MULTIPLICATIVE:
        return GlobalMultiplicativeCalibrator()

    if calibrator_type == CalibratorType.LOG_LINEAR:
        return LogLinearCalibrator(alpha=alpha)

    if calibrator_type == CalibratorType.STRATIFIED:
        if stratify_by is None:
            raise ValueError("stratify_by is required for StratifiedCalibrator")
        return StratifiedCalibrator(
            stratify_by=stratify_by,
            min_n_per_stratum=min_stations_per_stratum,
        )

    if calibrator_type == CalibratorType.SPATIALLY_WEIGHTED:
        raise NotImplementedError(
            "SpatiallyWeightedCalibrator is not yet implemented. "
            "Use 'global_multiplicative' or 'log_linear' instead."
        )

    raise ValueError(f"Unknown calibrator type: {calibrator_type}")
