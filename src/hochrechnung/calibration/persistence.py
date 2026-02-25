"""
Calibrator persistence (save/load).

Follows pattern from verification/persistence.py for consistency.
"""

import json
from pathlib import Path
from typing import Any

import joblib

from hochrechnung.calibration.base import CalibratorBase
from hochrechnung.calibration.validation import CalibrationResult
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


def save_calibrator(
    calibrator: CalibratorBase,
    output_path: Path,
    metrics: CalibrationResult | None = None,
) -> tuple[Path, Path]:
    """Save calibrator and metadata to disk.

    Creates two files:
        - {output_path}.calibrator.joblib: Pickled calibrator object
        - {output_path}.calibrator.json: Human-readable metadata

    Args:
        calibrator: Fitted calibrator to save.
        output_path: Base output path (without extension).
        metrics: Optional CalibrationResult with evaluation metrics.

    Returns:
        Tuple of (calibrator_path, metadata_path).
    """
    output_path = Path(output_path)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define output paths
    calibrator_path = output_path.with_suffix(".calibrator.joblib")
    metadata_path = output_path.with_suffix(".calibrator.json")

    # Save calibrator using joblib
    joblib.dump(calibrator, calibrator_path)
    log.info("Saved calibrator", path=str(calibrator_path))

    # Build metadata from calibrator params
    params = calibrator.get_params()
    metadata: dict[str, Any] = {
        "calibrator_type": params.calibrator_type,
        "params": params.params,
        "n_stations": params.n_stations,
        "fitted_at": params.fitted_at,
    }

    # Add metrics if provided
    if metrics is not None:
        metadata["uncalibrated"] = metrics.uncalibrated_metrics.to_dict()
        metadata["calibrated"] = metrics.calibrated_metrics.to_dict()
        metadata["improvement"] = metrics.improvement
        if metrics.loocv_metrics:
            metadata["loocv"] = metrics.loocv_metrics.to_dict()

    # Save metadata as JSON
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved calibrator metadata", path=str(metadata_path))

    return calibrator_path, metadata_path


def load_calibrator(path: Path) -> tuple[CalibratorBase, dict[str, Any]]:
    """Load calibrator and metadata from disk.

    Accepts either:
        - Path to .calibrator.joblib file directly
        - Base path (will append .calibrator.joblib)

    Args:
        path: Path to calibrator file or base path.

    Returns:
        Tuple of (calibrator, metadata_dict).

    Raises:
        FileNotFoundError: If calibrator file doesn't exist.
    """
    path = Path(path)

    # Determine correct paths
    if path.suffix == ".joblib":
        calibrator_path = path
        # Remove .calibrator.joblib to get base, then add .calibrator.json
        base = path.with_suffix("")  # removes .joblib
        if base.suffix == ".calibrator":
            base = base.with_suffix("")  # removes .calibrator
        metadata_path = base.with_suffix(".calibrator.json")
    else:
        calibrator_path = path.with_suffix(".calibrator.joblib")
        metadata_path = path.with_suffix(".calibrator.json")

    if not calibrator_path.exists():
        raise FileNotFoundError(f"Calibrator file not found: {calibrator_path}")

    # Load calibrator
    calibrator = joblib.load(calibrator_path)
    log.info("Loaded calibrator", path=str(calibrator_path))

    # Load metadata if available
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        log.info("Loaded calibrator metadata", path=str(metadata_path))
    else:
        log.warning("Calibrator metadata not found", path=str(metadata_path))

    return calibrator, metadata
