"""
Base classes for post-prediction calibrators.

Provides abstract interface for calibrating DTV predictions using
independent counting station data.

Scientific references:
    - Richter et al. (2025): GPS data bias in cycling populations
    - Kaiser et al. (2025): Sample count calibration halves error
    - Camacho-Torregrosa et al. (2021): Strava Usage Rate (SUR) as expansion factor
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class CalibrationParams:
    """Calibration parameters for persistence.

    Attributes:
        calibrator_type: Type identifier for the calibrator.
        params: Dictionary of fitted parameter values.
        n_stations: Number of stations used for fitting.
        fitted_at: ISO timestamp when calibrator was fitted.
    """

    calibrator_type: str
    params: dict
    n_stations: int
    fitted_at: str


class CalibratorBase(ABC):
    """Abstract base class for post-prediction calibrators.

    All calibrators follow a fit/predict pattern similar to scikit-learn:
    1. fit() - Learn calibration parameters from station data
    2. predict() - Apply calibration to new predictions

    The meta DataFrame is optional for simple calibrators but required
    for stratified approaches.
    """

    def __init__(self) -> None:
        self._is_fitted = False

    @abstractmethod
    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> "CalibratorBase":
        """Fit calibrator on station data.

        Args:
            y_pred: Uncalibrated predictions at station locations.
            y_true: Observed DTV values from counting stations.
            meta: Optional metadata for stratified calibration
                (e.g., infra_category, regiostar7).

        Returns:
            self (for method chaining)
        """
        ...

    @abstractmethod
    def predict(
        self,
        y_pred: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply calibration to predictions.

        Args:
            y_pred: Uncalibrated predictions to calibrate.
            meta: Optional metadata for stratified calibration.

        Returns:
            Calibrated predictions.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        ...

    @abstractmethod
    def get_params(self) -> CalibrationParams:
        """Get fitted parameters for persistence.

        Returns:
            CalibrationParams containing calibrator type and parameters.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """Whether the calibrator has been fitted."""
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Raise RuntimeError if not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before predict() or get_params()."
            )

    def _get_fitted_at(self) -> str:
        """Get current timestamp as ISO string."""
        return datetime.now().isoformat()
