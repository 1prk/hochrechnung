"""
Global multiplicative calibrator.

Simple linear scaling: y_cal = factor * y_pred

The factor is computed as median(y_true / y_pred) for robustness
against outliers.
"""

import numpy as np
import pandas as pd

from hochrechnung.calibration.base import CalibrationParams, CalibratorBase


class GlobalMultiplicativeCalibrator(CalibratorBase):
    """Simple multiplicative calibration: y_cal = factor * y_pred.

    The calibration factor is computed as the median ratio of observed
    to predicted values, clipped to reasonable bounds.

    Attributes:
        MIN_FACTOR: Minimum allowed calibration factor.
        MAX_FACTOR: Maximum allowed calibration factor.
        factor_: Fitted calibration factor.
    """

    MIN_FACTOR = 0.1
    MAX_FACTOR = 10.0

    def __init__(self) -> None:
        super().__init__()
        self.factor_: float | None = None
        self._n_stations: int = 0

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> "GlobalMultiplicativeCalibrator":
        """Fit calibrator by computing median ratio.

        Args:
            y_pred: Uncalibrated predictions at station locations.
            y_true: Observed DTV values from counting stations.
            meta: Ignored (not used by this calibrator).

        Returns:
            self (for method chaining)
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()

        # Guard against zeros and negative values
        mask = (y_pred > 0) & (y_true >= 0)
        if mask.sum() < 1:
            self.factor_ = 1.0
        else:
            ratios = y_true[mask] / y_pred[mask]
            self.factor_ = float(
                np.clip(np.median(ratios), self.MIN_FACTOR, self.MAX_FACTOR)
            )

        self._n_stations = len(y_pred)
        self._is_fitted = True
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply multiplicative calibration.

        Args:
            y_pred: Uncalibrated predictions to calibrate.
            meta: Ignored (not used by this calibrator).

        Returns:
            Calibrated predictions: factor * y_pred

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        self._check_is_fitted()
        return np.asarray(y_pred) * self.factor_

    def get_params(self) -> CalibrationParams:
        """Get fitted parameters.

        Returns:
            CalibrationParams with factor and metadata.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        self._check_is_fitted()
        return CalibrationParams(
            calibrator_type="global_multiplicative",
            params={"factor": self.factor_},
            n_stations=self._n_stations,
            fitted_at=self._get_fitted_at(),
        )
