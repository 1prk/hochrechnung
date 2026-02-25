"""
Log-linear calibrator.

Fits calibration in log1p space (consistent with model training):
    log1p(y_cal) = a * log1p(y_pred) + b

Uses Ridge regression for regularization to prevent overfitting
on small calibration datasets.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from hochrechnung.calibration.base import CalibrationParams, CalibratorBase


class LogLinearCalibrator(CalibratorBase):
    """Log-linear calibration in log1p space.

    Transforms predictions and targets to log1p space, fits a linear
    model, and transforms back. This is consistent with how DTV models
    are typically trained (with log1p target transformation).

    Model:
        log1p(y_cal) = a * log1p(y_pred) + b

    Attributes:
        alpha: Ridge regularization strength.
        a_: Fitted slope coefficient.
        b_: Fitted intercept.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Initialize calibrator.

        Args:
            alpha: Ridge regularization strength (default 1.0).
                Higher values = more regularization.
        """
        super().__init__()
        self.alpha = alpha
        self.a_: float | None = None
        self.b_: float | None = None
        self._n_stations: int = 0

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> "LogLinearCalibrator":
        """Fit log-linear calibration model.

        Args:
            y_pred: Uncalibrated predictions at station locations.
            y_true: Observed DTV values from counting stations.
            meta: Ignored (not used by this calibrator).

        Returns:
            self (for method chaining)
        """
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()

        # Transform to log1p space
        X = np.log1p(y_pred).reshape(-1, 1)
        y = np.log1p(y_true)

        # Fit Ridge regression
        model = Ridge(alpha=self.alpha)
        model.fit(X, y)

        self.a_ = float(model.coef_[0])
        self.b_ = float(model.intercept_)
        self._n_stations = len(y_pred)
        self._is_fitted = True
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply log-linear calibration.

        Args:
            y_pred: Uncalibrated predictions to calibrate.
            meta: Ignored (not used by this calibrator).

        Returns:
            Calibrated predictions (always non-negative).

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        self._check_is_fitted()

        y_pred = np.asarray(y_pred)
        log_pred = np.log1p(y_pred)
        log_cal = self.a_ * log_pred + self.b_

        # Inverse transform, ensure non-negative
        return np.maximum(np.expm1(log_cal), 0.0)

    def get_params(self) -> CalibrationParams:
        """Get fitted parameters.

        Returns:
            CalibrationParams with slope (a), intercept (b), and alpha.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        self._check_is_fitted()
        return CalibrationParams(
            calibrator_type="log_linear",
            params={
                "a": self.a_,
                "b": self.b_,
                "alpha": self.alpha,
            },
            n_stations=self._n_stations,
            fitted_at=self._get_fitted_at(),
        )
