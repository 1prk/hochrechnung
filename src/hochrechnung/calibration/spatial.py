"""
Spatially-weighted calibrator stub.

Not yet implemented. Will use inverse distance weighting to compute
location-specific calibration factors based on nearby counting stations.
"""

import numpy as np
import pandas as pd

from hochrechnung.calibration.base import CalibrationParams, CalibratorBase


class SpatiallyWeightedCalibrator(CalibratorBase):
    """Spatially-weighted calibration (not yet implemented).

    This calibrator will compute location-specific calibration factors
    using inverse distance weighting from nearby counting stations.

    Planned features:
        - IDW interpolation of calibration factors
        - Configurable distance kernel (e.g., Gaussian, linear)
        - Maximum interpolation distance cutoff
    """

    def __init__(self, bandwidth_m: float = 10000.0) -> None:
        """Initialize spatially-weighted calibrator.

        Args:
            bandwidth_m: Bandwidth in meters for distance weighting.
        """
        super().__init__()
        self.bandwidth_m = bandwidth_m

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> "SpatiallyWeightedCalibrator":
        """Fit spatially-weighted calibrator.

        Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SpatiallyWeightedCalibrator is not yet implemented. "
            "Use 'global_multiplicative' or 'log_linear' instead."
        )

    def predict(
        self,
        y_pred: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply spatially-weighted calibration.

        Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SpatiallyWeightedCalibrator is not yet implemented."
        )

    def get_params(self) -> CalibrationParams:
        """Get fitted parameters.

        Not yet implemented.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "SpatiallyWeightedCalibrator is not yet implemented."
        )
