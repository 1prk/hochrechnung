"""
Stratified calibrator.

Per-stratum calibration with fallback to global. Groups by specified
columns (e.g., infra_category, regiostar7) and fits separate calibrators
for each group. Falls back to global calibrator for strata with too few
samples.
"""

from typing import Any

import numpy as np
import pandas as pd

from hochrechnung.calibration.base import CalibrationParams, CalibratorBase
from hochrechnung.calibration.multiplicative import GlobalMultiplicativeCalibrator


class StratifiedCalibrator(CalibratorBase):
    """Per-stratum calibration with fallback to global.

    Fits separate calibrators for each stratum (combination of grouping
    columns). Strata with fewer than min_n_per_stratum samples fall back
    to the global calibrator.

    Attributes:
        stratify_by: Column names to group by.
        min_n_per_stratum: Minimum samples per stratum for dedicated calibrator.
        base_calibrator_class: Calibrator class to use for each stratum.
        stratum_calibrators_: Dict mapping stratum keys to fitted calibrators.
        global_calibrator_: Fallback calibrator for small strata.
    """

    def __init__(
        self,
        stratify_by: list[str],
        min_n_per_stratum: int = 3,
        base_calibrator_class: type[CalibratorBase] = GlobalMultiplicativeCalibrator,
    ) -> None:
        """Initialize stratified calibrator.

        Args:
            stratify_by: Column names to group by (e.g., ['infra_category']).
            min_n_per_stratum: Minimum stations per stratum (default 3).
                Strata with fewer samples use the global calibrator.
            base_calibrator_class: Calibrator class for each stratum
                (default GlobalMultiplicativeCalibrator).
        """
        super().__init__()
        self.stratify_by = stratify_by
        self.min_n_per_stratum = min_n_per_stratum
        self.base_calibrator_class = base_calibrator_class

        self.stratum_calibrators_: dict[tuple[Any, ...], CalibratorBase] = {}
        self.global_calibrator_: CalibratorBase | None = None
        self._n_stations: int = 0

    def fit(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> "StratifiedCalibrator":
        """Fit per-stratum calibrators.

        Args:
            y_pred: Uncalibrated predictions at station locations.
            y_true: Observed DTV values from counting stations.
            meta: Required DataFrame with stratification columns.

        Returns:
            self (for method chaining)

        Raises:
            ValueError: If meta is None or missing stratification columns.
        """
        if meta is None:
            raise ValueError("StratifiedCalibrator requires meta DataFrame")

        missing_cols = set(self.stratify_by) - set(meta.columns)
        if missing_cols:
            raise ValueError(f"Meta missing stratification columns: {missing_cols}")

        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()

        # Fit global fallback first
        self.global_calibrator_ = self.base_calibrator_class()
        self.global_calibrator_.fit(y_pred, y_true)

        # Build working dataframe
        df = meta.copy()
        df["_y_pred"] = y_pred
        df["_y_true"] = y_true

        # Fit per-stratum calibrators
        self.stratum_calibrators_ = {}
        for stratum_key, group in df.groupby(self.stratify_by, dropna=False):
            # Ensure stratum_key is always a tuple
            if not isinstance(stratum_key, tuple):
                stratum_key = (stratum_key,)

            if len(group) >= self.min_n_per_stratum:
                cal = self.base_calibrator_class()
                cal.fit(group["_y_pred"].values, group["_y_true"].values)
                self.stratum_calibrators_[stratum_key] = cal

        self._n_stations = len(y_pred)
        self._is_fitted = True
        return self

    def predict(
        self,
        y_pred: np.ndarray,
        meta: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Apply stratified calibration.

        Args:
            y_pred: Uncalibrated predictions to calibrate.
            meta: Required DataFrame with stratification columns.

        Returns:
            Calibrated predictions (using stratum-specific or global calibrator).

        Raises:
            RuntimeError: If calibrator has not been fitted.
            ValueError: If meta is None or missing stratification columns.
        """
        self._check_is_fitted()

        if meta is None:
            raise ValueError("StratifiedCalibrator requires meta DataFrame")

        missing_cols = set(self.stratify_by) - set(meta.columns)
        if missing_cols:
            raise ValueError(f"Meta missing stratification columns: {missing_cols}")

        y_pred = np.asarray(y_pred).ravel()
        result = np.zeros(len(y_pred))

        # Build working dataframe
        df = meta.copy()
        df["_idx"] = range(len(y_pred))

        for stratum_key, group in df.groupby(self.stratify_by, dropna=False):
            # Ensure stratum_key is always a tuple
            if not isinstance(stratum_key, tuple):
                stratum_key = (stratum_key,)

            idx = group["_idx"].values

            # Use stratum calibrator if available, else global fallback
            cal = self.stratum_calibrators_.get(stratum_key, self.global_calibrator_)
            result[idx] = cal.predict(y_pred[idx])

        return result

    def get_params(self) -> CalibrationParams:
        """Get fitted parameters.

        Returns:
            CalibrationParams with stratum calibrators and global fallback.

        Raises:
            RuntimeError: If calibrator has not been fitted.
        """
        self._check_is_fitted()

        # Build params dict with stratum info
        stratum_params = {}
        for stratum_key, cal in self.stratum_calibrators_.items():
            key_str = str(stratum_key)
            stratum_params[key_str] = cal.get_params().params

        global_params = (
            self.global_calibrator_.get_params().params
            if self.global_calibrator_
            else {}
        )

        return CalibrationParams(
            calibrator_type="stratified",
            params={
                "stratify_by": self.stratify_by,
                "min_n_per_stratum": self.min_n_per_stratum,
                "n_strata": len(self.stratum_calibrators_),
                "stratum_params": stratum_params,
                "global_params": global_params,
            },
            n_stations=self._n_stations,
            fitted_at=self._get_fitted_at(),
        )
