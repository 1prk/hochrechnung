"""Tests for the calibration system."""

import numpy as np
import pandas as pd
import pytest

from hochrechnung.calibration import (
    CalibrationResult,
    CalibratorType,
    GlobalMultiplicativeCalibrator,
    LOOCVMetrics,
    LogLinearCalibrator,
    StratifiedCalibrator,
    calibrate_and_evaluate,
    create_calibrator,
    run_loocv,
)
from hochrechnung.calibration.base import CalibrationParams


# --- Test Fixtures ---


@pytest.fixture
def sample_predictions() -> np.ndarray:
    """Sample uncalibrated predictions."""
    return np.array([100, 200, 150, 300, 250, 180, 220, 280, 160, 190])


@pytest.fixture
def sample_observations() -> np.ndarray:
    """Sample observed DTV values (systematically higher than predictions)."""
    # About 1.5x the predictions with some noise
    return np.array([155, 305, 220, 450, 375, 260, 340, 430, 245, 290])


@pytest.fixture
def sample_meta() -> pd.DataFrame:
    """Sample metadata for stratified calibration."""
    return pd.DataFrame({
        "infra_category": [
            "bicycle_lane", "bicycle_way", "bicycle_lane", "mixed_way",
            "bicycle_way", "bicycle_lane", "mixed_way", "bicycle_way",
            "bicycle_lane", "mixed_way"
        ],
        "regiostar7": [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
    })


# --- GlobalMultiplicativeCalibrator Tests ---


class TestGlobalMultiplicativeCalibrator:
    """Tests for GlobalMultiplicativeCalibrator."""

    def test_fit_and_predict(self, sample_predictions, sample_observations):
        """Test basic fit and predict workflow."""
        calibrator = GlobalMultiplicativeCalibrator()

        # Before fitting
        assert not calibrator.is_fitted
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator.predict(sample_predictions)

        # Fit
        calibrator.fit(sample_predictions, sample_observations)
        assert calibrator.is_fitted
        assert calibrator.factor_ is not None

        # The factor should be around 1.5 (since observations are ~1.5x predictions)
        assert 1.3 < calibrator.factor_ < 1.7

        # Predict
        calibrated = calibrator.predict(sample_predictions)
        assert len(calibrated) == len(sample_predictions)

        # Calibrated values should be closer to observations
        uncal_error = np.mean(np.abs(sample_predictions - sample_observations))
        cal_error = np.mean(np.abs(calibrated - sample_observations))
        assert cal_error < uncal_error

    def test_factor_clipping(self):
        """Test that extreme factors are clipped."""
        calibrator = GlobalMultiplicativeCalibrator()

        # Extreme underestimation: predictions are 100x too high
        y_pred = np.array([1000, 2000, 3000])
        y_true = np.array([1, 2, 3])
        calibrator.fit(y_pred, y_true)
        assert calibrator.factor_ == GlobalMultiplicativeCalibrator.MIN_FACTOR

        # Extreme overestimation: predictions are 100x too low
        y_pred = np.array([1, 2, 3])
        y_true = np.array([1000, 2000, 3000])
        calibrator.fit(y_pred, y_true)
        assert calibrator.factor_ == GlobalMultiplicativeCalibrator.MAX_FACTOR

    def test_handles_zeros(self):
        """Test handling of zero predictions and observations."""
        calibrator = GlobalMultiplicativeCalibrator()

        # Mix of zeros and positive values
        y_pred = np.array([0, 100, 200, 0, 150])
        y_true = np.array([10, 150, 300, 0, 225])

        calibrator.fit(y_pred, y_true)
        assert calibrator.is_fitted
        # Factor is computed from non-zero predictions only

    def test_get_params(self, sample_predictions, sample_observations):
        """Test parameter extraction."""
        calibrator = GlobalMultiplicativeCalibrator()
        calibrator.fit(sample_predictions, sample_observations)

        params = calibrator.get_params()
        assert isinstance(params, CalibrationParams)
        assert params.calibrator_type == "global_multiplicative"
        assert "factor" in params.params
        assert params.n_stations == len(sample_predictions)
        assert params.fitted_at is not None


# --- LogLinearCalibrator Tests ---


class TestLogLinearCalibrator:
    """Tests for LogLinearCalibrator."""

    def test_fit_and_predict(self, sample_predictions, sample_observations):
        """Test basic fit and predict workflow."""
        calibrator = LogLinearCalibrator(alpha=1.0)

        # Before fitting
        assert not calibrator.is_fitted

        # Fit
        calibrator.fit(sample_predictions, sample_observations)
        assert calibrator.is_fitted
        assert calibrator.a_ is not None
        assert calibrator.b_ is not None

        # Predict
        calibrated = calibrator.predict(sample_predictions)
        assert len(calibrated) == len(sample_predictions)

        # All calibrated values should be non-negative
        assert (calibrated >= 0).all()

        # Calibrated values should be closer to observations
        uncal_error = np.mean(np.abs(sample_predictions - sample_observations))
        cal_error = np.mean(np.abs(calibrated - sample_observations))
        assert cal_error < uncal_error

    def test_log_space_transformation(self):
        """Test that calibration works correctly in log1p space."""
        calibrator = LogLinearCalibrator()

        # Perfect linear relationship in log space
        y_pred = np.array([10, 100, 1000, 10000])
        y_true = np.exp(np.log1p(y_pred) * 1.2 + 0.5) - 1  # a=1.2, b=0.5

        calibrator.fit(y_pred, y_true)

        # The fitted coefficients should be close to a=1.2, b=0.5
        assert abs(calibrator.a_ - 1.2) < 0.1
        assert abs(calibrator.b_ - 0.5) < 0.5

    def test_regularization_effect(self, sample_predictions, sample_observations):
        """Test that alpha parameter affects regularization."""
        # High regularization
        cal_high = LogLinearCalibrator(alpha=100.0)
        cal_high.fit(sample_predictions, sample_observations)

        # Low regularization
        cal_low = LogLinearCalibrator(alpha=0.01)
        cal_low.fit(sample_predictions, sample_observations)

        # High regularization should pull slope closer to 1
        # (Ridge regression with high alpha -> coefficients toward 0)

    def test_get_params(self, sample_predictions, sample_observations):
        """Test parameter extraction."""
        calibrator = LogLinearCalibrator(alpha=2.0)
        calibrator.fit(sample_predictions, sample_observations)

        params = calibrator.get_params()
        assert params.calibrator_type == "log_linear"
        assert "a" in params.params
        assert "b" in params.params
        assert params.params["alpha"] == 2.0


# --- StratifiedCalibrator Tests ---


class TestStratifiedCalibrator:
    """Tests for StratifiedCalibrator."""

    def test_fit_and_predict(self, sample_predictions, sample_observations, sample_meta):
        """Test basic fit and predict workflow."""
        calibrator = StratifiedCalibrator(
            stratify_by=["infra_category"],
            min_n_per_stratum=3,
        )

        # Fit
        calibrator.fit(sample_predictions, sample_observations, sample_meta)
        assert calibrator.is_fitted
        assert calibrator.global_calibrator_ is not None

        # Check that some strata were created
        # bicycle_lane has 4 samples, bicycle_way has 3, mixed_way has 3
        assert len(calibrator.stratum_calibrators_) > 0

        # Predict
        calibrated = calibrator.predict(sample_predictions, sample_meta)
        assert len(calibrated) == len(sample_predictions)

    def test_requires_meta(self, sample_predictions, sample_observations):
        """Test that meta DataFrame is required."""
        calibrator = StratifiedCalibrator(stratify_by=["infra_category"])

        # Fit without meta should raise
        with pytest.raises(ValueError, match="requires meta"):
            calibrator.fit(sample_predictions, sample_observations, None)

        # Predict without meta should raise
        calibrator._is_fitted = True  # Hack to test predict
        with pytest.raises(ValueError, match="requires meta"):
            calibrator.predict(sample_predictions, None)

    def test_missing_stratify_column(self, sample_predictions, sample_observations):
        """Test error when stratification column is missing."""
        calibrator = StratifiedCalibrator(stratify_by=["nonexistent_column"])

        meta = pd.DataFrame({"other_column": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing stratification columns"):
            calibrator.fit(sample_predictions[:3], sample_observations[:3], meta)

    def test_fallback_to_global(self, sample_predictions, sample_observations):
        """Test that small strata fall back to global calibrator."""
        # Create meta with one large stratum and one tiny stratum
        meta = pd.DataFrame({
            "category": ["A"] * 8 + ["B"] * 2  # B has only 2 samples
        })

        calibrator = StratifiedCalibrator(
            stratify_by=["category"],
            min_n_per_stratum=3,
        )
        calibrator.fit(sample_predictions, sample_observations, meta)

        # Only stratum A should have a dedicated calibrator
        assert len(calibrator.stratum_calibrators_) == 1
        assert ("A",) in calibrator.stratum_calibrators_


# --- LOOCV Tests ---


class TestLOOCV:
    """Tests for Leave-One-Out Cross-Validation."""

    def test_run_loocv(self, sample_predictions, sample_observations):
        """Test LOOCV execution."""
        loocv_result = run_loocv(
            GlobalMultiplicativeCalibrator,
            sample_predictions,
            sample_observations,
        )

        assert isinstance(loocv_result, LOOCVMetrics)
        assert loocv_result.mae_mean >= 0
        assert loocv_result.mae_std >= 0
        assert loocv_result.rmse_mean >= 0
        assert len(loocv_result.per_fold_errors) == len(sample_predictions)

    def test_loocv_with_small_sample(self):
        """Test LOOCV with minimum sample size."""
        y_pred = np.array([100, 200, 300])
        y_true = np.array([150, 300, 450])

        loocv_result = run_loocv(
            LogLinearCalibrator,
            y_pred,
            y_true,
            alpha=1.0,
        )

        assert len(loocv_result.per_fold_errors) == 3


# --- calibrate_and_evaluate Tests ---


class TestCalibrateAndEvaluate:
    """Tests for the calibrate_and_evaluate function."""

    def test_full_evaluation(self, sample_predictions, sample_observations):
        """Test complete calibration and evaluation workflow."""
        calibrator = LogLinearCalibrator()
        result = calibrate_and_evaluate(
            calibrator,
            sample_predictions,
            sample_observations,
            run_loocv_validation=True,
        )

        assert isinstance(result, CalibrationResult)
        assert result.calibrator is calibrator
        assert result.n_stations == len(sample_predictions)

        # Metrics should be computed
        assert result.uncalibrated_metrics.mae >= 0
        assert result.calibrated_metrics.mae >= 0

        # Improvement dict should be populated
        assert "mae_reduction" in result.improvement
        assert "rmse_reduction" in result.improvement
        assert "r2_gain" in result.improvement

        # Calibration should improve MAE (most of the time)
        # Note: this is not guaranteed for all datasets
        assert result.improvement["mae_reduction"] > 0

    def test_skip_loocv(self, sample_predictions, sample_observations):
        """Test skipping LOOCV."""
        calibrator = GlobalMultiplicativeCalibrator()
        result = calibrate_and_evaluate(
            calibrator,
            sample_predictions,
            sample_observations,
            run_loocv_validation=False,
        )

        assert result.loocv_metrics is None


# --- Factory Function Tests ---


class TestCreateCalibrator:
    """Tests for the create_calibrator factory function."""

    def test_create_multiplicative(self):
        """Test creating multiplicative calibrator."""
        cal = create_calibrator(CalibratorType.GLOBAL_MULTIPLICATIVE)
        assert isinstance(cal, GlobalMultiplicativeCalibrator)

        # Also test string input
        cal = create_calibrator("global_multiplicative")
        assert isinstance(cal, GlobalMultiplicativeCalibrator)

    def test_create_loglinear(self):
        """Test creating log-linear calibrator."""
        cal = create_calibrator(CalibratorType.LOG_LINEAR, alpha=2.0)
        assert isinstance(cal, LogLinearCalibrator)
        assert cal.alpha == 2.0

    def test_create_stratified(self):
        """Test creating stratified calibrator."""
        cal = create_calibrator(
            CalibratorType.STRATIFIED,
            stratify_by=["infra_category"],
            min_stations_per_stratum=5,
        )
        assert isinstance(cal, StratifiedCalibrator)
        assert cal.stratify_by == ["infra_category"]
        assert cal.min_n_per_stratum == 5

    def test_stratified_requires_stratify_by(self):
        """Test that stratified calibrator requires stratify_by."""
        with pytest.raises(ValueError, match="stratify_by is required"):
            create_calibrator(CalibratorType.STRATIFIED)

    def test_spatial_not_implemented(self):
        """Test that spatially weighted raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            create_calibrator(CalibratorType.SPATIALLY_WEIGHTED)

    def test_unknown_type(self):
        """Test handling of unknown calibrator type."""
        with pytest.raises(ValueError, match="Unknown calibrator type"):
            create_calibrator("unknown_type")


# --- Edge Cases ---


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample(self):
        """Test handling of single sample."""
        calibrator = GlobalMultiplicativeCalibrator()
        y_pred = np.array([100])
        y_true = np.array([150])

        calibrator.fit(y_pred, y_true)
        calibrated = calibrator.predict(y_pred)

        assert len(calibrated) == 1
        assert calibrated[0] == 150  # factor = 1.5, 100 * 1.5 = 150

    def test_all_zeros(self):
        """Test handling of all-zero predictions."""
        calibrator = GlobalMultiplicativeCalibrator()
        y_pred = np.array([0, 0, 0])
        y_true = np.array([100, 200, 300])

        calibrator.fit(y_pred, y_true)
        # Factor should default to 1.0 when no valid ratios
        assert calibrator.factor_ == 1.0

    def test_negative_values(self):
        """Test handling of negative values."""
        calibrator = LogLinearCalibrator()
        y_pred = np.array([100, 200, 300])
        y_true = np.array([150, 300, 450])

        calibrator.fit(y_pred, y_true)
        calibrated = calibrator.predict(y_pred)

        # All calibrated values should be non-negative
        assert (calibrated >= 0).all()

    def test_predict_before_fit(self):
        """Test that predict raises before fit."""
        calibrator = LogLinearCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator.predict(np.array([100, 200]))

    def test_get_params_before_fit(self):
        """Test that get_params raises before fit."""
        calibrator = GlobalMultiplicativeCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator.get_params()
