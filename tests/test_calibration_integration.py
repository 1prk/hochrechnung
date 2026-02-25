"""Integration tests for the calibration system."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hochrechnung.calibration import (
    CalibrationDataLoader,
    GlobalMultiplicativeCalibrator,
    LogLinearCalibrator,
    calibrate_and_evaluate,
    load_calibrator,
    save_calibrator,
)


# --- Persistence Tests ---


class TestPersistence:
    """Tests for save/load calibrator functionality."""

    def test_save_and_load_multiplicative(self, tmp_path):
        """Test saving and loading multiplicative calibrator."""
        # Create and fit calibrator
        calibrator = GlobalMultiplicativeCalibrator()
        y_pred = np.array([100, 200, 300, 400, 500])
        y_true = np.array([150, 300, 450, 600, 750])
        calibrator.fit(y_pred, y_true)

        # Save
        output_path = tmp_path / "test_model"
        cal_path, meta_path = save_calibrator(calibrator, output_path)

        # Check files exist
        assert cal_path.exists()
        assert meta_path.exists()
        assert cal_path.suffix == ".joblib"
        assert meta_path.suffix == ".json"

        # Load
        loaded_cal, metadata = load_calibrator(output_path)

        # Verify loaded calibrator works
        assert loaded_cal.is_fitted
        assert loaded_cal.factor_ == calibrator.factor_

        # Verify predictions match
        original_pred = calibrator.predict(y_pred)
        loaded_pred = loaded_cal.predict(y_pred)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        # Verify metadata
        assert metadata["calibrator_type"] == "global_multiplicative"
        assert "factor" in metadata["params"]

    def test_save_and_load_loglinear(self, tmp_path):
        """Test saving and loading log-linear calibrator."""
        calibrator = LogLinearCalibrator(alpha=2.0)
        y_pred = np.array([100, 200, 300, 400, 500])
        y_true = np.array([150, 300, 450, 600, 750])
        calibrator.fit(y_pred, y_true)

        output_path = tmp_path / "loglinear_model"
        save_calibrator(calibrator, output_path)

        loaded_cal, metadata = load_calibrator(output_path)

        assert loaded_cal.is_fitted
        assert loaded_cal.a_ == calibrator.a_
        assert loaded_cal.b_ == calibrator.b_
        assert metadata["params"]["alpha"] == 2.0

    def test_save_with_metrics(self, tmp_path):
        """Test saving calibrator with evaluation metrics."""
        calibrator = LogLinearCalibrator()
        y_pred = np.array([100, 200, 300, 400, 500])
        y_true = np.array([150, 300, 450, 600, 750])

        result = calibrate_and_evaluate(
            calibrator, y_pred, y_true, run_loocv_validation=True
        )

        output_path = tmp_path / "with_metrics"
        cal_path, meta_path = save_calibrator(result.calibrator, output_path, result)

        # Load metadata and check metrics are present
        with open(meta_path) as f:
            metadata = json.load(f)

        assert "uncalibrated" in metadata
        assert "calibrated" in metadata
        assert "improvement" in metadata
        assert "loocv" in metadata

    def test_load_nonexistent(self, tmp_path):
        """Test loading non-existent calibrator."""
        with pytest.raises(FileNotFoundError):
            load_calibrator(tmp_path / "nonexistent")

    def test_load_with_joblib_extension(self, tmp_path):
        """Test loading by specifying full .joblib path."""
        calibrator = GlobalMultiplicativeCalibrator()
        calibrator.fit(np.array([100, 200]), np.array([150, 300]))

        output_path = tmp_path / "test"
        cal_path, _ = save_calibrator(calibrator, output_path)

        # Load using full .joblib path
        loaded_cal, _ = load_calibrator(cal_path)
        assert loaded_cal.is_fitted


# --- CalibrationDataLoader Tests ---


class TestCalibrationDataLoader:
    """Tests for CalibrationDataLoader."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading valid calibration CSV."""
        csv_content = """id,name,latitude,longitude,dtv
CAL001,Station A,50.1,8.6,1250
CAL002,Station B,50.2,8.7,980
CAL003,Station C,50.3,8.8,1500
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        df = loader.load()

        assert len(df) == 3
        assert "id" in df.columns
        assert "dtv" in df.columns
        assert df["dtv"].dtype in [np.float64, np.int64]

    def test_load_with_optional_columns(self, tmp_path):
        """Test loading CSV with optional stratification columns."""
        csv_content = """id,name,latitude,longitude,dtv,infra_category,regiostar7
CAL001,Station A,50.1,8.6,1250,bicycle_lane,1
CAL002,Station B,50.2,8.7,980,bicycle_way,2
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        df = loader.load()

        assert "infra_category" in df.columns
        assert "regiostar7" in df.columns

    def test_missing_required_column(self, tmp_path):
        """Test error when required column is missing."""
        csv_content = """id,name,latitude,longitude
CAL001,Station A,50.1,8.6
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load()

    def test_missing_dtv_values(self, tmp_path):
        """Test error when DTV has missing values."""
        csv_content = """id,name,latitude,longitude,dtv
CAL001,Station A,50.1,8.6,1250
CAL002,Station B,50.2,8.7,
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        with pytest.raises(ValueError, match="missing values"):
            loader.load()

    def test_negative_dtv(self, tmp_path):
        """Test error when DTV has negative values."""
        csv_content = """id,name,latitude,longitude,dtv
CAL001,Station A,50.1,8.6,-100
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        with pytest.raises(ValueError, match="negative values"):
            loader.load()

    def test_skip_validation(self, tmp_path):
        """Test loading without validation."""
        csv_content = """id,name,latitude,longitude,dtv
CAL001,Station A,50.1,8.6,-100
"""
        csv_path = tmp_path / "calibration.csv"
        csv_path.write_text(csv_content)

        loader = CalibrationDataLoader(csv_path)
        # Should not raise with validate=False
        df = loader.load(validate=False)
        assert len(df) == 1

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        loader = CalibrationDataLoader(tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load()


# --- End-to-End Workflow Tests ---


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_calibration_workflow(self, tmp_path):
        """Test complete calibration workflow: load -> calibrate -> save -> load."""
        # 1. Create calibration data
        np.random.seed(42)
        n_stations = 20

        # Simulate: predictions are systematically 30% low
        true_dtv = np.random.uniform(500, 2000, n_stations)
        predictions = true_dtv * 0.7 + np.random.normal(0, 50, n_stations)

        cal_data = pd.DataFrame({
            "id": [f"CAL{i:03d}" for i in range(n_stations)],
            "name": [f"Station {i}" for i in range(n_stations)],
            "latitude": np.random.uniform(49.5, 52.0, n_stations),
            "longitude": np.random.uniform(7.0, 10.0, n_stations),
            "dtv": true_dtv.astype(int),
        })

        # Save calibration data
        cal_csv_path = tmp_path / "calibration_stations.csv"
        cal_data.to_csv(cal_csv_path, index=False)

        # 2. Load calibration data
        loader = CalibrationDataLoader(cal_csv_path)
        loaded_data = loader.load()
        assert len(loaded_data) == n_stations

        # 3. Calibrate
        calibrator = LogLinearCalibrator(alpha=1.0)
        result = calibrate_and_evaluate(
            calibrator,
            y_pred=predictions,
            y_true=loaded_data["dtv"].values,
            run_loocv_validation=True,
        )

        # Calibration should improve metrics
        assert result.improvement["mae_reduction"] > 0
        assert result.loocv_metrics is not None

        # 4. Save calibrator
        model_path = tmp_path / "model_calibrator"
        cal_path, meta_path = save_calibrator(result.calibrator, model_path, result)

        assert cal_path.exists()
        assert meta_path.exists()

        # 5. Load and verify
        loaded_cal, metadata = load_calibrator(model_path)

        # Should produce same predictions
        original_pred = calibrator.predict(predictions)
        loaded_pred = loaded_cal.predict(predictions)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

        # Metadata should contain metrics
        assert metadata["improvement"]["mae_reduction"] == pytest.approx(
            result.improvement["mae_reduction"], rel=1e-5
        )

    def test_stratified_workflow(self, tmp_path):
        """Test stratified calibration workflow."""
        np.random.seed(123)

        # Create data with different calibration factors per infrastructure type
        infra_types = ["bicycle_lane", "bicycle_way", "mixed_way"]
        infra_factors = {"bicycle_lane": 1.5, "bicycle_way": 1.2, "mixed_way": 1.8}

        records = []
        for i in range(30):
            infra = infra_types[i % 3]
            true_dtv = np.random.uniform(500, 1500)
            pred = true_dtv / infra_factors[infra]  # Predictions are low by factor
            records.append({
                "id": f"CAL{i:03d}",
                "latitude": 50.0 + i * 0.01,
                "longitude": 8.0 + i * 0.01,
                "dtv": int(true_dtv),
                "infra_category": infra,
                "prediction": pred,
            })

        df = pd.DataFrame(records)

        # Create stratified calibrator
        from hochrechnung.calibration import StratifiedCalibrator

        calibrator = StratifiedCalibrator(
            stratify_by=["infra_category"],
            min_n_per_stratum=3,
        )

        result = calibrate_and_evaluate(
            calibrator,
            y_pred=df["prediction"].values,
            y_true=df["dtv"].values,
            meta=df[["infra_category"]],
            run_loocv_validation=False,  # Skip LOOCV for speed
        )

        # Each stratum should have its own calibrator
        assert len(calibrator.stratum_calibrators_) == 3

        # Calibration should improve significantly
        assert result.improvement["mae_reduction"] > 0.3

        # Save and load
        output_path = tmp_path / "stratified_model"
        save_calibrator(result.calibrator, output_path, result)

        loaded_cal, metadata = load_calibrator(output_path)

        # Verify stratified predictions work
        calibrated = loaded_cal.predict(
            df["prediction"].values,
            df[["infra_category"]],
        )
        assert len(calibrated) == len(df)


# --- Config Integration Tests ---


class TestConfigIntegration:
    """Tests for configuration integration."""

    def test_calibrator_type_enum(self):
        """Test CalibratorType enum values."""
        from hochrechnung.config.settings import CalibratorType

        assert CalibratorType.GLOBAL_MULTIPLICATIVE.value == "global_multiplicative"
        assert CalibratorType.LOG_LINEAR.value == "log_linear"
        assert CalibratorType.STRATIFIED.value == "stratified"
        assert CalibratorType.SPATIALLY_WEIGHTED.value == "spatially_weighted"

    def test_calibration_config_defaults(self):
        """Test CalibrationConfig default values."""
        from hochrechnung.config.settings import CalibrationConfig, CalibratorType

        config = CalibrationConfig()

        assert config.counter_locations is None
        assert config.calibrator == CalibratorType.LOG_LINEAR
        assert config.stratify_by is None
        assert config.min_stations_per_stratum == 3
        assert config.random_state == 1337

    def test_calibration_config_custom_values(self):
        """Test CalibrationConfig with custom values."""
        from pathlib import Path

        from hochrechnung.config.settings import CalibrationConfig, CalibratorType

        config = CalibrationConfig(
            counter_locations=Path("calibration/counters.csv"),
            calibrator=CalibratorType.STRATIFIED,
            stratify_by=["infra_category", "regiostar7"],
            min_stations_per_stratum=5,
        )

        assert config.counter_locations == Path("calibration/counters.csv")
        assert config.calibrator == CalibratorType.STRATIFIED
        assert config.stratify_by == ["infra_category", "regiostar7"]
        assert config.min_stations_per_stratum == 5
