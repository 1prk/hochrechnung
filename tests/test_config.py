"""Tests for configuration system."""

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest

from hochrechnung.config import (
    DataPathsConfig,
    RegionConfig,
    TemporalConfig,
    load_config,
)


class TestRegionConfig:
    """Tests for RegionConfig with ARS."""

    def test_valid_config(self) -> None:
        """Test creating valid region config with ARS."""
        config = RegionConfig(
            ars="060000000000",
            name="Hessen",
        )
        assert config.ars == "060000000000"
        assert config.name == "Hessen"
        assert config.land_code == "06"

    def test_ars_properties(self) -> None:
        """Test ARS code extraction properties."""
        config = RegionConfig(ars="064320001234")
        assert config.land_code == "06"
        assert config.regierungsbezirk_code == "064"
        assert config.kreis_code == "06432"
        assert config.gemeinde_code == "064320001234"

    def test_invalid_ars_length(self) -> None:
        """Test that invalid ARS length raises error."""
        with pytest.raises(ValueError, match="12-digit string"):
            RegionConfig(ars="06", name="Hessen")

    def test_invalid_ars_non_digit(self) -> None:
        """Test that non-digit ARS raises error."""
        with pytest.raises(ValueError, match="12-digit string"):
            RegionConfig(ars="06ABCD000000", name="Hessen")


class TestTemporalConfig:
    """Tests for TemporalConfig with simplified period."""

    def test_valid_config(self) -> None:
        """Test creating valid temporal config."""
        config = TemporalConfig(
            year=2024,
            period_start=date(2024, 5, 1),
            period_end=date(2024, 9, 30),
        )
        assert config.year == 2024
        assert config.period_start == date(2024, 5, 1)
        assert config.period_end == date(2024, 9, 30)

    def test_backwards_compatibility_aliases(self) -> None:
        """Test campaign_start/end aliases for backwards compatibility."""
        config = TemporalConfig(
            year=2024,
            period_start=date(2024, 5, 1),
            period_end=date(2024, 9, 30),
        )
        assert config.campaign_start == date(2024, 5, 1)
        assert config.campaign_end == date(2024, 9, 30)

    def test_period_days(self) -> None:
        """Test period_days calculation."""
        config = TemporalConfig(
            year=2024,
            period_start=date(2024, 5, 1),
            period_end=date(2024, 5, 10),
        )
        assert config.period_days == 10

    def test_invalid_year_range(self) -> None:
        """Test that year outside range raises error."""
        with pytest.raises(ValueError):
            TemporalConfig(
                year=2010,  # Too early
                period_start=date(2024, 5, 1),
                period_end=date(2024, 9, 30),
            )

    def test_invalid_period_dates(self) -> None:
        """Test that end before start raises error."""
        with pytest.raises(ValueError, match="period_end must be after"):
            TemporalConfig(
                year=2024,
                period_start=date(2024, 9, 30),
                period_end=date(2024, 5, 1),
            )


class TestDataPathsConfig:
    """Tests for DataPathsConfig."""

    def test_optional_counter_data(self) -> None:
        """Test that counter data is optional."""
        config = DataPathsConfig(
            traffic_volumes=Path("test.fgb"),
        )
        assert config.counter_locations is None
        assert config.counter_measurements is None

    def test_validate_for_training_missing_data(self) -> None:
        """Test validation fails when counter data missing for training."""
        config = DataPathsConfig(
            traffic_volumes=Path("test.fgb"),
        )
        with pytest.raises(ValueError, match="Training requires"):
            config.validate_for_training()

    def test_validate_for_training_with_data(self) -> None:
        """Test validation passes when counter data present."""
        config = DataPathsConfig(
            traffic_volumes=Path("test.fgb"),
            counter_locations=Path("locations.csv"),
            counter_measurements=Path("measurements.csv"),
        )
        config.validate_for_training()  # Should not raise


class TestLoadConfig:
    """Tests for config loading with new minimal format."""

    def test_load_minimal_config(self) -> None:
        """Test loading a minimal config file."""
        config_content = """
project: test-project

ars: "060000000000"
region_name: "Hessen"

year: 2024

period:
  start: "2024-05-01"
  end: "2024-09-30"

data:
  traffic_volumes: "traffic/vol.fgb"

features:
  raw_columns: ["count", "population"]
  model_features: ["count", "population"]

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: ["no", "bicycle_lane"]

models:
  enabled: ["Linear Regression"]
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.project == "test-project"
            assert config.region.ars == "060000000000"
            assert config.land_code == "06"
            assert config.temporal.year == 2024
            assert config.experiment_name == "test-project"  # Derived from project
        finally:
            temp_path.unlink()

    def test_load_training_config(self) -> None:
        """Test loading a training config with counter data."""
        config_content = """
project: hessen-2024

ars: "060000000000"
year: 2024

period:
  start: "2024-05-01"
  end: "2024-09-30"

data:
  traffic_volumes: "traffic/vol.fgb"
  counter_locations: "counters/loc.csv"
  counter_measurements: "counters/meas.csv"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

models:
  enabled: []
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.data_paths.counter_locations == Path("counters/loc.csv")
            config.data_paths.validate_for_training()  # Should not raise
        finally:
            temp_path.unlink()

    def test_output_paths_derived_from_project(self) -> None:
        """Test that output paths are derived from project name."""
        config_content = """
project: my-project

ars: "060000000000"
year: 2024

period:
  start: "2024-05-01"
  end: "2024-09-30"

data:
  traffic_volumes: "traffic/vol.fgb"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

models:
  enabled: []
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.predictions_dir == Path("./output/my-project/predictions")
            assert config.plots_dir == Path("./output/my-project/plots")
            assert config.cache_dir == Path("./output/my-project/cache")
        finally:
            temp_path.unlink()

    def test_env_var_interpolation(self) -> None:
        """Test environment variable interpolation."""
        os.environ["TEST_MLFLOW_URI"] = "http://test:5000"

        config_content = """
project: test

ars: "060000000000"
year: 2024

period:
  start: "2024-05-01"
  end: "2024-09-30"

data:
  traffic_volumes: "traffic/vol.fgb"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

models:
  enabled: []

mlflow:
  tracking_uri: "${TEST_MLFLOW_URI:http://default:5000}"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.mlflow.tracking_uri == "http://test:5000"
        finally:
            temp_path.unlink()
            del os.environ["TEST_MLFLOW_URI"]

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise errors."""
        # Missing project
        config_content = """
ars: "060000000000"
year: 2024
period:
  start: "2024-05-01"
  end: "2024-09-30"
data:
  traffic_volumes: "test.fgb"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="project"):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_missing_ars(self) -> None:
        """Test that missing ARS raises error."""
        config_content = """
project: test
year: 2024
period:
  start: "2024-05-01"
  end: "2024-09-30"
data:
  traffic_volumes: "test.fgb"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="ars"):
                load_config(temp_path)
        finally:
            temp_path.unlink()

    def test_backwards_compatibility_region_code(self) -> None:
        """Test that region_code property still works."""
        config_content = """
project: test

ars: "060000000000"
year: 2024

period:
  start: "2024-05-01"
  end: "2024-09-30"

data:
  traffic_volumes: "traffic/vol.fgb"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

models:
  enabled: []
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            # Backwards compatibility: region_code should return land_code
            assert config.region_code == "06"
        finally:
            temp_path.unlink()
