"""Tests for configuration system."""

import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
import yaml

from hochrechnung.config import (
    DataPathsConfig,
    PipelineConfig,
    RegionConfig,
    TemporalConfig,
    load_config,
)


class TestRegionConfig:
    """Tests for RegionConfig."""

    def test_valid_config(self) -> None:
        """Test creating valid region config."""
        config = RegionConfig(
            code="06",
            name="Hessen",
            bbox=(7.77, 49.39, 10.24, 51.66),
        )
        assert config.code == "06"
        assert config.name == "Hessen"

    def test_invalid_code_length(self) -> None:
        """Test that invalid code length raises error."""
        with pytest.raises(ValueError, match="2-digit string"):
            RegionConfig(code="6", name="Hessen", bbox=(0, 0, 1, 1))

    def test_invalid_bbox_min_max(self) -> None:
        """Test that invalid bbox (min > max) raises error."""
        with pytest.raises(ValueError, match="min_lon"):
            RegionConfig(code="06", name="Hessen", bbox=(10, 49, 7, 51))


class TestTemporalConfig:
    """Tests for TemporalConfig."""

    def test_valid_config(self) -> None:
        """Test creating valid temporal config."""
        config = TemporalConfig(
            year=2024,
            campaign_start=date(2024, 5, 1),
            campaign_end=date(2024, 9, 30),
            counter_period_start=date(2024, 5, 1),
            counter_period_end=date(2024, 9, 30),
        )
        assert config.year == 2024
        assert config.campaign_start == date(2024, 5, 1)

    def test_invalid_year_range(self) -> None:
        """Test that year outside range raises error."""
        with pytest.raises(ValueError):
            TemporalConfig(
                year=2010,  # Too early
                campaign_start=date(2024, 5, 1),
                campaign_end=date(2024, 9, 30),
                counter_period_start=date(2024, 5, 1),
                counter_period_end=date(2024, 9, 30),
            )


class TestLoadConfig:
    """Tests for config loading."""

    def test_load_simple_config(self) -> None:
        """Test loading a simple config file."""
        config_content = """
region:
  code: "06"
  name: "Hessen"
  bbox: [7.77, 49.39, 10.24, 51.66]

temporal:
  year: 2024
  campaign_start: "2024-05-01"
  campaign_end: "2024-09-30"
  counter_period_start: "2024-05-01"
  counter_period_end: "2024-09-30"

data_paths:
  data_root: "./data"
  counter_locations: "counters/loc.csv"
  counter_measurements: "counters/meas.csv"
  traffic_volumes: "traffic/vol.fgb"
  municipalities: "struct/vg250.gpkg"
  regiostar: "struct/regiostar.csv"
  city_centroids: "struct/centroids.gpkg"
  kommunen_stats: "stats/kommunen.shp"
  campaign_stats: "campaign/stats.csv"

features:
  raw_columns: ["count", "population"]
  model_features: ["count", "population"]

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: ["no", "bicycle_lane"]

training:
  test_size: 0.2
  cv_folds: 5

models:
  enabled: ["Linear Regression"]

mlflow:
  experiment_name: "test-experiment"

output:
  plots_dir: "./plots"
  predictions_dir: "./predictions"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.region.code == "06"
            assert config.temporal.year == 2024
            assert config.mlflow.experiment_name == "test-experiment"
        finally:
            temp_path.unlink()

    def test_env_var_interpolation(self) -> None:
        """Test environment variable interpolation."""
        os.environ["TEST_MLFLOW_URI"] = "http://test:5000"

        config_content = """
region:
  code: "06"
  name: "Hessen"
  bbox: [7.77, 49.39, 10.24, 51.66]

temporal:
  year: 2024
  campaign_start: "2024-05-01"
  campaign_end: "2024-09-30"
  counter_period_start: "2024-05-01"
  counter_period_end: "2024-09-30"

data_paths:
  data_root: "./data"
  counter_locations: "counters/loc.csv"
  counter_measurements: "counters/meas.csv"
  traffic_volumes: "traffic/vol.fgb"
  municipalities: "struct/vg250.gpkg"
  regiostar: "struct/regiostar.csv"
  city_centroids: "struct/centroids.gpkg"
  kommunen_stats: "stats/kommunen.shp"
  campaign_stats: "campaign/stats.csv"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

training: {}

models:
  enabled: []

mlflow:
  tracking_uri: "${TEST_MLFLOW_URI:http://default:5000}"
  experiment_name: "test"

output: {}
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

    def test_default_env_var(self) -> None:
        """Test default value when env var not set."""
        # Ensure env var is not set
        os.environ.pop("UNSET_VAR", None)

        config_content = """
region:
  code: "06"
  name: "Hessen"
  bbox: [0, 0, 1, 1]

temporal:
  year: 2024
  campaign_start: "2024-05-01"
  campaign_end: "2024-09-30"
  counter_period_start: "2024-05-01"
  counter_period_end: "2024-09-30"

data_paths:
  data_root: "./data"
  counter_locations: "a.csv"
  counter_measurements: "b.csv"
  traffic_volumes: "c.fgb"
  municipalities: "d.gpkg"
  regiostar: "e.csv"
  city_centroids: "f.gpkg"
  kommunen_stats: "g.shp"
  campaign_stats: "h.csv"

features:
  raw_columns: []
  model_features: []

preprocessing:
  infrastructure_mapping: {}
  valid_infrastructure_categories: []

training: {}

models:
  enabled: []

mlflow:
  tracking_uri: "${UNSET_VAR:http://default:5000}"
  experiment_name: "test"

output: {}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(config_content)
            temp_path = Path(f.name)

        try:
            config = load_config(temp_path)
            assert config.mlflow.tracking_uri == "http://default:5000"
        finally:
            temp_path.unlink()
