"""Tests for campaign data ingestion including JSON support."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from hochrechnung.ingestion.campaign import DemographicsLoader


class TestDemographicsLoaderJSON:
    """Tests for JSON loading functionality in DemographicsLoader."""

    def test_load_from_json_basic(self, tmp_path: Path) -> None:
        """Test basic JSON loading and column mapping."""
        # Create test JSON data
        json_data = [
            {
                "ars": "010010000000",
                "name": "Test Kommune 1",
                "users_n": 100,
                "trips_n": 500,
                "distance_km": 1234.56,
            },
            {
                "ars": "010020000000",
                "name": "Test Kommune 2",
                "users_n": 200,
                "trips_n": 1000,
                "distance_km": 2469.12,
            },
        ]

        json_path = tmp_path / "SR24_Commune_Statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        # Create mock config
        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("nonexistent.shp")
        config.year = 2024
        config.region.ars = "010000000000"

        # Test loading
        loader = DemographicsLoader(config)
        df = loader._load_from_json(json_path)

        # Verify column mapping
        assert "ARS" in df.columns
        assert "GEN" in df.columns
        assert "N_USERS" in df.columns
        assert "N_TRIPS" in df.columns
        assert "TOTAL_KM" in df.columns

        # Verify values
        assert df.iloc[0]["N_USERS"] == 100
        assert df.iloc[0]["N_TRIPS"] == 500
        assert df.iloc[0]["TOTAL_KM"] == 1234.56
        assert df.iloc[0]["ARS"] == "010010000000"

    def test_json_file_discovery(self, tmp_path: Path) -> None:
        """Test that JSON files are discovered when shapefile is missing."""
        # Create kommunen-stats directory with JSON
        kommunen_dir = tmp_path / "kommunen-stats"
        kommunen_dir.mkdir()

        json_data = [{"ars": "010010000000", "users_n": 100, "trips_n": 500, "distance_km": 1000.0}]
        json_path = kommunen_dir / "SR24_Commune_Statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        # Create mock config pointing to non-existent shapefile
        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("kommunen-stats/missing.shp")
        config.year = 2024
        config.region.ars = "010000000000"

        loader = DemographicsLoader(config)
        loader.resolve_path = lambda p: tmp_path / p  # Mock resolve_path

        found_path = loader._find_demographics_file()
        assert found_path == json_path

    def test_json_ars_padding(self, tmp_path: Path) -> None:
        """Test that ARS values are zero-padded to 12 digits."""
        # Create JSON with shorter ARS
        json_data = [
            {"ars": "10010000000", "users_n": 100, "trips_n": 500, "distance_km": 1000.0}
        ]

        json_path = tmp_path / "test.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.year = 2024
        config.region.ars = "010000000000"

        loader = DemographicsLoader(config)
        df = loader._load_from_json(json_path)

        # Verify ARS is padded to 12 digits
        assert df.iloc[0]["ARS"] == "010010000000"
        assert len(df.iloc[0]["ARS"]) == 12

    def test_json_preserves_extra_columns(self, tmp_path: Path) -> None:
        """Test that additional JSON columns are preserved."""
        json_data = [
            {
                "ars": "010010000000",
                "users_n": 100,
                "trips_n": 500,
                "distance_km": 1000.0,
                "avg_distance_km": 5.5,
                "average_speed": 15.2,
            }
        ]

        json_path = tmp_path / "test.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.year = 2024
        config.region.ars = "010000000000"

        loader = DemographicsLoader(config)
        df = loader._load_from_json(json_path)

        # Extra columns should be preserved
        assert "avg_distance_km" in df.columns
        assert "average_speed" in df.columns


class TestDemographicsLoaderComparison:
    """Tests comparing shapefile and JSON outputs."""

    def test_json_produces_required_columns(self, tmp_path: Path) -> None:
        """Test that JSON loading produces all columns needed for feature engineering."""
        required_columns = ["ARS", "N_USERS", "N_TRIPS", "TOTAL_KM"]

        json_data = [
            {
                "ars": "010010000000",
                "users_n": 634,
                "trips_n": 8805,
                "distance_km": 52769.63,
            }
        ]

        json_path = tmp_path / "test.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.year = 2024
        config.region.ars = "010000000000"

        loader = DemographicsLoader(config)
        df = loader._load_from_json(json_path)

        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_normalized_column_names_match(self, tmp_path: Path) -> None:
        """Test that normalized column names match expected schema."""
        # This test verifies the column renaming in _load_raw_geo
        json_data = [
            {
                "ars": "010010000000",
                "users_n": 100,
                "trips_n": 500,
                "distance_km": 1234.56,
            }
        ]

        # Create kommunen-stats directory
        kommunen_dir = tmp_path / "kommunen-stats"
        kommunen_dir.mkdir()

        json_path = kommunen_dir / "SR24_Commune_Statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("kommunen-stats/missing.shp")
        config.year = 2024
        config.region.ars = "010010000000"  # Match exactly

        loader = DemographicsLoader(config)
        loader.resolve_path = lambda p: tmp_path / p

        # Call _load_raw_geo which does the full loading and normalization
        df = loader._load_raw_geo()

        # After normalization, columns should be lowercase
        expected_columns = {"ars", "n_users", "n_trips", "total_km"}
        actual_columns = set(df.columns)

        for col in expected_columns:
            assert col in actual_columns, f"Missing normalized column: {col}"


class TestFileTypePriority:
    """Tests for file type discovery priority."""

    def test_shapefile_preferred_over_json(self, tmp_path: Path) -> None:
        """Test that configured shapefile path is preferred over JSON."""
        # Create both shapefile-like and JSON files
        kommunen_dir = tmp_path / "kommunen-stats"
        kommunen_dir.mkdir()

        # Create a marker file for shapefile (not a real shapefile)
        shp_path = kommunen_dir / "test.shp"
        shp_path.touch()

        # Create JSON file
        json_data = [{"ars": "010010000000", "users_n": 100, "trips_n": 500, "distance_km": 1000.0}]
        json_path = kommunen_dir / "SR24_Commune_Statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("kommunen-stats/test.shp")
        config.year = 2024

        loader = DemographicsLoader(config)
        loader.resolve_path = lambda p: tmp_path / p

        found_path = loader._find_demographics_file()

        # Should find shapefile first (configured path)
        assert found_path == shp_path

    def test_json_fallback_when_shapefile_missing(self, tmp_path: Path) -> None:
        """Test JSON is used when configured shapefile doesn't exist."""
        kommunen_dir = tmp_path / "kommunen-stats"
        kommunen_dir.mkdir()

        # Only create JSON file, no shapefile
        json_data = [{"ars": "010010000000", "users_n": 100, "trips_n": 500, "distance_km": 1000.0}]
        json_path = kommunen_dir / "SR24_Commune_Statistics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("kommunen-stats/missing.shp")
        config.year = 2024

        loader = DemographicsLoader(config)
        loader.resolve_path = lambda p: tmp_path / p

        found_path = loader._find_demographics_file()

        # Should fall back to JSON
        assert found_path == json_path

    def test_file_not_found_when_no_files_exist(self, tmp_path: Path) -> None:
        """Test FileNotFoundError when no demographics files exist."""
        kommunen_dir = tmp_path / "kommunen-stats"
        kommunen_dir.mkdir()

        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()

        config = MagicMock()
        config.data_paths.data_root = tmp_path
        config.data_paths.kommunen_stats = Path("kommunen-stats/missing.shp")
        config.year = 2024

        loader = DemographicsLoader(config)
        loader.resolve_path = lambda p: tmp_path / p

        with pytest.raises(FileNotFoundError):
            loader._find_demographics_file()
