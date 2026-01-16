"""Tests for validation module."""

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hochrechnung.config import PipelineConfig, load_config
from hochrechnung.validation import ValidationResult, ValidationRunner
from hochrechnung.validation.core import DATASET_SCHEMA_MAP


@pytest.fixture
def temp_data_dir() -> Path:
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def minimal_config_dict(temp_data_dir: Path) -> dict[str, Any]:
    """Create minimal config dictionary for testing."""
    return {
        "region": {
            "code": "06",
            "name": "Hessen",
            "bbox": [7.77, 49.39, 10.24, 51.66],
        },
        "temporal": {
            "year": 2024,
            "campaign_start": "2024-05-01",
            "campaign_end": "2024-09-30",
            "counter_period_start": "2024-05-01",
            "counter_period_end": "2024-09-30",
        },
        "data_paths": {
            "data_root": str(temp_data_dir),
            "counter_locations": "counter_locations.csv",
            "counter_measurements": "counter_measurements.csv",
            "traffic_volumes": "traffic_volumes.csv",
            "municipalities": "municipalities.csv",
            "regiostar": "regiostar.csv",
            "city_centroids": "city_centroids.csv",
            "kommunen_stats": "kommunen_stats.csv",
            "campaign_stats": "campaign_stats.csv",
        },
        "features": {
            "raw_columns": [],
            "model_features": [],
        },
        "preprocessing": {
            "infrastructure_mapping": {},
            "valid_infrastructure_categories": [],
        },
        "training": {},
        "models": {
            "enabled": [],
        },
        "mlflow": {
            "experiment_name": "test",
        },
        "output": {},
    }


@pytest.fixture
def valid_counter_locations_df() -> pd.DataFrame:
    """Create valid counter locations dataframe."""
    return pd.DataFrame(
        {
            "id": ["001", "002"],
            "name": ["Station A", "Station B"],
            "latitude": [50.1, 50.2],
            "longitude": [8.6, 8.7],
            "ars": ["064120000000", "063110000000"],  # Exactly 12 digits
        }
    )


@pytest.fixture
def invalid_counter_locations_df() -> pd.DataFrame:
    """Create invalid counter locations dataframe (missing required column)."""
    return pd.DataFrame(
        {
            "id": ["001", "002"],
            "name": ["Station A", "Station B"],
            # Missing latitude, longitude, ars
        }
    )


@pytest.fixture
def valid_counter_measurements_df() -> pd.DataFrame:
    """Create valid counter measurements dataframe."""
    return pd.DataFrame(
        {
            "counter_id": ["001", "001", "002"],
            "timestamp": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-01"]),
            "count": [100, 120, 80],
        }
    )


@pytest.fixture
def valid_regiostar_df() -> pd.DataFrame:
    """Create valid regiostar dataframe."""
    return pd.DataFrame(
        {
            "ars": ["064120000000", "063110000000"],  # Exactly 12 digits
            "RegioStaR5": [51, 52],  # Valid range 51-59
        }
    )


@pytest.fixture
def valid_campaign_metadata_df() -> pd.DataFrame:
    """Create valid campaign metadata dataframe."""
    return pd.DataFrame(
        {
            "ags": ["064120000000", "063110000000"],  # Exactly 12 digits
            "year": [2024, 2024],
            "start_date": pd.to_datetime(["2024-05-01", "2024-05-15"]),
            "end_date": pd.to_datetime(["2024-09-30", "2024-09-30"]),
        }
    )


class TestDatasetSchemaMapping:
    """Tests for the dataset schema mapping."""

    def test_all_mapped_schemas_exist(self) -> None:
        """Test that all mapped schemas exist in registry."""
        from hochrechnung.schemas.registry import SchemaRegistry

        for schema_name in DATASET_SCHEMA_MAP.values():
            assert schema_name in SchemaRegistry.list_schemas()


class TestValidationRunner:
    """Tests for ValidationRunner."""

    def test_runner_initialization(
        self, minimal_config_dict: dict[str, Any], temp_data_dir: Path
    ) -> None:
        """Test that runner initializes correctly."""
        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        assert runner.config == config

    def test_validation_missing_files(
        self, minimal_config_dict: dict[str, Any], temp_data_dir: Path
    ) -> None:
        """Test validation reports missing files correctly."""
        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        results = runner.run()

        # All files should be missing
        for result in results:
            if result.schema_name is not None:
                assert result.exists is False
                assert result.schema_valid is None
                assert result.error_message == "File not found"

    def test_validation_valid_file(
        self,
        minimal_config_dict: dict[str, Any],
        temp_data_dir: Path,
        valid_counter_locations_df: pd.DataFrame,
    ) -> None:
        """Test validation passes for valid file."""
        # Create valid file with explicit dtype to preserve leading zeros
        counter_locations_path = temp_data_dir / "counter_locations.csv"
        # Ensure ars is written as string to preserve leading zeros
        df = valid_counter_locations_df.copy()
        df['ars'] = df['ars'].astype(str)
        df.to_csv(counter_locations_path, index=False)

        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        results = runner.run()

        # Find counter_locations result
        counter_result = next(
            r for r in results if r.dataset_name == "counter_locations"
        )
        assert counter_result.exists is True
        assert counter_result.schema_valid is True
        assert counter_result.row_count == 2
        assert counter_result.error_message is None

    def test_validation_invalid_file(
        self,
        minimal_config_dict: dict[str, Any],
        temp_data_dir: Path,
        invalid_counter_locations_df: pd.DataFrame,
    ) -> None:
        """Test validation fails for invalid file."""
        # Create invalid file
        counter_locations_path = temp_data_dir / "counter_locations.csv"
        invalid_counter_locations_df.to_csv(counter_locations_path, index=False)

        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        results = runner.run()

        # Find counter_locations result
        counter_result = next(
            r for r in results if r.dataset_name == "counter_locations"
        )
        assert counter_result.exists is True
        assert counter_result.schema_valid is False
        assert counter_result.error_message is not None
        # Check for schema-related error message
        assert ("column" in counter_result.error_message.lower() or
                "error" in counter_result.error_message.lower())

    def test_validation_dataset_without_schema(
        self, minimal_config_dict: dict[str, Any], temp_data_dir: Path
    ) -> None:
        """Test validation skips datasets without schema mapping."""
        # kommunen_stats has no schema mapping
        kommunen_path = temp_data_dir / "kommunen_stats.csv"
        pd.DataFrame({"dummy": [1, 2, 3]}).to_csv(kommunen_path, index=False)

        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        results = runner.run()

        # Find kommunen_stats result
        kommunen_result = next(r for r in results if r.dataset_name == "kommunen_stats")
        assert kommunen_result.schema_name is None
        assert kommunen_result.schema_valid is None
        assert "No schema mapping" in kommunen_result.error_message

    def test_validation_multiple_files(
        self,
        minimal_config_dict: dict[str, Any],
        temp_data_dir: Path,
        valid_counter_locations_df: pd.DataFrame,
        valid_regiostar_df: pd.DataFrame,
    ) -> None:
        """Test validation handles multiple files correctly."""
        # Create multiple valid files, ensuring ars columns preserve leading zeros
        df_locations = valid_counter_locations_df.copy()
        df_locations['ars'] = df_locations['ars'].astype(str)
        (temp_data_dir / "counter_locations.csv").write_text(
            df_locations.to_csv(index=False), encoding="utf-8"
        )

        df_regiostar = valid_regiostar_df.copy()
        df_regiostar['ars'] = df_regiostar['ars'].astype(str)
        (temp_data_dir / "regiostar.csv").write_text(
            df_regiostar.to_csv(index=False), encoding="utf-8"
        )

        config = PipelineConfig(**minimal_config_dict)
        runner = ValidationRunner(config)
        results = runner.run()

        # Check specific results (counter_measurements is skipped - no schema mapping)
        valid_datasets = ["counter_locations", "regiostar"]
        for dataset_name in valid_datasets:
            result = next(r for r in results if r.dataset_name == dataset_name)
            assert result.exists is True
            assert result.schema_valid is True
            assert result.row_count is not None and result.row_count > 0


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a validation result."""
        result = ValidationResult(
            dataset_name="test_dataset",
            schema_name="test_schema",
            file_path=Path("/tmp/test.csv"),
            exists=True,
            schema_valid=True,
            row_count=100,
            error_message=None,
        )
        assert result.dataset_name == "test_dataset"
        assert result.schema_name == "test_schema"
        assert result.exists is True
        assert result.schema_valid is True
        assert result.row_count == 100
        assert result.error_message is None

    def test_result_with_error(self) -> None:
        """Test creating a result with error message."""
        result = ValidationResult(
            dataset_name="test_dataset",
            schema_name="test_schema",
            file_path=Path("/tmp/test.csv"),
            exists=True,
            schema_valid=False,
            row_count=100,
            error_message="Schema validation failed",
        )
        assert result.schema_valid is False
        assert result.error_message == "Schema validation failed"
