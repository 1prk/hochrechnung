"""
Core validation logic for data files.

Validates data files against registered Pandera schemas.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pandera as pa

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.traffic import TrafficVolumeLoader
from hochrechnung.schemas.registry import SchemaRegistry
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a single dataset."""

    dataset_name: str
    schema_name: str | None
    file_path: Path
    exists: bool
    schema_valid: bool | None
    row_count: int | None
    error_message: str | None


# Mapping from config dataset names to schema names
DATASET_SCHEMA_MAP: dict[str, str] = {
    "counter_locations": "counter_location",
    # counter_measurements uses wide format (one column per counter) - skip validation
    "traffic_volumes": "traffic_volume_raw",
    "municipalities": "municipality",
    "regiostar": "regiostar",
    "city_centroids": "city_centroid",
    "campaign_stats": "campaign_metadata",
    "kommunen_stats": "commune_statistics",
    "gebietseinheiten": "gebietseinheiten",
}


class ValidationRunner:
    """
    Runs validation for all configured datasets.

    Validates data files against their registered schemas and reports results.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize validation runner.

        Args:
            config: Pipeline configuration containing data paths.
        """
        self.config = config

    def run(self) -> list[ValidationResult]:
        """
        Run validation for all datasets in config.

        Returns:
            List of validation results, one per dataset.
        """
        results: list[ValidationResult] = []

        # Get all dataset field names from DataPathsConfig (excluding data_root)
        from hochrechnung.config.settings import DataPathsConfig

        dataset_attrs = [
            field_name
            for field_name in DataPathsConfig.model_fields.keys()
            if field_name != "data_root"
        ]

        for dataset_attr in dataset_attrs:
            result = self._validate_dataset(dataset_attr)
            results.append(result)

        return results

    def _validate_dataset(self, dataset_attr: str) -> ValidationResult:
        """
        Validate a single dataset.

        Args:
            dataset_attr: Attribute name from DataPathsConfig.

        Returns:
            ValidationResult for the dataset.
        """
        # Resolve file path
        file_path = self.config.data_paths.resolve(dataset_attr)

        # Check if schema mapping exists
        schema_name = DATASET_SCHEMA_MAP.get(dataset_attr)
        if schema_name is None:
            log.warning(
                "No schema mapping for dataset",
                dataset=dataset_attr,
                path=str(file_path),
            )
            return ValidationResult(
                dataset_name=dataset_attr,
                schema_name=None,
                file_path=file_path,
                exists=file_path.exists(),
                schema_valid=None,
                row_count=None,
                error_message="No schema mapping defined",
            )

        # Check file exists
        if not file_path.exists():
            log.warning("Data file not found", dataset=dataset_attr, path=str(file_path))
            return ValidationResult(
                dataset_name=dataset_attr,
                schema_name=schema_name,
                file_path=file_path,
                exists=False,
                schema_valid=None,
                row_count=None,
                error_message="File not found",
            )

        # Load and validate
        try:
            # Use specialized loaders for datasets that need column transformations
            if dataset_attr == "traffic_volumes":
                df = self._load_traffic_volumes()
            else:
                df = self._load_file(file_path)
            row_count = len(df)

            # Validate against schema
            SchemaRegistry.validate(df, schema_name)

            log.info(
                "Validation passed",
                dataset=dataset_attr,
                schema=schema_name,
                rows=row_count,
            )
            return ValidationResult(
                dataset_name=dataset_attr,
                schema_name=schema_name,
                file_path=file_path,
                exists=True,
                schema_valid=True,
                row_count=row_count,
                error_message=None,
            )

        except pa.errors.SchemaError as e:
            # Extract first few errors for reporting
            error_msg = self._format_schema_error(e)
            log.error(
                "Schema validation failed",
                dataset=dataset_attr,
                schema=schema_name,
                error=error_msg,
            )
            return ValidationResult(
                dataset_name=dataset_attr,
                schema_name=schema_name,
                file_path=file_path,
                exists=True,
                schema_valid=False,
                row_count=len(df) if "df" in locals() else None,
                error_message=error_msg,
            )

        except Exception as e:
            # Handle other errors (file read, etc.)
            error_msg = f"{type(e).__name__}: {e!s}"
            log.error("Validation error", dataset=dataset_attr, error=error_msg)
            return ValidationResult(
                dataset_name=dataset_attr,
                schema_name=schema_name,
                file_path=file_path,
                exists=True,
                schema_valid=False,
                row_count=None,
                error_message=error_msg,
            )

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a data file based on extension.

        Args:
            file_path: Path to the file.

        Returns:
            DataFrame with the loaded data.

        Raises:
            ValueError: If file format not supported.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            # Try UTF-8 first, fall back to Latin-1 for German umlauts
            # Read ars/ARS column as string to preserve leading zeros
            # Auto-detect delimiter (comma or semicolon)
            # Use on_bad_lines='warn' for malformed CSVs
            try:
                df = pd.read_csv(
                    file_path,
                    dtype={"ars": str, "ARS": str, "ags": str},
                    encoding="utf-8",
                    on_bad_lines="warn",
                    sep=None,  # Auto-detect separator
                    engine="python",
                )
            except UnicodeDecodeError:
                log.warning(
                    "UTF-8 decode failed, retrying with Latin-1",
                    path=str(file_path),
                )
                df = pd.read_csv(
                    file_path,
                    dtype={"ars": str, "ARS": str, "ags": str},
                    encoding="latin-1",
                    on_bad_lines="warn",
                    sep=None,  # Auto-detect separator
                    engine="python",
                )
            return self._normalize_ars(df)

        if suffix == ".fgb":
            gdf = gpd.read_file(file_path)
            # Drop geometry for schema validation (schemas are for tabular data)
            if "geometry" in gdf.columns:
                df = pd.DataFrame(gdf.drop(columns="geometry"))
            else:
                df = pd.DataFrame(gdf)
            return self._normalize_ars(df)

        if suffix == ".gpkg":
            gdf = gpd.read_file(file_path)
            if "geometry" in gdf.columns:
                df = pd.DataFrame(gdf.drop(columns="geometry"))
            else:
                df = pd.DataFrame(gdf)
            return self._normalize_ars(df)

        if suffix == ".shp":
            gdf = gpd.read_file(file_path)
            if "geometry" in gdf.columns:
                df = pd.DataFrame(gdf.drop(columns="geometry"))
            else:
                df = pd.DataFrame(gdf)
            return self._normalize_ars(df)

        if suffix == ".json":
            df = pd.read_json(file_path, dtype={"ars": str})
            return self._normalize_ars(df)

        msg = f"Unsupported file format: {suffix}"
        raise ValueError(msg)

    def _load_traffic_volumes(self) -> pd.DataFrame:
        """
        Load traffic volumes using TrafficVolumeLoader.

        Uses the specialized loader which handles German-to-English column
        renaming automatically.

        Returns:
            DataFrame with traffic volume data (geometry dropped for validation).
        """
        loader = TrafficVolumeLoader(self.config)
        gdf = loader._load_raw_geo()

        # Drop geometry for schema validation (schemas are for tabular data)
        if "geometry" in gdf.columns:
            return pd.DataFrame(gdf.drop(columns="geometry"))
        return pd.DataFrame(gdf)

    def _normalize_ars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ARS codes to exactly 12 digits.

        ARS must be 12 digits. Handles two truncation cases:
        1. Leading zeros dropped (e.g., 1234 → 000000001234)
        2. Trailing zeros dropped (e.g., 03000000 → 030000000000)

        Decision: If first 2 chars form valid state code (01-16), pad right.
        Otherwise, pad left.

        Args:
            df: DataFrame potentially containing ars/ARS/ags column.

        Returns:
            DataFrame with normalized ARS codes.
        """

        def normalize_ars_code(code: str) -> str:
            """Normalize a single ARS code to 12 digits."""
            code = str(code).strip()
            if len(code) >= 12:
                return code[:12]
            if len(code) == 0:
                return "000000000000"

            # Check if first 2 chars are valid state code (01-16)
            try:
                state_code = int(code[:2])
                if 1 <= state_code <= 16:
                    # Valid state code - pad with trailing zeros
                    return code.ljust(12, "0")
            except (ValueError, IndexError):
                pass

            # Not a valid state code - pad with leading zeros
            return code.zfill(12)

        # Check for ars column (lowercase)
        if "ars" in df.columns:
            df["ars"] = df["ars"].apply(normalize_ars_code)

        # Check for ARS column (uppercase)
        if "ARS" in df.columns:
            df["ARS"] = df["ARS"].apply(normalize_ars_code)

        # Check for ags column (9-digit code) - pad and rename to ars
        if "ags" in df.columns and "ars" not in df.columns:
            df["ars"] = df["ags"].apply(normalize_ars_code)

        # Check for gemrs_22 column (regiostar specific) - rename to ars
        if "gemrs_22" in df.columns and "ars" not in df.columns:
            df["ars"] = df["gemrs_22"].apply(normalize_ars_code)

        return df

    def _format_schema_error(self, error: pa.errors.SchemaError) -> str:
        """
        Format schema error for user-friendly display.

        Args:
            error: Pandera SchemaError.

        Returns:
            Formatted error message (first 5 violations).
        """
        # Get failure cases if available
        if hasattr(error, "failure_cases") and error.failure_cases is not None:
            failures = error.failure_cases

            # Handle case where failure_cases is a DataFrame
            if isinstance(failures, pd.DataFrame):
                n_failures = len(failures)

                # Show first 5 violations
                if n_failures > 5:
                    failures_str = failures.head(5).to_string(index=False)
                    return f"{n_failures} validation errors (showing first 5):\n{failures_str}"
                return f"{n_failures} validation error(s):\n{failures.to_string(index=False)}"

        # Fallback to error message
        return str(error).split("\n")[0][:200]  # First line, max 200 chars
