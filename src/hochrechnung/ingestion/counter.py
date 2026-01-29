"""
Counter (DZS) data ingestion.

Loads permanent bicycle counting station locations and measurements.
"""

from pathlib import Path

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import DataLoader
from hochrechnung.schemas.counter import CounterLocationSchema, CounterMeasurementSchema
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


class CounterLocationLoader(DataLoader[CounterLocationSchema]):
    """Loader for counter location data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize counter location loader."""
        super().__init__(config, CounterLocationSchema)

    def _load_raw(self) -> pd.DataFrame:
        """Load counter location CSV."""
        path = self.resolve_path(self.config.data_paths.counter_locations)

        if not path.exists():
            msg = f"Counter locations file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading counter locations", path=str(path))

        df = pd.read_csv(path)

        # Normalize column names
        column_mapping = {
            "ID": "id",
            "Name": "name",
            "name": "name",
            "latitude": "latitude",
            "Latitude": "latitude",
            "lat": "latitude",
            "longitude": "longitude",
            "Longitude": "longitude",
            "lon": "longitude",
            "ARS": "ars",
            "ars": "ars",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Ensure id is string
        if "id" in df.columns:
            df["id"] = df["id"].astype(str).str.zfill(3)

        return df


class CounterMeasurementLoader(DataLoader[CounterMeasurementSchema]):
    """Loader for counter measurement (daily count) data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize counter measurement loader."""
        super().__init__(config, CounterMeasurementSchema)

    def _load_raw(self) -> pd.DataFrame:
        """
        Load counter measurement CSV.

        The source file has a wide format with counters as columns.
        This loader transforms it to a long format.
        """
        path = self.resolve_path(self.config.data_paths.counter_measurements)

        if not path.exists():
            msg = f"Counter measurements file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading counter measurements", path=str(path))

        # Load with skiprows to handle header format
        df = pd.read_csv(path, skiprows=2)

        # Find timestamp column
        timestamp_col = None
        for col in ["Zeit", "Time", "timestamp", "date", "Date"]:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            msg = "Could not find timestamp column in counter measurements"
            raise ValueError(msg)

        # Parse dates - handle both German format and ISO format
        # German format: "Mo. 1. Mai 2023"
        # ISO format: "2024-05-01 00:00:00"
        sample_date = df[timestamp_col].astype(str).iloc[0] if len(df) > 0 else ""

        if sample_date and sample_date[0].isdigit():
            # Numeric format detected - could be ISO (YYYY-MM-DD) or European (DD-MM-YYYY)
            # Check if it starts with a 4-digit year
            if sample_date[:4].isdigit() and len(sample_date) >= 10 and sample_date[4] == "-":
                # ISO format: "2024-05-01 00:00:00"
                df[timestamp_col] = pd.to_datetime(
                    df[timestamp_col], format="%Y-%m-%d %H:%M:%S", errors="coerce"
                )
                log.debug("Parsed dates using ISO format")
            else:
                # European format: "01-05-2022 00:00:00" (DD-MM-YYYY)
                df[timestamp_col] = pd.to_datetime(
                    df[timestamp_col], format="%d-%m-%Y %H:%M:%S", errors="coerce"
                )
                log.debug("Parsed dates using European format (DD-MM-YYYY)")
        else:
            # German format: "Mo. 1. Mai 2023"
            german_months = {
                "Januar": "01", "Februar": "02", "März": "03", "April": "04",
                "Mai": "05", "Juni": "06", "Juli": "07", "August": "08",
                "September": "09", "Oktober": "10", "November": "11", "Dezember": "12",
            }

            # Strip weekday prefix (e.g., "Mo. ") - first 4 characters
            date_str = df[timestamp_col].astype(str).str.slice(4)
            # Remove periods from day: "1. Mai 2023" -> "1 Mai 2023"
            date_str = date_str.str.replace(".", "", regex=False)
            # Replace German month names with numbers
            for month_de, month_num in german_months.items():
                date_str = date_str.str.replace(month_de, month_num, regex=False)
            # Now format is "1 05 2023" - parse as day month year
            df[timestamp_col] = pd.to_datetime(date_str, format="%d %m %Y", errors="coerce")
            log.debug("Parsed dates using German format")

        # Identify counter columns (numeric columns that aren't the timestamp)
        counter_cols = [
            col
            for col in df.columns
            if col != timestamp_col and df[col].dtype in ["int64", "float64", "object"]
        ]

        # Melt to long format
        df_long = pd.melt(
            df,
            id_vars=[timestamp_col],
            value_vars=counter_cols,
            var_name="counter_id",
            value_name="count",
        )

        # Rename and clean
        df_long = df_long.rename(columns={timestamp_col: "timestamp"})
        df_long["count"] = pd.to_numeric(df_long["count"], errors="coerce")
        df_long = df_long.dropna(subset=["timestamp", "count"])
        df_long["count"] = df_long["count"].astype(int)

        # Counter IDs are the column names directly (e.g., "001", "064b", "1001")
        # Keep them as-is, just strip whitespace
        df_long["counter_id"] = df_long["counter_id"].str.strip()

        return df_long


def load_counter_locations(
    config: PipelineConfig, *, validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load counter locations.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with counter locations.
    """
    loader = CounterLocationLoader(config)
    return loader.load(validate=validate)


def load_counter_measurements(
    config: PipelineConfig, *, validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load counter measurements.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with counter measurements in long format.
    """
    loader = CounterMeasurementLoader(config)
    return loader.load(validate=validate)
