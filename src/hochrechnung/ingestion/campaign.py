"""
STADTRADELN campaign data ingestion.

Loads campaign metadata and participation demographics.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import DataLoader, GeoDataLoader
from hochrechnung.schemas.campaign import CampaignMetadataSchema, DemographicsSchema
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class CampaignMetadataLoader(DataLoader[CampaignMetadataSchema]):
    """Loader for STADTRADELN campaign metadata (dates per municipality)."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize campaign metadata loader."""
        super().__init__(config, CampaignMetadataSchema)

    def _load_raw(self) -> pd.DataFrame:
        """Load campaign metadata from CSV."""
        path = self.resolve_path(self.config.data_paths.campaign_stats)

        if not path.exists():
            msg = f"Campaign stats file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading campaign metadata", path=str(path))

        # Try different encodings
        try:
            df = pd.read_csv(path, encoding="cp1252", delimiter=";")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8", delimiter=",")

        # Normalize column names
        column_mapping = {
            "ags": "ars",
            "AGS": "ars",
            "ARS": "ars",
            "year": "year",
            "Year": "year",
            "start": "start_date",
            "Start": "start_date",
            "end": "end_date",
            "Ende": "end_date",
            "End": "end_date",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Filter to configured year
        if "year" in df.columns:
            year = self.config.temporal.year
            df = df[df["year"] == year]
            log.info("Filtered to year", year=year, rows=len(df))

        # Ensure ARS is 12-digit string
        if "ars" in df.columns:
            df["ars"] = df["ars"].astype(str).str.zfill(12)

        # Parse dates
        for col in ["start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df


class DemographicsLoader(GeoDataLoader[DemographicsSchema]):
    """Loader for STADTRADELN participation demographics."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize demographics loader."""
        super().__init__(config, DemographicsSchema)

    def _find_demographics_file(self) -> Path:
        """
        Find demographics file, checking multiple locations.

        Checks in order:
        1. Configured kommunen_stats path
        2. Year-specific CSV: campaign/SR_{year}.csv
        """
        data_root = self.config.data_paths.data_root

        # Try configured path first
        configured_path = self.resolve_path(self.config.data_paths.kommunen_stats)
        if configured_path.exists():
            return configured_path

        # Try year-specific CSV
        year = self.config.year
        year_csv = data_root / "campaign" / f"SR_{year}.csv"
        if year_csv.exists():
            return year_csv

        msg = (
            f"Demographics file not found. Tried:\n"
            f"  - {configured_path}\n"
            f"  - {year_csv}"
        )
        raise FileNotFoundError(msg)

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load demographics from shapefile or CSV."""
        import geopandas as gpd

        path = self._find_demographics_file()
        log.info("Loading demographics", path=str(path))

        # Load based on file type
        if path.suffix.lower() == ".csv":
            df = self._load_from_csv(path)
        else:
            df = gpd.read_file(path)

        # Filter to configured region using ARS prefix
        df = self._filter_and_aggregate_by_region(df)

        # Normalize column names
        column_mapping = {
            "ARS": "ars",
            "N_USERS": "n_users",
            "N_TRIPS": "n_trips",
            "TOTAL_KM": "total_km",
            "BUNDESLAND": "bundesland",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Ensure ARS is 12-digit string
        if "ars" in df.columns:
            df["ars"] = df["ars"].astype(str).str.zfill(12)

        return df

    def _load_from_csv(self, path: Path) -> pd.DataFrame:
        """Load demographics from CSV file."""
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")

        # Ensure ARS is string for filtering
        if "ARS" in df.columns:
            df["ARS"] = df["ARS"].astype(str).str.zfill(12)

        return df

    def _filter_and_aggregate_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter demographics to region and aggregate if needed.

        For county-level regions (ARS like 033540000000), aggregates
        all municipalities within the county (ARS starting with 03354).
        """
        region_ars = self.config.region.ars

        # Determine ARS column name
        ars_col = "ARS" if "ARS" in df.columns else "ars"
        if ars_col not in df.columns:
            log.warning("No ARS column found in demographics")
            return df

        # Ensure ARS is 12-digit string
        df[ars_col] = df[ars_col].astype(str).str.zfill(12)

        # Determine aggregation level from region ARS
        # If region ARS ends with zeros after position 5, it's a county
        # e.g., 033540000000 -> county prefix is 03354
        ars_str = str(region_ars).zfill(12)

        # Find where the trailing zeros start (minimum 5 chars for county)
        prefix_len = 12
        for i in range(11, 4, -1):  # Check from position 11 down to 5
            if ars_str[i] != "0":
                prefix_len = i + 1
                break
        else:
            # All zeros after some point, find the prefix
            for i in range(5, 12):
                if ars_str[i] == "0":
                    prefix_len = i
                    break

        ars_prefix = ars_str[:prefix_len]

        # Filter to matching municipalities
        mask = df[ars_col].str.startswith(ars_prefix)
        filtered = df[mask].copy()

        log.info(
            "Filtered demographics by ARS prefix",
            prefix=ars_prefix,
            matched=len(filtered),
            total=len(df),
        )

        if len(filtered) == 0:
            log.warning("No demographics found for region", ars_prefix=ars_prefix)
            return filtered

        # If we're aggregating at county level (prefix < 12 digits),
        # sum the user statistics and assign to all municipalities
        if prefix_len < 12 and len(filtered) > 1:
            agg_cols = ["N_USERS", "N_TRIPS", "TOTAL_KM", "n_users", "n_trips", "total_km"]
            agg_cols = [c for c in agg_cols if c in filtered.columns]

            if agg_cols:
                totals = {col: filtered[col].sum() for col in agg_cols}
                log.info(
                    "Aggregated demographics at county level",
                    n_municipalities=len(filtered),
                    totals={k: int(v) if isinstance(v, (int, float)) else v for k, v in totals.items()},
                )

                # Apply aggregated values to all rows
                for col, total in totals.items():
                    filtered[col] = total

                # Mark as county-level aggregation so downstream can handle population correctly
                filtered["_county_aggregated"] = True
                filtered["_county_n_municipalities"] = len(filtered)

        return filtered


def load_campaign_metadata(
    config: PipelineConfig, *, validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load campaign metadata.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with campaign metadata.
    """
    loader = CampaignMetadataLoader(config)
    return loader.load(validate=validate)


def load_demographics(
    config: PipelineConfig, *, validate: bool = True
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load participation demographics.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with participation demographics.
    """
    loader = DemographicsLoader(config)
    return loader.load(validate=validate)
