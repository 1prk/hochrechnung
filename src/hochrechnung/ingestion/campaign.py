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

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load demographics from shapefile."""
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.kommunen_stats)

        if not path.exists():
            msg = f"Kommunen stats file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading demographics", path=str(path))

        gdf = gpd.read_file(path)

        # Filter to configured region
        region_name = self.config.region.name
        if "BUNDESLAND" in gdf.columns:
            gdf = gdf[gdf["BUNDESLAND"] == region_name]
            log.info("Filtered to region", region=region_name, rows=len(gdf))

        # Normalize column names
        column_mapping = {
            "ARS": "ars",
            "ars": "ars",
            "N_USERS": "n_users",
            "N_TRIPS": "n_trips",
            "TOTAL_KM": "total_km",
            "BUNDESLAND": "bundesland",
        }

        gdf = gdf.rename(
            columns={k: v for k, v in column_mapping.items() if k in gdf.columns}
        )

        # Ensure ARS is 12-digit string
        if "ars" in gdf.columns:
            gdf["ars"] = gdf["ars"].astype(str).str.zfill(12)

        return gdf


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
