"""
Structural data ingestion.

Loads municipality boundaries, RegioStaR classification, and city centroids.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import DataLoader, GeoDataLoader
from hochrechnung.schemas.structural import (
    CityCentroidSchema,
    MunicipalitySchema,
    RegioStarSchema,
)
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class MunicipalityLoader(GeoDataLoader[MunicipalitySchema]):
    """Loader for municipality (VG250) data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize municipality loader."""
        super().__init__(config, MunicipalitySchema)

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load municipality boundaries from GeoPackage."""
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.municipalities)

        if not path.exists():
            msg = f"Municipalities file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading municipalities", path=str(path))

        # Load gemeinde layer from VG250
        gdf = gpd.read_file(path, layer="v_vg250_gem")

        # Filter by federal state (Land) using first 2 digits of ARS
        land_code = self.config.land_code
        gdf = gdf[gdf["Land"] == land_code]

        log.info("Filtered to region", land_code=land_code, rows=len(gdf))

        # Normalize column names
        column_mapping = {
            "Regionalschlüssel_ARS": "ars",
            "ARS": "ars",
            "GeografischerName_GEN": "name",
            "GEN": "name",
            "Einwohnerzahl_EWZ": "population",
            "EWZ": "population",
            "Land": "land",
        }

        gdf = gdf.rename(
            columns={k: v for k, v in column_mapping.items() if k in gdf.columns}
        )

        # Select essential columns
        keep_cols = ["ars", "name", "population", "land", "geometry"]
        gdf = gdf[[col for col in keep_cols if col in gdf.columns]]

        return gdf


class RegioStarLoader(DataLoader[RegioStarSchema]):
    """Loader for RegioStaR classification data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize RegioStaR loader."""
        super().__init__(config, RegioStarSchema)

    def _load_raw(self) -> pd.DataFrame:
        """Load RegioStaR classification from CSV."""
        path = self.resolve_path(self.config.data_paths.regiostar)

        if not path.exists():
            msg = f"RegioStaR file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading RegioStaR data", path=str(path))

        df = pd.read_csv(path, encoding="cp1252", delimiter=";")

        # Normalize column names
        column_mapping = {
            "gemrs_22": "ars",
            "RegioStaR5": "regiostar5",
            "RegioStaR7": "regiostar7",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Ensure ARS is 12-digit string
        if "ars" in df.columns:
            df["ars"] = df["ars"].astype(str).str.zfill(12)

        # Select essential columns
        keep_cols = ["ars", "regiostar5", "regiostar7"]
        df = df[[col for col in keep_cols if col in df.columns]]

        return df


class CityCentroidLoader(GeoDataLoader[CityCentroidSchema]):
    """Loader for city/town centroid data."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize city centroid loader."""
        super().__init__(config, CityCentroidSchema)

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load city centroids from GeoPackage."""
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.city_centroids)

        if not path.exists():
            msg = f"City centroids file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading city centroids", path=str(path))

        gdf = gpd.read_file(path)

        # Normalize column names
        column_mapping = {
            "ID": "id",
            "id": "id",
            "Name": "name",
            "name": "name",
        }

        gdf = gdf.rename(
            columns={k: v for k, v in column_mapping.items() if k in gdf.columns}
        )

        # Extract coordinates if not present
        if "latitude" not in gdf.columns and gdf.geometry is not None:
            gdf["latitude"] = gdf.geometry.y
        if "longitude" not in gdf.columns and gdf.geometry is not None:
            gdf["longitude"] = gdf.geometry.x

        return gdf


def load_municipalities(
    config: PipelineConfig, *, validate: bool = True
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load municipality data.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with municipality boundaries.
    """
    loader = MunicipalityLoader(config)
    return loader.load(validate=validate)


def load_regiostar(config: PipelineConfig, *, validate: bool = True) -> pd.DataFrame:
    """
    Convenience function to load RegioStaR data.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with RegioStaR classification.
    """
    loader = RegioStarLoader(config)
    return loader.load(validate=validate)


def load_city_centroids(
    config: PipelineConfig, *, validate: bool = True
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load city centroids.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with city centroids.
    """
    loader = CityCentroidLoader(config)
    return loader.load(validate=validate)
