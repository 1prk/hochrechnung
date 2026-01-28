"""
DE_Gebietseinheiten administrative boundary data ingestion.

Loads administrative boundaries at different levels (Land, Kreis, Verwaltungsgemeinschaft)
from the DE_Gebietseinheiten GeoPackage.
"""

from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import GeoDataLoader
from hochrechnung.schemas.structural import GebietseinheitenSchema
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class AdminLevel(str, Enum):
    """Administrative level for Gebietseinheiten."""

    LAND = "Land"
    KREIS = "Kreis"
    VERWALTUNGSGEMEINSCHAFT = "Verwaltungsgemeinschaft"

    @property
    def ars_prefix_length(self) -> int:
        """Get the significant ARS prefix length for this admin level."""
        return {
            AdminLevel.LAND: 2,
            AdminLevel.KREIS: 5,
            AdminLevel.VERWALTUNGSGEMEINSCHAFT: 9,
        }[self]

    @classmethod
    def from_string(cls, value: str) -> "AdminLevel":
        """Create AdminLevel from string value."""
        for level in cls:
            if level.value.lower() == value.lower():
                return level
        msg = f"Unknown admin level: {value}. Valid: {[l.value for l in cls]}"
        raise ValueError(msg)


class GebietseinheitenLoader(GeoDataLoader[GebietseinheitenSchema]):
    """Loader for DE_Gebietseinheiten administrative boundary data."""

    def __init__(
        self,
        config: PipelineConfig,
        admin_level: AdminLevel = AdminLevel.VERWALTUNGSGEMEINSCHAFT,
    ) -> None:
        """
        Initialize Gebietseinheiten loader.

        Args:
            config: Pipeline configuration.
            admin_level: Administrative level to filter (Land, Kreis, or Verwaltungsgemeinschaft).
        """
        super().__init__(config, GebietseinheitenSchema)
        self.admin_level = admin_level

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load Gebietseinheiten boundaries from GeoPackage."""
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.gebietseinheiten)

        if not path.exists():
            msg = f"Gebietseinheiten file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading Gebietseinheiten", path=str(path))

        # Load the GeoPackage
        gdf = gpd.read_file(path)

        log.info(
            "Loaded raw Gebietseinheiten",
            rows=len(gdf),
            columns=list(gdf.columns),
        )

        # Normalize column names - the GPKG may have different column names
        column_mapping = {
            "ARS": "ars",
            "Type": "admin_level",
            "Name": "name",
            "GEN": "name",
            "BEZ": "admin_level",
        }

        gdf = gdf.rename(
            columns={k: v for k, v in column_mapping.items() if k in gdf.columns}
        )

        # Ensure ARS is 12-digit string
        if "ars" in gdf.columns:
            gdf["ars"] = gdf["ars"].astype(str).str.zfill(12)

        # Filter by admin level (Type column)
        # Special handling: when VG level is requested, also include Kreisfreie Städte
        # (independent cities that have Type='Kreis' but no VG subdivisions)
        if "admin_level" in gdf.columns:
            if self.admin_level == AdminLevel.VERWALTUNGSGEMEINSCHAFT:
                # Get all VGs
                vg_mask = gdf["admin_level"] == "Verwaltungsgemeinschaft"
                vg_gdf = gdf[vg_mask]

                # Find Kreisfreie Städte: Kreise that have no VG with matching prefix
                kreis_gdf = gdf[gdf["admin_level"] == "Kreis"].copy()
                vg_prefixes = set(vg_gdf["ars"].str[:5].unique())
                kreisfreie_mask = ~kreis_gdf["ars"].str[:5].isin(vg_prefixes)
                kreisfreie_gdf = kreis_gdf[kreisfreie_mask]

                log.info(
                    "Identified Kreisfreie Städte",
                    n_kreisfreie=len(kreisfreie_gdf),
                    examples=kreisfreie_gdf["name"].head(5).tolist() if len(kreisfreie_gdf) > 0 else [],
                )

                # Combine VGs and Kreisfreie Städte
                gdf = gpd.GeoDataFrame(
                    pd.concat([vg_gdf, kreisfreie_gdf], ignore_index=True),
                    crs=gdf.crs,
                )
                log.info(
                    "Filtered by admin level (VG + Kreisfreie Städte)",
                    admin_level=self.admin_level.value,
                    n_vg=len(vg_gdf),
                    n_kreisfreie=len(kreisfreie_gdf),
                    total_rows=len(gdf),
                )
            else:
                gdf = gdf[gdf["admin_level"] == self.admin_level.value]
                log.info(
                    "Filtered by admin level",
                    admin_level=self.admin_level.value,
                    rows=len(gdf),
                )

        # Filter by federal state (Land) using first 2 digits of ARS
        land_code = self.config.land_code
        if "ars" in gdf.columns:
            gdf = gdf[gdf["ars"].str.startswith(land_code)]
            log.info("Filtered to region", land_code=land_code, rows=len(gdf))

        # Select essential columns
        keep_cols = ["ars", "admin_level", "name", "geometry"]
        gdf = gdf[[col for col in keep_cols if col in gdf.columns]]

        return gdf


def load_gebietseinheiten(
    config: PipelineConfig,
    admin_level: AdminLevel = AdminLevel.VERWALTUNGSGEMEINSCHAFT,
    *,
    validate: bool = True,
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load Gebietseinheiten data.

    Args:
        config: Pipeline configuration.
        admin_level: Administrative level to filter.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with administrative boundaries.
    """
    loader = GebietseinheitenLoader(config, admin_level=admin_level)
    return loader.load(validate=validate)
