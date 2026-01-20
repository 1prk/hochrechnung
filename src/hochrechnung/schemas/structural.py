"""
Pandera schemas for structural/administrative data.
"""

import pandera as pa
from pandera.typing import Series


class MunicipalitySchema(pa.DataFrameModel):
    """
    Schema for municipality (Gemeinde) data from VG250.

    Contains administrative boundaries and population data.
    Note: VG250 GPKG uses uppercase column names (ARS, GEN, etc.)
    """

    ARS: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    GEN: Series[str] = pa.Field(
        description="Municipality name (Gemeindename)",
    )

    class Config:
        """Schema configuration."""

        name = "MunicipalitySchema"
        strict = False  # Allow extra columns and geometry
        coerce = True


class RegioStarSchema(pa.DataFrameModel):
    """
    Schema for RegioStaR classification data.

    RegioStaR is a regional classification system by BBSR.
    """

    ars: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    RegioStaR5: Series[int] = pa.Field(
        ge=51,
        le=59,
        description="RegioStaR5 classification (51-59)",
    )

    class Config:
        """Schema configuration."""

        name = "RegioStarSchema"
        strict = False
        coerce = True


class CityCentroidSchema(pa.DataFrameModel):
    """
    Schema for city/town centroid data from OSM.

    Used for calculating distance-to-center features.
    Note: OSM data uses 'osm_id' and 'osm_type' columns.
    """

    osm_id: Series[int] = pa.Field(
        description="OSM identifier",
    )
    osm_type: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM type (node/way/relation)",
    )
    name: Series[str] | None = pa.Field(
        nullable=True,
        description="Place name",
    )

    class Config:
        """Schema configuration."""

        name = "CityCentroidSchema"
        strict = False  # Allow extra OSM columns and geometry
        coerce = True
