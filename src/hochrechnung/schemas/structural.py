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
    Only requires name and place columns.
    """

    name: Series[str] = pa.Field(
        description="Place name",
    )
    place: Series[str] = pa.Field(
        description="Place type (city, town, village, etc.)",
    )

    class Config:
        """Schema configuration."""

        name = "CityCentroidSchema"
        strict = False  # Allow extra OSM columns and geometry
        coerce = True


class GebietseinheitenSchema(pa.DataFrameModel):
    """
    Schema for DE_Gebietseinheiten administrative boundary data.

    Contains administrative units at different levels (Land, Kreis, Verwaltungsgemeinschaft)
    with 12-digit ARS codes where trailing digits are padded with zeros.
    """

    ARS: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit code with trailing zeros)",
        str_length={"min_value": 12, "max_value": 12},
    )
    Type: Series[str] = pa.Field(
        description="Administrative level (Land, Kreis, Verwaltungsgemeinschaft)",
    )
    Name: Series[str] = pa.Field(
        description="Name of the administrative unit",
    )

    class Config:
        """Schema configuration."""

        name = "GebietseinheitenSchema"
        strict = False  # Allow extra columns and geometry
        coerce = True
