"""
Pandera schemas for structural/administrative data.
"""

from typing import Optional

import pandera as pa
from pandera.typing import Series


class MunicipalitySchema(pa.DataFrameModel):
    """
    Schema for municipality (Gemeinde) data from VG250.

    Contains administrative boundaries and population data.
    """

    ars: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    name: Series[str] = pa.Field(
        description="Municipality name",
    )
    population: Series[int] = pa.Field(
        ge=0,
        description="Population (Einwohnerzahl)",
    )
    land: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 2},
        description="Federal state code (2-digit)",
    )

    class Config:
        """Schema configuration."""

        name = "MunicipalitySchema"
        strict = False  # Allow geometry column
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
    regiostar5: Series[int] = pa.Field(
        ge=1,
        le=5,
        description="RegioStaR5 classification (1-5)",
    )
    regiostar7: Optional[Series[int]] = pa.Field(
        ge=1,
        le=7,
        nullable=True,
        description="RegioStaR7 classification (1-7)",
    )

    class Config:
        """Schema configuration."""

        name = "RegioStarSchema"
        strict = False
        coerce = True


class CityCentroidSchema(pa.DataFrameModel):
    """
    Schema for city/town centroid data.

    Used for calculating distance-to-center features.
    """

    id: Series[int] = pa.Field(
        description="Unique centroid identifier",
    )
    name: Optional[Series[str]] = pa.Field(
        nullable=True,
        description="Place name",
    )
    latitude: Series[float] = pa.Field(
        ge=-90.0,
        le=90.0,
        description="Centroid latitude in WGS84",
    )
    longitude: Series[float] = pa.Field(
        ge=-180.0,
        le=180.0,
        description="Centroid longitude in WGS84",
    )

    class Config:
        """Schema configuration."""

        name = "CityCentroidSchema"
        strict = False  # Allow geometry column
        coerce = True
