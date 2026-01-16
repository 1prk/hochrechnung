"""
Pandera schemas for STADTRADELN campaign data.

STADTRADELN is a cycling campaign where participants track their rides.
"""

from typing import Optional

import pandera as pa
from pandera.typing import Series


class CampaignMetadataSchema(pa.DataFrameModel):
    """
    Schema for STADTRADELN campaign metadata.

    Contains temporal information about when each municipality participated.
    """

    ars: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit municipality code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    year: Series[int] = pa.Field(
        ge=2018,
        le=2030,
        description="Campaign year",
    )
    start_date: Optional[Series[pa.DateTime]] = pa.Field(
        nullable=True,
        description="Campaign start date for this municipality",
    )
    end_date: Optional[Series[pa.DateTime]] = pa.Field(
        nullable=True,
        description="Campaign end date for this municipality",
    )

    class Config:
        """Schema configuration."""

        name = "CampaignMetadataSchema"
        strict = False
        coerce = True


class DemographicsSchema(pa.DataFrameModel):
    """
    Schema for STADTRADELN participation demographics.

    Contains aggregated participation statistics per municipality.
    """

    ars: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit municipality code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    n_users: Series[int] = pa.Field(
        ge=0,
        description="Number of STADTRADELN participants",
    )
    n_trips: Series[int] = pa.Field(
        ge=0,
        description="Number of recorded trips",
    )
    total_km: Series[float] = pa.Field(
        ge=0.0,
        description="Total kilometers cycled",
    )
    bundesland: Optional[Series[str]] = pa.Field(
        nullable=True,
        description="Federal state name",
    )

    class Config:
        """Schema configuration."""

        name = "DemographicsSchema"
        strict = False
        coerce = True
