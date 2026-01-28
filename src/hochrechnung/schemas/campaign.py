"""
Pandera schemas for STADTRADELN campaign data.

STADTRADELN is a cycling campaign where participants track their rides.
"""

import pandera as pa
from pandera.typing import Series


class CampaignMetadataSchema(pa.DataFrameModel):
    """
    Schema for STADTRADELN campaign metadata.

    Contains temporal information about when each municipality participated.
    """

    ags: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit municipality code)",
        str_length={"min_value": 12, "max_value": 12},
    )
    year: Series[int] = pa.Field(
        ge=2018,
        le=2030,
        description="Campaign year",
    )
    start_date: Series[pa.DateTime] | None = pa.Field(
        nullable=True,
        description="Campaign start date for this municipality",
    )
    end_date: Series[pa.DateTime] | None = pa.Field(
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
    bundesland: Series[str] | None = pa.Field(
        nullable=True,
        description="Federal state name",
    )

    class Config:
        """Schema configuration."""

        name = "DemographicsSchema"
        strict = False
        coerce = True


class CommuneStatisticsSchema(pa.DataFrameModel):
    """
    Schema for STADTRADELN commune statistics JSON files.

    Contains per-municipality aggregated trip statistics including
    distributions for speed, duration, distance, time of day, etc.
    File format: SRxx_Commune_Statistics.json
    """

    ars: Series[str] = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit municipality code)",
    )
    name: Series[str] = pa.Field(
        nullable=True,
        description="Municipality name",
    )
    users_n: Series[int] = pa.Field(
        ge=0,
        description="Number of STADTRADELN participants",
    )
    trips_n: Series[int] = pa.Field(
        ge=0,
        description="Number of recorded trips",
    )
    distance_km: Series[float] = pa.Field(
        ge=0.0,
        description="Total distance cycled in km",
    )
    avg_distance_km: Series[float] = pa.Field(
        ge=0.0,
        description="Average trip distance in km",
    )
    median_distance_km: Series[float] = pa.Field(
        ge=0.0,
        description="Median trip distance in km",
    )
    average_speed: Series[float] = pa.Field(
        ge=0.0,
        description="Average cycling speed in km/h",
    )
    average_duration: Series[float] = pa.Field(
        ge=0.0,
        description="Average trip duration in minutes",
    )

    class Config:
        """Schema configuration."""

        name = "CommuneStatisticsSchema"
        strict = False  # Allow distribution columns (trip_distance_dist, etc.)
        coerce = True
