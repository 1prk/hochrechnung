"""
Pandera schemas for counter (DZS) data.

DZS = Dauerzählstelle (permanent counting station)
"""

import pandera.pandas as pa
from pandera.typing import Series


class CounterLocationSchema(pa.DataFrameModel):
    """
    Schema for counter location data.

    Contains geographic information about permanent bicycle counting stations.
    """

    id: Series[str] = pa.Field(
        description="Unique counter identifier (3-digit string, e.g., '001')",
        str_length={"min_value": 1, "max_value": 10},
    )
    name: Series[str] = pa.Field(
        description="Human-readable station name",
        nullable=True,
    )
    latitude: Series[float] = pa.Field(
        ge=-90.0,
        le=90.0,
        description="Station latitude in WGS84",
    )
    longitude: Series[float] = pa.Field(
        ge=-180.0,
        le=180.0,
        description="Station longitude in WGS84",
    )
    ars: Series[str] | None = pa.Field(
        description="Amtlicher Regionalschlüssel (12-digit municipality code)",
        str_length={"min_value": 12, "max_value": 12},
        nullable=True,
        default=None,
    )

    class Config:
        """Schema configuration."""

        name = "CounterLocationSchema"
        strict = False  # Allow extra columns
        coerce = True  # Coerce types where possible


class CounterMeasurementSchema(pa.DataFrameModel):
    """
    Schema for counter measurement data (daily counts).

    Contains time-series data from permanent counting stations.
    """

    timestamp: Series[pa.DateTime] = pa.Field(
        description="Measurement date",
    )
    counter_id: Series[str] = pa.Field(
        description="Counter identifier matching CounterLocationSchema.id",
    )
    count: Series[int] = pa.Field(
        ge=0,
        description="Daily bicycle count",
    )

    class Config:
        """Schema configuration."""

        name = "CounterMeasurementSchema"
        strict = False
        coerce = True
