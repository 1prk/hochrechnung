"""
Pandera schemas for OSM infrastructure data.
"""

import pandera.pandas as pa
from pandera.typing import Series


class OSMInfrastructureSchema(pa.DataFrameModel):
    """
    Schema for OSM bicycle infrastructure classification.

    Contains the raw OSM infrastructure attributes before simplification.
    """

    way_id: Series[int] = pa.Field(
        description="OSM way identifier",
    )
    highway: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM highway tag value",
    )
    cycleway: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM cycleway tag value",
    )
    cycleway_left: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM cycleway:left tag value",
    )
    cycleway_right: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM cycleway:right tag value",
    )
    bicycle: Series[str] | None = pa.Field(
        nullable=True,
        description="OSM bicycle tag value",
    )
    bicycle_infrastructure: Series[str] = pa.Field(
        description="Classified infrastructure category from osmcategorizer",
    )

    class Config:
        """Schema configuration."""

        name = "OSMInfrastructureSchema"
        strict = False
        coerce = True
