"""
Pandera schemas for traffic volume data.

This is the core ML input data with bicycle volumes per OSM edge.
"""

import pandera as pa
from pandera.typing import Series

# Valid infrastructure categories after OSM classification
VALID_INFRA_CATEGORIES = [
    "no",
    "mixed_way",
    "mit_road",
    "bicycle_lane",
    "bicycle_road",
    "bicycle_way",
]


class TrafficVolumeSchema(pa.DataFrameModel):
    """
    Schema for traffic volume data (core ML input).

    Contains bicycle traffic volumes aggregated per OSM edge from GPS traces.
    """

    edge_id: Series[int] = pa.Field(
        unique=True,
        description="Unique edge identifier",
    )
    base_id: Series[int] = pa.Field(
        description="OSM way ID",
    )
    count: Series[int] = pa.Field(
        ge=0,
        description="Aggregated bicycle traffic count from GPS traces",
    )
    bicycle_infrastructure: Series[str] = pa.Field(
        isin=VALID_INFRA_CATEGORIES,
        description="Classified bicycle infrastructure type",
    )

    class Config:
        """Schema configuration."""

        name = "TrafficVolumeSchema"
        strict = False  # Allow geometry column
        coerce = True


class TrafficVolumeRawSchema(pa.DataFrameModel):
    """
    Schema for raw traffic volume data before infrastructure mapping.

    Allows any infrastructure string which will be mapped to valid categories.
    """

    edge_id: Series[int] = pa.Field(
        unique=True,
        description="Unique edge identifier",
    )
    base_id: Series[int] = pa.Field(
        description="OSM way ID",
    )
    count: Series[int] = pa.Field(
        ge=0,
        description="Aggregated bicycle traffic count from GPS traces",
    )
    bicycle_infrastructure: Series[str] = pa.Field(
        nullable=True,
        description="Raw OSM bicycle infrastructure classification",
    )

    class Config:
        """Schema configuration."""

        name = "TrafficVolumeRawSchema"
        strict = False
        coerce = True
