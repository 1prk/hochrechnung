"""
Pandera schemas for traffic volume data.

This is the core ML input data with bicycle volumes per OSM edge.

Column naming conventions (choose one consistently):

German (STADTRADELN delivery format):
    KantenId            - Feature-ID der Straßennetzkante
    GrundlagenId        - ID der Kante des zugrundeliegenden Netzes [OSM]
    Verkehrsmenge       - Anzahl Radfahrende während STADTRADELN-Kampagne
    VerkehrsmengeIn     - Anzahl Radfahrende in Geometrierichtung
    VerkehrsmengeGegen  - Anzahl Radfahrende gegen Geometrierichtung
    GeschwIn            - Durchschnittsgeschwindigkeit in km/h in Geometrierichtung
    GeschwGegen         - Durchschnittsgeschwindigkeit in km/h gegen Geometrierichtung

English (short internal names):
    edge_id             - Unique edge identifier
    base_id             - OSM way ID
    count               - Total bicycle count during campaign
    count_forward       - Count in geometry direction
    count_backward      - Count against geometry direction
    speed_forward_kmh   - Average speed in km/h in geometry direction
    speed_backward_kmh  - Average speed in km/h against geometry direction
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

# =============================================================================
# Column name equivalents (for reference, not automatic mapping)
# =============================================================================
# Use this to understand column correspondence when working with different formats.

COLUMN_EQUIVALENTS: dict[str, str] = {
    # German -> English
    "KantenId": "edge_id",
    "GrundlagenId": "base_id",
    "Verkehrsmenge": "count",
    "VerkehrsmengeIn": "count_forward",
    "VerkehrsmengeGegen": "count_backward",
    "GeschwIn": "speed_forward_kmh",
    "GeschwGegen": "speed_backward_kmh",
}


# =============================================================================
# German column names (STADTRADELN delivery format)
# =============================================================================


class TrafficVolumeGermanSchema(pa.DataFrameModel):
    """
    Schema for traffic volume data with German column names.

    Use this when working directly with STADTRADELN delivery files.
    """

    KantenId: Series[int] = pa.Field(
        unique=True,
        description="Feature-ID der Straßennetzkante",
    )
    GrundlagenId: Series[int] = pa.Field(
        description="ID der Kante des zugrundeliegenden Netzes [OSM]",
    )
    Verkehrsmenge: Series[int] = pa.Field(
        ge=0,
        description="Anzahl Radfahrende während der STADTRADELN-Kampagne",
    )
    VerkehrsmengeIn: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Anzahl Radfahrende in Geometrierichtung",
    )
    VerkehrsmengeGegen: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Anzahl Radfahrende gegen Geometrierichtung",
    )
    GeschwIn: Series[float] = pa.Field(
        ge=0,
        nullable=True,
        description="Durchschnittsgeschwindigkeit in km/h in Geometrierichtung",
    )
    GeschwGegen: Series[float] = pa.Field(
        ge=0,
        nullable=True,
        description="Durchschnittsgeschwindigkeit in km/h gegen Geometrierichtung",
    )

    class Config:
        """Schema configuration."""

        name = "TrafficVolumeGermanSchema"
        strict = False  # Allow geometry and other columns
        coerce = True


# =============================================================================
# English column names (short internal names)
# =============================================================================


class TrafficVolumeSchema(pa.DataFrameModel):
    """
    Schema for traffic volume data with English column names.

    Required columns: edge_id, base_id, count
    Optional columns: count_forward, count_backward, speed_forward_kmh,
                      speed_backward_kmh, bicycle_infrastructure

    Use this for internal processing after column renaming (if desired).
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
        description="Total bicycle count during campaign",
    )
    bicycle_infrastructure: Series[str] = pa.Field(
        isin=VALID_INFRA_CATEGORIES,
        nullable=True,
        description="Classified bicycle infrastructure type",
    )

    class Config:
        """Schema configuration."""

        name = "TrafficVolumeSchema"
        strict = False  # Allow geometry and optional columns
        coerce = True


class TrafficVolumeExtendedSchema(pa.DataFrameModel):
    """
    Extended schema with directional counts and speeds.

    Includes all columns from STADTRADELN delivery.
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
        description="Total bicycle count during campaign",
    )
    count_forward: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Count in geometry direction",
    )
    count_backward: Series[int] = pa.Field(
        ge=0,
        nullable=True,
        description="Count against geometry direction",
    )
    speed_forward_kmh: Series[float] = pa.Field(
        ge=0,
        nullable=True,
        description="Average speed in km/h in geometry direction",
    )
    speed_backward_kmh: Series[float] = pa.Field(
        ge=0,
        nullable=True,
        description="Average speed in km/h against geometry direction",
    )
    bicycle_infrastructure: Series[str] = pa.Field(
        isin=VALID_INFRA_CATEGORIES,
        nullable=True,
        description="Classified bicycle infrastructure type",
    )

    class Config:
        """Schema configuration."""

        name = "TrafficVolumeExtendedSchema"
        strict = False  # Allow geometry and other columns
        coerce = True


class TrafficVolumeRawSchema(pa.DataFrameModel):
    """
    Schema for raw traffic volume data before infrastructure mapping.

    Minimal schema - accepts either German or English column names.
    Allows any infrastructure string which will be mapped to valid categories.
    """

    edge_id: Series[int] = pa.Field(
        unique=True,
        description="Unique edge identifier (or 'KantenId' in German)",
    )
    base_id: Series[int] = pa.Field(
        description="OSM way ID (or 'GrundlagenId' in German)",
    )
    count: Series[int] = pa.Field(
        ge=0,
        description="Total bicycle count (or 'Verkehrsmenge' in German)",
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
