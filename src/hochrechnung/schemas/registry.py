"""
Schema registry for versioning and discovery.

Provides centralized access to all schema definitions with version tracking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import pandera.pandas as pa

from hochrechnung.schemas.campaign import CampaignMetadataSchema, DemographicsSchema
from hochrechnung.schemas.counter import CounterLocationSchema, CounterMeasurementSchema
from hochrechnung.schemas.infrastructure import OSMInfrastructureSchema
from hochrechnung.schemas.structural import (
    CityCentroidSchema,
    MunicipalitySchema,
    RegioStarSchema,
)
from hochrechnung.schemas.traffic import (
    TrafficVolumeGermanSchema,
    TrafficVolumeRawSchema,
    TrafficVolumeSchema,
)

if TYPE_CHECKING:
    import pandas as pd


class DataRole(Enum):
    """Classification of data products by their role in the pipeline."""

    SOURCE = "source"  # Raw external data
    INTERMEDIATE = "intermediate"  # Processed but not final
    FEATURE = "feature"  # Feature-engineered data
    TARGET = "target"  # Target variable data
    OUTPUT = "output"  # Final model outputs
    LEGACY = "legacy"  # Deprecated/obsolete


@dataclass(frozen=True)
class SchemaInfo:
    """Metadata about a registered schema."""

    name: str
    schema: type[pa.DataFrameModel]
    version: str
    role: DataRole
    description: str


class SchemaRegistry:
    """
    Centralized registry for all data schemas.

    Provides version tracking and schema discovery.
    """

    _version = "1.0.0"

    _schemas: ClassVar[dict[str, SchemaInfo]] = {
        "counter_location": SchemaInfo(
            name="counter_location",
            schema=CounterLocationSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="Permanent bicycle counting station locations",
        ),
        "counter_measurement": SchemaInfo(
            name="counter_measurement",
            schema=CounterMeasurementSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="Daily counts from permanent stations",
        ),
        "campaign_metadata": SchemaInfo(
            name="campaign_metadata",
            schema=CampaignMetadataSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="STADTRADELN campaign dates per municipality",
        ),
        "demographics": SchemaInfo(
            name="demographics",
            schema=DemographicsSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="STADTRADELN participation statistics",
        ),
        "traffic_volume": SchemaInfo(
            name="traffic_volume",
            schema=TrafficVolumeSchema,
            version="1.1.0",
            role=DataRole.FEATURE,
            description="Bicycle traffic volumes per OSM edge (English column names)",
        ),
        "traffic_volume_german": SchemaInfo(
            name="traffic_volume_german",
            schema=TrafficVolumeGermanSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="Bicycle traffic volumes (German STADTRADELN column names)",
        ),
        "traffic_volume_raw": SchemaInfo(
            name="traffic_volume_raw",
            schema=TrafficVolumeRawSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="Raw traffic volumes before infrastructure mapping",
        ),
        "osm_infrastructure": SchemaInfo(
            name="osm_infrastructure",
            schema=OSMInfrastructureSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="OSM bicycle infrastructure classification",
        ),
        "municipality": SchemaInfo(
            name="municipality",
            schema=MunicipalitySchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="Municipality boundaries and population",
        ),
        "regiostar": SchemaInfo(
            name="regiostar",
            schema=RegioStarSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="RegioStaR regional classification",
        ),
        "city_centroid": SchemaInfo(
            name="city_centroid",
            schema=CityCentroidSchema,
            version="1.0.0",
            role=DataRole.SOURCE,
            description="City/town center points",
        ),
    }

    @classmethod
    def registry_version(cls) -> str:
        """Get the registry version."""
        return cls._version

    @classmethod
    def get(cls, name: str) -> type[pa.DataFrameModel]:
        """
        Get a schema by name.

        Args:
            name: Schema identifier.

        Returns:
            The Pandera DataFrameModel class.

        Raises:
            KeyError: If schema not found.
        """
        if name not in cls._schemas:
            available = ", ".join(cls._schemas.keys())
            msg = f"Unknown schema '{name}'. Available: {available}"
            raise KeyError(msg)
        return cls._schemas[name].schema

    @classmethod
    def get_info(cls, name: str) -> SchemaInfo:
        """
        Get full schema info by name.

        Args:
            name: Schema identifier.

        Returns:
            SchemaInfo with metadata.
        """
        if name not in cls._schemas:
            available = ", ".join(cls._schemas.keys())
            msg = f"Unknown schema '{name}'. Available: {available}"
            raise KeyError(msg)
        return cls._schemas[name]

    @classmethod
    def list_schemas(cls) -> list[str]:
        """List all registered schema names."""
        return list(cls._schemas.keys())

    @classmethod
    def list_by_role(cls, role: DataRole) -> list[str]:
        """List schemas filtered by their data role."""
        return [name for name, info in cls._schemas.items() if info.role == role]

    @classmethod
    def validate(cls, df: "pd.DataFrame", schema_name: str) -> "pd.DataFrame":
        """
        Validate a DataFrame against a registered schema.

        Args:
            df: DataFrame to validate.
            schema_name: Name of schema to validate against.

        Returns:
            Validated DataFrame.

        Raises:
            pandera.errors.SchemaError: If validation fails.
        """
        schema = cls.get(schema_name)
        return schema.validate(df)
