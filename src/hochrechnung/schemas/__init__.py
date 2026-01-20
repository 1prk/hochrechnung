"""
Schema definitions using Pandera for data validation.

All data contracts are defined here to ensure explicit,
validated data structures throughout the pipeline.
"""

from hochrechnung.schemas.campaign import CampaignMetadataSchema, DemographicsSchema
from hochrechnung.schemas.counter import CounterLocationSchema, CounterMeasurementSchema
from hochrechnung.schemas.infrastructure import OSMInfrastructureSchema
from hochrechnung.schemas.output import PredictionOutputSchema, TrainingOutputSchema
from hochrechnung.schemas.registry import SchemaRegistry
from hochrechnung.schemas.structural import MunicipalitySchema, RegioStarSchema
from hochrechnung.schemas.traffic import (
    COLUMN_EQUIVALENTS,
    TrafficVolumeGermanSchema,
    TrafficVolumeSchema,
)

__all__ = [
    "COLUMN_EQUIVALENTS",
    "CampaignMetadataSchema",
    "CounterLocationSchema",
    "CounterMeasurementSchema",
    "DemographicsSchema",
    "MunicipalitySchema",
    "OSMInfrastructureSchema",
    "PredictionOutputSchema",
    "RegioStarSchema",
    "SchemaRegistry",
    "TrafficVolumeGermanSchema",
    "TrafficVolumeSchema",
    "TrainingOutputSchema",
]
