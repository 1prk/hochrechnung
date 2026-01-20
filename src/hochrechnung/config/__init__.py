"""
Configuration management with typed Pydantic models.

Provides explicit year/region parameterization and
environment-aware configuration loading.
"""

from hochrechnung.config.loader import load_config
from hochrechnung.config.settings import (
    DataPathsConfig,
    FeatureConfig,
    MLflowConfig,
    PipelineConfig,
    RegionConfig,
    TemporalConfig,
    TrainingConfig,
)

__all__ = [
    "DataPathsConfig",
    "FeatureConfig",
    "MLflowConfig",
    "PipelineConfig",
    "RegionConfig",
    "TemporalConfig",
    "TrainingConfig",
    "load_config",
]
