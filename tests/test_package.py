"""Basic package tests to verify installation."""

import pytest


def test_package_imports() -> None:
    """Verify the main package can be imported."""
    import hochrechnung

    assert hochrechnung.__version__


def test_config_module_imports() -> None:
    """Verify config module structure is correct."""
    from hochrechnung.config import (
        DataPathsConfig,
        FeatureConfig,
        MLflowConfig,
        PipelineConfig,
        RegionConfig,
        TemporalConfig,
        TrainingConfig,
        load_config,
    )

    # Verify all exports are available
    assert PipelineConfig is not None
    assert RegionConfig is not None
    assert TemporalConfig is not None
    assert DataPathsConfig is not None
    assert FeatureConfig is not None
    assert TrainingConfig is not None
    assert MLflowConfig is not None
    assert load_config is not None


def test_schemas_module_imports() -> None:
    """Verify schemas module structure is correct."""
    from hochrechnung.schemas import (
        CampaignMetadataSchema,
        CounterLocationSchema,
        CounterMeasurementSchema,
        DemographicsSchema,
        MunicipalitySchema,
        OSMInfrastructureSchema,
        RegioStarSchema,
        SchemaRegistry,
        TrafficVolumeSchema,
    )

    # Verify all exports are available
    assert CampaignMetadataSchema is not None
    assert CounterLocationSchema is not None
    assert CounterMeasurementSchema is not None
    assert DemographicsSchema is not None
    assert MunicipalitySchema is not None
    assert OSMInfrastructureSchema is not None
    assert RegioStarSchema is not None
    assert SchemaRegistry is not None
    assert TrafficVolumeSchema is not None
