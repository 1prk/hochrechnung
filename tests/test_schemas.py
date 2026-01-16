"""Tests for Pandera schema definitions."""

import pandas as pd
import pandera as pa
import pytest

from hochrechnung.schemas import (
    CounterLocationSchema,
    CounterMeasurementSchema,
    TrafficVolumeSchema,
)
from hochrechnung.schemas.registry import DataRole, SchemaRegistry
from hochrechnung.schemas.traffic import VALID_INFRA_CATEGORIES


class TestCounterLocationSchema:
    """Tests for CounterLocationSchema."""

    def test_valid_data(self) -> None:
        """Test that valid data passes validation."""
        df = pd.DataFrame(
            {
                "id": ["001", "002"],
                "name": ["Station A", "Station B"],
                "latitude": [50.1, 50.2],
                "longitude": [8.6, 8.7],
            }
        )
        result = CounterLocationSchema.validate(df)
        assert len(result) == 2

    def test_invalid_latitude(self) -> None:
        """Test that invalid latitude fails validation."""
        df = pd.DataFrame(
            {
                "id": ["001"],
                "name": ["Station A"],
                "latitude": [100.0],  # Invalid: > 90
                "longitude": [8.6],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            CounterLocationSchema.validate(df)

    def test_invalid_longitude(self) -> None:
        """Test that invalid longitude fails validation."""
        df = pd.DataFrame(
            {
                "id": ["001"],
                "name": ["Station A"],
                "latitude": [50.1],
                "longitude": [200.0],  # Invalid: > 180
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            CounterLocationSchema.validate(df)

    def test_nullable_ars(self) -> None:
        """Test that ARS column can be null."""
        df = pd.DataFrame(
            {
                "id": ["001"],
                "name": ["Station A"],
                "latitude": [50.1],
                "longitude": [8.6],
                "ars": [None],
            }
        )
        result = CounterLocationSchema.validate(df)
        assert len(result) == 1


class TestTrafficVolumeSchema:
    """Tests for TrafficVolumeSchema."""

    def test_valid_data(self) -> None:
        """Test that valid data passes validation."""
        df = pd.DataFrame(
            {
                "edge_id": [1, 2, 3],
                "base_id": [100, 101, 102],
                "count": [50, 120, 30],
                "bicycle_infrastructure": ["bicycle_lane", "bicycle_way", "no"],
            }
        )
        result = TrafficVolumeSchema.validate(df)
        assert len(result) == 3

    def test_invalid_infrastructure(self) -> None:
        """Test that invalid infrastructure category fails."""
        df = pd.DataFrame(
            {
                "edge_id": [1],
                "base_id": [100],
                "count": [50],
                "bicycle_infrastructure": ["invalid_category"],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            TrafficVolumeSchema.validate(df)

    def test_negative_count(self) -> None:
        """Test that negative count fails validation."""
        df = pd.DataFrame(
            {
                "edge_id": [1],
                "base_id": [100],
                "count": [-10],  # Invalid: negative
                "bicycle_infrastructure": ["bicycle_lane"],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            TrafficVolumeSchema.validate(df)

    def test_duplicate_edge_id(self) -> None:
        """Test that duplicate edge_id fails validation."""
        df = pd.DataFrame(
            {
                "edge_id": [1, 1],  # Duplicate
                "base_id": [100, 101],
                "count": [50, 60],
                "bicycle_infrastructure": ["bicycle_lane", "no"],
            }
        )
        with pytest.raises(pa.errors.SchemaError):
            TrafficVolumeSchema.validate(df)

    def test_all_valid_infrastructure_categories(self) -> None:
        """Test all valid infrastructure categories pass."""
        df = pd.DataFrame(
            {
                "edge_id": list(range(len(VALID_INFRA_CATEGORIES))),
                "base_id": [100] * len(VALID_INFRA_CATEGORIES),
                "count": [50] * len(VALID_INFRA_CATEGORIES),
                "bicycle_infrastructure": VALID_INFRA_CATEGORIES,
            }
        )
        result = TrafficVolumeSchema.validate(df)
        assert len(result) == len(VALID_INFRA_CATEGORIES)


class TestSchemaRegistry:
    """Tests for SchemaRegistry."""

    def test_registry_version(self) -> None:
        """Test registry has a version."""
        version = SchemaRegistry.registry_version()
        assert version is not None
        assert isinstance(version, str)

    def test_list_schemas(self) -> None:
        """Test listing all schemas."""
        schemas = SchemaRegistry.list_schemas()
        assert len(schemas) > 0
        assert "traffic_volume" in schemas
        assert "counter_location" in schemas

    def test_get_schema(self) -> None:
        """Test getting a schema by name."""
        schema = SchemaRegistry.get("traffic_volume")
        assert schema == TrafficVolumeSchema

    def test_get_unknown_schema(self) -> None:
        """Test getting unknown schema raises KeyError."""
        with pytest.raises(KeyError):
            SchemaRegistry.get("nonexistent_schema")

    def test_get_info(self) -> None:
        """Test getting schema info."""
        info = SchemaRegistry.get_info("traffic_volume")
        assert info.name == "traffic_volume"
        assert info.role == DataRole.FEATURE
        assert "traffic" in info.description.lower()

    def test_list_by_role(self) -> None:
        """Test filtering schemas by role."""
        source_schemas = SchemaRegistry.list_by_role(DataRole.SOURCE)
        assert len(source_schemas) > 0
        assert "counter_location" in source_schemas

    def test_validate_method(self) -> None:
        """Test validation through registry."""
        df = pd.DataFrame(
            {
                "id": ["001"],
                "name": ["Test"],
                "latitude": [50.1],
                "longitude": [8.6],
            }
        )
        result = SchemaRegistry.validate(df, "counter_location")
        assert len(result) == 1
