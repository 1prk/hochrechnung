"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def sample_traffic_data() -> pd.DataFrame:
    """Create sample traffic volume data for testing."""
    return pd.DataFrame(
        {
            "edge_id": [1, 2, 3, 4, 5],
            "base_id": [100, 101, 102, 103, 104],
            "count": [50, 120, 30, 200, 80],
            "bicycle_infrastructure": [
                "bicycle_lane_left",
                "bicycle_way_both",
                "mixed_way",
                "mit_road",
                "no",
            ],
        }
    )


@pytest.fixture
def sample_counter_locations() -> pd.DataFrame:
    """Create sample counter location data for testing."""
    return pd.DataFrame(
        {
            "id": ["001", "002", "003"],
            "name": ["Station A", "Station B", "Station C"],
            "latitude": [50.1, 50.2, 50.3],
            "longitude": [8.6, 8.7, 8.8],
            "ars": ["060610000000", "060620000000", "060630000000"],
        }
    )


@pytest.fixture
def sample_municipality_data() -> pd.DataFrame:
    """Create sample municipality data for testing."""
    return pd.DataFrame(
        {
            "ars": ["060610000000", "060620000000", "060630000000"],
            "name": ["Frankfurt", "Wiesbaden", "Darmstadt"],
            "population": [750000, 280000, 160000],
            "regiostar5": [1, 2, 3],
        }
    )


@pytest.fixture
def base_config() -> dict[str, Any]:
    """Create a minimal configuration dictionary for testing."""
    return {
        "region": {
            "code": "06",
            "name": "Hessen",
            "bbox": [7.77, 49.39, 10.24, 51.66],
        },
        "temporal": {
            "year": 2024,
            "campaign_start": "2024-05-01",
            "campaign_end": "2024-09-30",
            "counter_period_start": "2024-05-01",
            "counter_period_end": "2024-09-30",
        },
        "data_paths": {
            "data_root": "./data",
            "counter_locations": "counter-locations/DZS_ecovisio_2024.csv",
            "counter_measurements": "counts/DZS_counts_ecovisio_2024.csv",
            "traffic_volumes": "trafficvolumes/SR24_Hessen_VM_assessed.fgb",
        },
    }
