"""
Declarative feature definitions.

Features are defined using a registry pattern for explicit dependencies
and testable formulas.
"""

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DerivedFeature:
    """
    Definition of a derived feature.

    Attributes:
        name: Feature name (column name in output).
        formula: Function that computes the feature from a DataFrame.
        dependencies: List of columns required for computation.
        description: Human-readable description.
        fill_value: Value to use for missing data (default: None = keep NaN).
    """

    name: str
    formula: Callable[[pd.DataFrame], pd.Series]
    dependencies: tuple[str, ...]
    description: str = ""
    fill_value: float | None = None


# Core derived features based on STADTRADELN metrics
DERIVED_FEATURES: list[DerivedFeature] = [
    DerivedFeature(
        name="participation_rate",
        formula=lambda df: df["n_users"] / df["population"],
        dependencies=("n_users", "population"),
        description="Relative STADTRADELN participation rate (TN_SR_relativ)",
    ),
    DerivedFeature(
        name="route_intensity",
        formula=lambda df: (df["n_users"] * df["stadtradeln_volume"])
        / df["population"],
        dependencies=("n_users", "stadtradeln_volume", "population"),
        description="Population-weighted route intensity (Streckengewicht_SR)",
    ),
    DerivedFeature(
        name="volume_per_trip",
        formula=lambda df: (df["stadtradeln_volume"] / df["n_trips"]) * df["n_users"],
        dependencies=("stadtradeln_volume", "n_trips", "n_users"),
        description="User-adjusted volume per trip (Erh_SR_relativ)",
    ),
    DerivedFeature(
        name="trips_per_user",
        formula=lambda df: df["n_trips"] / df["n_users"],
        dependencies=("n_trips", "n_users"),
        description="Average trips per participant",
    ),
    DerivedFeature(
        name="km_per_user",
        formula=lambda df: df["total_km"] / df["n_users"],
        dependencies=("total_km", "n_users"),
        description="Average kilometers per participant",
    ),
    DerivedFeature(
        name="volume_density",
        formula=lambda df: df["stadtradeln_volume"] / df["population"],
        dependencies=("stadtradeln_volume", "population"),
        description="Traffic volume normalized by population",
    ),
]


@dataclass
class FeatureRegistry:
    """
    Registry of all available features.

    Provides lookup and dependency checking for features.
    """

    features: dict[str, DerivedFeature] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize with default features."""
        for feature in DERIVED_FEATURES:
            self.register(feature)

    def register(self, feature: DerivedFeature) -> None:
        """
        Register a feature.

        Args:
            feature: Feature definition to register.
        """
        if feature.name in self.features:
            log.warning("Overwriting existing feature", name=feature.name)
        self.features[feature.name] = feature

    def get(self, name: str) -> DerivedFeature:
        """
        Get a feature by name.

        Args:
            name: Feature name.

        Returns:
            Feature definition.

        Raises:
            KeyError: If feature not found.
        """
        if name not in self.features:
            available = ", ".join(self.features.keys())
            msg = f"Unknown feature '{name}'. Available: {available}"
            raise KeyError(msg)
        return self.features[name]

    def list_features(self) -> list[str]:
        """List all registered feature names."""
        return list(self.features.keys())

    def get_dependencies(self, names: list[str]) -> set[str]:
        """
        Get all dependencies for a list of features.

        Args:
            names: List of feature names.

        Returns:
            Set of all required columns.
        """
        deps: set[str] = set()
        for name in names:
            feature = self.get(name)
            deps.update(feature.dependencies)
        return deps

    def check_dependencies(self, df: pd.DataFrame, names: list[str]) -> list[str]:
        """
        Check which dependencies are missing from DataFrame.

        Args:
            df: DataFrame to check.
            names: List of feature names to compute.

        Returns:
            List of missing column names.
        """
        required = self.get_dependencies(names)
        return [col for col in required if col not in df.columns]


# Global registry instance
_registry: FeatureRegistry | None = None


def get_registry() -> FeatureRegistry:
    """Get the global feature registry."""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry


def register_feature(feature: DerivedFeature) -> None:
    """Register a feature in the global registry."""
    get_registry().register(feature)


def get_feature(name: str) -> DerivedFeature:
    """Get a feature from the global registry."""
    return get_registry().get(name)
