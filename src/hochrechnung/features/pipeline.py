"""
Feature engineering pipeline.

Orchestrates feature computation from raw data to model-ready features.
"""

from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.features.definitions import DerivedFeature, get_registry
from hochrechnung.features.infrastructure import apply_infrastructure_mapping
from hochrechnung.normalization.columns import normalize_columns
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class FeaturePipeline:
    """
    Pipeline for computing features from raw data.

    Handles column normalization, infrastructure mapping,
    and derived feature calculation.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize feature pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.registry = get_registry()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run full feature processing pipeline.

        Args:
            df: Raw input DataFrame.

        Returns:
            DataFrame with all features computed.
        """
        log.info("Starting feature pipeline", rows=len(df))

        # 1. Normalize column names
        df = normalize_columns(df)

        # 2. Apply infrastructure mapping
        if "bicycle_infrastructure" in df.columns:
            df = apply_infrastructure_mapping(
                df,
                column="bicycle_infrastructure",
                output_column="infra_category",
                config=self.config.preprocessing,
            )

        # 3. Compute derived features
        df = self.compute_derived_features(df)

        log.info("Feature pipeline complete", columns=list(df.columns))
        return df

    def compute_derived_features(
        self,
        df: pd.DataFrame,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Compute derived features.

        Args:
            df: DataFrame with base columns.
            features: Optional list of feature names (default: all).

        Returns:
            DataFrame with derived features added.
        """
        if features is None:
            features = self.registry.list_features()

        df = df.copy()

        for name in features:
            try:
                feature = self.registry.get(name)
                df = self._compute_feature(df, feature)
            except KeyError:
                log.warning("Unknown feature, skipping", name=name)
            except Exception as e:
                log.warning("Error computing feature", name=name, error=str(e))

        return df

    def _compute_feature(
        self,
        df: pd.DataFrame,
        feature: DerivedFeature,
    ) -> pd.DataFrame:
        """
        Compute a single derived feature.

        Args:
            df: Input DataFrame.
            feature: Feature definition.

        Returns:
            DataFrame with feature column added.
        """
        # Check dependencies
        missing = [col for col in feature.dependencies if col not in df.columns]
        if missing:
            log.debug(
                "Missing dependencies for feature",
                feature=feature.name,
                missing=missing,
            )
            return df

        try:
            df[feature.name] = feature.formula(df)

            # Apply fill value if specified
            if feature.fill_value is not None:
                df[feature.name] = df[feature.name].fillna(feature.fill_value)

            non_null = df[feature.name].notna().sum()
            log.debug(
                "Computed feature",
                name=feature.name,
                non_null=int(non_null),
                total=len(df),
            )

        except Exception as e:
            log.warning(
                "Error computing feature",
                feature=feature.name,
                error=str(e),
            )

        return df

    def get_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract model-ready feature matrix.

        Args:
            df: DataFrame with all columns.

        Returns:
            DataFrame with only model feature columns.
        """
        model_features = self.config.features.model_features

        # Check for missing features
        missing = [f for f in model_features if f not in df.columns]
        if missing:
            log.warning("Missing model features", missing=missing)

        # Select available features
        available = [f for f in model_features if f in df.columns]
        return df[available]

    def validate_features(self, df: pd.DataFrame) -> dict[str, bool]:
        """
        Validate that required features are present and non-null.

        Args:
            df: DataFrame to validate.

        Returns:
            Dictionary of feature_name -> is_valid.
        """
        model_features = self.config.features.model_features
        result = {}

        for feature in model_features:
            if feature not in df.columns:
                result[feature] = False
            else:
                # Feature is valid if at least some values are non-null
                result[feature] = df[feature].notna().any()

        return result


def compute_features(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    Convenience function to compute features.

    Args:
        df: Input DataFrame.
        config: Pipeline configuration.

    Returns:
        DataFrame with features computed.
    """
    pipeline = FeaturePipeline(config)
    return pipeline.process(df)
