"""
Inference pipeline for generating predictions.

Handles prediction generation with validated output schema.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.features.pipeline import FeaturePipeline
from hochrechnung.ingestion.campaign import load_demographics
from hochrechnung.ingestion.osm import load_osm_infrastructure
from hochrechnung.ingestion.structural import (
    load_city_centroids,
    load_municipalities,
    load_regiostar,
)
from hochrechnung.ingestion.traffic import load_traffic_volumes
from hochrechnung.normalization.spatial import (
    calculate_distances_to_centroids,
    spatial_join_municipalities,
)
from hochrechnung.utils.cache import CacheManager
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Container for prediction results.

    Attributes:
        predictions: DataFrame with predictions.
        model_name: Name of model used.
        feature_names: Features used for prediction.
        n_predictions: Number of predictions made.
        n_failures: Number of failed predictions.
    """

    predictions: pd.DataFrame
    model_name: str
    feature_names: list[str]
    n_predictions: int
    n_failures: int


def predict_traffic_volumes(
    model: Any,
    features: pd.DataFrame,
    original_data: pd.DataFrame | None = None,
    *,
    batch_size: int | None = None,
) -> PredictionResult:
    """
    Generate predictions for traffic volumes.

    Args:
        model: Fitted model (sklearn pipeline).
        features: Feature DataFrame.
        original_data: Original data to merge with predictions.
        batch_size: Optional batch size for large datasets.

    Returns:
        PredictionResult with predictions.
    """
    log.info("Generating predictions", n_samples=len(features))

    # Handle missing values
    n_missing = features.isna().any(axis=1).sum()
    if n_missing > 0:
        log.warning("Features contain missing values", n_missing=int(n_missing))

    # Generate predictions
    if batch_size and len(features) > batch_size:
        predictions = _predict_in_batches(model, features, batch_size)
    else:
        try:
            predictions = model.predict(features)
        except Exception as e:
            log.error("Prediction failed", error=str(e))
            predictions = np.full(len(features), np.nan)

    # Build result DataFrame
    result_df = pd.DataFrame({"predicted_dtv": predictions})

    # Add original data if provided
    if original_data is not None:
        # Align indices
        result_df.index = features.index
        for col in original_data.columns:
            if col in features.index.names or col == features.index.name:
                continue
            result_df[col] = original_data[col].values

    n_failures = int(np.isnan(predictions).sum())

    result = PredictionResult(
        predictions=result_df,
        model_name=_get_model_name(model),
        feature_names=list(features.columns),
        n_predictions=len(predictions) - n_failures,
        n_failures=n_failures,
    )

    log.info(
        "Predictions complete",
        n_predictions=result.n_predictions,
        n_failures=result.n_failures,
    )

    return result


def _predict_in_batches(
    model: Any,
    features: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Generate predictions in batches for memory efficiency."""
    predictions = []

    for i in range(0, len(features), batch_size):
        batch = features.iloc[i : i + batch_size]
        try:
            batch_pred = model.predict(batch)
            predictions.append(batch_pred)
        except Exception as e:
            log.warning(f"Batch {i} failed: {e}")
            predictions.append(np.full(len(batch), np.nan))

        log.debug(f"Processed batch {i // batch_size + 1}")

    return np.concatenate(predictions)


def _get_model_name(model: Any) -> str:
    """Extract model name from pipeline."""
    if hasattr(model, "regressor"):
        # TransformedTargetRegressor
        inner = model.regressor
        if hasattr(inner, "named_steps"):
            return type(inner.named_steps.get("model", inner)).__name__
        return type(inner).__name__
    if hasattr(model, "named_steps"):
        return type(model.named_steps.get("model", model)).__name__
    return type(model).__name__


def save_predictions(
    result: PredictionResult,
    output_path: Path,
    format: str = "csv",
) -> Path:
    """
    Save predictions to file.

    Args:
        result: Prediction result.
        output_path: Output file path.
        format: Output format ('csv', 'parquet', 'fgb').

    Returns:
        Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.predictions

    if format == "csv":
        output_file = output_path.with_suffix(".csv")
        df.to_csv(output_file, index=False)
    elif format == "parquet":
        output_file = output_path.with_suffix(".parquet")
        df.to_parquet(output_file)
    elif format == "fgb":
        import geopandas as gpd

        output_file = output_path.with_suffix(".fgb")
        if "geometry" in df.columns or hasattr(df, "geometry"):
            gdf = gpd.GeoDataFrame(df)
            gdf.to_file(output_file, driver="FlatGeobuf")
        else:
            # Fall back to CSV if no geometry
            output_file = output_path.with_suffix(".csv")
            df.to_csv(output_file, index=False)
    else:
        output_file = output_path.with_suffix(".csv")
        df.to_csv(output_file, index=False)

    log.info("Saved predictions", path=str(output_file), n_rows=len(df))
    return output_file


def load_model(model_path: Path | str) -> Any:
    """
    Load a saved model.

    Args:
        model_path: Path to model file or MLflow URI.

    Returns:
        Loaded model.
    """
    model_path_str = str(model_path)

    # Check if it's an MLflow URI
    if model_path_str.startswith("runs:/") or model_path_str.startswith("models:/"):
        import mlflow

        return mlflow.sklearn.load_model(model_path_str)

    # Load from file
    import joblib

    return joblib.load(model_path)


@dataclass
class PredictionPipelineResult:
    """
    Result of prediction pipeline execution.

    Attributes:
        predictions: GeoDataFrame with predictions.
        n_edges: Total number of edges processed.
        n_predictions: Number of successful predictions.
        n_failures: Number of failed predictions.
        model_name: Name of model used.
        output_path: Path where predictions were saved (if any).
    """

    predictions: "gpd.GeoDataFrame"
    n_edges: int
    n_predictions: int
    n_failures: int
    model_name: str
    output_path: Path | None = None


def _get_prediction_cache(config: PipelineConfig) -> CacheManager:
    """Get cache manager for prediction operations."""
    cache_dir = config.output.cache_dir / "prediction"
    return CacheManager(cache_dir)


class PredictionPipeline:
    """
    Pipeline for generating predictions on all traffic edges.

    Loads traffic volumes, joins structural data, computes features,
    and applies a trained model to generate DTV predictions for all edges.

    Performance optimizations:
    - Caching of expensive intermediate results (feature data)
    - Cache invalidation based on source file changes
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        use_cache: bool = True,
    ) -> None:
        """
        Initialize prediction pipeline.

        Args:
            config: Pipeline configuration.
            use_cache: Whether to cache intermediate feature data.
        """
        self.config = config
        self.feature_pipeline = FeaturePipeline(config)
        self.use_cache = use_cache
        self._cache = _get_prediction_cache(config) if use_cache else None

    def run(
        self,
        model: Any,
        output_path: Path | None = None,
        output_format: str = "fgb",
    ) -> PredictionPipelineResult:
        """
        Run prediction pipeline on all traffic edges.

        Args:
            model: Trained model (sklearn pipeline).
            output_path: Optional path to save predictions.
            output_format: Output format ('fgb', 'csv', 'parquet').

        Returns:
            PredictionPipelineResult with predictions and statistics.
        """
        import geopandas as gpd

        log.info(
            "Starting prediction pipeline", year=self.config.year, cache=self.use_cache
        )

        # Try to load cached feature data
        cached_data = self._load_cached_features()

        if cached_data is not None:
            log.info("Using cached feature data")
            feature_df, geometry, edge_ids, original_crs, n_edges, traffic_gdf = (
                cached_data
            )
            features = self._prepare_features(feature_df)
        else:
            # Build feature data from scratch
            # Step 1: Load traffic volumes (all edges)
            log.info("Step 1: Loading traffic volumes")
            traffic_gdf = load_traffic_volumes(self.config, validate=False)
            n_edges = len(traffic_gdf)
            log.info("Loaded traffic edges", n_edges=n_edges)

            # Store geometry and identifiers for output
            geometry = traffic_gdf.geometry.copy()
            original_crs = traffic_gdf.crs

            # Capture edge identifiers (edge_id may not always be present)
            edge_id_cols = ["base_id"]
            if "edge_id" in traffic_gdf.columns:
                edge_id_cols.append("edge_id")
            edge_ids = traffic_gdf[edge_id_cols].copy()

            # Step 2: Join OSM infrastructure if not present
            if "bicycle_infrastructure" not in traffic_gdf.columns:
                log.info("Step 2: Loading and joining OSM infrastructure")
                osm_df = load_osm_infrastructure(self.config)
                traffic_gdf = self._join_osm_infrastructure(traffic_gdf, osm_df)
            else:
                log.info("Step 2: OSM infrastructure already present")

            # Step 3: Join structural data
            log.info("Step 3: Adding structural data")
            traffic_gdf = self._add_structural_data(traffic_gdf)

            # Step 4: Compute derived features
            log.info("Step 4: Computing features")
            feature_df = self.feature_pipeline.process(traffic_gdf)

            # Step 5: Prepare features for model
            log.info("Step 5: Preparing features for model")
            features = self._prepare_features(feature_df)

            # Cache the feature data
            self._cache_features(
                feature_df, geometry, edge_ids, original_crs, n_edges, traffic_gdf
            )

        # Step 6: Generate predictions
        log.info("Step 6: Generating predictions")
        pred_result = predict_traffic_volumes(model, features)

        # Step 7: Build output GeoDataFrame
        log.info("Step 7: Building output")
        output_data: dict[str, Any] = {
            "base_id": edge_ids["base_id"].values,
            "predicted_dtv": pred_result.predictions["predicted_dtv"].values,
            "geometry": geometry.values,
        }
        if "edge_id" in edge_ids.columns:
            output_data["edge_id"] = edge_ids["edge_id"].values

        output_gdf = gpd.GeoDataFrame(output_data, crs=original_crs)

        # Add original count for reference
        if "count" in traffic_gdf.columns:
            output_gdf["stadtradeln_count"] = traffic_gdf["count"].values
        if "bicycle_infrastructure" in traffic_gdf.columns:
            output_gdf["bicycle_infrastructure"] = traffic_gdf[
                "bicycle_infrastructure"
            ].values

        # Save if output path provided
        saved_path = None
        if output_path is not None:
            saved_path = save_predictions_geo(output_gdf, output_path, output_format)

        log.info(
            "Prediction pipeline complete",
            n_edges=n_edges,
            n_predictions=pred_result.n_predictions,
            n_failures=pred_result.n_failures,
        )

        return PredictionPipelineResult(
            predictions=output_gdf,
            n_edges=n_edges,
            n_predictions=pred_result.n_predictions,
            n_failures=pred_result.n_failures,
            model_name=pred_result.model_name,
            output_path=saved_path,
        )

    def _join_osm_infrastructure(
        self, traffic_gdf: "gpd.GeoDataFrame", osm_df: pd.DataFrame
    ) -> "gpd.GeoDataFrame":
        """Join OSM infrastructure attributes with traffic volumes."""
        import geopandas as gpd

        if "base_id" not in traffic_gdf.columns:
            log.warning("Traffic data missing base_id, cannot join OSM infrastructure")
            return traffic_gdf

        if "osm_id" not in osm_df.columns:
            log.warning("OSM data missing osm_id, cannot join infrastructure")
            return traffic_gdf

        # Convert base_id to same type for matching
        traffic_gdf = traffic_gdf.copy()
        traffic_gdf["base_id"] = (
            traffic_gdf["base_id"].fillna(-1).astype(int).astype(str)
        )
        traffic_gdf.loc[traffic_gdf["base_id"] == "-1", "base_id"] = None

        osm_df = osm_df.copy()
        osm_df["osm_id"] = osm_df["osm_id"].astype(str)

        # Merge on base_id = osm_id
        original_crs = traffic_gdf.crs
        joined = traffic_gdf.merge(
            osm_df[["osm_id", "bicycle_infrastructure"]],
            left_on="base_id",
            right_on="osm_id",
            how="left",
        )

        if "osm_id" in joined.columns:
            joined = joined.drop(columns=["osm_id"])

        if not isinstance(joined, gpd.GeoDataFrame):
            joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=original_crs)

        log.info(
            "Joined OSM infrastructure",
            matched=joined["bicycle_infrastructure"].notna().sum(),
            total=len(joined),
        )

        return joined

    def _add_structural_data(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """Add municipality, RegioStaR, demographics, and distance features."""
        # Load structural data
        municipalities = load_municipalities(self.config, validate=False)
        regiostar = load_regiostar(self.config, validate=False)
        centroids = load_city_centroids(self.config, validate=False)

        # Spatial join with municipalities
        gdf = spatial_join_municipalities(gdf, municipalities, how="left")

        # Join RegioStaR on ARS
        if "ars" in gdf.columns and "ars" in regiostar.columns:
            gdf["ars_12"] = gdf["ars"].astype(str).str[:12]
            regiostar["ars_12"] = regiostar["ars"].astype(str).str[:12]

            gdf = gdf.merge(
                regiostar[["ars_12", "regiostar5", "regiostar7"]],
                on="ars_12",
                how="left",
            )

        # Calculate distances to city centroids
        gdf = calculate_distances_to_centroids(gdf, centroids)

        # Load demographics for participation metrics
        try:
            demographics = load_demographics(self.config, validate=False)
            if "ars" in gdf.columns and "ars" in demographics.columns:
                demo_cols = ["ars", "n_users", "n_trips", "total_km", "population"]
                demo_cols = [c for c in demo_cols if c in demographics.columns]
                demographics["ars_12"] = demographics["ars"].astype(str).str[:12]

                gdf = gdf.merge(
                    demographics[["ars_12"] + [c for c in demo_cols if c != "ars"]],
                    on="ars_12",
                    how="left",
                    suffixes=("", "_demo"),
                )
        except FileNotFoundError:
            log.warning("Demographics data not found, skipping")

        return gdf

    def _get_cache_key(self) -> str:
        """Generate cache key for feature data."""
        return f"prediction_features_{self.config.year}"

    def _get_source_files(self) -> list[Path]:
        """Get list of source files that affect feature data."""
        data_root = self.config.data_paths.data_root
        return [
            data_root / self.config.data_paths.traffic_volumes,
            data_root / self.config.data_paths.municipalities,
            data_root / self.config.data_paths.regiostar,
            data_root / self.config.data_paths.city_centroids,
        ]

    def _load_cached_features(
        self,
    ) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any, int, Any] | None:
        """
        Load cached feature data if available and valid.

        Returns:
            Tuple of (feature_df, geometry, edge_ids, crs, n_edges, traffic_gdf)
            or None if cache miss.
        """
        if self._cache is None:
            return None

        cache_key = self._get_cache_key()
        source_files = self._get_source_files()
        config_hash = self._cache.compute_config_hash(self.config.region)

        cached = self._cache.get(
            cache_key,
            source_files=source_files,
            config_hash=config_hash,
        )

        if cached is not None:
            log.info("Loaded prediction features from cache")
            return cached

        return None

    def _cache_features(
        self,
        feature_df: pd.DataFrame,
        geometry: Any,
        edge_ids: pd.DataFrame,
        crs: Any,
        n_edges: int,
        traffic_gdf: Any,
    ) -> None:
        """
        Cache feature data for future prediction runs.

        Args:
            feature_df: Computed feature DataFrame.
            geometry: Geometry series.
            edge_ids: Edge identifier DataFrame.
            crs: Coordinate reference system.
            n_edges: Total number of edges.
            traffic_gdf: Traffic GeoDataFrame (for count/infra columns).
        """
        if self._cache is None:
            return

        cache_key = self._get_cache_key()
        source_files = self._get_source_files()
        config_hash = self._cache.compute_config_hash(self.config.region)

        # Cache as tuple
        cache_data = (feature_df, geometry, edge_ids, crs, n_edges, traffic_gdf)

        self._cache.set(
            cache_key,
            cache_data,
            source_files=source_files,
            config_hash=config_hash,
        )
        log.info("Cached prediction feature data")

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model prediction.

        Maps columns to the format expected by the trained model.
        Uses the same feature names that training data uses.
        """
        df = df.copy()

        # Column name mapping for prediction pipeline output to training data names
        # After normalization in feature pipeline:
        #   count -> stadtradeln_volume
        #   bicycle_infrastructure -> infra_category (via infrastructure mapping)
        # These need to match what the model was trained with
        column_aliases = {
            # Prediction pipeline name -> config/model name alternatives
            "stadtradeln_volume": ["count", "Erh_SR"],
            "infra_category": ["bicycle_infrastructure", "OSM_Radinfra"],
            "dist_to_center_m": ["dist_to_centroid_m", "HubDist"],
            "regiostar5": ["RegioStaR5"],
            "participation_rate": ["TN_SR_relativ"],
            "route_intensity": ["Streckengewicht_SR"],
        }

        # Get expected features from config or use defaults
        model_features = self.config.features.model_features
        if not model_features:
            # Default feature set matching the model training features
            model_features = [
                "infra_category",
                "participation_rate",
                "route_intensity",
                "regiostar5",
                "stadtradeln_volume",
                "dist_to_center_m",
            ]

        # Resolve feature names: find available columns that match requested features
        feature_mapping: dict[str, str] = {}  # feature_name -> actual_column_name
        for feature in model_features:
            if feature in df.columns:
                feature_mapping[feature] = feature
            else:
                # Check if this feature is an alias for an available column
                for actual_col, aliases in column_aliases.items():
                    if feature in aliases and actual_col in df.columns:
                        feature_mapping[feature] = actual_col
                        break
                # Also check reverse: if feature is the actual column and an alias is in model_features
                if feature not in feature_mapping:
                    for actual_col, aliases in column_aliases.items():
                        if feature == actual_col:
                            for alias in aliases:
                                if alias in df.columns:
                                    feature_mapping[feature] = alias
                                    break
                            if feature in feature_mapping:
                                break

        available = list(feature_mapping.keys())
        missing = [f for f in model_features if f not in feature_mapping]

        if missing:
            log.warning("Missing features for prediction", missing=missing)

        if not available:
            msg = "No features available for prediction"
            raise ValueError(msg)

        # Select features using resolved column names
        # Rename columns to match what model expects
        feature_df = df[[feature_mapping[f] for f in available]].copy()
        feature_df.columns = pd.Index(available)

        # Clean up data: replace inf with nan, handle extreme values
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        log.info("Prepared features", n_features=len(available), features=available)

        return feature_df


def save_predictions_geo(
    gdf: "gpd.GeoDataFrame",
    output_path: Path,
    format: str = "fgb",
) -> Path:
    """
    Save predictions GeoDataFrame to file.

    Args:
        gdf: GeoDataFrame with predictions.
        output_path: Output file path (without extension).
        format: Output format ('fgb', 'csv', 'parquet', 'gpkg').

    Returns:
        Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "fgb":
        output_file = output_path.with_suffix(".fgb")
        gdf.to_file(output_file, driver="FlatGeobuf")
    elif format == "gpkg":
        output_file = output_path.with_suffix(".gpkg")
        gdf.to_file(output_file, driver="GPKG")
    elif format == "parquet":
        output_file = output_path.with_suffix(".parquet")
        gdf.to_parquet(output_file)
    elif format == "csv":
        output_file = output_path.with_suffix(".csv")
        # For CSV, we need to convert geometry to WKT
        df = gdf.copy()
        df["geometry"] = df["geometry"].apply(lambda g: g.wkt if g else None)
        df.to_csv(output_file, index=False)
    else:
        output_file = output_path.with_suffix(".fgb")
        gdf.to_file(output_file, driver="FlatGeobuf")

    log.info("Saved predictions", path=str(output_file), n_rows=len(gdf))
    return output_file


def run_prediction(
    config: PipelineConfig,
    model: Any,
    output_path: Path | None = None,
    output_format: str = "fgb",
) -> PredictionPipelineResult:
    """
    Convenience function to run prediction pipeline.

    Args:
        config: Pipeline configuration.
        model: Trained model.
        output_path: Optional path to save predictions.
        output_format: Output format ('fgb', 'csv', 'parquet').

    Returns:
        PredictionPipelineResult with predictions and statistics.
    """
    pipeline = PredictionPipeline(config)
    return pipeline.run(model, output_path=output_path, output_format=output_format)
