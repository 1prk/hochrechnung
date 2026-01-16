"""
ETL pipeline implementation.

Orchestrates all data loading, transformation, and feature engineering
to produce a training-ready dataset.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.features.pipeline import FeaturePipeline
from hochrechnung.ingestion.campaign import load_demographics
from hochrechnung.ingestion.counter import (
    load_counter_locations,
    load_counter_measurements,
)
from hochrechnung.ingestion.structural import (
    load_city_centroids,
    load_municipalities,
    load_regiostar,
)
from hochrechnung.ingestion.traffic import load_traffic_volumes
from hochrechnung.normalization.columns import normalize_columns
from hochrechnung.normalization.spatial import (
    calculate_distances_to_centroids,
    match_counters_to_edges,
    spatial_join_municipalities,
)
from hochrechnung.targets.dtv import (
    calculate_dtv,
    dtv_results_to_dataframe,
    filter_dtv_by_quality,
)
from hochrechnung.utils.cache import CacheManager
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


def _get_etl_cache(config: PipelineConfig) -> CacheManager:
    """Get cache manager for ETL operations."""
    cache_dir = config.output.cache_dir / "etl"
    return CacheManager(cache_dir)


@dataclass
class ETLResult:
    """
    Result of ETL pipeline execution.

    Attributes:
        training_data: Final training-ready DataFrame.
        n_counters: Number of counters with valid DTV.
        n_edges: Number of edges in traffic data.
        n_matched: Number of counters matched to edges.
        output_path: Path where data was saved (if any).
    """

    training_data: pd.DataFrame
    n_counters: int
    n_edges: int
    n_matched: int
    output_path: Path | None = None


class ETLPipeline:
    """
    ETL pipeline for bicycle traffic estimation.

    Loads all data sources, performs spatial matching, joins with
    structural data, and outputs a training-ready dataset.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize ETL pipeline.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.feature_pipeline = FeaturePipeline(config)

    def run(self, output_path: Path | None = None) -> ETLResult:
        """
        Run the full ETL pipeline.

        Args:
            output_path: Optional path to save output CSV.

        Returns:
            ETLResult with training data and statistics.
        """
        log.info("Starting ETL pipeline", year=self.config.year)

        # Step 1: Load and compute DTV from counter data
        log.info("Step 1: Loading counter data and computing DTV")
        dtv_df = self._load_and_compute_dtv()

        # Step 2: Load counter locations and create spatial DataFrame
        log.info("Step 2: Loading counter locations")
        counters_gdf = self._load_counter_locations()

        # Step 3: Merge DTV with counter locations
        log.info("Step 3: Merging DTV with counter locations")
        counters_with_dtv = self._merge_dtv_with_locations(dtv_df, counters_gdf)
        n_counters = len(counters_with_dtv)

        # Step 4: Load traffic volumes
        log.info("Step 4: Loading traffic volumes")
        traffic_gdf = self._load_traffic_volumes()
        n_edges = len(traffic_gdf)

        # Step 5: Match counters to edges and capture edge attributes
        # This captures base_id, count, and bicycle_infrastructure directly
        # from the matched edge - no separate join needed
        log.info("Step 5: Matching counters to edges")
        edge_columns = ["base_id", "count", "bicycle_infrastructure"]
        matched_counters = match_counters_to_edges(
            counters_with_dtv,
            traffic_gdf,
            max_distance_m=50.0,
            edge_columns=edge_columns,
        )

        # Filter to only matched counters (those with base_id)
        matched_df = matched_counters[matched_counters["base_id"].notna()].copy()
        n_matched = len(matched_df)
        log.info("Filtered to matched counters", n_matched=n_matched)

        # Step 6: Load and join structural data
        log.info("Step 6: Adding structural data")
        joined_df = self._add_structural_data(matched_df)

        # Step 7: Compute derived features
        log.info("Step 7: Computing features")
        feature_df = self.feature_pipeline.process(joined_df)

        # Step 8: Prepare final output
        log.info("Step 8: Preparing final output")
        training_data = self._prepare_output(feature_df)

        # Save if output path provided
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            training_data.to_csv(output_path, index=False)
            log.info(
                "Saved training data", path=str(output_path), rows=len(training_data)
            )

        log.info(
            "ETL pipeline complete",
            n_counters=n_counters,
            n_edges=n_edges,
            n_matched=int(n_matched),
            n_training_rows=len(training_data),
        )

        return ETLResult(
            training_data=training_data,
            n_counters=n_counters,
            n_edges=n_edges,
            n_matched=int(n_matched),
            output_path=output_path,
        )

    def _load_and_compute_dtv(self) -> pd.DataFrame:
        """Load counter measurements and compute DTV."""
        measurements = load_counter_measurements(self.config, validate=True)

        dtv_results = calculate_dtv(measurements, self.config.temporal)
        valid_results = filter_dtv_by_quality(
            dtv_results,
            min_quality=0.5,
            min_observations=7,
        )

        dtv_df = dtv_results_to_dataframe(valid_results)
        log.info(
            "DTV computation complete",
            valid_counters=len(valid_results),
            total_counters=len(dtv_results),
        )

        return dtv_df

    def _load_counter_locations(self) -> "gpd.GeoDataFrame":
        """Load counter locations as GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import Point

        locations_df = load_counter_locations(self.config, validate=True)

        # Create geometry from lat/lon
        geometry = [
            Point(lon, lat)
            for lon, lat in zip(
                locations_df["longitude"], locations_df["latitude"], strict=True
            )
        ]

        gdf = gpd.GeoDataFrame(locations_df, geometry=geometry, crs="EPSG:4326")
        return gdf

    def _merge_dtv_with_locations(
        self, dtv_df: pd.DataFrame, counters_gdf: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """Merge DTV values with counter locations."""
        import geopandas as gpd

        # The counter_id in dtv_df matches the 'name' column in counters_gdf
        # Both are strings like "001", "064b", "1001", etc.
        dtv_df = dtv_df.copy()
        counters_gdf = counters_gdf.copy()

        # Preserve CRS before merge
        original_crs = counters_gdf.crs

        # Ensure both are strings
        dtv_df["counter_id"] = dtv_df["counter_id"].astype(str).str.strip()
        counters_gdf["name"] = counters_gdf["name"].astype(str).str.strip()

        log.debug(
            "Merging counters",
            dtv_counter_ids=dtv_df["counter_id"].head(10).tolist(),
            location_names=counters_gdf["name"].head(10).tolist(),
        )

        # Merge on counter name
        merged = counters_gdf.merge(
            dtv_df[["counter_id", "dtv", "quality_score", "is_valid"]],
            left_on="name",
            right_on="counter_id",
            how="inner",
        )

        # Restore GeoDataFrame with CRS (merge may return DataFrame)
        if not isinstance(merged, gpd.GeoDataFrame):
            merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=original_crs)
        elif merged.crs is None:
            merged = merged.set_crs(original_crs)

        # Keep only valid DTV
        merged = merged[merged["is_valid"] == True]  # noqa: E712

        log.info(
            "Merged DTV with locations",
            merged_rows=len(merged),
            original_counters=len(counters_gdf),
        )

        return merged

    def _load_traffic_volumes(self) -> "gpd.GeoDataFrame":
        """Load traffic volume data."""
        return load_traffic_volumes(self.config, validate=False)

    def _join_counters_with_traffic(
        self, counters: "gpd.GeoDataFrame", traffic: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """
        Join matched counters with traffic edge data.

        Uses fid for one-to-one matching to avoid row multiplication
        when multiple edges share the same base_id.
        """
        # Filter to only matched counters
        matched = counters[counters["matched_fid"].notna()].copy()

        # Convert matched_fid to same type as traffic fid
        if "fid" in traffic.columns:
            matched["matched_fid"] = matched["matched_fid"].astype(traffic["fid"].dtype)

        # Merge with traffic data using fid (unique per edge)
        traffic_cols = [
            col for col in traffic.columns if col not in ["geometry"]
        ]
        traffic_subset = traffic[traffic_cols].copy()

        joined = matched.merge(
            traffic_subset,
            left_on="matched_fid",
            right_on="fid",
            how="left",
            suffixes=("", "_traffic"),
        )

        # Drop redundant fid column (keep matched_fid)
        if "fid" in joined.columns and "matched_fid" in joined.columns:
            joined = joined.drop(columns=["fid"])

        log.info("Joined counters with traffic", rows=len(joined))
        return joined

    def _add_structural_data(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """Add municipality, RegioStaR, and distance features."""
        # Load structural data
        municipalities = load_municipalities(self.config, validate=False)
        regiostar = load_regiostar(self.config, validate=False)
        centroids = load_city_centroids(self.config, validate=False)

        # Spatial join with municipalities
        gdf = spatial_join_municipalities(gdf, municipalities, how="left")

        # Join RegioStaR on ARS
        if "ars" in gdf.columns and "ars" in regiostar.columns:
            # Extract 12-digit ARS for matching
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
                demo_cols = ["ars", "n_users", "n_trips", "total_km"]
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

    def _prepare_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final output DataFrame with columns matching legacy format.

        Legacy columns:
        - OSM_Radinfra: Infrastructure type (simplified categories)
        - TN_SR_relativ: Participation rate
        - Streckengewicht_SR: Route intensity
        - RegioStaR5: Regional classification
        - Erh_SR: STADTRADELN volume
        - HubDist: Distance to city center
        - id: DZS counter ID (e.g., "071c", "064b")
        - base_id: OSM edge ID
        - lat: Latitude
        - lon: Longitude
        - DZS_mean_SR: DTV value
        """
        import re

        df = df.copy()

        # Normalize column names first
        df = normalize_columns(df)

        # Map bicycle_infrastructure to simplified categories
        # Matches the old mapping dict from data_prep.py
        infra_mapping = {
            r"^bicycle_lane_.*$": "bicycle_lane",
            r"^bus_lane_.*$": "bicycle_lane",
            r"^bicycle_way_.*$": "bicycle_way",
            r"^bicycle_road.*$": "bicycle_road",
            r"^mit_road.*$": "mit_road",
            r"^mixed_way.*$": "mixed_way",
            r"^path_not_forbidden$|^pedestrian_both$|^service_misc$": "no",
        }

        if "bicycle_infrastructure" in df.columns:
            infra_col = df["bicycle_infrastructure"].fillna("no").astype(str)
            for pattern, replacement in infra_mapping.items():
                mask = infra_col.str.contains(pattern, regex=True, na=False)
                infra_col = infra_col.where(~mask, replacement)
            # Anything not matched becomes "no"
            valid_categories = {"bicycle_lane", "bicycle_way", "bicycle_road",
                               "mit_road", "mixed_way", "no"}
            infra_col = infra_col.where(infra_col.isin(valid_categories), "no")
            df["bicycle_infrastructure"] = infra_col

        # Map to legacy output format
        # Note: normalize_columns renames count→stadtradeln_volume, so map that
        output_mapping = {
            "bicycle_infrastructure": "OSM_Radinfra",
            "infra_category": "OSM_Radinfra",
            "participation_rate": "TN_SR_relativ",
            "route_intensity": "Streckengewicht_SR",
            "regiostar5": "RegioStaR5",
            "stadtradeln_volume": "Erh_SR",
            "count": "Erh_SR",  # fallback if not normalized
            "dist_to_center_m": "HubDist",
            "latitude": "lat",
            "longitude": "lon",
            "dtv": "DZS_mean_SR",
        }

        # Rename columns that exist
        rename_dict = {k: v for k, v in output_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Use DZS counter ID (name column) for id, not ARS
        if "name" in df.columns and "id" not in df.columns:
            df["id"] = df["name"].astype(str)
        elif "counter_id" in df.columns and "id" not in df.columns:
            df["id"] = df["counter_id"].astype(str)

        # Select output columns in legacy order
        output_columns = [
            "OSM_Radinfra",
            "TN_SR_relativ",
            "Streckengewicht_SR",
            "RegioStaR5",
            "Erh_SR",
            "HubDist",
            "id",
            "base_id",
            "lat",
            "lon",
            "DZS_mean_SR",
        ]

        # Keep only columns that exist
        available_cols = [c for c in output_columns if c in df.columns]
        missing_cols = [c for c in output_columns if c not in df.columns]

        if missing_cols:
            log.warning("Missing output columns", missing=missing_cols)

        return df[available_cols]


def run_etl(config: PipelineConfig, output_path: Path | None = None) -> ETLResult:
    """
    Convenience function to run ETL pipeline.

    Args:
        config: Pipeline configuration.
        output_path: Optional path to save output CSV.

    Returns:
        ETLResult with training data and statistics.
    """
    pipeline = ETLPipeline(config)
    return pipeline.run(output_path=output_path)
