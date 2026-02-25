"""
ETL pipeline implementation.

Orchestrates all data loading, transformation, and feature engineering
to produce a training-ready dataset.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from hochrechnung.config.settings import PipelineConfig, StatisticsApproach
from hochrechnung.features.pipeline import FeaturePipeline
from hochrechnung.ingestion.campaign import load_aggregated_statistics, load_demographics
from hochrechnung.ingestion.counter import (
    load_counter_locations,
    load_counter_measurements,
)
from hochrechnung.ingestion.osm import load_osm_infrastructure
from hochrechnung.ingestion.structural import (
    load_city_centroids,
    load_municipalities,
    load_regiostar,
)
from hochrechnung.ingestion.traffic import load_traffic_volumes
from hochrechnung.normalization.columns import normalize_columns
from hochrechnung.normalization.spatial import (
    calculate_distances_to_centroids,
    find_candidate_edges,
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
from hochrechnung.verification.outliers import run_all_detection
from hochrechnung.verification.persistence import load_verified_counters

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)

ETLMode = Literal["production", "verification"]


def _get_etl_cache(config: PipelineConfig) -> CacheManager:
    """Get cache manager for ETL operations."""
    cache_dir = config.cache_dir / "etl"
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
        mode: ETL mode ('production' or 'verification').
        used_verified_counters: Whether verified counters were loaded.
        n_flagged_outliers: Number of counters flagged as outliers (verification mode only).
        verification_data: DataFrame with verification data (verification mode only).
    """

    training_data: pd.DataFrame
    n_counters: int
    n_edges: int
    n_matched: int
    output_path: Path | None = None
    mode: ETLMode = "production"
    used_verified_counters: bool = False
    n_flagged_outliers: int | None = None
    verification_data: pd.DataFrame | None = None


class ETLPipeline:
    """
    ETL pipeline for bicycle traffic estimation.

    Loads all data sources, performs spatial matching, joins with
    structural data, and outputs a training-ready dataset.

    Performance optimizations:
    - Parallel loading of independent data sources
    - Caching of expensive intermediate results
    - Lazy CRS conversion with reuse
    - Copy-on-write semantics for DataFrames
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        use_cache: bool = True,
        parallel_loading: bool = True,
        max_workers: int = 4,
        mode: ETLMode = "production",
    ) -> None:
        """
        Initialize ETL pipeline.

        Args:
            config: Pipeline configuration.
            use_cache: Whether to cache intermediate results.
            parallel_loading: Whether to load data sources in parallel.
            max_workers: Maximum number of parallel workers for data loading.
            mode: ETL mode - 'production' uses verified counters, 'verification' creates them.
        """
        self.config = config
        self.feature_pipeline = FeaturePipeline(config)
        self.use_cache = use_cache
        self.parallel_loading = parallel_loading
        self.max_workers = max_workers
        self.mode = mode
        self._cache = _get_etl_cache(config) if use_cache else None

        # Enable copy-on-write for memory efficiency (pandas 2.0+)
        pd.options.mode.copy_on_write = True

    def run(self, output_path: Path | None = None) -> ETLResult:
        """
        Run the full ETL pipeline.

        In production mode: Uses verified counter dataset if available.
        In verification mode: Performs spatial matching and flags outliers for review.

        Args:
            output_path: Optional path to save output CSV.

        Returns:
            ETLResult with training data and statistics.
        """
        log.info(
            "Starting ETL pipeline",
            year=self.config.year,
            mode=self.mode,
            parallel=self.parallel_loading,
            cache=self.use_cache,
        )

        # Steps 1-4.5: Load all independent data sources
        # These can run in parallel as they have no dependencies on each other
        if self.parallel_loading:
            dtv_df, counters_gdf, traffic_gdf, osm_infra = self._load_data_parallel()
        else:
            dtv_df, counters_gdf, traffic_gdf, osm_infra = self._load_data_sequential()

        # Step 3: Merge DTV with counter locations
        log.info("Step 3: Merging DTV with counter locations")
        counters_with_dtv = self._merge_dtv_with_locations(dtv_df, counters_gdf)
        n_counters = len(counters_with_dtv)

        n_edges = len(traffic_gdf)

        # Join OSM infrastructure with traffic volumes
        log.info("Joining OSM infrastructure with traffic volumes")
        traffic_gdf = self._join_osm_infrastructure(traffic_gdf, osm_infra)

        # Prune unused columns from traffic data to reduce memory
        # Only keep columns needed for matching and feature engineering
        # Include count_forward/count_backward for verification UI if available
        edge_columns = ["base_id", "count", "bicycle_infrastructure"]
        optional_edge_columns = ["count_forward", "count_backward"]
        available_optional = [
            c for c in optional_edge_columns if c in traffic_gdf.columns
        ]
        edge_columns = edge_columns + available_optional

        traffic_gdf = self._prune_columns(
            traffic_gdf,
            required_columns=[*edge_columns, "geometry"],
            dataset_name="traffic",
        )

        # Step 5: Load verified counters OR perform spatial matching
        used_verified = False
        verification_data = None
        n_flagged_outliers = None

        # Try to load verified counters in production mode
        if self.mode == "production":
            verified_counters = load_verified_counters(
                self.config.data_paths.data_root, self.config.year,
                project=self.config.project,
            )

            if verified_counters is not None:
                log.info("Using verified counter dataset (production mode)")
                # Join DTV with verified spatial assignments
                matched_df = counters_with_dtv.merge(
                    verified_counters[
                        ["counter_id", "base_id", "count", "bicycle_infrastructure"]
                    ],
                    left_on="counter_id",
                    right_on="counter_id",
                    how="inner",
                )
                used_verified = True
            else:
                log.warning(
                    "No verified counter dataset found, falling back to spatial matching"
                )
                # Fall through to spatial matching below

        # Perform spatial matching if not using verified counters
        if not used_verified:
            log.info("Step 5: Matching counters to edges")
            matched_counters = match_counters_to_edges(
                counters_with_dtv,
                traffic_gdf,
                max_distance_m=50.0,
                edge_columns=edge_columns,
            )

            # In verification mode: run all detection methods
            if self.mode == "verification":
                log.info("Running verification detection (verification mode)")

                # Find candidate edges for ambiguity detection
                matched_counters = find_candidate_edges(
                    matched_counters,
                    traffic_gdf,
                    max_distance_m=50.0,
                    edge_columns=edge_columns,
                )

                # Merge previous verifications to preserve session state
                matched_counters = self._merge_previous_verifications(matched_counters)

                # Run all detection methods (ratios, outliers, no_volume, campaign_bias, ambiguous)
                # skip_verified=True to preserve previous verifications
                matched_counters, outlier_result = run_all_detection(
                    matched_counters, skip_verified=True
                )
                n_flagged_outliers = outlier_result.n_flagged

                log.info(
                    "Detection complete",
                    n_flagged=n_flagged_outliers,
                    median_ratio=outlier_result.median_ratio,
                    severity_breakdown=outlier_result.n_by_severity,
                )

                # Store verification data for export
                verification_data = matched_counters.copy()

            matched_df = matched_counters

        # Filter to only matched counters (those with base_id)
        matched_df = matched_df[matched_df["base_id"].notna()].copy()
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
            mode=self.mode,
            used_verified=used_verified,
        )

        return ETLResult(
            training_data=training_data,
            n_counters=n_counters,
            n_edges=n_edges,
            n_matched=int(n_matched),
            output_path=output_path,
            mode=self.mode,
            used_verified_counters=used_verified,
            n_flagged_outliers=n_flagged_outliers,
            verification_data=verification_data,
        )

    def _load_data_parallel(
        self,
    ) -> tuple[pd.DataFrame, "gpd.GeoDataFrame", "gpd.GeoDataFrame", pd.DataFrame]:
        """
        Load all independent data sources in parallel.

        Returns:
            Tuple of (dtv_df, counters_gdf, traffic_gdf, osm_infra).
        """
        log.info("Loading data sources in parallel", workers=self.max_workers)

        results: dict[str, pd.DataFrame | None] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_and_compute_dtv): "dtv",
                executor.submit(self._load_counter_locations): "counters",
                executor.submit(self._load_traffic_volumes_cached): "traffic",
                executor.submit(self._load_osm_infrastructure): "osm",
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    log.debug(f"Loaded {name} successfully")
                except Exception as e:
                    log.error(f"Failed to load {name}", error=str(e))
                    raise

        return (
            results["dtv"],  # type: ignore[return-value]
            results["counters"],  # type: ignore[return-value]
            results["traffic"],  # type: ignore[return-value]
            results["osm"],  # type: ignore[return-value]
        )

    def _load_data_sequential(
        self,
    ) -> tuple[pd.DataFrame, "gpd.GeoDataFrame", "gpd.GeoDataFrame", pd.DataFrame]:
        """
        Load all data sources sequentially.

        Returns:
            Tuple of (dtv_df, counters_gdf, traffic_gdf, osm_infra).
        """
        log.info("Step 1: Loading counter data and computing DTV")
        dtv_df = self._load_and_compute_dtv()

        log.info("Step 2: Loading counter locations")
        counters_gdf = self._load_counter_locations()

        log.info("Step 4: Loading traffic volumes")
        traffic_gdf = self._load_traffic_volumes_cached()

        log.info("Step 4.5: Loading OSM infrastructure")
        osm_infra = self._load_osm_infrastructure()

        return dtv_df, counters_gdf, traffic_gdf, osm_infra

    def _load_and_compute_dtv(self) -> pd.DataFrame:
        """Load counter measurements and compute DTV.

        If counter_measurements is not configured, falls back to extracting
        pre-computed DTV values from the counter locations CSV (e.g., DZS_mean_SR).
        """
        if self.config.data_paths.counter_measurements is not None:
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

        # Fallback: extract pre-computed DTV from counter locations CSV
        log.info("No counter measurements configured, extracting DTV from counter locations")
        locations = load_counter_locations(self.config, validate=False)

        if "dtv" not in locations.columns:
            msg = "Counter locations must have a 'dtv' column when counter_measurements is not configured"
            raise ValueError(msg)

        dtv_df = pd.DataFrame({
            "counter_id": locations["name"].astype(str).str.strip(),
            "dtv": pd.to_numeric(locations["dtv"], errors="coerce"),
            "quality_score": 1.0,
            "is_valid": True,
        })
        dtv_df = dtv_df.dropna(subset=["dtv"])

        log.info(
            "Extracted DTV from counter locations",
            valid_counters=len(dtv_df),
            total_counters=len(locations),
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

        # Drop columns from counters_gdf that will come from dtv_df to avoid
        # suffix conflicts (e.g., when counter locations CSV already has a 'dtv' column)
        dtv_merge_cols = {"dtv", "quality_score", "is_valid", "counter_id"}
        overlap_cols = [c for c in counters_gdf.columns if c in dtv_merge_cols]
        if overlap_cols:
            counters_gdf = counters_gdf.drop(columns=overlap_cols)

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

    def _load_traffic_volumes_cached(self) -> "gpd.GeoDataFrame":
        """
        Load traffic volumes with caching support.

        Uses pickle cache for the expensive GeoDataFrame to avoid
        repeated file parsing on subsequent runs.
        """
        if self._cache is None:
            return self._load_traffic_volumes()

        # Build cache key from traffic file path
        traffic_path = (
            self.config.data_paths.data_root / self.config.data_paths.traffic_volumes
        )
        cache_key = f"traffic_volumes_{self.config.year}"

        # Try to get from cache
        cached = self._cache.get(
            cache_key,
            source_files=[traffic_path],
            config_hash=self._cache.compute_config_hash(self.config.region),
        )

        if cached is not None:
            log.info("Loaded traffic volumes from cache", rows=len(cached))
            return cached

        # Load fresh and cache
        log.info("Loading traffic volumes (cache miss)")
        gdf = self._load_traffic_volumes()

        self._cache.set(
            cache_key,
            gdf,
            source_files=[traffic_path],
            config_hash=self._cache.compute_config_hash(self.config.region),
        )

        return gdf

    def _load_osm_infrastructure(self) -> pd.DataFrame:
        """Load OSM infrastructure classifications."""
        osm_df = load_osm_infrastructure(self.config)
        log.info("Loaded OSM infrastructure", rows=len(osm_df))
        return osm_df

    def _merge_previous_verifications(
        self, matched_counters: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """
        Merge previous verification results to preserve session state.

        Loads existing verified counters and applies their:
        - verification_status (verified/carryover)
        - verified base_id overrides
        - is_discarded flag
        - verification_metadata

        This ensures that when regenerating tiles, previous verifications
        are preserved rather than starting fresh.

        Args:
            matched_counters: Counters after spatial matching.

        Returns:
            GeoDataFrame with previous verifications merged in.
        """
        verified_counters = load_verified_counters(
            self.config.data_paths.data_root, self.config.year,
            project=self.config.project,
        )

        if verified_counters is None:
            log.info("No previous verifications found, starting fresh")
            # Initialize verification columns with defaults
            matched_counters["verification_status"] = "unverified"
            matched_counters["verified_at"] = None
            matched_counters["verification_metadata"] = ""
            matched_counters["is_discarded"] = False
            return matched_counters

        log.info(
            "Merging previous verifications",
            n_verified=len(verified_counters),
            n_matched=len(matched_counters),
        )

        # Columns to merge from verified counters
        merge_cols = [
            "counter_id",
            "base_id",
            "count",
            "verification_status",
            "verified_at",
            "verification_metadata",
            "is_discarded",
        ]
        # Only include columns that exist
        merge_cols = [c for c in merge_cols if c in verified_counters.columns]

        # Merge on counter_id, keeping all matched counters (left join)
        # Suffix _verified for columns that might conflict
        merged = matched_counters.merge(
            verified_counters[merge_cols],
            on="counter_id",
            how="left",
            suffixes=("", "_verified"),
        )

        # Determine column names after merge - suffix only added if conflict existed
        # If matched_counters had no verification_status, it won't have _verified suffix
        status_col = (
            "verification_status_verified"
            if "verification_status_verified" in merged.columns
            else "verification_status"
        )

        # For verified counters, use their verified base_id and count
        if status_col in merged.columns:
            verified_mask = merged[status_col].isin(["verified", "carryover"])
        else:
            verified_mask = pd.Series([False] * len(merged))

        if verified_mask.any():
            # Override base_id with verified value where available
            base_id_col = (
                "base_id_verified" if "base_id_verified" in merged.columns else None
            )
            if base_id_col:
                merged.loc[verified_mask, "base_id"] = merged.loc[
                    verified_mask, base_id_col
                ]
            # Override count with verified value where available
            count_col = "count_verified" if "count_verified" in merged.columns else None
            if count_col:
                verified_count_mask = verified_mask & merged[count_col].notna()
                merged.loc[verified_count_mask, "count"] = merged.loc[
                    verified_count_mask, count_col
                ]

        # Apply verification status - handle both suffixed and non-suffixed cases
        if "verification_status_verified" in merged.columns:
            # Had conflict - use _verified column and drop it
            merged["verification_status"] = merged[
                "verification_status_verified"
            ].fillna("unverified")
            merged = merged.drop(columns=["verification_status_verified"])
        elif "verification_status" in merged.columns:
            # No conflict - column came from verified_counters directly
            merged["verification_status"] = merged["verification_status"].fillna(
                "unverified"
            )
        else:
            merged["verification_status"] = "unverified"

        # Apply other verification fields - handle both cases
        if "verified_at_verified" in merged.columns:
            merged["verified_at"] = merged["verified_at_verified"]
            merged = merged.drop(columns=["verified_at_verified"])
        elif "verified_at" not in merged.columns:
            merged["verified_at"] = None

        if "verification_metadata_verified" in merged.columns:
            merged["verification_metadata"] = merged[
                "verification_metadata_verified"
            ].fillna("")
            merged = merged.drop(columns=["verification_metadata_verified"])
        elif "verification_metadata" in merged.columns:
            merged["verification_metadata"] = merged["verification_metadata"].fillna("")
        else:
            merged["verification_metadata"] = ""

        if "is_discarded_verified" in merged.columns:
            merged["is_discarded"] = merged["is_discarded_verified"].fillna(False)
            merged = merged.drop(columns=["is_discarded_verified"])
        elif "is_discarded" in merged.columns:
            merged["is_discarded"] = merged["is_discarded"].fillna(False)
        else:
            merged["is_discarded"] = False

        # Clean up any remaining _verified suffix columns
        verified_cols = [c for c in merged.columns if c.endswith("_verified")]
        if verified_cols:
            merged = merged.drop(columns=verified_cols)

        n_preserved = verified_mask.sum()
        log.info(
            "Merged previous verifications",
            n_preserved=int(n_preserved),
            n_new=len(merged) - int(n_preserved),
        )

        return merged

    def _prune_columns(
        self,
        gdf: "gpd.GeoDataFrame",
        required_columns: list[str],
        dataset_name: str = "data",
    ) -> "gpd.GeoDataFrame":
        """
        Drop unused columns from a GeoDataFrame to reduce memory.

        Args:
            gdf: GeoDataFrame to prune.
            required_columns: Columns to keep.
            dataset_name: Name for logging.

        Returns:
            GeoDataFrame with only required columns.
        """
        available = set(gdf.columns)
        keep = [c for c in required_columns if c in available]
        drop = [c for c in available if c not in keep]

        if drop:
            log.debug(
                f"Pruning {len(drop)} columns from {dataset_name}",
                dropped=drop[:5],  # Log first 5
                kept=len(keep),
            )
            return gdf[keep]

        return gdf

    def _join_osm_infrastructure(
        self, traffic_gdf: "gpd.GeoDataFrame", osm_df: pd.DataFrame
    ) -> "gpd.GeoDataFrame":
        """
        Join OSM infrastructure attributes with traffic volumes.

        Matches on base_id (OSM way ID).

        Note: Some traffic files (e.g., *_assessed.fgb) already include
        bicycle_infrastructure. In that case, the join is skipped.
        """
        import geopandas as gpd

        # Skip join if traffic data already has bicycle_infrastructure
        if "bicycle_infrastructure" in traffic_gdf.columns:
            log.info(
                "Traffic data already has bicycle_infrastructure, skipping OSM join",
                rows_with_infra=traffic_gdf["bicycle_infrastructure"].notna().sum(),
            )
            return traffic_gdf

        # Ensure base_id exists in both datasets
        if "base_id" not in traffic_gdf.columns:
            log.warning("Traffic data missing base_id, cannot join OSM infrastructure")
            return traffic_gdf

        if "osm_id" not in osm_df.columns:
            log.warning("OSM data missing osm_id, cannot join infrastructure")
            return traffic_gdf

        # Convert base_id to same type for matching
        # base_id might be float (e.g., 4865862.0), so convert to int first to avoid ".0" suffix
        traffic_gdf = traffic_gdf.copy()
        traffic_gdf["base_id"] = (
            traffic_gdf["base_id"].fillna(-1).astype(int).astype(str)
        )
        # Remove the -1 placeholder for NaN values
        traffic_gdf.loc[traffic_gdf["base_id"] == "-1", "base_id"] = None

        osm_df = osm_df.copy()
        osm_df["osm_id"] = osm_df["osm_id"].astype(str)

        # Merge on base_id = osm_id
        joined = traffic_gdf.merge(
            osm_df[["osm_id", "bicycle_infrastructure"]],
            left_on="base_id",
            right_on="osm_id",
            how="left",
        )

        # Drop redundant osm_id column
        if "osm_id" in joined.columns:
            joined = joined.drop(columns=["osm_id"])

        # Ensure it's still a GeoDataFrame
        if not isinstance(joined, gpd.GeoDataFrame):
            joined = gpd.GeoDataFrame(
                joined, geometry=traffic_gdf.geometry, crs=traffic_gdf.crs
            )

        log.info(
            "Joined OSM infrastructure",
            matched=joined["bicycle_infrastructure"].notna().sum(),
            total=len(joined),
        )

        return joined

    def _join_counters_with_traffic(
        self, counters: "gpd.GeoDataFrame", traffic: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """
        Join matched counters with traffic edge data.

        Uses edge_id for one-to-one matching to avoid row multiplication
        when multiple edges share the same base_id.
        """
        # Filter to only matched counters
        matched = counters[counters["matched_edge_id"].notna()].copy()

        # Convert matched_edge_id to same type as traffic edge_id
        if "edge_id" in traffic.columns:
            matched["matched_edge_id"] = matched["matched_edge_id"].astype(
                traffic["edge_id"].dtype
            )

        # Merge with traffic data using edge_id (unique per edge)
        traffic_cols = [col for col in traffic.columns if col not in ["geometry"]]
        traffic_subset = traffic[traffic_cols].copy()

        joined = matched.merge(
            traffic_subset,
            left_on="matched_edge_id",
            right_on="edge_id",
            how="left",
            suffixes=("", "_traffic"),
        )

        # Drop redundant edge_id column (keep matched_edge_id)
        if "edge_id" in joined.columns and "matched_edge_id" in joined.columns:
            joined = joined.drop(columns=["edge_id"])

        log.info("Joined counters with traffic", rows=len(joined))
        return joined

    def _add_structural_data(self, gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """Add municipality, RegioStaR, and distance features.

        Switches between legacy and gebietseinheiten approaches based on config.
        """
        if self.config.stats.approach == StatisticsApproach.GEBIETSEINHEITEN:
            return self._add_structural_data_gebietseinheiten(gdf)
        return self._add_structural_data_legacy(gdf)

    def _add_structural_data_legacy(
        self, gdf: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """Add structural data using legacy approach (VG250 + kommunen_stats)."""
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
        # Note: VG250 uses 12-digit ARS with sub-identifiers (e.g., 064310001001)
        # while demographics JSON uses trailing zeros (e.g., 064310001000).
        # Matching on first 9 characters (municipality prefix) gives best results.
        try:
            demographics = load_demographics(self.config, validate=False)
            if "ars" in gdf.columns and "ars" in demographics.columns:
                demo_cols = ["ars", "n_users", "n_trips", "total_km"]
                demo_cols = [c for c in demo_cols if c in demographics.columns]

                # Use 9-character prefix for matching (municipality level)
                gdf["ars_9"] = gdf["ars"].astype(str).str[:9]
                demographics["ars_9"] = demographics["ars"].astype(str).str[:9]

                gdf = gdf.merge(
                    demographics[["ars_9"] + [c for c in demo_cols if c != "ars"]],
                    on="ars_9",
                    how="left",
                    suffixes=("", "_demo"),
                )
        except FileNotFoundError:
            log.warning("Demographics data not found, skipping")

        return gdf

    def _add_structural_data_gebietseinheiten(
        self, gdf: "gpd.GeoDataFrame"
    ) -> "gpd.GeoDataFrame":
        """Add structural data using Gebietseinheiten approach.

        Uses DE_Gebietseinheiten boundaries with aggregated commune statistics.
        """
        from hochrechnung.ingestion.gebietseinheiten import (
            AdminLevel,
            load_gebietseinheiten,
        )

        # Parse admin level from config
        admin_level = AdminLevel.from_string(self.config.stats.admin_level)
        log.info(
            "Using Gebietseinheiten approach",
            admin_level=admin_level.value,
            prefix_len=admin_level.ars_prefix_length,
        )

        # Load Gebietseinheiten boundaries at configured admin level
        gebietseinheiten = load_gebietseinheiten(
            self.config, admin_level=admin_level, validate=False
        )

        # Load aggregated statistics at same admin level
        aggregated_stats = load_aggregated_statistics(
            self.config, admin_level=admin_level, validate=False
        )

        # Load RegioStaR and city centroids (same as legacy)
        regiostar = load_regiostar(self.config, validate=False)
        centroids = load_city_centroids(self.config, validate=False)

        # Spatial join features with Gebietseinheiten boundaries
        import geopandas as gpd

        # Ensure both are in same CRS
        if gdf.crs != gebietseinheiten.crs:
            gebietseinheiten = gebietseinheiten.to_crs(gdf.crs)

        # Spatial join to get admin unit ARS for each feature
        # Use "intersects" instead of "within" to catch edge cases
        gdf = gpd.sjoin(
            gdf,
            gebietseinheiten[["ars", "name", "geometry"]].rename(
                columns={"ars": "admin_ars", "name": "admin_name"}
            ),
            how="left",
            predicate="intersects",
        )

        # Drop index_right from sjoin if present
        if "index_right" in gdf.columns:
            gdf = gdf.drop(columns=["index_right"])

        # Handle potential duplicates from "intersects" (points on boundaries)
        id_col = "counter_id" if "counter_id" in gdf.columns else "name"
        if id_col in gdf.columns:
            n_before = len(gdf)
            gdf = gdf.drop_duplicates(subset=[id_col], keep="first")
            n_after = len(gdf)
            if n_before != n_after:
                log.info(
                    "Removed duplicate rows from boundary intersections",
                    before=n_before,
                    after=n_after,
                )

        # Debug logging: show mapping for each counter
        if "counter_id" in gdf.columns or "name" in gdf.columns:
            id_col = "counter_id" if "counter_id" in gdf.columns else "name"
            for _, row in gdf.iterrows():
                counter_id = row.get(id_col)
                admin_ars = row.get("admin_ars")
                admin_name = row.get("admin_name", "N/A")
                if pd.notna(admin_ars):
                    log.info(
                        "Mapping counter to Gebietseinheit",
                        counter_id=counter_id,
                        admin_ars=admin_ars,
                        admin_name=admin_name,
                    )
                else:
                    # Log counters that failed to match - these are the problem cases
                    lat = row.get("latitude", row.get("lat", "N/A"))
                    lon = row.get("longitude", row.get("lon", "N/A"))
                    log.warning(
                        "Counter not matched to any Gebietseinheit",
                        counter_id=counter_id,
                        latitude=lat,
                        longitude=lon,
                    )

        n_matched = gdf["admin_ars"].notna().sum()
        n_unmatched = gdf["admin_ars"].isna().sum()
        log.info(
            "Spatial joined with Gebietseinheiten",
            matched=n_matched,
            unmatched=n_unmatched,
            total=len(gdf),
        )

        # Join aggregated statistics on admin ARS
        if "admin_ars" in gdf.columns and "ars" in aggregated_stats.columns:
            gdf = gdf.merge(
                aggregated_stats[["ars", "n_users", "n_trips", "total_km"]],
                left_on="admin_ars",
                right_on="ars",
                how="left",
                suffixes=("", "_stats"),
            )
            # Drop duplicate ars column from stats
            if "ars_stats" in gdf.columns:
                gdf = gdf.drop(columns=["ars_stats"])

            # Debug logging for statistics join
            if "counter_id" in gdf.columns or "name" in gdf.columns:
                id_col = "counter_id" if "counter_id" in gdf.columns else "name"
                for _, row in gdf.iterrows():
                    counter_id = row.get(id_col)
                    n_users = row.get("n_users")
                    admin_ars = row.get("admin_ars")
                    if pd.isna(n_users):
                        log.warning(
                            "Counter missing STADTRADELN stats",
                            counter_id=counter_id,
                            admin_ars=admin_ars,
                        )
                    else:
                        log.debug(
                            "Counter STADTRADELN stats assigned",
                            counter_id=counter_id,
                            n_users=n_users,
                            n_trips=row.get("n_trips"),
                        )

            n_with_stats = gdf["n_users"].notna().sum()
            n_missing_stats = gdf["n_users"].isna().sum()
            log.info(
                "Joined aggregated STADTRADELN statistics",
                with_stats=n_with_stats,
                missing_stats=n_missing_stats,
            )

        # Aggregate population from VG250 municipalities by ARS prefix
        population_agg = self._aggregate_population_by_admin_level(admin_level)
        if "admin_ars" in gdf.columns and "ars" in population_agg.columns:
            gdf = gdf.merge(
                population_agg[["ars", "population"]],
                left_on="admin_ars",
                right_on="ars",
                how="left",
                suffixes=("", "_pop"),
            )
            # Drop duplicate ars column
            if "ars_pop" in gdf.columns:
                gdf = gdf.drop(columns=["ars_pop"])

            # Exclude counters in gemeindefreie Gebiete (population = 0)
            # These are uninhabited areas (forests, military zones) where
            # derived features like participation_rate produce infinity values
            if "population" in gdf.columns:
                zero_pop_mask = gdf["population"] == 0
                n_zero_pop = zero_pop_mask.sum()
                if n_zero_pop > 0:
                    id_col = "counter_id" if "counter_id" in gdf.columns else "name"
                    excluded_names = gdf.loc[zero_pop_mask, id_col].tolist()
                    log.warning(
                        "Excluding counters in gemeindefreie Gebiete (population=0)",
                        n_excluded=n_zero_pop,
                        counters=excluded_names,
                    )
                    gdf = gdf[~zero_pop_mask]

        # Join RegioStaR - requires municipality-level ARS
        # RegioStaR is defined at municipality level, so we need to spatial join
        # with municipalities to get the correct ARS for matching
        municipalities = load_municipalities(self.config, validate=False)

        # Ensure same CRS
        if gdf.crs != municipalities.crs:
            municipalities = municipalities.to_crs(gdf.crs)

        # Spatial join to get municipality ARS
        # Use "intersects" instead of "within" to catch edge cases
        gdf = gpd.sjoin(
            gdf,
            municipalities[["ars", "geometry"]].rename(columns={"ars": "municipality_ars"}),
            how="left",
            predicate="intersects",
        )
        if "index_right" in gdf.columns:
            gdf = gdf.drop(columns=["index_right"])

        # Handle potential duplicates from "intersects" (points on boundaries)
        id_col = "counter_id" if "counter_id" in gdf.columns else "name"
        if id_col in gdf.columns:
            n_before = len(gdf)
            gdf = gdf.drop_duplicates(subset=[id_col], keep="first")
            n_after = len(gdf)
            if n_before != n_after:
                log.info(
                    "Removed duplicate rows from municipality boundary intersections",
                    before=n_before,
                    after=n_after,
                )

        # Debug logging for municipality join
        if "counter_id" in gdf.columns or "name" in gdf.columns:
            id_col = "counter_id" if "counter_id" in gdf.columns else "name"
            for _, row in gdf.iterrows():
                counter_id = row.get(id_col)
                muni_ars = row.get("municipality_ars")
                if pd.isna(muni_ars):
                    lat = row.get("latitude", row.get("lat", "N/A"))
                    lon = row.get("longitude", row.get("lon", "N/A"))
                    log.warning(
                        "Counter not matched to any municipality (VG250)",
                        counter_id=counter_id,
                        latitude=lat,
                        longitude=lon,
                    )
                else:
                    log.debug(
                        "Mapping counter to municipality",
                        counter_id=counter_id,
                        municipality_ars=muni_ars,
                    )

        n_muni_matched = gdf["municipality_ars"].notna().sum()
        n_muni_unmatched = gdf["municipality_ars"].isna().sum()
        log.info(
            "Spatial joined with municipalities (VG250)",
            matched=n_muni_matched,
            unmatched=n_muni_unmatched,
        )

        # Now join RegioStaR on municipality ARS
        if "municipality_ars" in gdf.columns and "ars" in regiostar.columns:
            gdf["ars_12"] = gdf["municipality_ars"].astype(str).str[:12]
            regiostar["ars_12"] = regiostar["ars"].astype(str).str[:12]

            # Debug: show ARS lookup details
            gdf_ars_codes = set(gdf["ars_12"].dropna().unique())
            regiostar_ars_codes = set(regiostar["ars_12"].unique())
            missing_in_regiostar = gdf_ars_codes - regiostar_ars_codes
            if missing_in_regiostar:
                log.warning(
                    "Municipality ARS codes not found in RegioStaR",
                    missing_codes=list(missing_in_regiostar)[:10],
                    total_missing=len(missing_in_regiostar),
                )

            gdf = gdf.merge(
                regiostar[["ars_12", "regiostar5", "regiostar7"]],
                on="ars_12",
                how="left",
            )

            # Debug logging for RegioStaR join
            if "counter_id" in gdf.columns or "name" in gdf.columns:
                id_col = "counter_id" if "counter_id" in gdf.columns else "name"
                for _, row in gdf.iterrows():
                    counter_id = row.get(id_col)
                    regiostar5 = row.get("regiostar5")
                    muni_ars = row.get("municipality_ars")
                    ars_12 = row.get("ars_12")
                    if pd.isna(regiostar5):
                        log.warning(
                            "Counter missing RegioStaR5",
                            counter_id=counter_id,
                            municipality_ars=muni_ars,
                            ars_12=ars_12,
                        )
                    else:
                        log.debug(
                            "Counter RegioStaR assigned",
                            counter_id=counter_id,
                            regiostar5=regiostar5,
                        )

            n_regiostar = gdf["regiostar5"].notna().sum()
            n_missing_regiostar = gdf["regiostar5"].isna().sum()
            log.info(
                "Joined RegioStaR",
                with_regiostar=n_regiostar,
                missing_regiostar=n_missing_regiostar,
            )

        # Calculate distances to city centroids (same as legacy)
        gdf = calculate_distances_to_centroids(gdf, centroids)

        # Copy admin_ars to ars for downstream compatibility if ars doesn't exist
        if "ars" not in gdf.columns and "admin_ars" in gdf.columns:
            gdf["ars"] = gdf["admin_ars"]

        return gdf

    def _aggregate_population_by_admin_level(
        self, admin_level: "AdminLevel"
    ) -> pd.DataFrame:
        """Aggregate municipality populations to admin level.

        Args:
            admin_level: Administrative level for aggregation.

        Returns:
            DataFrame with aggregated population by admin unit ARS.
        """
        from hochrechnung.ingestion.gebietseinheiten import AdminLevel

        # Load municipalities from VG250
        municipalities = load_municipalities(self.config, validate=False)

        if "population" not in municipalities.columns:
            log.warning("No population column in municipalities")
            return pd.DataFrame(columns=["ars", "population"])

        prefix_len = admin_level.ars_prefix_length

        # Create aggregation key matching Gebietseinheiten ARS format
        municipalities["ars_agg"] = (
            municipalities["ars"].astype(str).str[:prefix_len].str.ljust(12, "0")
        )

        # Sum population by admin unit
        pop_agg = (
            municipalities.groupby("ars_agg", as_index=False)["population"]
            .sum()
            .rename(columns={"ars_agg": "ars"})
        )

        log.info(
            "Aggregated population",
            admin_level=admin_level.value,
            municipalities=len(municipalities),
            aggregated_units=len(pop_agg),
        )

        return pop_agg

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
        df = df.copy()

        # Normalize column names first
        df = normalize_columns(df)

        # Map infra_category to simplified categories (after normalization)
        # Matches the old mapping dict from data_prep.py
        # Note: normalize_columns renames bicycle_infrastructure → infra_category
        infra_mapping = {
            r"^bicycle_lane_.*$": "bicycle_lane",
            r"^bus_lane_.*$": "bicycle_lane",
            r"^bicycle_way_.*$": "bicycle_way",
            r"^bicycle_road.*$": "bicycle_road",
            r"^mit_road.*$": "mit_road",
            r"^mixed_way.*$": "mixed_way",
            r"^path_not_forbidden$|^pedestrian_both$|^service_misc$": "no",
        }

        # Apply mapping to infra_category (not bicycle_infrastructure, which is already renamed)
        if "infra_category" in df.columns:
            infra_col = df["infra_category"].fillna("no").astype(str)
            for pattern, replacement in infra_mapping.items():
                mask = infra_col.str.contains(pattern, regex=True, na=False)
                infra_col = infra_col.where(~mask, replacement)
            # Anything not matched becomes "no"
            valid_categories = {
                "bicycle_lane",
                "bicycle_way",
                "bicycle_road",
                "mit_road",
                "mixed_way",
                "no",
            }
            infra_col = infra_col.where(infra_col.isin(valid_categories), "no")
            df["infra_category"] = infra_col

        # Map to legacy output format
        # Note: normalize_columns renames count→stadtradeln_volume, so map that
        # Use infra_category (simplified) for OSM_Radinfra, not bicycle_infrastructure
        output_mapping = {
            "infra_category": "OSM_Radinfra",
            "participation_rate": "TN_SR_relativ",
            "route_intensity": "Streckengewicht_SR",
            "regiostar5": "RegioStaR5",
            "regiostar7": "RegioStaR7",
            "stadtradeln_volume": "Erh_SR",
            "count": "Erh_SR",  # fallback if not normalized
            "dist_to_center_m": "HubDist",
            "latitude": "lat",
            "longitude": "lon",
            "dtv": "DZS_mean_SR",
        }

        # Drop bicycle_infrastructure if infra_category exists (avoid duplicates)
        if "infra_category" in df.columns and "bicycle_infrastructure" in df.columns:
            df = df.drop(columns=["bicycle_infrastructure"])

        # Rename columns that exist
        rename_dict = {k: v for k, v in output_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Use DZS counter ID (name column) for id, overwrite any edge id from traffic data
        if "name" in df.columns:
            df["id"] = df["name"].astype(str)
        elif "counter_id" in df.columns:
            df["id"] = df["counter_id"].astype(str)

        # Select output columns in legacy order
        output_columns = [
            "OSM_Radinfra",
            "TN_SR_relativ",
            "Streckengewicht_SR",
            "RegioStaR5",
            "RegioStaR7",
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


def run_etl(
    config: PipelineConfig,
    output_path: Path | None = None,
    mode: ETLMode = "production",
) -> ETLResult:
    """
    Convenience function to run ETL pipeline.

    Args:
        config: Pipeline configuration.
        output_path: Optional path to save output CSV.
        mode: ETL mode - 'production' uses verified counters, 'verification' creates them.

    Returns:
        ETLResult with training data and statistics.
    """
    pipeline = ETLPipeline(config, mode=mode)
    return pipeline.run(output_path=output_path)
