"""
Spatial matching for counter-to-OSM edge alignment.

Provides functions to spatially join counters with traffic data and
calculate distances to city centers.

Performance optimizations:
- Cached CRS conversions to avoid repeated transforms
- Reusable spatial indices
- Vectorized distance calculations
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)

# Standard CRS for metric calculations (Web Mercator)
METRIC_CRS = "EPSG:3857"
# Standard CRS for geographic coordinates
GEOGRAPHIC_CRS = "EPSG:4326"


def ensure_metric_crs(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Ensure GeoDataFrame is in metric CRS for distance calculations.

    Uses EPSG:3857 (Web Mercator) which provides meter-based coordinates
    suitable for distance calculations in most regions.

    Args:
        gdf: GeoDataFrame to convert.

    Returns:
        GeoDataFrame in EPSG:3857 CRS. Returns same object if already
        in correct CRS to avoid unnecessary copies.
    """
    if gdf.crs is None:
        log.warning("GeoDataFrame has no CRS, assuming EPSG:4326")
        gdf = gdf.set_crs(GEOGRAPHIC_CRS)

    # Check if already in target CRS
    epsg = gdf.crs.to_epsg() if gdf.crs is not None else None
    if epsg == 3857:
        return gdf

    return gdf.to_crs(METRIC_CRS)


def ensure_geographic_crs(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Ensure GeoDataFrame is in geographic CRS (WGS84).

    Args:
        gdf: GeoDataFrame to convert.

    Returns:
        GeoDataFrame in EPSG:4326 CRS.
    """
    if gdf.crs is None:
        log.warning("GeoDataFrame has no CRS, assuming EPSG:4326")
        return gdf.set_crs(GEOGRAPHIC_CRS)

    if gdf.crs.to_epsg() == 4326:
        return gdf

    return gdf.to_crs(GEOGRAPHIC_CRS)


def match_counters_to_edges(
    counters: "gpd.GeoDataFrame",
    edges: "gpd.GeoDataFrame",
    max_distance_m: float = 50.0,
    edge_columns: list[str] | None = None,
    *,
    edges_metric: "gpd.GeoDataFrame | None" = None,
) -> "gpd.GeoDataFrame":
    """
    Match counter locations to nearest traffic edges.

    Captures attributes from the matched edge directly during spatial matching
    to ensure one-to-one correspondence (no row multiplication from joins).

    Args:
        counters: GeoDataFrame with counter point geometries.
        edges: GeoDataFrame with edge line geometries.
        max_distance_m: Maximum matching distance in meters.
        edge_columns: Columns to capture from matched edges. Defaults to
            ["base_id", "count", "bicycle_infrastructure"].
        edges_metric: Pre-converted edges in metric CRS (EPSG:3857).
            If provided, avoids redundant CRS conversion.

    Returns:
        Counters with matched edge information.
    """
    from shapely.strtree import STRtree

    if edge_columns is None:
        edge_columns = ["base_id", "count", "bicycle_infrastructure"]

    log.info(
        "Matching counters to edges",
        n_counters=len(counters),
        n_edges=len(edges),
        max_distance=max_distance_m,
        capturing_columns=edge_columns,
    )

    # Convert to metric CRS for distance calculations
    # Use pre-converted edges if provided to avoid redundant conversion
    counters_m = ensure_metric_crs(counters)
    edges_m = edges_metric if edges_metric is not None else ensure_metric_crs(edges)

    # Build spatial index on edges
    valid_edges = edges_m.geometry.notna() & (~edges_m.geometry.is_empty)
    edge_geoms = np.array(edges_m.loc[valid_edges, "geometry"].values, dtype=object)
    tree = STRtree(edge_geoms)

    # Get valid counter points
    valid_pts = counters_m.geometry.notna() & (~counters_m.geometry.is_empty)
    points = np.array(counters_m.loc[valid_pts, "geometry"].values, dtype=object)

    # Query nearest edges
    (pt_idx, edge_idx), distances = tree.query_nearest(
        points, all_matches=False, return_distance=True
    )

    # Filter by max distance
    within_threshold = distances <= max_distance_m

    # Map results back to DataFrame
    result = counters.copy()
    result["nearest_edge_dist_m"] = np.nan

    valid_indices = counters.index[valid_pts].to_numpy()

    # Set matched values
    matched_pt_idx = pt_idx[within_threshold]
    matched_edge_idx = edge_idx[within_threshold]
    matched_distances = distances[within_threshold]

    result.loc[valid_indices[matched_pt_idx], "nearest_edge_dist_m"] = matched_distances

    # Get the actual edge indices in the original edges dataframe
    edge_original_indices = edges.index[valid_edges].to_numpy()[matched_edge_idx]

    # Capture all requested columns from matched edges
    for col in edge_columns:
        if col in edges.columns:
            matched_values = edges.loc[edge_original_indices, col].values
            result.loc[valid_indices[matched_pt_idx], col] = matched_values
        else:
            log.warning(f"Column '{col}' not found in edges, skipping")

    matched_count = within_threshold.sum()
    log.info(
        "Counter matching complete",
        matched=int(matched_count),
        unmatched=int(len(counters) - matched_count),
    )

    return result


def find_candidate_edges(
    counters: "gpd.GeoDataFrame",
    edges: "gpd.GeoDataFrame",
    max_distance_m: float = 50.0,
    edge_columns: list[str] | None = None,
    *,
    edges_metric: "gpd.GeoDataFrame | None" = None,
) -> "gpd.GeoDataFrame":
    """
    Find all candidate edges within threshold for each counter.

    Unlike match_counters_to_edges which returns only the nearest edge,
    this function returns all edges within the distance threshold,
    enabling ambiguity detection.

    Args:
        counters: GeoDataFrame with counter point geometries.
        edges: GeoDataFrame with edge line geometries.
        max_distance_m: Maximum matching distance in meters.
        edge_columns: Columns to capture from matched edges. Defaults to
            ["base_id", "count", "bicycle_infrastructure"].
        edges_metric: Pre-converted edges in metric CRS (EPSG:3857).

    Returns:
        Counters with 'candidate_edges' column containing list of dicts:
        [{"base_id": int, "count": int, "distance": float, ...}, ...]
    """
    from shapely.strtree import STRtree

    if edge_columns is None:
        edge_columns = ["base_id", "count", "bicycle_infrastructure"]

    log.info(
        "Finding candidate edges for counters",
        n_counters=len(counters),
        n_edges=len(edges),
        max_distance=max_distance_m,
    )

    # Convert to metric CRS for distance calculations
    counters_m = ensure_metric_crs(counters)
    edges_m = edges_metric if edges_metric is not None else ensure_metric_crs(edges)

    # Build spatial index on edges
    valid_edges_mask = edges_m.geometry.notna() & (~edges_m.geometry.is_empty)
    valid_edge_indices = edges.index[valid_edges_mask].to_numpy()
    edge_geoms = np.array(
        edges_m.loc[valid_edges_mask, "geometry"].values, dtype=object
    )
    tree = STRtree(edge_geoms)

    # Get valid counter points
    valid_pts_mask = counters_m.geometry.notna() & (~counters_m.geometry.is_empty)
    valid_counter_indices = counters.index[valid_pts_mask].to_numpy()
    points = np.array(counters_m.loc[valid_pts_mask, "geometry"].values, dtype=object)

    # Query all edges within distance for each point
    # Returns indices of geometries within distance
    results = tree.query(points, predicate="dwithin", distance=max_distance_m)

    # results is a 2D array: [[point_indices], [edge_indices]]
    pt_indices = results[0]
    edge_indices = results[1]

    # Calculate actual distances for matched pairs
    # points and edge_geoms contain shapely geometries stored as object dtype
    distances = np.array(
        [
            points[pi].distance(edge_geoms[ei])  # type: ignore[union-attr]
            for pi, ei in zip(pt_indices, edge_indices)
        ]
    )

    # Group by counter
    result = counters.copy()
    result["candidate_edges"] = None
    result["candidate_edges"] = result["candidate_edges"].astype(object)

    # Build candidate lists
    for i, counter_idx in enumerate(valid_counter_indices):
        # Find all matches for this counter
        mask = pt_indices == i
        if not mask.any():
            result.at[counter_idx, "candidate_edges"] = []
            continue

        candidates = []
        for edge_local_idx, dist in zip(edge_indices[mask], distances[mask]):
            edge_original_idx = valid_edge_indices[edge_local_idx]
            candidate = {"distance": float(dist)}

            # Add requested columns from edge
            for col in edge_columns:
                if col in edges.columns:
                    val = edges.at[edge_original_idx, col]
                    # Handle numpy types for JSON serialization
                    if hasattr(val, "item"):
                        val = val.item()
                    candidate[col] = val

            candidates.append(candidate)

        # Sort by distance
        candidates.sort(key=lambda x: x["distance"])
        result.at[counter_idx, "candidate_edges"] = candidates

    # Count statistics
    n_with_candidates = sum(
        1
        for idx in valid_counter_indices
        if result.at[idx, "candidate_edges"]
        and len(result.at[idx, "candidate_edges"]) > 0
    )
    n_ambiguous = sum(
        1
        for idx in valid_counter_indices
        if result.at[idx, "candidate_edges"]
        and len(result.at[idx, "candidate_edges"]) > 1
    )

    log.info(
        "Candidate edge search complete",
        n_with_candidates=n_with_candidates,
        n_with_multiple=n_ambiguous,
    )

    return result


def calculate_distances_to_centroids(
    gdf: "gpd.GeoDataFrame",
    centroids: "gpd.GeoDataFrame",
    *,
    gdf_metric: "gpd.GeoDataFrame | None" = None,
    centroids_metric: "gpd.GeoDataFrame | None" = None,
) -> "gpd.GeoDataFrame":
    """
    Calculate distance from each feature to nearest city centroid.

    Args:
        gdf: GeoDataFrame with geometries.
        centroids: GeoDataFrame with city centroid points.
        gdf_metric: Pre-converted gdf in metric CRS (EPSG:3857).
            If provided, avoids redundant CRS conversion.
        centroids_metric: Pre-converted centroids in metric CRS.
            If provided, avoids redundant CRS conversion.

    Returns:
        GeoDataFrame with 'dist_to_center_m' column.
    """
    from shapely.strtree import STRtree

    log.info("Calculating distances to centroids", n_features=len(gdf))

    # Convert to metric CRS, reusing pre-converted if available
    gdf_m = gdf_metric if gdf_metric is not None else ensure_metric_crs(gdf)
    centroids_m = (
        centroids_metric
        if centroids_metric is not None
        else ensure_metric_crs(centroids)
    )

    # Build spatial index on centroids
    centroid_geoms = np.array(centroids_m.geometry.values, dtype=object)
    tree = STRtree(centroid_geoms)

    # Get valid geometries (use centroid for lines/polygons)
    valid_mask = gdf_m.geometry.notna() & (~gdf_m.geometry.is_empty)
    geoms = gdf_m.loc[valid_mask, "geometry"]

    # Get representative points (centroid for lines)
    points = np.array(geoms.centroid.values, dtype=object)

    # Query nearest centroids
    (pt_idx, _cen_idx), distances = tree.query_nearest(
        points, all_matches=False, return_distance=True
    )

    # Map back to original DataFrame
    result = gdf.copy()
    result["dist_to_center_m"] = np.nan

    valid_indices = gdf.index[valid_mask].to_numpy()
    result.loc[valid_indices[pt_idx], "dist_to_center_m"] = distances

    log.info(
        "Distance calculation complete",
        mean_distance=float(result["dist_to_center_m"].mean()),
    )

    return result


def spatial_join_municipalities(
    gdf: "gpd.GeoDataFrame",
    municipalities: "gpd.GeoDataFrame",
    how: Literal["left", "right", "inner"] = "left",
) -> "gpd.GeoDataFrame":
    """
    Spatially join features with municipality boundaries.

    Args:
        gdf: GeoDataFrame with geometries.
        municipalities: GeoDataFrame with municipality polygons.
        how: Join type ('left', 'right', 'inner').

    Returns:
        GeoDataFrame with municipality attributes.
    """
    import geopandas as gpd

    log.info("Spatial join with municipalities", n_features=len(gdf))

    # Ensure same CRS - convert municipalities to match gdf
    if gdf.crs is not None and gdf.crs != municipalities.crs:
        municipalities = municipalities.to_crs(gdf.crs)

    # Perform spatial join
    result = gpd.sjoin(gdf, municipalities, how=how, predicate="within")

    # Clean up index column
    if "index_right" in result.columns:
        result = result.drop("index_right", axis=1)

    matched = result["ars"].notna().sum() if "ars" in result.columns else 0
    log.info("Spatial join complete", matched=int(matched), total=len(result))

    # Ensure we return a GeoDataFrame
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result)

    return result
