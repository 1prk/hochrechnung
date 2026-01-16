"""
Spatial matching for counter-to-OSM edge alignment.

Provides functions to spatially join counters with traffic data and
calculate distances to city centers.
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


def match_counters_to_edges(
    counters: "gpd.GeoDataFrame",
    edges: "gpd.GeoDataFrame",
    max_distance_m: float = 50.0,
) -> "gpd.GeoDataFrame":
    """
    Match counter locations to nearest traffic edges.

    Args:
        counters: GeoDataFrame with counter point geometries.
        edges: GeoDataFrame with edge line geometries.
        max_distance_m: Maximum matching distance in meters.

    Returns:
        Counters with matched edge information.
    """
    import geopandas as gpd
    from shapely.strtree import STRtree

    log.info(
        "Matching counters to edges",
        n_counters=len(counters),
        n_edges=len(edges),
        max_distance=max_distance_m,
    )

    # Convert to metric CRS for distance calculations
    counters_m = counters.to_crs(3857)
    edges_m = edges.to_crs(3857)

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
    result["nearest_edge_idx"] = pd.NA
    result["nearest_edge_dist_m"] = np.nan
    result["matched_base_id"] = pd.NA

    valid_indices = counters.index[valid_pts].to_numpy()

    # Set matched values
    matched_pt_idx = pt_idx[within_threshold]
    matched_edge_idx = edge_idx[within_threshold]
    matched_distances = distances[within_threshold]

    result.loc[valid_indices[matched_pt_idx], "nearest_edge_dist_m"] = matched_distances

    if "base_id" in edges.columns:
        matched_base_ids = edges.iloc[
            edges.index[valid_edges].to_numpy()[matched_edge_idx]
        ]["base_id"].values
        result.loc[valid_indices[matched_pt_idx], "matched_base_id"] = matched_base_ids

    matched_count = within_threshold.sum()
    log.info(
        "Counter matching complete",
        matched=int(matched_count),
        unmatched=int(len(counters) - matched_count),
    )

    return result


def calculate_distances_to_centroids(
    gdf: "gpd.GeoDataFrame",
    centroids: "gpd.GeoDataFrame",
) -> "gpd.GeoDataFrame":
    """
    Calculate distance from each feature to nearest city centroid.

    Args:
        gdf: GeoDataFrame with geometries.
        centroids: GeoDataFrame with city centroid points.

    Returns:
        GeoDataFrame with 'dist_to_center_m' column.
    """
    from shapely.strtree import STRtree

    log.info("Calculating distances to centroids", n_features=len(gdf))

    # Convert to metric CRS
    gdf_m = gdf.to_crs(3857)
    centroids_m = centroids.to_crs(3857)

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
    how: str = "left",
) -> "gpd.GeoDataFrame":
    """
    Spatially join features with municipality boundaries.

    Args:
        gdf: GeoDataFrame with geometries.
        municipalities: GeoDataFrame with municipality polygons.
        how: Join type ('left', 'inner').

    Returns:
        GeoDataFrame with municipality attributes.
    """
    import geopandas as gpd

    log.info("Spatial join with municipalities", n_features=len(gdf))

    # Ensure same CRS
    if gdf.crs != municipalities.crs:
        municipalities = municipalities.to_crs(gdf.crs)

    # Perform spatial join
    result = gpd.sjoin(gdf, municipalities, how=how, predicate="within")

    # Clean up index column
    if "index_right" in result.columns:
        result = result.drop("index_right", axis=1)

    matched = result["ars"].notna().sum() if "ars" in result.columns else 0
    log.info("Spatial join complete", matched=int(matched), total=len(result))

    return result
