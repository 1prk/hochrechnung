"""
MBTiles generation for verification interface.

Creates lightweight vector tiles with traffic volumes around flagged counters.
Uses ogr2ogr from bundled GDAL installation.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from hochrechnung.utils.gdal import (
    check_gdal_installation,
    get_gdal_env,
    get_ogr2ogr_path,
)
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


def generate_verification_mbtiles(
    counters_gdf: "gpd.GeoDataFrame",
    edges_gdf: "gpd.GeoDataFrame",
    output_path: Path,
    buffer_m: float = 250.0,
    no_volume_buffer_m: float = 1000.0,
    min_zoom: int = 10,
    max_zoom: int = 16,
    *,
    only_flagged: bool = True,
) -> Path:
    """
    Generate MBTiles with edges around flagged counters.

    Creates lightweight tiles by only including edges within buffer distance
    of flagged counters. This keeps file size manageable (5-20 MB vs 500+ MB
    for full coverage).

    Args:
        counters_gdf: GeoDataFrame with counter points.
        edges_gdf: GeoDataFrame with traffic volume edges.
        output_path: Path for output MBTiles file.
        buffer_m: Buffer distance in meters around counters.
        no_volume_buffer_m: Larger buffer for no_volume counters (default 1km).
        min_zoom: Minimum zoom level.
        max_zoom: Maximum zoom level.
        only_flagged: If True, only buffer around flagged counters.

    Returns:
        Path to generated MBTiles file.

    Raises:
        FileNotFoundError: If ogr2ogr/GDAL is not installed.
        RuntimeError: If ogr2ogr fails.
    """
    log.info(
        "Generating verification MBTiles",
        n_counters=len(counters_gdf),
        n_edges=len(edges_gdf),
        buffer_m=buffer_m,
        only_flagged=only_flagged,
    )

    # Check ogr2ogr availability
    if not check_gdal_installation():
        import sys

        if sys.platform == "win32":
            msg = (
                "ogr2ogr not found. Download GISInternals GDAL release and "
                "extract to bin/release-1930-x64-gdal-3-11-3-mapserver-8-4-0/"
            )
        else:
            msg = (
                "ogr2ogr not found. Install GDAL:\n"
                "  Ubuntu/Debian: sudo apt install gdal-bin\n"
                "  macOS: brew install gdal"
            )
        raise FileNotFoundError(msg)

    # Filter to counters needing tiles if requested
    # Include: outliers, non-ok severity, AND verified counters (to show their edges)
    if only_flagged:
        needs_tiles = counters_gdf["is_outlier"] == True  # noqa: E712
        if "flag_severity" in counters_gdf.columns:
            # Include counters with non-ok severity (ambiguous, no_volume, etc.)
            # Also include verified counters so their edges are visible
            needs_tiles = needs_tiles | ~counters_gdf["flag_severity"].isin(["ok"])
        if "verification_status" in counters_gdf.columns:
            # Also include any previously verified counters
            needs_tiles = needs_tiles | counters_gdf["verification_status"].isin(
                ["verified", "carryover"]
            )
        flagged = counters_gdf[needs_tiles]
        if len(flagged) == 0:
            log.warning("No flagged counters found, using all counters")
            flagged = counters_gdf
    else:
        flagged = counters_gdf

    log.info("Filtering edges around counters", n_flagged=len(flagged))

    # Ensure edges have CRS set (traffic volume files may lack CRS metadata)
    if edges_gdf.crs is None:
        log.warning("Edges GeoDataFrame has no CRS, assuming EPSG:4326")
        edges_gdf = edges_gdf.set_crs(epsg=4326)

    # Convert to metric CRS for buffering
    flagged_metric = flagged.to_crs(epsg=3857)
    edges_metric = edges_gdf.to_crs(epsg=3857)

    # Create union of buffers with larger buffer for no_volume counters
    if "flag_severity" in flagged_metric.columns:
        no_volume_mask = flagged_metric["flag_severity"] == "no_volume"
        regular_counters = flagged_metric[~no_volume_mask]
        no_volume_counters = flagged_metric[no_volume_mask]

        search_areas = []
        if len(regular_counters) > 0:
            search_areas.append(regular_counters.buffer(buffer_m).unary_union)
        if len(no_volume_counters) > 0:
            log.info(
                "Using larger buffer for no_volume counters",
                n_no_volume=len(no_volume_counters),
                buffer_m=no_volume_buffer_m,
            )
            search_areas.append(
                no_volume_counters.buffer(no_volume_buffer_m).unary_union
            )

        if search_areas:
            from shapely.ops import unary_union as shapely_union

            search_area = shapely_union(search_areas)
        else:
            search_area = flagged_metric.buffer(buffer_m).unary_union
    else:
        search_area = flagged_metric.buffer(buffer_m).unary_union

    # Filter edges to search area
    relevant_edges = edges_metric[edges_metric.intersects(search_area)].copy()

    log.info(
        "Filtered edges",
        n_edges_original=len(edges_gdf),
        n_edges_filtered=len(relevant_edges),
        reduction_pct=round(
            100 * (1 - len(relevant_edges) / len(edges_gdf))
            if len(edges_gdf) > 0
            else 0,
            1,
        ),
    )

    # Convert back to WGS84 for tiles
    relevant_edges = relevant_edges.to_crs(epsg=4326)

    # Prepare properties for tiles
    # Keep only essential properties to reduce tile size
    # Include count_forward/count_backward for directional visualization
    keep_cols = [
        "base_id",
        "count",
        "count_forward",
        "count_backward",
        "bicycle_infrastructure",
        "geometry",
    ]
    relevant_edges = relevant_edges[
        [c for c in keep_cols if c in relevant_edges.columns]
    ]

    # Export to temporary GeoJSON for ogr2ogr
    # Use mkstemp to avoid Windows file locking issues with NamedTemporaryFile
    fd, tmp_path_str = tempfile.mkstemp(suffix=".geojson")
    tmp_path = Path(tmp_path_str)
    try:
        # Close file descriptor - geopandas will open the file itself
        os.close(fd)

        # Convert to GeoJSON
        relevant_edges.to_file(tmp_path, driver="GeoJSON")

        log.info(
            "Exported GeoJSON",
            path=str(tmp_path),
            size_mb=tmp_path.stat().st_size / 1e6,
        )
        # Run ogr2ogr
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove existing file (ogr2ogr MVT driver doesn't overwrite cleanly)
        if output_path.exists():
            output_path.unlink()

        ogr2ogr = get_ogr2ogr_path()
        env = get_gdal_env()

        cmd = [
            str(ogr2ogr),
            "-f",
            "MVT",
            str(output_path),
            str(tmp_path),
            "-dsco",
            f"MINZOOM={min_zoom}",
            "-dsco",
            f"MAXZOOM={max_zoom}",
            "-dsco",
            "COMPRESS=YES",
            "-dsco",
            "MAX_SIZE=500000",  # Max tile size in bytes
            "-nln",
            "volumes",  # Layer name
        ]

        log.info("Running ogr2ogr", cmd=" ".join(cmd))

        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minutes
        )

        if result.stderr:
            log.debug("ogr2ogr stderr", output=result.stderr[:500])

    except subprocess.CalledProcessError as e:
        log.error("ogr2ogr failed", stderr=e.stderr[:500] if e.stderr else "")
        msg = f"ogr2ogr failed: {e.stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        log.error("ogr2ogr timed out")
        msg = "ogr2ogr timed out after 5 minutes"
        raise RuntimeError(msg) from e
    finally:
        # Clean up temporary file
        tmp_path.unlink(missing_ok=True)

    if not output_path.exists():
        msg = f"MBTiles generation failed: {output_path} was not created"
        raise RuntimeError(msg)

    file_size_mb = output_path.stat().st_size / 1e6
    log.info(
        "Generated MBTiles",
        path=str(output_path),
        size_mb=round(file_size_mb, 2),
    )

    return output_path


def generate_counter_geojson(
    counters_gdf: "gpd.GeoDataFrame",
    output_path: Path,
) -> Path:
    """
    Generate GeoJSON with counter points for the verification UI.

    Args:
        counters_gdf: GeoDataFrame with counter points.
        output_path: Path for output GeoJSON file.

    Returns:
        Path to generated GeoJSON file.
    """
    log.info("Generating counter GeoJSON", n_counters=len(counters_gdf))

    # Ensure WGS84
    if counters_gdf.crs is not None and counters_gdf.crs.to_epsg() != 4326:
        counters_gdf = counters_gdf.to_crs(epsg=4326)

    # Select relevant columns
    keep_cols = [
        "counter_id",
        "name",
        "latitude",
        "longitude",
        "dtv",
        "base_id",
        "count",
        "bicycle_infrastructure",
        "dtv_volume_ratio",
        "is_outlier",
        "flag_severity",
        "verification_status",
        "geometry",
    ]

    export_cols = [c for c in keep_cols if c in counters_gdf.columns]
    counters_export = counters_gdf[export_cols].copy()

    # Convert to GeoJSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counters_export.to_file(output_path, driver="GeoJSON")

    file_size_kb = output_path.stat().st_size / 1e3
    log.info(
        "Generated counter GeoJSON",
        path=str(output_path),
        size_kb=round(file_size_kb, 2),
    )

    return output_path
