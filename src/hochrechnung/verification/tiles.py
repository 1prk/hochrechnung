"""
MBTiles generation for verification interface.

Creates lightweight vector tiles with traffic volumes around flagged counters.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


def generate_verification_mbtiles(
    counters_gdf: "gpd.GeoDataFrame",
    edges_gdf: "gpd.GeoDataFrame",
    output_path: Path,
    buffer_m: float = 250.0,
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
        min_zoom: Minimum zoom level.
        max_zoom: Maximum zoom level.
        only_flagged: If True, only buffer around flagged counters.

    Returns:
        Path to generated MBTiles file.

    Raises:
        FileNotFoundError: If tippecanoe is not installed.
        RuntimeError: If tippecanoe fails.
    """
    log.info(
        "Generating verification MBTiles",
        n_counters=len(counters_gdf),
        n_edges=len(edges_gdf),
        buffer_m=buffer_m,
        only_flagged=only_flagged,
    )

    # Check tippecanoe availability
    if not _check_tippecanoe():
        msg = (
            "tippecanoe not found. Install with: "
            "brew install tippecanoe (macOS) or "
            "apt install tippecanoe (Debian/Ubuntu)"
        )
        raise FileNotFoundError(msg)

    # Filter to flagged counters if requested
    if only_flagged and "is_outlier" in counters_gdf.columns:
        flagged = counters_gdf[counters_gdf["is_outlier"] == True]  # noqa: E712
        if len(flagged) == 0:
            log.warning("No flagged counters found, using all counters")
            flagged = counters_gdf
    else:
        flagged = counters_gdf

    log.info("Filtering edges around counters", n_flagged=len(flagged))

    # Convert to metric CRS for buffering
    flagged_metric = flagged.to_crs(epsg=3857)
    edges_metric = edges_gdf.to_crs(epsg=3857)

    # Create union of buffers
    search_area = flagged_metric.buffer(buffer_m).unary_union

    # Filter edges to search area
    relevant_edges = edges_metric[edges_metric.intersects(search_area)].copy()

    log.info(
        "Filtered edges",
        n_edges_original=len(edges_gdf),
        n_edges_filtered=len(relevant_edges),
        reduction_pct=round(
            100 * (1 - len(relevant_edges) / len(edges_gdf)) if len(edges_gdf) > 0 else 0,
            1,
        ),
    )

    # Convert back to WGS84 for tiles
    relevant_edges = relevant_edges.to_crs(epsg=4326)

    # Prepare properties for tiles
    # Keep only essential properties to reduce tile size
    keep_cols = ["base_id", "count", "bicycle_infrastructure", "geometry"]
    relevant_edges = relevant_edges[[c for c in keep_cols if c in relevant_edges.columns]]

    # Export to temporary GeoJSON for tippecanoe
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".geojson", delete=False
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

        # Convert to GeoJSON
        relevant_edges.to_file(tmp_path, driver="GeoJSON")

        log.info("Exported GeoJSON", path=str(tmp_path), size_mb=tmp_path.stat().st_size / 1e6)

    try:
        # Run tippecanoe
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "tippecanoe",
            "-o",
            str(output_path),
            "-Z",
            str(min_zoom),
            "-z",
            str(max_zoom),
            "-l",
            "volumes",  # Layer name
            "--drop-densest-as-needed",  # Handle dense areas
            "--force",  # Overwrite existing
            "--no-feature-limit",  # Don't limit features per tile
            "--no-tile-size-limit",  # Don't limit tile size
            str(tmp_path),
        ]

        log.info("Running tippecanoe", cmd=" ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,  # 5 minutes
        )

        if result.stderr:
            log.debug("Tippecanoe stderr", output=result.stderr[:500])

    except subprocess.CalledProcessError as e:
        log.error("Tippecanoe failed", stderr=e.stderr[:500])
        msg = f"tippecanoe failed: {e.stderr}"
        raise RuntimeError(msg) from e
    except subprocess.TimeoutExpired as e:
        log.error("Tippecanoe timed out")
        msg = "tippecanoe timed out after 5 minutes"
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


def _check_tippecanoe() -> bool:
    """Check if tippecanoe is available."""
    try:
        subprocess.run(
            ["tippecanoe", "--version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


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
