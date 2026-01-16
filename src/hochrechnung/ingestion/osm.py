"""
OSM bicycle infrastructure ingestion.

Provides automated OSM infrastructure classification using external categorizer.
Eliminates manual merging of *_assessed.fgb files.
"""

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class OSMCategorizerError(Exception):
    """Error from OSM categorizer tool."""


def find_osmcategorizer() -> Path | None:
    """
    Find the osmcategorizer_rust executable.

    Searches in:
    1. PATH
    2. Common installation locations

    Returns:
        Path to executable or None if not found.
    """
    # Check PATH first
    exe_name = "osmcategorizer_rust"
    if (path := shutil.which(exe_name)) is not None:
        return Path(path)

    # Check Windows variant
    exe_name_win = "osmcategorizer_rust.exe"
    if (path := shutil.which(exe_name_win)) is not None:
        return Path(path)

    # Check common locations
    common_paths = [
        Path.home() / ".cargo" / "bin" / exe_name,
        Path.home() / ".cargo" / "bin" / exe_name_win,
        Path("/usr/local/bin") / exe_name,
    ]

    for path in common_paths:
        if path.exists():
            return path

    return None


def categorize_osm_infrastructure(
    pbf_path: Path,
    output_path: Path,
    bbox: tuple[float, float, float, float] | None = None,
    *,
    categorizer_path: Path | None = None,
) -> Path:
    """
    Run OSM categorizer on a PBF file.

    Args:
        pbf_path: Path to input .osm.pbf file.
        output_path: Path for output (FlatGeoBuf or GeoPackage).
        bbox: Optional bounding box (minx, miny, maxx, maxy).
        categorizer_path: Optional path to categorizer executable.

    Returns:
        Path to output file.

    Raises:
        OSMCategorizerError: If categorization fails.
        FileNotFoundError: If categorizer not found.
    """
    # Find categorizer
    exe_path = categorizer_path or find_osmcategorizer()
    if exe_path is None:
        msg = (
            "osmcategorizer_rust not found. "
            "Install it from https://github.com/your-org/osmcategorizer_rust"
        )
        raise FileNotFoundError(msg)

    if not pbf_path.exists():
        msg = f"PBF file not found: {pbf_path}"
        raise FileNotFoundError(msg)

    # Build command
    cmd = [str(exe_path), str(pbf_path), str(output_path)]

    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        cmd.extend(["--bbox", f"{min_lon},{min_lat},{max_lon},{max_lat}"])

    log.info(
        "Running OSM categorizer",
        input=str(pbf_path),
        output=str(output_path),
    )

    # Run categorizer
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600,  # 1 hour timeout
        )
        log.debug("Categorizer output", stdout=result.stdout)
    except subprocess.CalledProcessError as e:
        msg = f"OSM categorizer failed: {e.stderr}"
        raise OSMCategorizerError(msg) from e
    except subprocess.TimeoutExpired as e:
        msg = "OSM categorizer timed out after 1 hour"
        raise OSMCategorizerError(msg) from e

    if not output_path.exists():
        msg = f"Categorizer did not create output file: {output_path}"
        raise OSMCategorizerError(msg)

    log.info("OSM categorization complete", output=str(output_path))
    return output_path


def load_osm_infrastructure(
    config: PipelineConfig,
    pbf_path: Path | None = None,
    *,
    use_cached: bool = True,
) -> "gpd.GeoDataFrame":
    """
    Load OSM infrastructure with automatic categorization.

    If a pre-categorized file exists and use_cached=True, loads that.
    Otherwise, runs the categorizer on the PBF file.

    Args:
        config: Pipeline configuration.
        pbf_path: Optional path to .osm.pbf file.
        use_cached: Whether to use existing categorized file.

    Returns:
        GeoDataFrame with categorized infrastructure.
    """
    import geopandas as gpd

    # Check for existing categorized file
    traffic_path = config.data_paths.data_root / config.data_paths.traffic_volumes

    if use_cached and traffic_path.exists():
        log.info("Loading cached categorized OSM data", path=str(traffic_path))
        return gpd.read_file(traffic_path, bbox=config.region.bbox)

    # Need to categorize - require pbf_path
    if pbf_path is None:
        msg = (
            f"No cached categorized file at {traffic_path} "
            "and no PBF path provided for categorization"
        )
        raise FileNotFoundError(msg)

    # Run categorization
    output_path = traffic_path.parent / f"{traffic_path.stem}_categorized.fgb"
    categorize_osm_infrastructure(
        pbf_path=pbf_path,
        output_path=output_path,
        bbox=config.region.bbox,
    )

    return gpd.read_file(output_path, bbox=config.region.bbox)
