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
    2. Project bin directory
    3. Common installation locations

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
        Path.cwd() / "bin" / exe_name_win,  # Project bin directory (Windows)
        Path.cwd() / "bin" / exe_name,  # Project bin directory (Unix)
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
    export_geometries: bool = True,
) -> Path:
    """
    Run OSM categorizer on a PBF file.

    Args:
        pbf_path: Path to input .osm.pbf file.
        output_path: Path for output CSV file.
        bbox: Optional bounding box (currently unused - tool processes full file).
        categorizer_path: Optional path to categorizer executable.
        export_geometries: Whether to export WKT linestring geometries.

    Returns:
        Path to output file.

    Raises:
        OSMCategorizerError: If categorization fails.
        FileNotFoundError: If categorizer not found.

    Note:
        The osmcategorizer_rust tool does not support bbox filtering.
        Full file will be processed; filter results afterward if needed.
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

    # Build command with flags as per tool's CLI interface
    cmd = [
        str(exe_path),
        "-i",
        str(pbf_path),
        "-o",
        str(output_path),
    ]

    # Add geometry export flag if requested
    if export_geometries:
        cmd.append("-g")

    log.info(
        "Running OSM categorizer",
        input=str(pbf_path),
        output=str(output_path),
        export_geometries=export_geometries,
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
) -> pd.DataFrame:
    """
    Load OSM infrastructure categorization data.

    Returns categorization mapping (osm_id -> bicycle_infrastructure, surface_cat)
    that can be joined with traffic volumes by base_id.

    If a pre-categorized CSV exists and use_cached=True, loads that.
    Otherwise, runs the categorizer on the PBF file.

    Args:
        config: Pipeline configuration.
        pbf_path: Optional path to .osm.pbf file. Falls back to config.data_paths.osm_pbf.
        use_cached: Whether to use existing categorized file.

    Returns:
        DataFrame with columns: osm_id, bicycle_infrastructure, surface_cat.
    """
    # Determine paths
    data_root = config.data_paths.data_root
    assessed_csv = data_root / "osm-data" / f"hessen-{config.year % 100:02d}0101-assessed.csv"

    # Check for existing assessed file
    if use_cached and assessed_csv.exists():
        log.info("Loading cached OSM categorization", path=str(assessed_csv))
        return pd.read_csv(assessed_csv)

    # Need to categorize - get pbf_path from config if not provided
    if pbf_path is None and config.data_paths.osm_pbf is not None:
        pbf_path = data_root / config.data_paths.osm_pbf

    if pbf_path is None:
        msg = (
            f"No cached assessed file at {assessed_csv} "
            "and no PBF path in config or provided for categorization"
        )
        raise FileNotFoundError(msg)

    # Run categorization (no geometries needed - just osm_id mapping)
    log.info("Running OSM categorization", pbf=str(pbf_path), output=str(assessed_csv))
    assessed_csv.parent.mkdir(parents=True, exist_ok=True)

    categorize_osm_infrastructure(
        pbf_path=pbf_path,
        output_path=assessed_csv,
        export_geometries=False,  # Don't need geometries - just categorization
    )

    # Load and return the categorization data
    return pd.read_csv(assessed_csv)
