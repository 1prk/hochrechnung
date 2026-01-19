"""
GDAL/OGR environment setup for standalone binaries.

Configures environment for running ogr2ogr from bundled GISInternals release.
"""

import os
import subprocess
from pathlib import Path

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# GISInternals release bundled in bin/
_GDAL_RELEASE_NAME = "release-1930-x64-gdal-3-11-3-mapserver-8-4-0"


def _find_project_root() -> Path:
    """
    Find project root by looking for pyproject.toml.

    Returns:
        Path to project root directory.

    Raises:
        FileNotFoundError: If project root cannot be determined.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = "Could not find project root (no pyproject.toml found)"
    raise FileNotFoundError(msg)


def get_gdal_root() -> Path:
    """
    Get path to bundled GDAL installation.

    Returns:
        Path to GDAL bin directory containing DLLs and executables.

    Raises:
        FileNotFoundError: If GDAL installation not found.
    """
    project_root = _find_project_root()
    gdal_root = project_root / "bin" / _GDAL_RELEASE_NAME / "bin"

    if not gdal_root.exists():
        msg = (
            f"GDAL installation not found at {gdal_root}. "
            f"Download GISInternals {_GDAL_RELEASE_NAME} and extract to bin/"
        )
        raise FileNotFoundError(msg)

    return gdal_root


def get_ogr2ogr_path() -> Path:
    """
    Get path to ogr2ogr executable.

    Returns:
        Path to ogr2ogr.exe.

    Raises:
        FileNotFoundError: If ogr2ogr not found.
    """
    gdal_root = get_gdal_root()
    ogr2ogr = gdal_root / "gdal" / "apps" / "ogr2ogr.exe"

    if not ogr2ogr.exists():
        msg = f"ogr2ogr.exe not found at {ogr2ogr}"
        raise FileNotFoundError(msg)

    return ogr2ogr


def get_gdal_env() -> dict[str, str]:
    """
    Build environment dict for running GDAL tools.

    Sets PATH, GDAL_DATA, and PROJ_LIB to bundled locations.

    Returns:
        Environment dictionary ready for subprocess.run(env=...).

    Raises:
        FileNotFoundError: If GDAL installation not found.
    """
    gdal_root = get_gdal_root()

    gdal_data = gdal_root / "gdal-data"
    proj_lib = gdal_root / "proj9" / "share"

    if not gdal_data.exists():
        msg = f"GDAL_DATA not found at {gdal_data}"
        raise FileNotFoundError(msg)

    if not proj_lib.exists():
        msg = f"PROJ_LIB not found at {proj_lib}"
        raise FileNotFoundError(msg)

    env = os.environ.copy()

    # Prepend GDAL bin to PATH so DLLs are found
    env["PATH"] = str(gdal_root) + os.pathsep + env.get("PATH", "")
    env["GDAL_DATA"] = str(gdal_data)
    env["PROJ_LIB"] = str(proj_lib)

    return env


def check_gdal_installation() -> bool:
    """
    Check if GDAL installation is available and working.

    Returns:
        True if ogr2ogr can be executed successfully.
    """
    try:
        ogr2ogr = get_ogr2ogr_path()
        env = get_gdal_env()

        result = subprocess.run(
            [str(ogr2ogr), "--version"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            log.debug("GDAL installation verified", version=version)
            return True

        log.warning("ogr2ogr returned non-zero", stderr=result.stderr)
        return False

    except FileNotFoundError as e:
        log.warning("GDAL installation not found", error=str(e))
        return False
    except subprocess.TimeoutExpired:
        log.warning("ogr2ogr timed out")
        return False
