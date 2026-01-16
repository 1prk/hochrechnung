"""
Persistent verified counter dataset management.

Handles loading and saving of verified counter-to-OSM-edge assignments
that persist across ETL runs.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

VerificationStatus = Literal["auto", "verified", "carryover", "unverified"]


@dataclass
class VerifiedCounter:
    """
    Verified counter with OSM way assignment.

    Attributes:
        counter_id: Unique counter identifier.
        name: Counter name.
        latitude: WGS84 latitude.
        longitude: WGS84 longitude.
        ars: 12-digit municipality code.
        base_id: OSM way ID (verified).
        count: Bicycle volume on the edge (verified).
        bicycle_infrastructure: Infrastructure type.
        osm_source: OSM data source identifier (e.g., 'osm-2023-01-01').
        verification_status: Status of verification.
        verified_at: Timestamp of verification.
        verification_metadata: Free-text explanation.
    """

    counter_id: str
    name: str
    latitude: float
    longitude: float
    ars: str
    base_id: int
    count: int
    bicycle_infrastructure: str
    osm_source: str
    verification_status: VerificationStatus
    verified_at: datetime
    verification_metadata: str


def load_verified_counters(
    data_root: Path, year: int, *, must_exist: bool = False
) -> pd.DataFrame | None:
    """
    Load verified counter dataset for a specific year.

    Args:
        data_root: Root data directory.
        year: Campaign year.
        must_exist: If True, raises FileNotFoundError if file doesn't exist.

    Returns:
        DataFrame with verified counters, or None if file doesn't exist
        and must_exist=False.

    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True.
    """
    verified_dir = data_root / "verified"
    verified_path = verified_dir / f"counters_verified_{year}.csv"

    if not verified_path.exists():
        if must_exist:
            msg = f"Verified counter dataset not found: {verified_path}"
            raise FileNotFoundError(msg)
        log.info(
            "No verified counter dataset found",
            path=str(verified_path),
            year=year,
        )
        return None

    log.info(
        "Loading verified counter dataset",
        path=str(verified_path),
        year=year,
    )

    df = pd.read_csv(verified_path)

    # Parse verified_at timestamp
    if "verified_at" in df.columns:
        df["verified_at"] = pd.to_datetime(df["verified_at"])

    log.info(
        "Loaded verified counters",
        n_counters=len(df),
        n_verified=len(df[df["verification_status"] == "verified"]),
        n_auto=len(df[df["verification_status"] == "auto"]),
        n_carryover=len(df[df["verification_status"] == "carryover"]),
    )

    return df


def save_verified_counters(
    verified_df: pd.DataFrame,
    data_root: Path,
    year: int,
    *,
    add_etl_version: bool = True,
) -> Path:
    """
    Save verified counter dataset.

    Args:
        verified_df: DataFrame with verified counter data.
        data_root: Root data directory.
        year: Campaign year.
        add_etl_version: If True, adds git commit hash as etl_version.

    Returns:
        Path to saved file.
    """
    verified_dir = data_root / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)

    verified_path = verified_dir / f"counters_verified_{year}.csv"

    # Add git version if requested
    df = verified_df.copy()
    if add_etl_version and "etl_version" not in df.columns:
        git_hash = _get_git_commit_hash()
        df["etl_version"] = git_hash

    # Ensure consistent column order
    column_order = [
        "counter_id",
        "name",
        "latitude",
        "longitude",
        "ars",
        "base_id",
        "count",
        "bicycle_infrastructure",
        "osm_source",
        "verification_status",
        "verified_at",
        "verification_metadata",
    ]
    if "etl_version" in df.columns:
        column_order.append("etl_version")

    # Keep only columns that exist
    available_cols = [c for c in column_order if c in df.columns]
    df = df[available_cols]

    # Sort by counter_id for stable diffs
    df = df.sort_values("counter_id")

    # Save with consistent formatting
    df.to_csv(
        verified_path,
        index=False,
        encoding="utf-8",
        line_terminator="\n",  # Unix-style for git
    )

    log.info(
        "Saved verified counter dataset",
        path=str(verified_path),
        n_counters=len(df),
        year=year,
    )

    return verified_path


def _get_git_commit_hash() -> str:
    """
    Get current git commit hash.

    Returns:
        Short git commit hash (7 chars), or 'unknown' if not in a git repo.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        log.warning("Could not get git commit hash")
        return "unknown"


def init_verified_counters_from_previous_year(
    data_root: Path,
    current_year: int,
    previous_year: int,
    new_traffic_volumes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bootstrap verified counters for a new year from previous year.

    Inherits OSM way assignments (base_id) but updates bicycle volumes
    from new STADTRADELN data. Marks all as 'carryover' status for review.

    Args:
        data_root: Root data directory.
        current_year: Year to create verified dataset for.
        previous_year: Year to inherit from.
        new_traffic_volumes: Traffic volume data for current year.

    Returns:
        DataFrame with initialized verified counters.

    Raises:
        FileNotFoundError: If previous year's verified dataset doesn't exist.
    """
    log.info(
        "Initializing verified counters from previous year",
        current_year=current_year,
        previous_year=previous_year,
    )

    # Load previous year
    prev_verified = load_verified_counters(data_root, previous_year, must_exist=True)

    if prev_verified is None:
        msg = f"Cannot initialize: no verified counters for {previous_year}"
        raise FileNotFoundError(msg)

    # Keep spatial assignments but update volumes
    keep_cols = [
        "counter_id",
        "name",
        "latitude",
        "longitude",
        "ars",
        "base_id",
        "bicycle_infrastructure",
    ]

    new_verified = prev_verified[keep_cols].copy()

    # Join with new traffic volumes to get updated counts
    volumes_subset = new_traffic_volumes[["base_id", "count"]].drop_duplicates(
        subset=["base_id"]
    )

    new_verified = new_verified.merge(volumes_subset, on="base_id", how="left")

    # Add verification metadata
    new_verified["osm_source"] = prev_verified["osm_source"].iloc[0]  # Inherit
    new_verified["verification_status"] = "carryover"
    new_verified["verified_at"] = datetime.now()
    new_verified[
        "verification_metadata"
    ] = f"Inherited from {previous_year}, needs review"

    log.info(
        "Initialized verified counters",
        n_counters=len(new_verified),
        n_with_volumes=new_verified["count"].notna().sum(),
    )

    return new_verified
