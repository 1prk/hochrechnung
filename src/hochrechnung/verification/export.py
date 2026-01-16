"""
Export verification data for the UI.

Creates JSON files with counter data for the verification interface.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


@dataclass
class VerificationExport:
    """
    Verification data export metadata.

    Attributes:
        year: Campaign year.
        n_counters: Total number of counters.
        n_flagged: Number of flagged counters.
        counters_geojson: Path to counters GeoJSON file.
        volumes_mbtiles: Path to volumes MBTiles file.
        median_ratio: Median DTV/volume ratio.
        outlier_threshold_lower: Lower outlier threshold.
        outlier_threshold_upper: Upper outlier threshold.
    """

    year: int
    n_counters: int
    n_flagged: int
    counters_geojson: str
    volumes_mbtiles: str
    median_ratio: float
    outlier_threshold_lower: float
    outlier_threshold_upper: float


def export_verification_data(
    counters_df: "pd.DataFrame",
    output_dir: Path,
    year: int,
    outlier_threshold_lower: float,
    outlier_threshold_upper: float,
    median_ratio: float,
) -> Path:
    """
    Export verification data to JSON.

    Creates a JSON file with counter data for the verification UI.
    Counters are sorted by flag_severity (critical first) and ratio (highest first).

    Args:
        counters_df: DataFrame with counter data including outlier flags.
        output_dir: Output directory.
        year: Campaign year.
        outlier_threshold_lower: Lower outlier threshold.
        outlier_threshold_upper: Upper outlier threshold.
        median_ratio: Median DTV/volume ratio.

    Returns:
        Path to exported JSON file.
    """
    log.info("Exporting verification data", n_counters=len(counters_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification_data.json"

    # Prepare counter data
    counters_export = []

    # Sort: critical first, then warnings, then by ratio descending
    severity_order = {"critical": 0, "warning": 1, "ok": 2}
    df_sorted = counters_df.copy()
    df_sorted["_severity_sort"] = df_sorted.get("flag_severity", "ok").map(
        lambda x: severity_order.get(x, 2)
    )
    df_sorted = df_sorted.sort_values(
        ["_severity_sort", "dtv_volume_ratio"], ascending=[True, False]
    )

    for _, row in df_sorted.iterrows():
        counter_data = {
            "counter_id": str(row["counter_id"]),
            "name": str(row.get("name", "")),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "dtv": float(row["dtv"]) if "dtv" in row and row["dtv"] is not None else None,
            "base_id": int(row["base_id"]) if "base_id" in row and row["base_id"] is not None else None,
            "count": int(row["count"]) if "count" in row and row["count"] is not None else None,
            "bicycle_infrastructure": str(row.get("bicycle_infrastructure", "")),
            "ratio": (
                float(row["dtv_volume_ratio"])
                if "dtv_volume_ratio" in row and row["dtv_volume_ratio"] is not None
                else None
            ),
            "is_outlier": bool(row.get("is_outlier", False)),
            "flag_severity": str(row.get("flag_severity", "ok")),
            "verification_status": str(row.get("verification_status", "unverified")),
            "verification_metadata": str(row.get("verification_metadata", "")),
        }
        counters_export.append(counter_data)

    # Create export metadata
    n_flagged = int(df_sorted.get("is_outlier", pd.Series([False])).sum())

    export_data = {
        "metadata": {
            "year": year,
            "n_counters": len(counters_df),
            "n_flagged": n_flagged,
            "median_ratio": float(median_ratio),
            "outlier_threshold_lower": float(outlier_threshold_lower),
            "outlier_threshold_upper": float(outlier_threshold_upper),
        },
        "counters": counters_export,
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    file_size_kb = output_path.stat().st_size / 1e3
    log.info(
        "Exported verification data",
        path=str(output_path),
        size_kb=round(file_size_kb, 2),
        n_counters=len(counters_export),
        n_flagged=n_flagged,
    )

    return output_path
