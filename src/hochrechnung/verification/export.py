"""
Export verification data for the UI.

Creates JSON files with counter data for the verification interface.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


def _is_valid(value: Any) -> bool:
    """Check if a value is valid (not NaN/None)."""
    return bool(pd.notna(value))


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize data for JSON serialization.

    Replaces NaN/Infinity with None (becomes null in JSON).
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    # Handle numpy types
    if hasattr(obj, "item"):  # numpy scalar
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    return obj


# Severity display configuration
SEVERITY_CONFIG = {
    "critical": {"color": "#dc3545", "label": "Critical", "priority": 0},
    "ambiguous": {"color": "#6f42c1", "label": "Ambiguous", "priority": 1},
    "no_volume": {"color": "#0d6efd", "label": "No Volume", "priority": 2},
    "warning": {"color": "#ffc107", "label": "Warning", "priority": 3},
    "campaign_bias": {"color": "#fd7e14", "label": "Campaign Bias", "priority": 4},
    "carryover": {"color": "#6c757d", "label": "Carryover", "priority": 5},
    "ok": {"color": "#6c757d", "label": "OK", "priority": 6},
    "verified": {"color": "#28a745", "label": "Verified", "priority": 7},
}


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
        n_by_severity: Count of counters per severity level.
    """

    year: int
    n_counters: int
    n_flagged: int
    counters_geojson: str
    volumes_mbtiles: str
    median_ratio: float
    outlier_threshold_lower: float
    outlier_threshold_upper: float
    n_by_severity: dict[str, int] | None = None


def export_verification_data(
    counters_df: "pd.DataFrame",
    output_dir: Path,
    year: int,
    outlier_threshold_lower: float,
    outlier_threshold_upper: float,
    median_ratio: float,
    n_by_severity: dict[str, int] | None = None,
    images_db_path: Path | None = None,
) -> Path:
    """
    Export verification data to JSON.

    Creates a JSON file with counter data for the verification UI.
    Counters are sorted by flag_severity priority and ratio.

    Args:
        counters_df: DataFrame with counter data including outlier flags.
        output_dir: Output directory.
        year: Campaign year.
        outlier_threshold_lower: Lower outlier threshold.
        outlier_threshold_upper: Upper outlier threshold.
        median_ratio: Median DTV/volume ratio.
        n_by_severity: Count of counters per severity level.

    Returns:
        Path to exported JSON file.
    """
    log.info("Exporting verification data", n_counters=len(counters_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification_data.json"

    # Build image count lookup from SQLite database
    image_counts: dict[str, int] = {}
    if images_db_path and images_db_path.exists():
        import sqlite3

        try:
            conn = sqlite3.connect(images_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT counter_id, COUNT(*) FROM images GROUP BY counter_id"
            )
            for row in cursor.fetchall():
                image_counts[str(row[0])] = row[1]
            conn.close()
            log.info("Loaded image counts", n_counters_with_images=len(image_counts))
        except Exception as e:
            log.warning("Failed to load image counts", error=str(e))

    # Prepare counter data
    counters_export = []

    # Sort by severity priority (from SEVERITY_CONFIG), then by ratio descending
    df_sorted = counters_df.copy()
    flag_severity_col = (
        df_sorted["flag_severity"]
        if "flag_severity" in df_sorted.columns
        else pd.Series(["ok"] * len(df_sorted))
    )
    df_sorted["_severity_sort"] = flag_severity_col.map(
        lambda x: SEVERITY_CONFIG.get(str(x), {}).get("priority", 99)
    )
    df_sorted = df_sorted.sort_values(
        ["_severity_sort", "dtv_volume_ratio"], ascending=[True, False]
    )

    for _, row in df_sorted.iterrows():
        # Original numeric ID (Counter_ID_orig) for image DB lookups
        original_id = str(row["id"]) if "id" in row and _is_valid(row.get("id")) else None

        counter_data = {
            "counter_id": str(row["counter_id"]),
            "original_id": original_id,
            "name": str(row.get("name", "")),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "dtv": float(row["dtv"])
            if "dtv" in row and _is_valid(row["dtv"])
            else None,
            "base_id": (
                int(row["base_id"])
                if "base_id" in row and _is_valid(row["base_id"])
                else None
            ),
            "count": (
                int(row["count"])
                if "count" in row and _is_valid(row["count"])
                else None
            ),
            "bicycle_infrastructure": (
                str(row["bicycle_infrastructure"])
                if _is_valid(row.get("bicycle_infrastructure"))
                else ""
            ),
            "ratio": (
                float(row["dtv_volume_ratio"])
                if "dtv_volume_ratio" in row and _is_valid(row["dtv_volume_ratio"])
                else None
            ),
            "is_outlier": bool(row.get("is_outlier", False)),
            "flag_severity": str(row.get("flag_severity", "ok")),
            "verification_status": str(row.get("verification_status", "unverified")),
            "verification_metadata": str(row.get("verification_metadata", "")),
            # New fields for extended verification
            "has_no_volume": bool(row.get("has_no_volume", False)),
            "has_campaign_bias": bool(row.get("has_campaign_bias", False)),
            "is_ambiguous": bool(row.get("is_ambiguous", False)),
            "is_discarded": bool(row.get("is_discarded", False)),
            "candidate_edges": (
                row["candidate_edges"]
                if "candidate_edges" in row and row["candidate_edges"] is not None
                else []
            ),
            "image_count": image_counts.get(original_id, 0) if original_id else 0,
        }
        counters_export.append(counter_data)

    # Create export metadata
    # Count flagged as anything not 'ok' or 'verified'
    if "flag_severity" in df_sorted.columns:
        flagged_count = len(
            df_sorted[~df_sorted["flag_severity"].isin(["ok", "verified"])]
        )
    else:
        flagged_count = int(
            df_sorted["is_outlier"].sum() if "is_outlier" in df_sorted.columns else 0
        )

    # Calculate severity breakdown if not provided
    if n_by_severity is None and "flag_severity" in df_sorted.columns:
        n_by_severity = df_sorted["flag_severity"].value_counts().to_dict()

    export_data = {
        "metadata": {
            "year": year,
            "n_counters": len(counters_df),
            "n_flagged": flagged_count,
            "median_ratio": float(median_ratio),
            "outlier_threshold_lower": float(outlier_threshold_lower),
            "outlier_threshold_upper": float(outlier_threshold_upper),
            "n_by_severity": n_by_severity or {},
        },
        "severity_config": SEVERITY_CONFIG,
        "counters": counters_export,
    }

    # Sanitize and write JSON (NaN/Infinity -> null for JS compatibility)
    sanitized_data = _sanitize_for_json(export_data)
    with output_path.open("w") as f:
        json.dump(sanitized_data, f, indent=2, allow_nan=False)

    file_size_kb = output_path.stat().st_size / 1e3
    log.info(
        "Exported verification data",
        path=str(output_path),
        size_kb=round(file_size_kb, 2),
        n_counters=len(counters_export),
        n_flagged=flagged_count,
    )

    return output_path
