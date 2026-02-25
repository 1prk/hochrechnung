"""
Export calibration data for the verification UI.

Creates JSON files with calibration station data for manual validation.
Adapts the verification module's export format for calibration purposes.
"""

import json
import math
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


# Severity configuration for calibration stations
# Simpler than training counter verification - mainly checking location/DTV validity
CALIBRATION_SEVERITY_CONFIG = {
    "critical": {"color": "#dc3545", "label": "Critical", "priority": 0},
    "warning": {"color": "#ffc107", "label": "Warning", "priority": 1},
    "ok": {"color": "#6c757d", "label": "OK", "priority": 2},
    "verified": {"color": "#28a745", "label": "Verified", "priority": 3},
}


def flag_calibration_stations(
    df: pd.DataFrame,
    predicted_dtv: pd.Series | None = None,
    dtv_column: str = "dtv",
) -> pd.DataFrame:
    """
    Flag calibration stations for potential issues.

    Checks for:
    - Missing DTV values
    - Suspicious DTV values (very low or very high)
    - Large prediction residuals (if predictions provided)

    Args:
        df: DataFrame with calibration station data.
        predicted_dtv: Optional predicted DTV values for comparison.
        dtv_column: Name of the DTV column.

    Returns:
        DataFrame with flag_severity and flag_reason columns added.
    """
    df = df.copy()

    # Initialize flags
    df["flag_severity"] = "ok"
    df["flag_reason"] = ""

    # Check for missing DTV
    missing_dtv = df[dtv_column].isna()
    df.loc[missing_dtv, "flag_severity"] = "critical"
    df.loc[missing_dtv, "flag_reason"] = "Missing DTV value"

    # Check for very low DTV (< 25)
    low_dtv = (df[dtv_column] < 25) & ~missing_dtv
    df.loc[low_dtv, "flag_severity"] = "warning"
    df.loc[low_dtv, "flag_reason"] = "Very low DTV (< 25)"

    # Check for very high DTV (> 10000)
    high_dtv = (df[dtv_column] > 10000) & ~missing_dtv
    df.loc[high_dtv, "flag_severity"] = "warning"
    df.loc[high_dtv, "flag_reason"] = "Very high DTV (> 10000)"

    # If predictions provided, check for large residuals
    if predicted_dtv is not None:
        df["predicted_dtv"] = predicted_dtv

        # Calculate residual ratio (actual / predicted)
        with pd.option_context("mode.use_inf_as_na", True):
            ratio = df[dtv_column] / predicted_dtv
            df["dtv_pred_ratio"] = ratio

            # Flag extreme ratios (more than 3x or less than 0.33x)
            extreme_ratio = (ratio > 3.0) | (ratio < 0.33)
            extreme_ratio = extreme_ratio & ~missing_dtv

            # Only upgrade to warning if currently ok
            upgrade_mask = extreme_ratio & (df["flag_severity"] == "ok")
            df.loc[upgrade_mask, "flag_severity"] = "warning"
            df.loc[upgrade_mask, "flag_reason"] = (
                "Large prediction discrepancy (ratio > 3x or < 0.33x)"
            )

    return df


def export_calibration_data(
    stations_df: pd.DataFrame,
    output_dir: Path,
    year: int,
    region: str,
    dtv_column: str = "dtv",
    id_column: str | None = None,
) -> Path:
    """
    Export calibration station data to JSON for verification UI.

    Args:
        stations_df: DataFrame with calibration station data.
        output_dir: Output directory.
        year: Campaign year.
        region: Region name.
        dtv_column: Name of the DTV column.
        id_column: Name of the ID column (auto-detected if None).

    Returns:
        Path to exported JSON file.
    """
    log.info("Exporting calibration data", n_stations=len(stations_df))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification_data.json"

    # Auto-detect ID column
    if id_column is None:
        id_candidates = ["id", "counter_id", "station_id", "DZS_id", "Zaehlst_id"]
        for col in id_candidates:
            if col in stations_df.columns:
                id_column = col
                break
        if id_column is None:
            # Use index as ID
            stations_df = stations_df.copy()
            stations_df["id"] = stations_df.index.astype(str)
            id_column = "id"

    # Auto-detect coordinate columns
    lon_col = "longitude" if "longitude" in stations_df.columns else "lon"
    lat_col = "latitude" if "latitude" in stations_df.columns else "lat"

    # Auto-detect name column
    name_col = None
    name_candidates = ["name", "Name", "DZS_name", "Zaehlstellenname"]
    for col in name_candidates:
        if col in stations_df.columns:
            name_col = col
            break

    # Prepare counter data for export
    counters_export = []

    # Sort by severity priority if flagged, then by DTV descending
    df_sorted = stations_df.copy()
    if "flag_severity" in df_sorted.columns:
        df_sorted["_severity_sort"] = df_sorted["flag_severity"].map(
            lambda x: CALIBRATION_SEVERITY_CONFIG.get(str(x), {}).get("priority", 99)
        )
        df_sorted = df_sorted.sort_values(
            ["_severity_sort", dtv_column], ascending=[True, False]
        )
    else:
        df_sorted = df_sorted.sort_values(dtv_column, ascending=False)

    for _, row in df_sorted.iterrows():
        counter_data = {
            "counter_id": str(row[id_column]),
            "name": str(row[name_col]) if name_col and _is_valid(row.get(name_col)) else "",
            "latitude": float(row[lat_col]) if _is_valid(row.get(lat_col)) else None,
            "longitude": float(row[lon_col]) if _is_valid(row.get(lon_col)) else None,
            "dtv": float(row[dtv_column]) if _is_valid(row.get(dtv_column)) else None,
            # Verification fields
            "flag_severity": str(row.get("flag_severity", "ok")),
            "flag_reason": str(row.get("flag_reason", "")),
            "verification_status": str(row.get("verification_status", "unverified")),
            "verification_metadata": str(row.get("verification_metadata", "")),
            "is_discarded": bool(row.get("is_discarded", False)),
            # Calibration-specific fields
            "predicted_dtv": (
                float(row["predicted_dtv"])
                if "predicted_dtv" in row and _is_valid(row["predicted_dtv"])
                else None
            ),
            "dtv_pred_ratio": (
                float(row["dtv_pred_ratio"])
                if "dtv_pred_ratio" in row and _is_valid(row["dtv_pred_ratio"])
                else None
            ),
        }

        # Include any additional columns that might be useful
        for extra_col in ["bicycle_infrastructure", "infra_category", "regiostar7"]:
            if extra_col in row:
                counter_data[extra_col] = (
                    str(row[extra_col]) if _is_valid(row[extra_col]) else ""
                )

        counters_export.append(counter_data)

    # Calculate severity breakdown
    n_by_severity = {}
    if "flag_severity" in df_sorted.columns:
        n_by_severity = df_sorted["flag_severity"].value_counts().to_dict()
        n_flagged = len(
            df_sorted[~df_sorted["flag_severity"].isin(["ok", "verified"])]
        )
    else:
        n_flagged = 0

    # Calculate median ratio (for UI compatibility)
    median_ratio = 1.0
    if "dtv_pred_ratio" in df_sorted.columns:
        valid_ratios = df_sorted["dtv_pred_ratio"].dropna()
        if len(valid_ratios) > 0:
            median_ratio = float(valid_ratios.median())

    export_data = {
        "metadata": {
            "year": year,
            "region": region,
            "n_counters": len(stations_df),
            "n_flagged": n_flagged,
            "mode": "calibration",
            "n_by_severity": n_by_severity,
            # Fields expected by verification UI
            "median_ratio": median_ratio,
            "outlier_threshold_lower": 0.33,  # Used for flagging
            "outlier_threshold_upper": 3.0,   # Used for flagging
        },
        "severity_config": CALIBRATION_SEVERITY_CONFIG,
        "counters": counters_export,
    }

    # Sanitize and write JSON
    sanitized_data = _sanitize_for_json(export_data)
    with output_path.open("w") as f:
        json.dump(sanitized_data, f, indent=2, allow_nan=False)

    file_size_kb = output_path.stat().st_size / 1e3
    log.info(
        "Exported calibration data",
        path=str(output_path),
        size_kb=round(file_size_kb, 2),
        n_stations=len(counters_export),
        n_flagged=n_flagged,
    )

    return output_path
