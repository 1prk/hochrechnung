"""
Column name normalization.

Provides canonical column naming to ensure consistency throughout the pipeline.
"""

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# Canonical column names for the pipeline
# Maps various source names to standardized internal names
COLUMN_MAPPING: dict[str, str] = {
    # Population
    "Einwohnerzahl_EWZ": "population",
    "EWZ": "population",
    "ewz": "population",
    "Bev_insg": "population",
    # Traffic counts
    "count": "stadtradeln_volume",
    "n": "stadtradeln_volume",
    "Erh_SR": "stadtradeln_volume",
    # Infrastructure
    "bicycle_infrastructure": "infra_category",
    "OSM_Radinfra": "infra_category",
    # Identifiers
    "Regionalschlüssel_ARS": "ars",
    "ARS": "ars",
    "ags": "ars",
    # Geographic
    "GeografischerName_GEN": "municipality_name",
    "GEN": "municipality_name",
    # STADTRADELN metrics
    "N_USERS": "n_users",
    "N_TRIPS": "n_trips",
    "TOTAL_KM": "total_km",
    # Regional classification
    "RegioStaR5": "regiostar5",
    "RegioStaR7": "regiostar7",
    # Distance features
    "dist_to_centroid_m": "dist_to_center_m",
    # Target
    "dtv_value": "dtv",
    "DZS_mean_SR": "dtv",
    "DTV": "dtv",
}

# Reverse mapping for export (internal -> common external)
EXPORT_MAPPING: dict[str, str] = {
    "population": "Bev_insg",
    "stadtradeln_volume": "Erh_SR",
    "infra_category": "OSM_Radinfra",
    "dtv": "DZS_mean_SR",
}


def normalize_columns(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Normalize column names to canonical form.

    Args:
        df: DataFrame to normalize.
        mapping: Optional custom mapping (defaults to COLUMN_MAPPING).

    Returns:
        DataFrame with normalized column names.
    """
    mapping = mapping or COLUMN_MAPPING

    # Build rename dict for columns that exist
    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}

    if rename_dict:
        log.debug("Normalizing columns", renamed=list(rename_dict.keys()))
        df = df.rename(columns=rename_dict)

    return df


def export_columns(
    df: pd.DataFrame,
    mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Convert canonical column names back to export format.

    Args:
        df: DataFrame with canonical columns.
        mapping: Optional custom mapping (defaults to EXPORT_MAPPING).

    Returns:
        DataFrame with export column names.
    """
    mapping = mapping or EXPORT_MAPPING

    rename_dict = {k: v for k, v in mapping.items() if k in df.columns}

    if rename_dict:
        log.debug("Exporting columns", renamed=list(rename_dict.keys()))
        df = df.rename(columns=rename_dict)

    return df


def validate_required_columns(
    df: pd.DataFrame,
    required: list[str],
    *,
    raise_on_missing: bool = True,
) -> list[str]:
    """
    Check that required columns are present.

    Args:
        df: DataFrame to check.
        required: List of required column names.
        raise_on_missing: Whether to raise error if columns missing.

    Returns:
        List of missing columns.

    Raises:
        ValueError: If raise_on_missing and columns are missing.
    """
    missing = [col for col in required if col not in df.columns]

    if missing and raise_on_missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    if missing:
        log.warning("Missing columns", missing=missing)

    return missing
