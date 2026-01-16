"""
OSM bicycle infrastructure category mapping.

Maps raw OSM infrastructure tags to simplified categories.
"""

import re

import pandas as pd

from hochrechnung.config.settings import PreprocessingConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# Default mapping patterns (regex -> category)
DEFAULT_INFRASTRUCTURE_MAPPING: dict[str, str] = {
    r"^bicycle_lane_.*$": "bicycle_lane",
    r"^bus_lane_.*$": "bicycle_lane",
    r"^bicycle_way_.*$": "bicycle_way",
    r"^mit_road.*$": "mit_road",
    r"^mixed_way.*$": "mixed_way",
    r"^path_not_forbidden$|^pedestrian_both$|^service_misc$": "no",
}

# Valid output categories
VALID_CATEGORIES = [
    "no",
    "mixed_way",
    "mit_road",
    "bicycle_lane",
    "bicycle_road",
    "bicycle_way",
]


def map_infrastructure_category(
    value: str | None,
    mapping: dict[str, str] | None = None,
    default: str = "no",
) -> str:
    """
    Map a single infrastructure value to a category.

    Args:
        value: Raw infrastructure string.
        mapping: Pattern -> category mapping.
        default: Default category for unmatched values.

    Returns:
        Mapped category.
    """
    if value is None or pd.isna(value):
        return default

    mapping = mapping or DEFAULT_INFRASTRUCTURE_MAPPING
    value_str = str(value)

    for pattern, category in mapping.items():
        if re.match(pattern, value_str):
            return category

    # If value is already a valid category, keep it
    if value_str in VALID_CATEGORIES:
        return value_str

    return default


def apply_infrastructure_mapping(
    df: pd.DataFrame,
    column: str = "bicycle_infrastructure",
    output_column: str | None = None,
    config: PreprocessingConfig | None = None,
) -> pd.DataFrame:
    """
    Apply infrastructure category mapping to a DataFrame.

    Args:
        df: DataFrame with infrastructure column.
        column: Name of input column.
        output_column: Name for output column (default: same as input).
        config: Optional preprocessing config for custom mapping.

    Returns:
        DataFrame with mapped infrastructure categories.
    """
    if column not in df.columns:
        log.warning("Infrastructure column not found", column=column)
        return df

    mapping = (
        config.infrastructure_mapping if config else DEFAULT_INFRASTRUCTURE_MAPPING
    )
    output_column = output_column or column

    df = df.copy()

    # Apply mapping
    original_values = df[column].unique()
    df[output_column] = df[column].apply(
        lambda x: map_infrastructure_category(x, mapping)
    )

    # Log mapping results
    new_values = df[output_column].value_counts().to_dict()
    log.info(
        "Applied infrastructure mapping",
        original_unique=len(original_values),
        categories=new_values,
    )

    # Validate all values are in valid categories
    invalid = df[~df[output_column].isin(VALID_CATEGORIES)][output_column].unique()
    if len(invalid) > 0:
        log.warning("Found invalid infrastructure categories", invalid=list(invalid))

    return df


def get_infrastructure_distribution(
    df: pd.DataFrame,
    column: str = "infra_category",
) -> dict[str, int]:
    """
    Get distribution of infrastructure categories.

    Args:
        df: DataFrame with infrastructure column.
        column: Name of infrastructure column.

    Returns:
        Dictionary of category -> count.
    """
    if column not in df.columns:
        return {}
    return df[column].value_counts().to_dict()
