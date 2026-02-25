"""
Curated data loading for model training.

Loads pre-curated Germany-wide counter data with verified OSM assignments
and pre-computed features. This bypasses the regional ETL pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.normalization.spatial import calculate_distances_to_centroids
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


# Column mapping: curated data columns -> model-friendly names
CURATED_TO_MODEL: dict[str, str] = {
    "DZS_mean_SR": "dtv",
    "Erh_SR": "stadtradeln_volume",
    "OSM_Radinfra_Simple": "infra_category",
    "RegioStaR7": "regiostar7",
    "N_USERS": "n_users",
    "Bev_insg": "population",
    "lat": "latitude",
    "lon": "longitude",
    "Counter_ID": "counter_id",
    "OSM-ID": "base_id",
}

# Features available in curated data (after derivation)
CURATED_FEATURES = [
    "stadtradeln_volume",
    "route_intensity",
    "participation_rate",
    "infra_category",
    "regiostar7",
    "dist_to_center_m",
]


@dataclass
class CuratedTrainingData:
    """Container for curated training data with metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    n_samples: int
    target_stats: dict[str, float]
    year: int


def load_curated_data(
    path: Path,
    config: PipelineConfig,
    centroids_path: Path,
    *,
    year: int | None = None,
) -> CuratedTrainingData:
    """
    Load pre-curated Germany-wide counter data for model training.

    Args:
        path: Path to curated CSV file.
        config: Pipeline configuration.
        centroids_path: Path to city centroids GeoPackage.
        year: Filter to specific year. If None, uses config.year.

    Returns:
        CuratedTrainingData with train/test splits and metadata.

    Raises:
        FileNotFoundError: If data or centroids file not found.
        ValueError: If required columns missing or no valid data.
    """
    if not path.exists():
        msg = f"Curated data file not found: {path}"
        raise FileNotFoundError(msg)

    if not centroids_path.exists():
        msg = f"Centroids file not found: {centroids_path}"
        raise FileNotFoundError(msg)

    target_year = year if year is not None else config.year
    log.info("Loading curated data", path=str(path), year=target_year)

    # Load curated data (semicolon-delimited, Latin-1 encoded for German umlauts)
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    log.info("Loaded raw curated data", rows=len(df), columns=len(df.columns))

    # Filter by year
    if "year" in df.columns:
        df = df.loc[df["year"] == target_year].copy()
        log.info("Filtered by year", year=target_year, rows=len(df))
    else:
        log.warning("No 'year' column in curated data, using all rows")

    if len(df) == 0:
        msg = f"No data for year {target_year}"
        raise ValueError(msg)

    # Rename columns to model-friendly names
    df = _rename_columns(cast("pd.DataFrame", df))

    # Derive features
    df = _derive_features(df)

    # Calculate distance to centroids
    df = _calculate_centroids_distance(df, centroids_path)

    # Simplify infrastructure categories
    df = _simplify_infrastructure(df, config)

    # Filter by DTV bounds
    df = _filter_by_dtv(df, config)

    # Validate required columns
    _validate_columns(df)

    # Prepare X and y
    feature_cols = [f for f in CURATED_FEATURES if f in df.columns]
    X = df[feature_cols].copy()
    y: pd.Series[Any] = df["dtv"].copy()  # type: ignore[assignment]

    # Split data (or use all data for CV-only mode)
    if config.training.test_size > 0:
        split_result = train_test_split(
            X,
            y,
            test_size=config.training.test_size,
            random_state=config.training.random_state,
        )
        X_train = cast("pd.DataFrame", split_result[0])
        X_test = cast("pd.DataFrame", split_result[1])
        y_train = cast("pd.Series", split_result[2])
        y_test = cast("pd.Series", split_result[3])
    else:
        # CV-only: all data used for training, no holdout
        X_train = X
        X_test = cast("pd.DataFrame", X.iloc[:0])
        y_train = y
        y_test = cast("pd.Series", y.iloc[:0])

    # Compute target statistics
    target_stats = _compute_target_stats(y)

    log.info(
        "Prepared curated training data",
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(feature_cols),
        target_mean=f"{target_stats['mean']:.2f}",
    )

    return CuratedTrainingData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        n_samples=len(df),
        target_stats=target_stats,
        year=target_year,
    )


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename curated columns to model-friendly names."""
    rename_dict = {k: v for k, v in CURATED_TO_MODEL.items() if k in df.columns}

    if rename_dict:
        log.debug("Renaming columns", mapping=rename_dict)
        df = df.rename(columns=rename_dict)

    return df


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive computed features from raw columns."""
    # participation_rate = n_users / population
    if "n_users" in df.columns and "population" in df.columns:
        df["participation_rate"] = df["n_users"] / df["population"]
        # Handle division by zero
        df["participation_rate"] = df["participation_rate"].replace(
            [np.inf, -np.inf], np.nan
        )
        log.debug("Derived participation_rate")

    # route_intensity = (n_users * stadtradeln_volume) / population
    if all(c in df.columns for c in ["n_users", "stadtradeln_volume", "population"]):
        df["route_intensity"] = (df["n_users"] * df["stadtradeln_volume"]) / df[
            "population"
        ]
        # Handle division by zero
        df["route_intensity"] = df["route_intensity"].replace([np.inf, -np.inf], np.nan)
        log.debug("Derived route_intensity")

    return df


def _calculate_centroids_distance(
    df: pd.DataFrame, centroids_path: Path
) -> pd.DataFrame:
    """Calculate distance from each counter to nearest city centroid."""
    if "latitude" not in df.columns or "longitude" not in df.columns:
        log.warning("Missing lat/lon columns, skipping centroid distance calculation")
        return df

    # Load centroids
    centroids = gpd.read_file(centroids_path)
    log.info("Loaded centroids", n_centroids=len(centroids))

    # Create GeoDataFrame from counter locations
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Calculate distances
    gdf = calculate_distances_to_centroids(gdf, centroids)

    # Copy distance back to DataFrame
    df["dist_to_center_m"] = gdf["dist_to_center_m"].values

    log.info(
        "Calculated centroid distances",
        mean_distance=float(df["dist_to_center_m"].mean()),
    )

    return df


def _simplify_infrastructure(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Simplify infrastructure categories using config mapping."""
    import re

    if "infra_category" not in df.columns:
        log.warning("No infra_category column, skipping simplification")
        return df

    # Apply regex-based mapping from config
    mapping = config.preprocessing.infrastructure_mapping

    def map_infra(value: Any) -> str:
        if value is None:
            return "no"
        if isinstance(value, float) and np.isnan(value):  # type: ignore[arg-type]
            return "no"
        value_str = str(value).lower()
        for pattern, category in mapping.items():
            if re.match(pattern, value_str, re.IGNORECASE):
                return category
        return "no"

    df["infra_category"] = df["infra_category"].apply(map_infra)

    return df


def _filter_by_dtv(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Filter data by min/max DTV bounds from config."""
    original_len = len(df)

    # Filter out NaN values in target
    result: pd.DataFrame = df.loc[df["dtv"].notna()].copy()  # type: ignore[assignment]

    # Filter out NaN/invalid stadtradeln_volume (required feature)
    if "stadtradeln_volume" in result.columns:
        result = result.loc[result["stadtradeln_volume"].notna()].copy()  # type: ignore[assignment]
        result = result.loc[result["stadtradeln_volume"] > 0].copy()  # type: ignore[assignment]

    # Apply min_dtv filter
    result = result.loc[result["dtv"] >= config.training.min_dtv].copy()  # type: ignore[assignment]

    # Apply max_dtv filter (only if specified)
    if config.training.max_dtv is not None:
        result = result.loc[result["dtv"] <= config.training.max_dtv].copy()  # type: ignore[assignment]

    filtered_len = len(result)
    if filtered_len < original_len:
        log.info(
            "Filtered by DTV bounds",
            original=original_len,
            filtered=filtered_len,
            min_dtv=config.training.min_dtv,
            max_dtv=config.training.max_dtv,
        )

    return result


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    required = ["dtv", "stadtradeln_volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)


def _compute_target_stats(y: pd.Series) -> dict[str, float]:
    """Compute statistics about the target variable."""
    return {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "median": float(y.median()),
        "q25": float(np.percentile(y, 25)),
        "q75": float(np.percentile(y, 75)),
    }
