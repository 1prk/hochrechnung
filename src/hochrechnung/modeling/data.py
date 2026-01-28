"""
Training data loading and preparation.

Loads ETL output, maps columns to model-friendly names, and prepares
train/test splits for model training.

Supports two model variants:
- baseline: Single predictor (stadtradeln_volume only) - y ~ x
- enhanced: All configured features - y ~ x1, x2, ...
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


class ModelVariant(str, Enum):
    """Model variant type for training."""

    BASELINE = "baseline"  # Single predictor: y ~ stadtradeln_volume
    ENHANCED = "enhanced"  # All features: y ~ x1, x2, ...


# Baseline model uses only the STADTRADELN volume as predictor
BASELINE_FEATURE = "stadtradeln_volume"
BASELINE_FEATURE_ETL = "Erh_SR"  # ETL legacy name

# Enhanced model features (all predictors from original model)
# These are the ETL column names - will be renamed if rename_columns=True
ENHANCED_FEATURES_ETL = [
    "Erh_SR",  # stadtradeln_volume - Box-Cox transformed
    "Streckengewicht_SR",  # route_intensity - StandardScaler
    "TN_SR_relativ",  # participation_rate - RobustScaler
    "OSM_Radinfra",  # infra_category - OrdinalEncoder
    "RegioStaR5",  # regiostar5
    "HubDist",  # dist_to_center_m
]

# Enhanced model features (model-friendly names after renaming)
ENHANCED_FEATURES = [
    "stadtradeln_volume",
    "route_intensity",
    "participation_rate",
    "infra_category",
    "regiostar5",
    "dist_to_center_m",
]

# Column mapping: ETL legacy names -> model feature names
ETL_TO_MODEL_COLUMNS: dict[str, str] = {
    "OSM_Radinfra": "infra_category",
    "TN_SR_relativ": "participation_rate",
    "Streckengewicht_SR": "route_intensity",
    "RegioStaR5": "regiostar5",
    "Erh_SR": "stadtradeln_volume",
    "HubDist": "dist_to_center_m",
    "DZS_mean_SR": "dtv",
}

# Reverse mapping for reference
MODEL_TO_ETL_COLUMNS: dict[str, str] = {v: k for k, v in ETL_TO_MODEL_COLUMNS.items()}


@dataclass
class TrainingData:
    """
    Container for training data with metadata.

    Attributes:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
        feature_names: List of feature column names.
        n_samples: Total number of samples.
        target_stats: Statistics about the target variable.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    n_samples: int
    target_stats: dict[str, float]


def load_training_data(
    path: Path,
    config: PipelineConfig,
    *,
    rename_columns: bool = True,
    variant: ModelVariant = ModelVariant.ENHANCED,
) -> TrainingData:
    """
    Load training data from ETL output CSV.

    Args:
        path: Path to ETL output CSV file.
        config: Pipeline configuration.
        rename_columns: Whether to rename ETL columns to model-friendly names.
        variant: Model variant (baseline or enhanced).

    Returns:
        TrainingData with train/test splits and metadata.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        msg = f"Training data file not found: {path}"
        raise FileNotFoundError(msg)

    log.info("Loading training data", path=str(path), variant=variant.value)
    df = pd.read_csv(path)
    log.info("Loaded raw data", rows=len(df), columns=list(df.columns))

    # Rename columns if requested
    if rename_columns:
        df = _rename_columns(df)

    # Determine target column
    target_col = _get_target_column(df, config)

    # Filter by DTV bounds
    df = _filter_by_dtv(df, target_col, config)

    # Get feature columns based on variant
    feature_cols = _get_feature_columns(df, target_col, config, variant=variant)

    # Validate required columns exist
    _validate_columns(df, feature_cols, target_col)

    # Drop rows with NaN in features (e.g., counters matched to edges with no volume)
    before_dropna = len(df)
    df = df.dropna(subset=feature_cols)
    after_dropna = len(df)
    if after_dropna < before_dropna:
        log.info(
            "Dropped rows with missing features",
            dropped=before_dropna - after_dropna,
            remaining=after_dropna,
        )

    # Prepare X and y
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
    )

    # Compute target statistics
    target_stats = _compute_target_stats(y)

    log.info(
        "Prepared training data",
        n_train=len(X_train),
        n_test=len(X_test),
        n_features=len(feature_cols),
        target_mean=f"{target_stats['mean']:.2f}",
    )

    return TrainingData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        n_samples=len(df),
        target_stats=target_stats,
    )


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ETL columns to model-friendly names."""
    rename_dict = {k: v for k, v in ETL_TO_MODEL_COLUMNS.items() if k in df.columns}

    if rename_dict:
        log.debug("Renaming columns", mapping=rename_dict)
        df = df.rename(columns=rename_dict)

    return df


def _get_target_column(df: pd.DataFrame, config: PipelineConfig) -> str:
    """Determine the target column name."""
    # Try model-friendly name first
    if "dtv" in df.columns:
        return "dtv"

    # Try ETL legacy name
    if "DZS_mean_SR" in df.columns:
        return "DZS_mean_SR"

    # Try config target column
    if config.preprocessing.target_column in df.columns:
        return config.preprocessing.target_column

    msg = "Target column not found. Expected 'dtv' or 'DZS_mean_SR'"
    raise ValueError(msg)


def _filter_by_dtv(
    df: pd.DataFrame,
    target_col: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Filter data by min/max DTV bounds from config."""
    original_len = len(df)

    # Filter out NaN values in target
    df = df[df[target_col].notna()].copy()

    # Apply min_dtv filter (min_dtv is always set, default 25)
    df = df[df[target_col] >= config.training.min_dtv].copy()

    # Apply max_dtv filter (only if specified)
    if config.training.max_dtv is not None:
        df = df[df[target_col] <= config.training.max_dtv].copy()

    filtered_len = len(df)
    if filtered_len < original_len:
        log.info(
            "Filtered by DTV bounds",
            original=original_len,
            filtered=filtered_len,
            min_dtv=config.training.min_dtv,
            max_dtv=config.training.max_dtv,
        )

    return df


def _get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    _config: PipelineConfig,
    *,
    variant: ModelVariant = ModelVariant.ENHANCED,
) -> list[str]:
    """
    Get list of feature columns to use based on model variant.

    Args:
        df: DataFrame with available columns.
        target_col: Target column name (to exclude).
        config: Pipeline configuration.
        variant: Model variant (baseline uses single feature, enhanced uses all).

    Returns:
        List of feature column names.
    """
    # Baseline model: single predictor only
    if variant == ModelVariant.BASELINE:
        # Try model-friendly name first, then ETL legacy name
        if BASELINE_FEATURE in df.columns:
            return [BASELINE_FEATURE]
        if BASELINE_FEATURE_ETL in df.columns:
            return [BASELINE_FEATURE_ETL]
        msg = (
            f"Baseline feature not found. Expected '{BASELINE_FEATURE}' "
            f"or '{BASELINE_FEATURE_ETL}'"
        )
        raise ValueError(msg)

    # Enhanced model: use explicit feature list from original model
    # Try model-friendly names first (after column renaming)
    available_enhanced = [f for f in ENHANCED_FEATURES if f in df.columns]
    if len(available_enhanced) == len(ENHANCED_FEATURES):
        return available_enhanced

    # Try ETL legacy names (before column renaming)
    available_etl = [f for f in ENHANCED_FEATURES_ETL if f in df.columns]
    if len(available_etl) == len(ENHANCED_FEATURES_ETL):
        return available_etl

    # Mixed: some columns renamed, some not - collect what's available
    available_features = []
    for model_name, etl_name in zip(ENHANCED_FEATURES, ENHANCED_FEATURES_ETL):
        if model_name in df.columns:
            available_features.append(model_name)
        elif etl_name in df.columns:
            available_features.append(etl_name)

    if available_features:
        log.warning(
            "Not all enhanced features available",
            expected=len(ENHANCED_FEATURES),
            found=len(available_features),
            features=available_features,
        )
        return available_features

    # Fallback: use all non-metadata columns
    exclude_cols = {
        target_col,
        "id",
        "base_id",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "name",
        "counter_id",
        "geometry",
    }

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    log.warning(
        "Using fallback feature selection",
        n_features=len(feature_cols),
    )

    return feature_cols


def _validate_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> None:
    """Validate that required columns exist."""
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        msg = f"Missing feature columns: {missing_features}"
        raise ValueError(msg)

    if target_col not in df.columns:
        msg = f"Missing target column: {target_col}"
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


def auto_detect_data_path(config: PipelineConfig) -> Path:
    """
    Auto-detect ETL output path from config.

    Looks for training_data_{year}.csv in the cache directory.

    Args:
        config: Pipeline configuration.

    Returns:
        Path to detected data file.

    Raises:
        FileNotFoundError: If no data file is found.
    """
    cache_dir = config.cache_dir
    expected_path = cache_dir / f"training_data_{config.year}.csv"

    if expected_path.exists():
        return expected_path

    # Try without year suffix
    fallback_path = cache_dir / "training_data.csv"
    if fallback_path.exists():
        return fallback_path

    msg = (
        f"No training data found. Expected: {expected_path}\n"
        f"Run ETL first: uv run hochrechnung etl --config <config.yaml>"
    )
    raise FileNotFoundError(msg)
