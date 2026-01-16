"""
Outlier detection for counter verification.

Calculates DTV/volume ratios and flags counters with suspicious assignments.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class OutlierResult:
    """
    Result of outlier detection.

    Attributes:
        n_counters: Total number of counters analyzed.
        n_flagged: Number of counters flagged as outliers.
        threshold_lower: Lower bound of ratio threshold.
        threshold_upper: Upper bound of ratio threshold.
        median_ratio: Median DTV/volume ratio.
    """

    n_counters: int
    n_flagged: int
    threshold_lower: float
    threshold_upper: float
    median_ratio: float


def calculate_ratios(
    counters_df: pd.DataFrame,
    dtv_col: str = "dtv",
    volume_col: str = "count",
    ratio_col: str = "dtv_volume_ratio",
) -> pd.DataFrame:
    """
    Calculate DTV/bicycle_volume ratio for each counter.

    Args:
        counters_df: DataFrame with counter data.
        dtv_col: Name of DTV column.
        volume_col: Name of bicycle volume column.
        ratio_col: Name for output ratio column.

    Returns:
        DataFrame with added ratio column.
    """
    df = counters_df.copy()

    # Handle division by zero
    df[ratio_col] = df[dtv_col] / df[volume_col].replace(0, np.nan)

    # Handle infinite values (volume = 0 but DTV > 0)
    df.loc[np.isinf(df[ratio_col]), ratio_col] = np.nan

    log.info(
        "Calculated DTV/volume ratios",
        n_counters=len(df),
        n_valid=df[ratio_col].notna().sum(),
        median_ratio=float(df[ratio_col].median()),
    )

    return df


def flag_outliers(
    counters_df: pd.DataFrame,
    ratio_col: str = "dtv_volume_ratio",
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    *,
    skip_verified: bool = True,
) -> tuple[pd.DataFrame, OutlierResult]:
    """
    Flag counters with outlier ratios.

    Uses IQR (Interquartile Range) method by default for robust outlier detection.

    Args:
        counters_df: DataFrame with ratio column.
        ratio_col: Name of ratio column.
        method: Method for outlier detection ('iqr' or 'zscore').
        iqr_multiplier: Multiplier for IQR (1.5 = standard, 3.0 = extreme only).
        skip_verified: If True, don't flag counters with verification_status='verified'.

    Returns:
        Tuple of (DataFrame with outlier flags, OutlierResult summary).
    """
    df = counters_df.copy()

    # Initialize flags
    df["is_outlier"] = False
    df["flag_severity"] = "ok"

    # Get valid ratios
    valid_ratios = df[ratio_col].dropna()

    if len(valid_ratios) == 0:
        log.warning("No valid ratios to analyze for outliers")
        result = OutlierResult(
            n_counters=len(df),
            n_flagged=0,
            threshold_lower=0.0,
            threshold_upper=0.0,
            median_ratio=0.0,
        )
        return df, result

    if method == "iqr":
        # IQR method (robust to extreme outliers)
        Q1 = valid_ratios.quantile(0.25)
        Q3 = valid_ratios.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        # Flag outliers
        is_outlier = (df[ratio_col] < lower_bound) | (df[ratio_col] > upper_bound)

    elif method == "zscore":
        # Z-score method (assumes normal distribution)
        mean = valid_ratios.mean()
        std = valid_ratios.std()

        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        is_outlier = (df[ratio_col] < lower_bound) | (df[ratio_col] > upper_bound)

    else:
        msg = f"Unknown outlier detection method: {method}"
        raise ValueError(msg)

    # Skip already verified counters if requested
    if skip_verified and "verification_status" in df.columns:
        already_verified = df["verification_status"] == "verified"
        is_outlier = is_outlier & ~already_verified
        log.info(
            "Skipping verified counters",
            n_verified=already_verified.sum(),
        )

    df["is_outlier"] = is_outlier

    # Assign severity levels
    df.loc[df["is_outlier"], "flag_severity"] = "warning"

    # Critical: very extreme outliers (beyond 2x upper threshold)
    critical_upper = upper_bound * 2
    df.loc[df[ratio_col] > critical_upper, "flag_severity"] = "critical"

    n_flagged = int(is_outlier.sum())
    n_critical = int((df["flag_severity"] == "critical").sum())

    log.info(
        "Flagged outliers",
        n_flagged=n_flagged,
        n_critical=n_critical,
        method=method,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
    )

    result = OutlierResult(
        n_counters=len(df),
        n_flagged=n_flagged,
        threshold_lower=float(lower_bound),
        threshold_upper=float(upper_bound),
        median_ratio=float(valid_ratios.median()),
    )

    return df, result
