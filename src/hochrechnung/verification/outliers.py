"""
Outlier detection for counter verification.

Calculates DTV/volume ratios and flags counters with suspicious assignments.

Severity Hierarchy (highest to lowest priority):
- critical: Extreme ratio outlier (beyond 2x IQR bounds)
- ambiguous: Multiple candidate edges within distance threshold
- no_volume: Matched edge has no volume data (NULL or 0)
- warning: Ratio outlier (beyond 1.5x IQR bounds)
- campaign_bias: Inverse ratio outlier (high volume, low DTV)
- carryover: Inherited from previous year, needs review
- ok: Normal, no issues detected
- verified: Human-approved assignment
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# Severity levels in priority order (lower index = higher priority)
SEVERITY_PRIORITY = [
    "critical",
    "ambiguous",
    "no_volume",
    "warning",
    "campaign_bias",
    "carryover",
    "ok",
    "verified",
]


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
        n_by_severity: Count of counters per severity level.
    """

    n_counters: int
    n_flagged: int
    threshold_lower: float
    threshold_upper: float
    median_ratio: float
    n_by_severity: dict[str, int] = field(default_factory=dict)


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


def flag_no_volume(
    counters_df: pd.DataFrame,
    volume_col: str = "count",
) -> pd.DataFrame:
    """
    Flag counters where matched edge has no volume data.

    Args:
        counters_df: DataFrame with counter data.
        volume_col: Name of volume column.

    Returns:
        DataFrame with 'has_no_volume' flag column.
    """
    df = counters_df.copy()

    # Check for NULL or 0 volume
    df["has_no_volume"] = df[volume_col].isna() | (df[volume_col] == 0)

    n_no_volume = int(df["has_no_volume"].sum())
    log.info("Flagged counters with no volume", n_no_volume=n_no_volume)

    return df


def flag_campaign_bias(
    counters_df: pd.DataFrame,
    dtv_col: str = "dtv",
    volume_col: str = "count",
    iqr_multiplier: float = 1.5,
    *,
    skip_verified: bool = True,
) -> pd.DataFrame:
    """
    Flag counters with potential campaign bias (high volume, low DTV).

    Uses IQR method on inverse ratio (volume/DTV) to detect edges
    that are disproportionately popular in the cycling campaign
    relative to general traffic.

    Args:
        counters_df: DataFrame with counter data.
        dtv_col: Name of DTV column.
        volume_col: Name of volume column.
        iqr_multiplier: Multiplier for IQR bounds.
        skip_verified: If True, don't flag already verified counters.

    Returns:
        DataFrame with 'has_campaign_bias' flag column.
    """
    df = counters_df.copy()

    # Calculate inverse ratio (volume / DTV)
    df["inverse_ratio"] = df[volume_col] / df[dtv_col].replace(0, np.nan)
    df.loc[np.isinf(df["inverse_ratio"]), "inverse_ratio"] = np.nan

    # Get valid ratios
    valid_ratios = df["inverse_ratio"].dropna()

    if len(valid_ratios) == 0:
        df["has_campaign_bias"] = False
        return df

    # IQR method - only flag high inverse ratios (campaign hotspots)
    Q1 = valid_ratios.quantile(0.25)
    Q3 = valid_ratios.quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + iqr_multiplier * IQR

    # Flag only upper outliers (high volume relative to DTV)
    is_campaign_bias = df["inverse_ratio"] > upper_bound

    # Skip already verified counters if requested
    if skip_verified and "verification_status" in df.columns:
        already_verified = df["verification_status"] == "verified"
        is_campaign_bias = is_campaign_bias & ~already_verified

    df["has_campaign_bias"] = is_campaign_bias

    n_campaign_bias = int(is_campaign_bias.sum())
    log.info(
        "Flagged counters with campaign bias",
        n_campaign_bias=n_campaign_bias,
        inverse_ratio_upper_bound=float(upper_bound),
    )

    return df


def flag_ambiguous_matches(
    counters_df: pd.DataFrame,
    distance_ratio_threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Flag counters with ambiguous edge matches.

    A match is considered ambiguous when multiple candidate edges
    are at similar distances (ratio of distances below threshold).

    Args:
        counters_df: DataFrame with counter data including candidate_edges.
        distance_ratio_threshold: If second_distance / first_distance < this,
            flag as ambiguous.

    Returns:
        DataFrame with 'is_ambiguous' flag column.
    """
    df = counters_df.copy()
    df["is_ambiguous"] = False

    # Check if candidate_edges column exists
    if "candidate_edges" not in df.columns:
        log.info("No candidate_edges column, skipping ambiguity detection")
        return df

    for idx, row in df.iterrows():
        candidates = row.get("candidate_edges")
        if candidates is None or len(candidates) < 2:
            continue

        # Sort candidates by distance
        sorted_candidates = sorted(
            candidates, key=lambda x: x.get("distance", float("inf"))
        )

        if len(sorted_candidates) >= 2:
            d1 = sorted_candidates[0].get("distance", 0)
            d2 = sorted_candidates[1].get("distance", float("inf"))

            # Avoid division by zero
            if d1 > 0 and d2 / d1 < distance_ratio_threshold:
                df.at[idx, "is_ambiguous"] = True

    n_ambiguous = int(df["is_ambiguous"].sum())
    log.info(
        "Flagged ambiguous matches",
        n_ambiguous=n_ambiguous,
        distance_ratio_threshold=distance_ratio_threshold,
    )

    return df


def assign_final_severity(
    counters_df: pd.DataFrame,
    *,
    skip_verified: bool = True,
) -> pd.DataFrame:
    """
    Assign final flag_severity based on all detection results.

    Applies severity hierarchy: critical > ambiguous > no_volume >
    warning > campaign_bias > carryover > ok.

    Args:
        counters_df: DataFrame with all flag columns.
        skip_verified: If True, preserve 'verified' status.

    Returns:
        DataFrame with final 'flag_severity' column.
    """
    df = counters_df.copy()

    # Start with 'ok' as default
    df["flag_severity"] = "ok"

    # Apply in reverse priority order (lowest priority first)
    # so higher priority flags overwrite lower ones

    # Carryover status
    if "verification_status" in df.columns:
        df.loc[df["verification_status"] == "carryover", "flag_severity"] = "carryover"

    # Campaign bias (soft flag)
    if "has_campaign_bias" in df.columns:
        df.loc[df["has_campaign_bias"] == True, "flag_severity"] = "campaign_bias"  # noqa: E712

    # Warning (ratio outlier)
    if "is_outlier" in df.columns:
        df.loc[df["is_outlier"] == True, "flag_severity"] = "warning"  # noqa: E712

    # No volume
    if "has_no_volume" in df.columns:
        df.loc[df["has_no_volume"] == True, "flag_severity"] = "no_volume"  # noqa: E712

    # Ambiguous match
    if "is_ambiguous" in df.columns:
        df.loc[df["is_ambiguous"] == True, "flag_severity"] = "ambiguous"  # noqa: E712

    # Critical (extreme ratio outlier) - check ratio directly
    if "dtv_volume_ratio" in df.columns:
        # Recalculate critical threshold from valid ratios
        valid_ratios = df["dtv_volume_ratio"].dropna()
        if len(valid_ratios) > 0:
            Q1 = valid_ratios.quantile(0.25)
            Q3 = valid_ratios.quantile(0.75)
            IQR = Q3 - Q1
            critical_upper = (Q3 + 1.5 * IQR) * 2
            df.loc[df["dtv_volume_ratio"] > critical_upper, "flag_severity"] = (
                "critical"
            )

    # Preserve verified status if requested
    if skip_verified and "verification_status" in df.columns:
        df.loc[df["verification_status"] == "verified", "flag_severity"] = "verified"

    # Log severity distribution
    severity_counts = df["flag_severity"].value_counts().to_dict()
    log.info("Assigned final severities", **severity_counts)

    return df


def run_all_detection(
    counters_df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
    distance_ratio_threshold: float = 2.0,
    *,
    skip_verified: bool = True,
) -> tuple[pd.DataFrame, OutlierResult]:
    """
    Run all detection methods and assign final severity.

    This is the main entry point for the verification pipeline.

    Args:
        counters_df: DataFrame with counter data.
        iqr_multiplier: IQR multiplier for outlier detection.
        distance_ratio_threshold: Threshold for ambiguous match detection.
        skip_verified: If True, preserve verified counters.

    Returns:
        Tuple of (DataFrame with all flags and severity, OutlierResult summary).
    """
    log.info("Running all detection methods", n_counters=len(counters_df))

    # Step 1: Calculate ratios
    df = calculate_ratios(counters_df)

    # Step 2: Flag ratio outliers
    df, outlier_result = flag_outliers(
        df,
        iqr_multiplier=iqr_multiplier,
        skip_verified=skip_verified,
    )

    # Step 3: Flag no volume
    df = flag_no_volume(df)

    # Step 4: Flag campaign bias
    df = flag_campaign_bias(
        df,
        iqr_multiplier=iqr_multiplier,
        skip_verified=skip_verified,
    )

    # Step 5: Flag ambiguous matches
    df = flag_ambiguous_matches(
        df,
        distance_ratio_threshold=distance_ratio_threshold,
    )

    # Step 6: Assign final severity
    df = assign_final_severity(df, skip_verified=skip_verified)

    # Update result with severity counts
    severity_counts = df["flag_severity"].value_counts().to_dict()
    outlier_result.n_by_severity = severity_counts

    # Count all flagged (anything not 'ok' or 'verified')
    n_flagged = len(df[~df["flag_severity"].isin(["ok", "verified"])])
    outlier_result.n_flagged = n_flagged

    log.info(
        "Detection complete",
        n_flagged=n_flagged,
        severity_breakdown=severity_counts,
    )

    return df, outlier_result
