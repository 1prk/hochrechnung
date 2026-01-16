"""
Data quality assessment for target variables.

Provides quality metrics and filtering for training data.
"""

from dataclasses import dataclass

import pandas as pd

from hochrechnung.config.settings import TrainingConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class QualityMetrics:
    """
    Quality metrics for a dataset.

    Attributes:
        total_samples: Total number of samples.
        valid_samples: Number of samples meeting quality criteria.
        missing_target_count: Number of samples with missing target.
        below_min_count: Number of samples below minimum threshold.
        above_max_count: Number of samples above maximum threshold.
        outlier_count: Number of statistical outliers.
    """

    total_samples: int
    valid_samples: int
    missing_target_count: int
    below_min_count: int
    above_max_count: int
    outlier_count: int

    @property
    def valid_ratio(self) -> float:
        """Ratio of valid to total samples."""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples


def assess_target_quality(
    df: pd.DataFrame,
    target_column: str,
    config: TrainingConfig,
) -> QualityMetrics:
    """
    Assess quality of target variable.

    Args:
        df: DataFrame with target column.
        target_column: Name of target column.
        config: Training configuration with thresholds.

    Returns:
        QualityMetrics object.
    """
    total = len(df)

    if target_column not in df.columns:
        log.warning("Target column not found", column=target_column)
        return QualityMetrics(
            total_samples=total,
            valid_samples=0,
            missing_target_count=total,
            below_min_count=0,
            above_max_count=0,
            outlier_count=0,
        )

    target = df[target_column]

    # Count issues
    missing = int(target.isna().sum())
    below_min = int((target < config.min_dtv).sum())
    above_max = int((target > config.max_dtv).sum()) if config.max_dtv else 0

    # Detect outliers using IQR
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    outliers = int(((target < lower) | (target > upper)).sum())

    # Calculate valid samples
    valid_mask = target.notna()
    valid_mask &= target >= config.min_dtv
    if config.max_dtv:
        valid_mask &= target <= config.max_dtv

    valid = int(valid_mask.sum())

    metrics = QualityMetrics(
        total_samples=total,
        valid_samples=valid,
        missing_target_count=missing,
        below_min_count=below_min,
        above_max_count=above_max,
        outlier_count=outliers,
    )

    log.info(
        "Target quality assessment",
        total=total,
        valid=valid,
        valid_ratio=f"{metrics.valid_ratio:.2%}",
        missing=missing,
    )

    return metrics


def filter_by_target_quality(
    df: pd.DataFrame,
    target_column: str,
    config: TrainingConfig,
    *,
    remove_outliers: bool = False,
) -> pd.DataFrame:
    """
    Filter DataFrame by target quality criteria.

    Args:
        df: DataFrame to filter.
        target_column: Name of target column.
        config: Training configuration with thresholds.
        remove_outliers: Whether to remove statistical outliers.

    Returns:
        Filtered DataFrame.
    """
    if target_column not in df.columns:
        log.warning("Target column not found, returning empty", column=target_column)
        return df.head(0)

    initial_count = len(df)
    result = df.copy()

    # Remove missing
    result = result[result[target_column].notna()]

    # Apply min threshold
    result = result[result[target_column] >= config.min_dtv]

    # Apply max threshold if set
    if config.max_dtv:
        result = result[result[target_column] <= config.max_dtv]

    # Remove outliers if requested
    if remove_outliers:
        target = result[target_column]
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        result = result[(target >= lower) & (target <= upper)]

    log.info(
        "Filtered by target quality",
        before=initial_count,
        after=len(result),
        removed=initial_count - len(result),
    )

    return result
