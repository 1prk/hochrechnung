"""
DTV (Daily Traffic Volume) calculation.

Computes average daily traffic from counter measurements with quality tracking.
"""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from hochrechnung.config.settings import TemporalConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DTVResult:
    """
    Result of DTV calculation for a single counter.

    Attributes:
        counter_id: Counter identifier.
        value: Calculated DTV value.
        observation_count: Number of daily observations used.
        missing_days: Number of missing days in period.
        zero_days: Number of days with zero count.
        quality_score: Quality score (0-1) based on completeness.
        period_start: Start of measurement period.
        period_end: End of measurement period.
    """

    counter_id: str
    value: float
    observation_count: int
    missing_days: int
    zero_days: int
    quality_score: float
    period_start: date
    period_end: date

    @property
    def is_valid(self) -> bool:
        """Check if DTV result is valid based on quality."""
        return self.quality_score >= 0.5 and self.observation_count >= 7


def calculate_dtv(
    measurements: pd.DataFrame,
    temporal: TemporalConfig,
    counter_id_column: str = "counter_id",
    count_column: str = "count",
    date_column: str = "timestamp",
) -> list[DTVResult]:
    """
    Calculate DTV for all counters in measurements.

    Args:
        measurements: DataFrame with daily counter measurements.
        temporal: Temporal configuration with period dates.
        counter_id_column: Name of counter ID column.
        count_column: Name of count column.
        date_column: Name of date column.

    Returns:
        List of DTVResult for each counter.
    """
    log.info("Calculating DTV values")

    # Filter to campaign period
    df = measurements.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    start = pd.Timestamp(temporal.campaign_start)
    end = pd.Timestamp(temporal.campaign_end)

    df = df[(df[date_column] >= start) & (df[date_column] <= end)]

    # Calculate expected days
    expected_days = (temporal.campaign_end - temporal.campaign_start).days + 1

    results: list[DTVResult] = []

    for counter_id, group in df.groupby(counter_id_column):
        counts = group[count_column].dropna()

        observation_count = len(counts)
        missing_days = expected_days - observation_count
        zero_days = int((counts == 0).sum())

        # Calculate mean (DTV) and round to match legacy format
        if observation_count > 0:
            dtv_value = float(round(counts.mean()))
        else:
            dtv_value = 0.0

        # Quality score based on completeness and zero ratio
        completeness = observation_count / expected_days
        zero_ratio = zero_days / max(observation_count, 1)
        quality_score = completeness * (1 - zero_ratio * 0.5)

        result = DTVResult(
            counter_id=str(counter_id),
            value=dtv_value,
            observation_count=observation_count,
            missing_days=missing_days,
            zero_days=zero_days,
            quality_score=quality_score,
            period_start=temporal.campaign_start,
            period_end=temporal.campaign_end,
        )

        results.append(result)

    log.info(
        "DTV calculation complete",
        n_counters=len(results),
        valid_counters=sum(1 for r in results if r.is_valid),
    )

    return results


def dtv_results_to_dataframe(results: list[DTVResult]) -> pd.DataFrame:
    """
    Convert DTV results to DataFrame.

    Args:
        results: List of DTVResult objects.

    Returns:
        DataFrame with DTV data. Returns empty DataFrame with proper columns if no results.
    """
    data = [
        {
            "counter_id": r.counter_id,
            "dtv": r.value,
            "observation_count": r.observation_count,
            "missing_days": r.missing_days,
            "zero_days": r.zero_days,
            "quality_score": r.quality_score,
            "is_valid": r.is_valid,
        }
        for r in results
    ]

    # If empty, create DataFrame with proper columns
    if not data:
        return pd.DataFrame(columns=[
            "counter_id",
            "dtv",
            "observation_count",
            "missing_days",
            "zero_days",
            "quality_score",
            "is_valid",
        ])

    return pd.DataFrame(data)


def filter_dtv_by_quality(
    results: list[DTVResult],
    min_quality: float = 0.5,
    min_observations: int = 7,
) -> list[DTVResult]:
    """
    Filter DTV results by quality criteria.

    Args:
        results: List of DTVResult objects.
        min_quality: Minimum quality score.
        min_observations: Minimum observation count.

    Returns:
        Filtered list of DTVResult.
    """
    filtered = [
        r
        for r in results
        if r.quality_score >= min_quality and r.observation_count >= min_observations
    ]

    log.info(
        "Filtered DTV results",
        before=len(results),
        after=len(filtered),
        min_quality=min_quality,
        min_observations=min_observations,
    )

    return filtered
