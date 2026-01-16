"""
Temporal alignment for campaign periods.

Handles date normalization and campaign period filtering.
"""

from datetime import date

import pandas as pd

from hochrechnung.config.settings import TemporalConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


def filter_to_campaign_period(
    df: pd.DataFrame,
    temporal: TemporalConfig,
    date_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Filter DataFrame to campaign period.

    Args:
        df: DataFrame with date column.
        temporal: Temporal configuration.
        date_column: Name of date column.

    Returns:
        Filtered DataFrame.
    """
    if date_column not in df.columns:
        log.warning("Date column not found, skipping filter", column=date_column)
        return df

    start = pd.Timestamp(temporal.campaign_start)
    end = pd.Timestamp(temporal.campaign_end)

    # Ensure column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    mask = (df[date_column] >= start) & (df[date_column] <= end)
    filtered = df[mask]

    log.info(
        "Filtered to campaign period",
        start=str(start.date()),
        end=str(end.date()),
        rows_before=len(df),
        rows_after=len(filtered),
    )

    return filtered


def filter_to_counter_period(
    df: pd.DataFrame,
    temporal: TemporalConfig,
    date_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Filter DataFrame to counter measurement period.

    Args:
        df: DataFrame with date column.
        temporal: Temporal configuration.
        date_column: Name of date column.

    Returns:
        Filtered DataFrame.
    """
    if date_column not in df.columns:
        log.warning("Date column not found, skipping filter", column=date_column)
        return df

    start = pd.Timestamp(temporal.counter_period_start)
    end = pd.Timestamp(temporal.counter_period_end)

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    mask = (df[date_column] >= start) & (df[date_column] <= end)
    filtered = df[mask]

    log.info(
        "Filtered to counter period",
        start=str(start.date()),
        end=str(end.date()),
        rows_before=len(df),
        rows_after=len(filtered),
    )

    return filtered


def add_holiday_flag(
    df: pd.DataFrame,
    temporal: TemporalConfig,
    date_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Add flag indicating if date falls within school holidays.

    Args:
        df: DataFrame with date column.
        temporal: Temporal configuration with holiday dates.
        date_column: Name of date column.

    Returns:
        DataFrame with 'is_holiday' column.
    """
    if temporal.holiday_start is None or temporal.holiday_end is None:
        log.debug("No holiday dates configured, skipping flag")
        df = df.copy()
        df["is_holiday"] = False
        return df

    if date_column not in df.columns:
        log.warning("Date column not found, adding False flag", column=date_column)
        df = df.copy()
        df["is_holiday"] = False
        return df

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    start = pd.Timestamp(temporal.holiday_start)
    end = pd.Timestamp(temporal.holiday_end)

    df["is_holiday"] = (df[date_column] >= start) & (df[date_column] <= end)

    holiday_count = df["is_holiday"].sum()
    log.info("Added holiday flag", holiday_days=int(holiday_count))

    return df


def calculate_campaign_days(temporal: TemporalConfig) -> int:
    """
    Calculate number of days in campaign period.

    Args:
        temporal: Temporal configuration.

    Returns:
        Number of days.
    """
    return (temporal.campaign_end - temporal.campaign_start).days + 1


def get_campaign_dates(temporal: TemporalConfig) -> tuple[date, date]:
    """
    Get campaign start and end dates.

    Args:
        temporal: Temporal configuration.

    Returns:
        Tuple of (start_date, end_date).
    """
    return (temporal.campaign_start, temporal.campaign_end)
