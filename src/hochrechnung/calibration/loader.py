"""
Calibration data loader.

Loads calibration counter data with pre-calculated DTV values.
Handles both raw and verified calibration datasets.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


class CalibrationDataLoader:
    """Load calibration counter data with pre-calculated DTV.

    Expected CSV format:
        id,name,latitude,longitude,dtv
        CAL001,Station A,50.1,8.6,1250
        CAL002,Station B,50.2,8.7,980

    Optional columns (used by stratified calibrators):
        - infra_category: Infrastructure category
        - regiostar7: RegioStaR urban/rural classification

    Attributes:
        REQUIRED_COLUMNS: Columns that must be present in the CSV.
    """

    REQUIRED_COLUMNS = ["id", "latitude", "longitude", "dtv"]

    def __init__(self, path: Path) -> None:
        """Initialize loader.

        Args:
            path: Path to calibration counter CSV.
        """
        self.path = Path(path)

    def load(self, *, validate: bool = True) -> pd.DataFrame:
        """Load calibration counter data.

        Args:
            validate: Whether to validate required columns and values.

        Returns:
            DataFrame with calibration counter data.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If validation fails (missing columns, invalid values).
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Calibration data not found: {self.path}")

        log.info("Loading calibration data", path=str(self.path))
        df = pd.read_csv(self.path)

        if validate:
            self._validate(df)

        log.info(
            "Loaded calibration stations",
            n_stations=len(df),
            columns=list(df.columns),
        )
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate calibration data.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If validation fails.
        """
        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for missing DTV values
        n_missing_dtv = df["dtv"].isna().sum()
        if n_missing_dtv > 0:
            raise ValueError(
                f"DTV column contains {n_missing_dtv} missing values"
            )

        # Check for negative DTV values
        n_negative = (df["dtv"] < 0).sum()
        if n_negative > 0:
            raise ValueError(
                f"DTV column contains {n_negative} negative values"
            )

        # Check coordinate bounds (roughly Germany)
        lat_ok = df["latitude"].between(47.0, 56.0).all()
        lon_ok = df["longitude"].between(5.5, 15.5).all()
        if not lat_ok or not lon_ok:
            log.warning(
                "Some coordinates outside Germany bounds",
                lat_range=(df["latitude"].min(), df["latitude"].max()),
                lon_range=(df["longitude"].min(), df["longitude"].max()),
            )


def load_verified_calibration_counters(
    data_root: Path,
    region: str,
    year: int,
    *,
    must_exist: bool = True,
) -> pd.DataFrame | None:
    """Load verified calibration counters for a region/year.

    Looks for file at: {data_root}/verified/calibration_{region}_{year}.csv

    This follows the pattern from verification/persistence.py for
    consistency with the verified counter datasets.

    Args:
        data_root: Root data directory.
        region: Region name (e.g., 'hessen').
        year: Year for the calibration data.
        must_exist: If True, raises FileNotFoundError if file doesn't exist.

    Returns:
        DataFrame with verified calibration counters, or None if file
        doesn't exist and must_exist=False.

    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True.
    """
    # Normalize region name for file path
    region_normalized = region.lower().replace(" ", "_")
    verified_path = (
        data_root / "verified" / f"calibration_{region_normalized}_{year}.csv"
    )

    if not verified_path.exists():
        if must_exist:
            raise FileNotFoundError(
                f"Verified calibration data not found: {verified_path}"
            )
        log.info(
            "No verified calibration data found",
            path=str(verified_path),
            region=region,
            year=year,
        )
        return None

    log.info(
        "Loading verified calibration counters",
        path=str(verified_path),
        region=region,
        year=year,
    )

    df = pd.read_csv(verified_path)

    log.info(
        "Loaded verified calibration counters",
        n_counters=len(df),
        region=region,
        year=year,
    )

    return df


def save_verified_calibration_counters(
    df: pd.DataFrame,
    data_root: Path,
    region: str,
    year: int,
) -> Path:
    """Save verified calibration counters for a region/year.

    Saves to: {data_root}/verified/calibration_{region}_{year}.csv

    Args:
        df: DataFrame with calibration counter data.
        data_root: Root data directory.
        region: Region name (e.g., 'leipzig').
        year: Year for the calibration data.

    Returns:
        Path to saved file.
    """
    # Normalize region name for file path
    region_normalized = region.lower().replace(" ", "_")
    verified_dir = data_root / "verified"
    verified_dir.mkdir(parents=True, exist_ok=True)
    verified_path = verified_dir / f"calibration_{region_normalized}_{year}.csv"

    # Add verification timestamp if not present
    if "verified_at" not in df.columns:
        df = df.copy()
        df["verified_at"] = datetime.now().isoformat()

    # Ensure is_discarded column exists
    if "is_discarded" not in df.columns:
        df = df.copy()
        df["is_discarded"] = False

    # Filter out discarded stations before saving
    # (keep them in the file but mark them)
    df.to_csv(verified_path, index=False, encoding="utf-8", lineterminator="\n")

    n_discarded = df["is_discarded"].sum() if "is_discarded" in df.columns else 0
    log.info(
        "Saved verified calibration counters",
        path=str(verified_path),
        n_counters=len(df),
        n_discarded=n_discarded,
        region=region,
        year=year,
    )

    return verified_path
