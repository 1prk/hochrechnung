"""
STADTRADELN campaign data ingestion.

Loads campaign metadata and participation demographics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import DataLoader, GeoDataLoader
from hochrechnung.schemas.campaign import CampaignMetadataSchema, DemographicsSchema
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd
    from hochrechnung.ingestion.gebietseinheiten import AdminLevel

log = get_logger(__name__)


class CampaignMetadataLoader(DataLoader[CampaignMetadataSchema]):
    """Loader for STADTRADELN campaign metadata (dates per municipality)."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize campaign metadata loader."""
        super().__init__(config, CampaignMetadataSchema)

    def _load_raw(self) -> pd.DataFrame:
        """Load campaign metadata from CSV."""
        path = self.resolve_path(self.config.data_paths.campaign_stats)

        if not path.exists():
            msg = f"Campaign stats file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading campaign metadata", path=str(path))

        # Try different encodings
        try:
            df = pd.read_csv(path, encoding="cp1252", delimiter=";")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8", delimiter=",")

        # Normalize column names
        column_mapping = {
            "ags": "ars",
            "AGS": "ars",
            "ARS": "ars",
            "year": "year",
            "Year": "year",
            "start": "start_date",
            "Start": "start_date",
            "end": "end_date",
            "Ende": "end_date",
            "End": "end_date",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Filter to configured year
        if "year" in df.columns:
            year = self.config.temporal.year
            df = df[df["year"] == year]
            log.info("Filtered to year", year=year, rows=len(df))

        # Ensure ARS is 12-digit string
        if "ars" in df.columns:
            df["ars"] = df["ars"].astype(str).str.zfill(12)

        # Parse dates
        for col in ["start_date", "end_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        return df


class DemographicsLoader(GeoDataLoader[DemographicsSchema]):
    """Loader for STADTRADELN participation demographics."""

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize demographics loader."""
        super().__init__(config, DemographicsSchema)

    def _find_demographics_file(self) -> Path:
        """
        Find demographics file, checking multiple locations.

        Checks in order:
        1. Configured kommunen_stats path
        2. Year-specific JSON: kommunen-stats/SR{YY}_Commune_Statistics.json
        3. Year-specific CSV: campaign/SR_{year}.csv
        """
        data_root = self.config.data_paths.data_root
        year = self.config.year
        year_short = str(year)[-2:]  # e.g., 2024 -> "24"

        # Try configured path first
        configured_path = self.resolve_path(self.config.data_paths.kommunen_stats)
        if configured_path.exists():
            return configured_path

        # Try year-specific JSON (new format)
        year_json = data_root / "kommunen-stats" / f"SR{year_short}_Commune_Statistics.json"
        if year_json.exists():
            return year_json

        # Try year-specific CSV
        year_csv = data_root / "campaign" / f"SR_{year}.csv"
        if year_csv.exists():
            return year_csv

        msg = (
            f"Demographics file not found. Tried:\n"
            f"  - {configured_path}\n"
            f"  - {year_json}\n"
            f"  - {year_csv}"
        )
        raise FileNotFoundError(msg)

    def _load_raw_geo(self) -> pd.DataFrame:
        """Load demographics from shapefile, JSON, or CSV."""
        import geopandas as gpd

        path = self._find_demographics_file()
        log.info("Loading demographics", path=str(path))

        # Load based on file type
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = self._load_from_csv(path)
        elif suffix == ".json":
            df = self._load_from_json(path)
        else:
            df = gpd.read_file(path)

        # Filter to configured region using ARS prefix
        df = self._filter_and_aggregate_by_region(df)

        # Normalize column names
        column_mapping = {
            "ARS": "ars",
            "N_USERS": "n_users",
            "N_TRIPS": "n_trips",
            "TOTAL_KM": "total_km",
            "BUNDESLAND": "bundesland",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Ensure ARS is 12-digit string
        if "ars" in df.columns:
            df["ars"] = df["ars"].astype(str).str.zfill(12)

        return df

    def _load_from_csv(self, path: Path) -> pd.DataFrame:
        """Load demographics from CSV file."""
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp1252")

        # Ensure ARS is string for filtering
        if "ARS" in df.columns:
            df["ARS"] = df["ARS"].astype(str).str.zfill(12)

        return df

    def _load_from_json(self, path: Path) -> pd.DataFrame:
        """
        Load demographics from JSON file (new format).

        JSON format has columns: ars, name, users_n, trips_n, distance_km, etc.
        Maps to expected column names: ARS, GEN, N_USERS, N_TRIPS, TOTAL_KM.
        """
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        df = pd.DataFrame(data)

        # Map JSON column names to expected shapefile-style names
        column_mapping = {
            "ars": "ARS",
            "name": "GEN",
            "users_n": "N_USERS",
            "trips_n": "N_TRIPS",
            "distance_km": "TOTAL_KM",
        }

        df = df.rename(
            columns={k: v for k, v in column_mapping.items() if k in df.columns}
        )

        # Ensure ARS is 12-digit string for filtering
        if "ARS" in df.columns:
            df["ARS"] = df["ARS"].astype(str).str.zfill(12)

        log.info("Loaded demographics from JSON", rows=len(df))

        return df

    def _filter_and_aggregate_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter demographics to region, keeping municipality-level statistics.

        German ARS structure (12 digits):
        - Positions 0-1: State (Land) - e.g., 06 = Hessen
        - Positions 2-4: District (Regierungsbezirk) - 000 if unspecified
        - Positions 5-7: County (Kreis) - 000 if unspecified
        - Positions 8-11: Municipality (Gemeinde) - 0000 if unspecified

        For state-level regions (e.g., 060000000000), includes all municipalities
        in that state with their individual statistics preserved.
        """
        region_ars = self.config.region.ars

        # Determine ARS column name
        ars_col = "ARS" if "ARS" in df.columns else "ars"
        if ars_col not in df.columns:
            log.warning("No ARS column found in demographics")
            return df

        # Ensure ARS is 12-digit string
        df[ars_col] = df[ars_col].astype(str).str.zfill(12)

        # Determine the administrative level by checking trailing zeros
        ars_str = str(region_ars).zfill(12)

        # Standard German admin levels: 2 (state), 5 (district), 8 (county), 12 (municipality)
        if ars_str[2:] == "0" * 10:
            # State level: e.g., 060000000000 -> prefix "06"
            prefix_len = 2
        elif ars_str[5:] == "0" * 7:
            # District level: e.g., 064000000000 -> prefix based on actual content
            prefix_len = 5 if ars_str[2:5] != "000" else 2
        elif ars_str[8:] == "0" * 4:
            # County level: e.g., 064110000000 -> prefix "06411"
            prefix_len = 8 if ars_str[5:8] != "000" else (5 if ars_str[2:5] != "000" else 2)
        else:
            # Municipality level: use full ARS
            prefix_len = 12

        ars_prefix = ars_str[:prefix_len]

        # Filter to matching municipalities (keep individual statistics)
        mask = df[ars_col].str.startswith(ars_prefix)
        filtered = df[mask].copy()

        log.info(
            "Filtered demographics by ARS prefix",
            prefix=ars_prefix,
            prefix_len=prefix_len,
            matched=len(filtered),
            total=len(df),
        )

        if len(filtered) == 0:
            log.warning("No demographics found for region", ars_prefix=ars_prefix)

        return filtered


def load_campaign_metadata(
    config: PipelineConfig, *, validate: bool = True
) -> pd.DataFrame:
    """
    Convenience function to load campaign metadata.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with campaign metadata.
    """
    loader = CampaignMetadataLoader(config)
    return loader.load(validate=validate)


def load_demographics(
    config: PipelineConfig, *, validate: bool = True
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load participation demographics.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with participation demographics.
    """
    loader = DemographicsLoader(config)
    return loader.load(validate=validate)


class AggregatedStatisticsLoader(DataLoader[DemographicsSchema]):
    """
    Loader for STADTRADELN statistics aggregated to a specific admin level.

    Aggregates commune-level statistics (n_users, n_trips, total_km) to the
    specified administrative level (Land, Kreis, or Verwaltungsgemeinschaft)
    by matching ARS prefixes.
    """

    def __init__(
        self,
        config: PipelineConfig,
        admin_level: "AdminLevel | None" = None,
    ) -> None:
        """
        Initialize aggregated statistics loader.

        Args:
            config: Pipeline configuration.
            admin_level: Administrative level for aggregation. If None, uses config.stats.admin_level.
        """
        super().__init__(config, DemographicsSchema)
        if admin_level is None:
            from hochrechnung.ingestion.gebietseinheiten import AdminLevel

            self.admin_level = AdminLevel.from_string(config.stats.admin_level)
        else:
            self.admin_level = admin_level

    def _load_raw(self) -> pd.DataFrame:
        """Load and aggregate commune-level statistics to admin level.

        Handles Kreisfreie Städte specially: when VG level is requested, these
        cities are aggregated at Kreis level (5-digit prefix) since they don't
        have VG subdivisions.
        """
        from hochrechnung.ingestion.gebietseinheiten import AdminLevel

        # Load commune-level data via DemographicsLoader
        demographics_loader = DemographicsLoader(self.config)
        commune_df = demographics_loader.load(validate=False)

        if len(commune_df) == 0:
            log.warning("No commune statistics found")
            return pd.DataFrame(columns=["ars", "n_users", "n_trips", "total_km"])

        # Ensure we have the required columns
        required_cols = ["ars", "n_users", "n_trips", "total_km"]
        missing_cols = [c for c in required_cols if c not in commune_df.columns]
        if missing_cols:
            log.warning("Missing columns in commune data", missing=missing_cols)
            # Add missing columns with zeros
            for col in missing_cols:
                if col != "ars":
                    commune_df[col] = 0

        # For VG level, we need to handle Kreisfreie Städte specially
        if self.admin_level == AdminLevel.VERWALTUNGSGEMEINSCHAFT:
            return self._aggregate_with_kreisfreie(commune_df)

        # Simple case: aggregate at the specified level
        prefix_len = self.admin_level.ars_prefix_length
        commune_df["ars_agg"] = (
            commune_df["ars"].astype(str).str[:prefix_len].str.ljust(12, "0")
        )

        # Aggregate statistics by admin level
        agg_df = (
            commune_df.groupby("ars_agg", as_index=False)
            .agg(
                {
                    "n_users": "sum",
                    "n_trips": "sum",
                    "total_km": "sum",
                }
            )
            .rename(columns={"ars_agg": "ars"})
        )

        log.info(
            "Aggregated statistics",
            admin_level=self.admin_level.value,
            prefix_len=prefix_len,
            communes=len(commune_df),
            aggregated_units=len(agg_df),
        )

        return agg_df

    def _aggregate_with_kreisfreie(self, commune_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate statistics handling Kreisfreie Städte.

        For VG level aggregation:
        - Kreisfreie Städte (cities without VG subdivisions) aggregate at Kreis level (5 digits)
        - Regular municipalities aggregate at VG level (9 digits)

        The loader identifies Kreisfreie Städte by checking if any VG exists with
        the same Kreis prefix (first 5 digits).
        """
        from hochrechnung.ingestion.gebietseinheiten import AdminLevel, load_gebietseinheiten

        # Load both VGs and Kreise from Gebietseinheiten to identify Kreisfreie Städte
        # We need to load at Kreis level to find which Kreise have no VGs
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.gebietseinheiten)
        gdf = gpd.read_file(path)

        # Normalize ARS column
        ars_col = "ARS" if "ARS" in gdf.columns else "ars"
        gdf["ars"] = gdf[ars_col].astype(str).str.zfill(12)

        # Normalize Type column
        type_col = "Type" if "Type" in gdf.columns else "admin_level"

        # Get VG prefixes (first 5 digits of Kreise that have VGs)
        vg_gdf = gdf[gdf[type_col] == "Verwaltungsgemeinschaft"]
        vg_kreis_prefixes = set(vg_gdf["ars"].str[:5].unique())

        # Get Kreis-level entries
        kreis_gdf = gdf[gdf[type_col] == "Kreis"]

        # Identify Kreisfreie Städte: Kreise without any VG children
        land_code = self.config.land_code
        kreisfreie_prefixes = set()
        for _, row in kreis_gdf.iterrows():
            kreis_ars = row["ars"]
            if kreis_ars.startswith(land_code):
                kreis_prefix = kreis_ars[:5]
                if kreis_prefix not in vg_kreis_prefixes:
                    kreisfreie_prefixes.add(kreis_prefix)

        log.info(
            "Identified Kreisfreie Städte prefixes for aggregation",
            n_kreisfreie=len(kreisfreie_prefixes),
            examples=list(kreisfreie_prefixes)[:5],
        )

        # Create aggregation key based on whether commune is in a Kreisfreie Stadt
        def get_agg_key(ars: str) -> str:
            kreis_prefix = ars[:5]
            if kreis_prefix in kreisfreie_prefixes:
                # Kreisfreie Stadt: aggregate at Kreis level (5 digits)
                return kreis_prefix.ljust(12, "0")
            else:
                # Regular VG: aggregate at VG level (9 digits)
                return ars[:9].ljust(12, "0")

        commune_df = commune_df.copy()
        commune_df["ars_agg"] = commune_df["ars"].astype(str).apply(get_agg_key)

        # Aggregate statistics
        agg_df = (
            commune_df.groupby("ars_agg", as_index=False)
            .agg(
                {
                    "n_users": "sum",
                    "n_trips": "sum",
                    "total_km": "sum",
                }
            )
            .rename(columns={"ars_agg": "ars"})
        )

        # Count aggregation levels
        n_kreis_level = sum(1 for ars in agg_df["ars"] if ars[:5] in kreisfreie_prefixes)
        n_vg_level = len(agg_df) - n_kreis_level

        log.info(
            "Aggregated statistics (VG + Kreisfreie)",
            communes=len(commune_df),
            aggregated_units=len(agg_df),
            vg_level=n_vg_level,
            kreis_level=n_kreis_level,
        )

        return agg_df


def load_aggregated_statistics(
    config: PipelineConfig,
    admin_level: "AdminLevel | None" = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to load aggregated STADTRADELN statistics.

    Args:
        config: Pipeline configuration.
        admin_level: Administrative level for aggregation. If None, uses config.stats.admin_level.
        validate: Whether to validate against schema.

    Returns:
        DataFrame with aggregated statistics.
    """
    loader = AggregatedStatisticsLoader(config, admin_level=admin_level)
    return loader.load(validate=validate)
