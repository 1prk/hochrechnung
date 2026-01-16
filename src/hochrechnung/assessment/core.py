"""
Core assessment logic for ETL output verification.

Compares ETL output values against original source data to verify
data integrity and transformation correctness.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.counter import (
    load_counter_locations,
    load_counter_measurements,
)
from hochrechnung.ingestion.structural import load_regiostar
from hochrechnung.ingestion.traffic import load_traffic_volumes
from hochrechnung.targets.dtv import calculate_dtv, dtv_results_to_dataframe
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


class CheckStatus(Enum):
    """Status of an individual check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class CheckResult:
    """
    Result of a single assessment check.

    Attributes:
        name: Check name.
        status: Pass/warn/fail/skip.
        message: Human-readable description.
        details: Optional additional details.
        n_checked: Number of rows checked.
        n_passed: Number of rows passing.
        n_failed: Number of rows failing.
        sample_failures: Sample of failure cases for debugging.
    """

    name: str
    status: CheckStatus
    message: str
    details: str | None = None
    n_checked: int = 0
    n_passed: int = 0
    n_failed: int = 0
    sample_failures: pd.DataFrame | None = None


@dataclass
class AssessmentResult:
    """
    Result of full ETL assessment.

    Attributes:
        etl_path: Path to ETL output file.
        checks: List of individual check results.
        overall_status: Aggregated status across all checks.
    """

    etl_path: Path
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def overall_status(self) -> CheckStatus:
        """Determine overall status from individual checks."""
        if any(c.status == CheckStatus.FAIL for c in self.checks):
            return CheckStatus.FAIL
        if any(c.status == CheckStatus.WARN for c in self.checks):
            return CheckStatus.WARN
        if all(c.status == CheckStatus.SKIP for c in self.checks):
            return CheckStatus.SKIP
        return CheckStatus.PASS

    @property
    def n_passed(self) -> int:
        """Count checks that passed."""
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def n_failed(self) -> int:
        """Count checks that failed."""
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    @property
    def n_warned(self) -> int:
        """Count checks with warnings."""
        return sum(1 for c in self.checks if c.status == CheckStatus.WARN)


class AssessmentRunner:
    """
    Runs ETL output assessment against source data.

    Verifies that ETL transformations produced correct values by
    comparing output with original source data.
    """

    # Tolerance for floating point comparisons
    FLOAT_RTOL = 1e-5  # Relative tolerance
    FLOAT_ATOL = 1e-8  # Absolute tolerance

    # Threshold for pass/warn/fail
    WARN_THRESHOLD = 0.95  # Warn if < 95% match
    FAIL_THRESHOLD = 0.80  # Fail if < 80% match

    def __init__(
        self,
        config: PipelineConfig,
        etl_output_path: Path | None = None,
    ) -> None:
        """
        Initialize assessment runner.

        Args:
            config: Pipeline configuration.
            etl_output_path: Path to ETL output CSV. If None, uses default cache path.
        """
        self.config = config
        self.etl_path = etl_output_path or (
            config.output.cache_dir / f"training_data_{config.year}.csv"
        )

    def run(self) -> AssessmentResult:
        """
        Run full assessment suite.

        Returns:
            AssessmentResult with all check results.
        """
        log.info("Starting ETL assessment", path=str(self.etl_path))

        result = AssessmentResult(etl_path=self.etl_path)

        # Load ETL output
        if not self.etl_path.exists():
            result.checks.append(
                CheckResult(
                    name="etl_output_exists",
                    status=CheckStatus.FAIL,
                    message=f"ETL output file not found: {self.etl_path}",
                )
            )
            return result

        try:
            etl_df = pd.read_csv(self.etl_path)
            log.info(
                "Loaded ETL output", rows=len(etl_df), columns=list(etl_df.columns)
            )
        except Exception as e:
            result.checks.append(
                CheckResult(
                    name="etl_output_readable",
                    status=CheckStatus.FAIL,
                    message=f"Failed to read ETL output: {e}",
                )
            )
            return result

        # Run individual checks
        result.checks.append(self._check_dtv_values(etl_df))
        result.checks.append(self._check_counter_locations(etl_df))
        result.checks.append(self._check_infrastructure_values(etl_df))
        result.checks.append(self._check_regiostar_values(etl_df))
        result.checks.append(self._check_traffic_volume_values(etl_df))
        result.checks.append(self._check_derived_features(etl_df))
        result.checks.append(self._check_hub_distances(etl_df))
        result.checks.append(self._check_value_ranges(etl_df))

        log.info(
            "Assessment complete",
            overall=result.overall_status.value,
            passed=result.n_passed,
            failed=result.n_failed,
            warned=result.n_warned,
        )

        return result

    def _check_dtv_values(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check DZS_mean_SR values against recalculated DTV from measurements.

        The DTV (Daily Traffic Volume) is the mean of daily counts during
        the campaign period. This check verifies the ETL correctly computed
        these values.
        """
        if "DZS_mean_SR" not in etl_df.columns:
            return CheckResult(
                name="dtv_values",
                status=CheckStatus.SKIP,
                message="DZS_mean_SR column not in ETL output",
            )

        try:
            # Load measurements and recalculate DTV
            measurements = load_counter_measurements(self.config, validate=False)
            dtv_results = calculate_dtv(measurements, self.config.temporal)
            dtv_df = dtv_results_to_dataframe(dtv_results)

            # Create lookup by counter_id
            dtv_lookup = dtv_df.set_index("counter_id")["dtv"].to_dict()

            # Match ETL rows to expected DTV
            # ETL uses 'id' column for counter ID (3-char code like "071")
            if "id" not in etl_df.columns:
                return CheckResult(
                    name="dtv_values",
                    status=CheckStatus.SKIP,
                    message="id column not in ETL output for DTV matching",
                )

            matches = []
            mismatches = []

            for _, row in etl_df.iterrows():
                counter_id = str(row["id"]).strip()
                etl_dtv = row["DZS_mean_SR"]

                if pd.isna(etl_dtv):
                    continue

                # Try to find matching DTV
                expected_dtv = dtv_lookup.get(counter_id)
                if expected_dtv is None:
                    # Try with zero-padding
                    expected_dtv = dtv_lookup.get(counter_id.zfill(3))

                if expected_dtv is not None:
                    if np.isclose(etl_dtv, expected_dtv, rtol=self.FLOAT_RTOL):
                        matches.append(
                            {"id": counter_id, "etl": etl_dtv, "expected": expected_dtv}
                        )
                    else:
                        mismatches.append(
                            {
                                "id": counter_id,
                                "etl_value": etl_dtv,
                                "expected_value": expected_dtv,
                                "diff": abs(etl_dtv - expected_dtv),
                                "diff_pct": abs(etl_dtv - expected_dtv)
                                / max(expected_dtv, 1)
                                * 100,
                            }
                        )

            n_checked = len(matches) + len(mismatches)
            if n_checked == 0:
                return CheckResult(
                    name="dtv_values",
                    status=CheckStatus.WARN,
                    message="No DTV values could be matched to source data",
                )

            match_rate = len(matches) / n_checked
            status = self._rate_to_status(match_rate)

            sample_failures = None
            if mismatches:
                sample_failures = pd.DataFrame(mismatches[:10])

            return CheckResult(
                name="dtv_values",
                status=status,
                message=f"DTV values: {len(matches)}/{n_checked} match ({match_rate:.1%})",
                details="Compares DZS_mean_SR with recalculated DTV from counter measurements",
                n_checked=n_checked,
                n_passed=len(matches),
                n_failed=len(mismatches),
                sample_failures=sample_failures,
            )

        except Exception as e:
            log.error("DTV check failed", error=str(e))
            return CheckResult(
                name="dtv_values",
                status=CheckStatus.FAIL,
                message=f"DTV check failed: {e}",
            )

    def _check_counter_locations(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check lat/lon values against counter location source data.
        """
        if "lat" not in etl_df.columns or "lon" not in etl_df.columns:
            return CheckResult(
                name="counter_locations",
                status=CheckStatus.SKIP,
                message="lat/lon columns not in ETL output",
            )

        try:
            locations_df = load_counter_locations(self.config, validate=False)

            # Create lookup by name (which corresponds to id in ETL)
            location_lookup = {
                str(row["name"]).strip(): (row["latitude"], row["longitude"])
                for _, row in locations_df.iterrows()
            }

            matches = []
            mismatches = []

            for _, row in etl_df.iterrows():
                counter_id = str(row.get("id", "")).strip()
                etl_lat, etl_lon = row["lat"], row["lon"]

                if pd.isna(etl_lat) or pd.isna(etl_lon):
                    continue

                expected = location_lookup.get(counter_id)
                if expected is None:
                    expected = location_lookup.get(counter_id.zfill(3))

                if expected is not None:
                    exp_lat, exp_lon = expected
                    # Use small tolerance for coordinates (6 decimal places ~ 0.1m)
                    lat_match = np.isclose(etl_lat, exp_lat, atol=1e-6)
                    lon_match = np.isclose(etl_lon, exp_lon, atol=1e-6)

                    if lat_match and lon_match:
                        matches.append({"id": counter_id})
                    else:
                        mismatches.append(
                            {
                                "id": counter_id,
                                "etl_lat": etl_lat,
                                "etl_lon": etl_lon,
                                "expected_lat": exp_lat,
                                "expected_lon": exp_lon,
                            }
                        )

            n_checked = len(matches) + len(mismatches)
            if n_checked == 0:
                return CheckResult(
                    name="counter_locations",
                    status=CheckStatus.WARN,
                    message="No locations could be matched to source data",
                )

            match_rate = len(matches) / n_checked
            status = self._rate_to_status(match_rate)

            sample_failures = None
            if mismatches:
                sample_failures = pd.DataFrame(mismatches[:10])

            return CheckResult(
                name="counter_locations",
                status=status,
                message=f"Counter locations: {len(matches)}/{n_checked} match ({match_rate:.1%})",
                details="Compares lat/lon with counter location CSV",
                n_checked=n_checked,
                n_passed=len(matches),
                n_failed=len(mismatches),
                sample_failures=sample_failures,
            )

        except Exception as e:
            log.error("Location check failed", error=str(e))
            return CheckResult(
                name="counter_locations",
                status=CheckStatus.FAIL,
                message=f"Location check failed: {e}",
            )

    def _check_infrastructure_values(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check OSM_Radinfra values against traffic volume source data.

        The infrastructure values come from the traffic volume FlatGeoBuf file
        and are mapped to simplified categories (no, mixed_way, mit_road, etc.).
        """
        if "OSM_Radinfra" not in etl_df.columns:
            return CheckResult(
                name="infrastructure_values",
                status=CheckStatus.SKIP,
                message="OSM_Radinfra column not in ETL output",
            )

        try:
            traffic_gdf = load_traffic_volumes(self.config, validate=False)

            if "base_id" not in etl_df.columns or "base_id" not in traffic_gdf.columns:
                return CheckResult(
                    name="infrastructure_values",
                    status=CheckStatus.SKIP,
                    message="base_id column missing for infrastructure matching",
                )

            # Valid infrastructure categories from config
            valid_categories = {
                "no",
                "mixed_way",
                "mit_road",
                "bicycle_lane",
                "bicycle_road",
                "bicycle_way",
            }

            # Check that all ETL values are valid categories
            etl_infra = etl_df["OSM_Radinfra"].dropna()
            invalid = etl_infra[~etl_infra.isin(valid_categories)]

            n_checked = len(etl_infra)
            n_valid = n_checked - len(invalid)

            if n_checked == 0:
                return CheckResult(
                    name="infrastructure_values",
                    status=CheckStatus.WARN,
                    message="No infrastructure values to check",
                )

            match_rate = n_valid / n_checked
            status = self._rate_to_status(match_rate)

            sample_failures = None
            if len(invalid) > 0:
                sample_failures = pd.DataFrame(
                    {"invalid_value": invalid.head(10).tolist()}
                )

            return CheckResult(
                name="infrastructure_values",
                status=status,
                message=f"Infrastructure categories: {n_valid}/{n_checked} valid ({match_rate:.1%})",
                details=f"Valid categories: {sorted(valid_categories)}",
                n_checked=n_checked,
                n_passed=n_valid,
                n_failed=len(invalid),
                sample_failures=sample_failures,
            )

        except Exception as e:
            log.error("Infrastructure check failed", error=str(e))
            return CheckResult(
                name="infrastructure_values",
                status=CheckStatus.FAIL,
                message=f"Infrastructure check failed: {e}",
            )

    def _check_regiostar_values(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check RegioStaR5 values against structural data source.

        RegioStaR5 is a 5-level regional classification (1-5) from BBSR.
        """
        if "RegioStaR5" not in etl_df.columns:
            return CheckResult(
                name="regiostar_values",
                status=CheckStatus.SKIP,
                message="RegioStaR5 column not in ETL output",
            )

        try:
            # Load regiostar to verify source data exists
            _ = load_regiostar(self.config, validate=False)

            # RegioStaR5 should be integer values 51-55 (Hessen = federal state 5x)
            # or 11-15, 21-25, etc. depending on encoding
            etl_values = etl_df["RegioStaR5"].dropna()

            # Check value range (should be integer-like, typically 11-55)
            valid_range = etl_values.between(10, 60)
            n_checked = len(etl_values)
            n_valid = valid_range.sum()

            if n_checked == 0:
                return CheckResult(
                    name="regiostar_values",
                    status=CheckStatus.WARN,
                    message="No RegioStaR5 values to check",
                )

            # Also check distribution
            value_counts = etl_values.value_counts().head(10).to_dict()

            match_rate = n_valid / n_checked
            status = self._rate_to_status(match_rate)

            return CheckResult(
                name="regiostar_values",
                status=status,
                message=f"RegioStaR5 values: {n_valid}/{n_checked} in valid range ({match_rate:.1%})",
                details=f"Value distribution: {value_counts}",
                n_checked=n_checked,
                n_passed=int(n_valid),
                n_failed=n_checked - int(n_valid),
            )

        except Exception as e:
            log.error("RegioStaR check failed", error=str(e))
            return CheckResult(
                name="regiostar_values",
                status=CheckStatus.FAIL,
                message=f"RegioStaR check failed: {e}",
            )

    def _check_traffic_volume_values(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check Erh_SR (STADTRADELN volume) values against traffic volume source.

        Erh_SR is the raw count from the FlatGeoBuf traffic volume file,
        representing the number of STADTRADELN GPS traces on that edge.
        """
        if "Erh_SR" not in etl_df.columns:
            return CheckResult(
                name="traffic_volume_values",
                status=CheckStatus.SKIP,
                message="Erh_SR column not in ETL output",
            )

        try:
            traffic_gdf = load_traffic_volumes(self.config, validate=False)

            if "base_id" not in etl_df.columns or "base_id" not in traffic_gdf.columns:
                return CheckResult(
                    name="traffic_volume_values",
                    status=CheckStatus.SKIP,
                    message="base_id column missing for traffic volume matching",
                )

            # Create lookup from traffic data
            volume_lookup = traffic_gdf.set_index("base_id")["count"].to_dict()

            matches = []
            mismatches = []

            for _, row in etl_df.iterrows():
                base_id = row["base_id"]
                etl_volume = row["Erh_SR"]

                if pd.isna(etl_volume) or pd.isna(base_id):
                    continue

                expected = volume_lookup.get(base_id)
                if expected is not None:
                    if np.isclose(etl_volume, expected, rtol=self.FLOAT_RTOL):
                        matches.append({"base_id": base_id})
                    else:
                        mismatches.append(
                            {
                                "base_id": base_id,
                                "etl_value": etl_volume,
                                "expected_value": expected,
                                "diff": abs(etl_volume - expected),
                            }
                        )

            n_checked = len(matches) + len(mismatches)
            if n_checked == 0:
                return CheckResult(
                    name="traffic_volume_values",
                    status=CheckStatus.WARN,
                    message="No traffic volumes could be matched",
                )

            match_rate = len(matches) / n_checked
            status = self._rate_to_status(match_rate)

            sample_failures = None
            if mismatches:
                sample_failures = pd.DataFrame(mismatches[:10])

            return CheckResult(
                name="traffic_volume_values",
                status=status,
                message=f"Traffic volumes (Erh_SR): {len(matches)}/{n_checked} match ({match_rate:.1%})",
                details="Compares Erh_SR with count from traffic volume source",
                n_checked=n_checked,
                n_passed=len(matches),
                n_failed=len(mismatches),
                sample_failures=sample_failures,
            )

        except Exception as e:
            log.error("Traffic volume check failed", error=str(e))
            return CheckResult(
                name="traffic_volume_values",
                status=CheckStatus.FAIL,
                message=f"Traffic volume check failed: {e}",
            )

    def _check_derived_features(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check derived feature calculations (TN_SR_relativ, Streckengewicht_SR).

        These are computed as:
        - TN_SR_relativ (participation_rate) = n_users / population
        - Streckengewicht_SR (route_intensity) = (n_users * count) / population
        """
        checks_passed = 0
        checks_failed = 0
        details = []

        # Check TN_SR_relativ if dependencies exist
        if all(
            col in etl_df.columns for col in ["TN_SR_relativ", "n_users", "population"]
        ):
            expected = etl_df["n_users"] / etl_df["population"]
            matches = np.isclose(
                etl_df["TN_SR_relativ"].fillna(0),
                expected.fillna(0),
                rtol=self.FLOAT_RTOL,
                equal_nan=True,
            )
            n_match = matches.sum()
            n_total = len(etl_df)
            details.append(f"TN_SR_relativ: {n_match}/{n_total}")
            checks_passed += n_match
            checks_failed += n_total - n_match

        # Check Streckengewicht_SR if dependencies exist
        if all(
            col in etl_df.columns
            for col in ["Streckengewicht_SR", "n_users", "Erh_SR", "population"]
        ):
            expected = (etl_df["n_users"] * etl_df["Erh_SR"]) / etl_df["population"]
            matches = np.isclose(
                etl_df["Streckengewicht_SR"].fillna(0),
                expected.fillna(0),
                rtol=self.FLOAT_RTOL,
                equal_nan=True,
            )
            n_match = matches.sum()
            n_total = len(etl_df)
            details.append(f"Streckengewicht_SR: {n_match}/{n_total}")
            checks_passed += n_match
            checks_failed += n_total - n_match

        n_checked = checks_passed + checks_failed
        if n_checked == 0:
            return CheckResult(
                name="derived_features",
                status=CheckStatus.SKIP,
                message="Derived feature columns or dependencies not available",
            )

        match_rate = checks_passed / n_checked
        status = self._rate_to_status(match_rate)

        return CheckResult(
            name="derived_features",
            status=status,
            message=f"Derived features: {checks_passed}/{n_checked} match ({match_rate:.1%})",
            details="; ".join(details),
            n_checked=n_checked,
            n_passed=checks_passed,
            n_failed=checks_failed,
        )

    def _check_hub_distances(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check HubDist (distance to city center) values.

        HubDist should be positive distances in meters, typically 0-50000m.
        """
        if "HubDist" not in etl_df.columns:
            return CheckResult(
                name="hub_distances",
                status=CheckStatus.SKIP,
                message="HubDist column not in ETL output",
            )

        distances = etl_df["HubDist"].dropna()
        n_checked = len(distances)

        if n_checked == 0:
            return CheckResult(
                name="hub_distances",
                status=CheckStatus.WARN,
                message="No HubDist values to check",
            )

        # Validate: positive, reasonable range (< 100km)
        valid = (distances >= 0) & (distances < 100000)
        n_valid = valid.sum()

        stats = {
            "min": float(distances.min()),
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "median": float(distances.median()),
        }

        match_rate = n_valid / n_checked
        status = self._rate_to_status(match_rate)

        return CheckResult(
            name="hub_distances",
            status=status,
            message=f"HubDist values: {n_valid}/{n_checked} in valid range ({match_rate:.1%})",
            details=f"Stats: min={stats['min']:.0f}m, max={stats['max']:.0f}m, mean={stats['mean']:.0f}m",
            n_checked=n_checked,
            n_passed=int(n_valid),
            n_failed=n_checked - int(n_valid),
        )

    def _check_value_ranges(self, etl_df: pd.DataFrame) -> CheckResult:
        """
        Check that all numeric values are within reasonable ranges.
        """
        range_checks = {
            "DZS_mean_SR": (0, 50000),  # DTV: 0-50k bikes/day
            "Erh_SR": (0, 1e9),  # STADTRADELN count
            "TN_SR_relativ": (0, 1),  # Participation rate 0-100%
            "RegioStaR5": (1, 99),  # Classification code
            "lat": (47, 56),  # Germany latitude bounds
            "lon": (5, 16),  # Germany longitude bounds
        }

        results = []
        for col, (min_val, max_val) in range_checks.items():
            if col not in etl_df.columns:
                continue

            values = etl_df[col].dropna()
            if len(values) == 0:
                continue

            in_range = (values >= min_val) & (values <= max_val)
            n_valid = in_range.sum()
            n_total = len(values)
            results.append(
                {
                    "column": col,
                    "n_valid": n_valid,
                    "n_total": n_total,
                    "rate": n_valid / n_total if n_total > 0 else 1.0,
                }
            )

        if not results:
            return CheckResult(
                name="value_ranges",
                status=CheckStatus.SKIP,
                message="No numeric columns to check",
            )

        n_checked = sum(r["n_total"] for r in results)
        n_passed = sum(r["n_valid"] for r in results)
        match_rate = n_passed / n_checked if n_checked > 0 else 1.0

        status = self._rate_to_status(match_rate)
        details = ", ".join(
            f"{r['column']}: {r['n_valid']}/{r['n_total']}" for r in results
        )

        return CheckResult(
            name="value_ranges",
            status=status,
            message=f"Value ranges: {n_passed}/{n_checked} in bounds ({match_rate:.1%})",
            details=details,
            n_checked=n_checked,
            n_passed=n_passed,
            n_failed=n_checked - n_passed,
        )

    def _rate_to_status(self, rate: float) -> CheckStatus:
        """Convert match rate to status."""
        if rate >= self.WARN_THRESHOLD:
            return CheckStatus.PASS
        if rate >= self.FAIL_THRESHOLD:
            return CheckStatus.WARN
        return CheckStatus.FAIL
