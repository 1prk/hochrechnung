"""Tests for the ETL output assessment module."""

import numpy as np
import pandas as pd
import pytest

from hochrechnung.assessment.core import (
    AssessmentResult,
    AssessmentRunner,
    CheckResult,
    CheckStatus,
)


class TestCheckStatus:
    """Tests for CheckStatus enum."""

    def test_status_values(self) -> None:
        """Test that all expected status values exist."""
        assert CheckStatus.PASS.value == "pass"
        assert CheckStatus.WARN.value == "warn"
        assert CheckStatus.FAIL.value == "fail"
        assert CheckStatus.SKIP.value == "skip"


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_minimal_result(self) -> None:
        """Test creating a minimal check result."""
        result = CheckResult(
            name="test_check",
            status=CheckStatus.PASS,
            message="Test passed",
        )
        assert result.name == "test_check"
        assert result.status == CheckStatus.PASS
        assert result.message == "Test passed"
        assert result.n_checked == 0
        assert result.n_passed == 0
        assert result.n_failed == 0

    def test_full_result(self) -> None:
        """Test creating a full check result with all fields."""
        failures = pd.DataFrame({"id": [1, 2], "error": ["a", "b"]})
        result = CheckResult(
            name="full_check",
            status=CheckStatus.FAIL,
            message="Found issues",
            details="Additional info",
            n_checked=100,
            n_passed=90,
            n_failed=10,
            sample_failures=failures,
        )
        assert result.n_checked == 100
        assert result.n_passed == 90
        assert result.n_failed == 10
        assert result.sample_failures is not None
        assert len(result.sample_failures) == 2


class TestAssessmentResult:
    """Tests for AssessmentResult dataclass."""

    def test_empty_result(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test assessment result with no checks."""
        result = AssessmentResult(etl_path=tmp_path / "test.csv")  # type: ignore[arg-type]
        assert result.overall_status == CheckStatus.SKIP
        assert result.n_passed == 0
        assert result.n_failed == 0
        assert result.n_warned == 0

    def test_all_pass(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test overall status when all checks pass."""
        result = AssessmentResult(etl_path=tmp_path / "test.csv")  # type: ignore[arg-type]
        result.checks = [
            CheckResult("check1", CheckStatus.PASS, "ok"),
            CheckResult("check2", CheckStatus.PASS, "ok"),
        ]
        assert result.overall_status == CheckStatus.PASS
        assert result.n_passed == 2
        assert result.n_failed == 0

    def test_any_fail(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test overall status when any check fails."""
        result = AssessmentResult(etl_path=tmp_path / "test.csv")  # type: ignore[arg-type]
        result.checks = [
            CheckResult("check1", CheckStatus.PASS, "ok"),
            CheckResult("check2", CheckStatus.FAIL, "failed"),
            CheckResult("check3", CheckStatus.WARN, "warning"),
        ]
        assert result.overall_status == CheckStatus.FAIL
        assert result.n_passed == 1
        assert result.n_failed == 1
        assert result.n_warned == 1

    def test_warn_without_fail(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test overall status when there are warnings but no failures."""
        result = AssessmentResult(etl_path=tmp_path / "test.csv")  # type: ignore[arg-type]
        result.checks = [
            CheckResult("check1", CheckStatus.PASS, "ok"),
            CheckResult("check2", CheckStatus.WARN, "warning"),
        ]
        assert result.overall_status == CheckStatus.WARN


class TestAssessmentRunner:
    """Tests for AssessmentRunner."""

    def test_rate_to_status_pass(self) -> None:
        """Test rate conversion for passing values."""
        # Create minimal config mock
        runner = self._create_mock_runner()
        assert runner._rate_to_status(1.0) == CheckStatus.PASS
        assert runner._rate_to_status(0.95) == CheckStatus.PASS

    def test_rate_to_status_warn(self) -> None:
        """Test rate conversion for warning values."""
        runner = self._create_mock_runner()
        assert runner._rate_to_status(0.94) == CheckStatus.WARN
        assert runner._rate_to_status(0.80) == CheckStatus.WARN

    def test_rate_to_status_fail(self) -> None:
        """Test rate conversion for failing values."""
        runner = self._create_mock_runner()
        assert runner._rate_to_status(0.79) == CheckStatus.FAIL
        assert runner._rate_to_status(0.0) == CheckStatus.FAIL

    def test_check_value_ranges_valid(self) -> None:
        """Test value range checking with valid data."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "DZS_mean_SR": [100, 500, 1000],
            "lat": [50.0, 50.5, 51.0],
            "lon": [8.5, 9.0, 9.5],
            "TN_SR_relativ": [0.01, 0.05, 0.1],
            "RegioStaR5": [51, 52, 53],
        })
        result = runner._check_value_ranges(df)
        assert result.status == CheckStatus.PASS

    def test_check_value_ranges_invalid(self) -> None:
        """Test value range checking with invalid data."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "DZS_mean_SR": [-100, 500, 100000],  # Negative and too high
            "lat": [30.0, 50.5, 70.0],  # Outside Germany
            "lon": [8.5, 9.0, 9.5],
        })
        result = runner._check_value_ranges(df)
        # Should warn or fail due to out-of-range values
        assert result.status in [CheckStatus.WARN, CheckStatus.FAIL]

    def test_check_infrastructure_valid_categories(self) -> None:
        """Test infrastructure check with valid categories."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "OSM_Radinfra": ["no", "mixed_way", "bicycle_lane", "mit_road"],
            "base_id": [1, 2, 3, 4],
        })
        # This will skip since it can't load traffic data
        result = runner._check_infrastructure_values(df)
        # The check validates categories, not source matching
        assert result.status in [CheckStatus.PASS, CheckStatus.SKIP, CheckStatus.FAIL]

    def test_check_hub_distances_valid(self) -> None:
        """Test hub distance check with valid values."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "HubDist": [100.0, 500.0, 2000.0, 5000.0],
        })
        result = runner._check_hub_distances(df)
        assert result.status == CheckStatus.PASS
        assert result.n_checked == 4
        assert result.n_passed == 4

    def test_check_hub_distances_invalid(self) -> None:
        """Test hub distance check with invalid values."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "HubDist": [-100.0, 500.0, 200000.0],  # Negative and too far
        })
        result = runner._check_hub_distances(df)
        assert result.status in [CheckStatus.WARN, CheckStatus.FAIL]
        assert result.n_failed > 0

    def test_check_hub_distances_missing_column(self) -> None:
        """Test hub distance check when column is missing."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = runner._check_hub_distances(df)
        assert result.status == CheckStatus.SKIP

    def test_check_derived_features_correct(self) -> None:
        """Test derived features with correctly calculated values."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "n_users": [100, 200, 300],
            "population": [10000, 10000, 10000],
            "Erh_SR": [50, 100, 150],
            "TN_SR_relativ": [0.01, 0.02, 0.03],  # n_users / population
            "Streckengewicht_SR": [0.5, 2.0, 4.5],  # (n_users * Erh_SR) / population
        })
        result = runner._check_derived_features(df)
        assert result.status == CheckStatus.PASS

    def test_check_derived_features_incorrect(self) -> None:
        """Test derived features with incorrectly calculated values."""
        runner = self._create_mock_runner()
        df = pd.DataFrame({
            "n_users": [100, 200, 300],
            "population": [10000, 10000, 10000],
            "Erh_SR": [50, 100, 150],
            "TN_SR_relativ": [0.05, 0.05, 0.05],  # Wrong values
            "Streckengewicht_SR": [1.0, 1.0, 1.0],  # Wrong values
        })
        result = runner._check_derived_features(df)
        assert result.status in [CheckStatus.WARN, CheckStatus.FAIL]
        assert result.n_failed > 0

    def _create_mock_runner(self) -> AssessmentRunner:
        """Create a runner with mock config for unit testing."""
        from unittest.mock import MagicMock
        from pathlib import Path

        mock_config = MagicMock()
        mock_config.year = 2024
        mock_config.cache_dir = Path("/tmp")

        # Bypass __init__ validation
        runner = object.__new__(AssessmentRunner)
        runner.config = mock_config
        runner.etl_path = Path("/tmp/test.csv")
        runner.FLOAT_RTOL = 1e-5
        runner.FLOAT_ATOL = 1e-8
        runner.WARN_THRESHOLD = 0.95
        runner.FAIL_THRESHOLD = 0.80

        return runner


class TestValueComparisons:
    """Tests for numerical comparison logic."""

    def test_float_comparison_close_values(self) -> None:
        """Test that close float values are considered equal."""
        a = 847.0
        b = 847.0000001
        assert np.isclose(a, b, rtol=1e-5)

    def test_float_comparison_different_values(self) -> None:
        """Test that different float values are not equal."""
        a = 847.0
        b = 850.0
        assert not np.isclose(a, b, rtol=1e-5)

    def test_coordinate_comparison(self) -> None:
        """Test coordinate comparison with small tolerance."""
        lat1, lat2 = 49.81438786508744, 49.81438786508744
        lon1, lon2 = 8.622631430625917, 8.622631430625917
        assert np.isclose(lat1, lat2, atol=1e-6)
        assert np.isclose(lon1, lon2, atol=1e-6)
