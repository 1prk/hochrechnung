"""Tests for the modeling data loading module."""

from pathlib import Path

import pandas as pd
import pytest

from hochrechnung.config.settings import (
    DataPathsConfig,
    FeatureConfig,
    MLflowConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    RegionConfig,
    TemporalConfig,
    TrainingConfig,
)
from hochrechnung.modeling.data import (
    ETL_TO_MODEL_COLUMNS,
    MODEL_TO_ETL_COLUMNS,
    TrainingData,
    _compute_target_stats,
    _filter_by_dtv,
    _get_feature_columns,
    _get_target_column,
    _rename_columns,
    auto_detect_data_path,
    load_training_data,
)


@pytest.fixture
def sample_etl_output() -> pd.DataFrame:
    """Create sample ETL output data matching legacy format."""
    return pd.DataFrame(
        {
            "OSM_Radinfra": [
                "bicycle_lane",
                "bicycle_way",
                "mixed_way",
                "no",
                "mit_road",
            ],
            "TN_SR_relativ": [0.05, 0.03, 0.08, 0.02, 0.04],
            "Streckengewicht_SR": [1.2, 0.8, 1.5, 0.5, 1.0],
            "RegioStaR5": [1, 2, 1, 3, 2],
            "Erh_SR": [150, 80, 200, 50, 120],
            "HubDist": [1000.0, 2500.0, 500.0, 5000.0, 1500.0],
            "id": ["001", "002", "003", "004", "005"],
            "base_id": [100, 101, 102, 103, 104],
            "lat": [50.1, 50.2, 50.3, 50.4, 50.5],
            "lon": [8.6, 8.7, 8.8, 8.9, 9.0],
            "DZS_mean_SR": [150.0, 80.0, 250.0, 30.0, 120.0],
        }
    )


@pytest.fixture
def test_config() -> PipelineConfig:
    """Create a minimal test configuration."""
    from datetime import date

    return PipelineConfig(
        project_name="Test",
        project_version="0.1.0",
        region=RegionConfig(
            code="06",
            name="Hessen",
            bbox=(7.77, 49.39, 10.24, 51.66),
        ),
        temporal=TemporalConfig(
            year=2023,
            campaign_start=date(2023, 5, 1),
            campaign_end=date(2023, 9, 30),
            counter_period_start=date(2023, 5, 1),
            counter_period_end=date(2023, 9, 30),
        ),
        data_paths=DataPathsConfig(
            data_root=Path("./data"),
            counter_locations=Path("counter-locations/test.csv"),
            counter_measurements=Path("counts/test.csv"),
            traffic_volumes=Path("trafficvolumes/test.fgb"),
            municipalities=Path("structural-data/DE_VG250.gpkg"),
            regiostar=Path("structural-data/regiostar.csv"),
            city_centroids=Path("structural-data/centroids.gpkg"),
            kommunen_stats=Path("kommunen-stats/test.shp"),
            campaign_stats=Path("campaign/test.csv"),
        ),
        features=FeatureConfig(
            raw_columns=["count", "population"],
            derived={},
            model_features=["infra_category", "participation_rate", "route_intensity"],
        ),
        preprocessing=PreprocessingConfig(
            infrastructure_mapping={},
            valid_infrastructure_categories=["no", "bicycle_lane"],
            target_column="dtv",
            target_transformation="log1p",
        ),
        training=TrainingConfig(
            test_size=0.2,
            cv_folds=5,
            random_state=42,
            min_dtv=25,
            max_dtv=None,
        ),
        models=ModelConfig(
            enabled=["Random Forest"],
            hyperparameters={},
        ),
        mlflow=MLflowConfig(
            tracking_uri="http://localhost:5000",
            artifact_location=Path("./mlartifacts"),
            experiment_name="test",
        ),
        output=OutputConfig(
            plots_dir=Path("./plots"),
            predictions_dir=Path("./predictions"),
            cache_dir=Path("./cache"),
        ),
    )


class TestColumnMapping:
    """Tests for column mapping dictionaries."""

    def test_etl_to_model_columns_complete(self) -> None:
        """Column mapping should cover all expected ETL columns."""
        expected_etl_columns = {
            "OSM_Radinfra",
            "TN_SR_relativ",
            "Streckengewicht_SR",
            "RegioStaR5",
            "Erh_SR",
            "HubDist",
            "DZS_mean_SR",
        }
        assert expected_etl_columns == set(ETL_TO_MODEL_COLUMNS.keys())

    def test_reverse_mapping_consistent(self) -> None:
        """Reverse mapping should be consistent with forward mapping."""
        for etl_name, model_name in ETL_TO_MODEL_COLUMNS.items():
            assert MODEL_TO_ETL_COLUMNS[model_name] == etl_name


class TestRenameColumns:
    """Tests for _rename_columns function."""

    def test_renames_all_mapped_columns(self, sample_etl_output: pd.DataFrame) -> None:
        """Should rename all columns that have mappings."""
        df = _rename_columns(sample_etl_output)

        # Check renamed columns exist
        assert "infra_category" in df.columns
        assert "participation_rate" in df.columns
        assert "route_intensity" in df.columns
        assert "regiostar5" in df.columns
        assert "stadtradeln_volume" in df.columns
        assert "dist_to_center_m" in df.columns
        assert "dtv" in df.columns

        # Check original names no longer exist
        assert "OSM_Radinfra" not in df.columns
        assert "DZS_mean_SR" not in df.columns

    def test_preserves_unmapped_columns(self, sample_etl_output: pd.DataFrame) -> None:
        """Should preserve columns without mappings."""
        df = _rename_columns(sample_etl_output)

        assert "id" in df.columns
        assert "base_id" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns


class TestGetTargetColumn:
    """Tests for _get_target_column function."""

    def test_finds_dtv_column(self, test_config: PipelineConfig) -> None:
        """Should find 'dtv' column in renamed data."""
        df = pd.DataFrame({"dtv": [100], "other": [1]})
        assert _get_target_column(df, test_config) == "dtv"

    def test_finds_legacy_column(self, test_config: PipelineConfig) -> None:
        """Should find legacy 'DZS_mean_SR' column."""
        df = pd.DataFrame({"DZS_mean_SR": [100], "other": [1]})
        assert _get_target_column(df, test_config) == "DZS_mean_SR"

    def test_raises_on_missing_target(self, test_config: PipelineConfig) -> None:
        """Should raise ValueError when target column is missing."""
        df = pd.DataFrame({"other": [1]})
        with pytest.raises(ValueError, match="Target column not found"):
            _get_target_column(df, test_config)


class TestFilterByDtv:
    """Tests for _filter_by_dtv function."""

    def test_filters_by_min_dtv(self, test_config: PipelineConfig) -> None:
        """Should filter out rows below min_dtv."""
        df = pd.DataFrame({"dtv": [10, 30, 50, 100, 20]})
        filtered = _filter_by_dtv(df, "dtv", test_config)

        # min_dtv=25 in test_config, so 10 and 20 should be filtered
        assert len(filtered) == 3
        assert (filtered["dtv"] >= 25).all()

    def test_filters_nan_values(self, test_config: PipelineConfig) -> None:
        """Should filter out NaN values in target."""
        df = pd.DataFrame({"dtv": [30.0, None, 50.0]})
        filtered = _filter_by_dtv(df, "dtv", test_config)

        assert len(filtered) == 2
        assert bool(filtered["dtv"].notna().all())

    def test_respects_max_dtv(self) -> None:
        """Should filter out rows above max_dtv when specified."""
        from datetime import date

        config = PipelineConfig(
            project_name="Test",
            project_version="0.1.0",
            region=RegionConfig(code="06", name="Test", bbox=(0, 0, 1, 1)),
            temporal=TemporalConfig(
                year=2023,
                campaign_start=date(2023, 1, 1),
                campaign_end=date(2023, 12, 31),
                counter_period_start=date(2023, 1, 1),
                counter_period_end=date(2023, 12, 31),
            ),
            data_paths=DataPathsConfig(
                data_root=Path(),
                counter_locations=Path("test.csv"),
                counter_measurements=Path("test.csv"),
                traffic_volumes=Path("test.fgb"),
                municipalities=Path("test.gpkg"),
                regiostar=Path("test.csv"),
                city_centroids=Path("test.gpkg"),
                kommunen_stats=Path("test.shp"),
                campaign_stats=Path("test.csv"),
            ),
            features=FeatureConfig(raw_columns=[], derived={}, model_features=[]),
            preprocessing=PreprocessingConfig(
                infrastructure_mapping={},
                valid_infrastructure_categories=[],
            ),
            training=TrainingConfig(min_dtv=0, max_dtv=100),
            models=ModelConfig(enabled=[], hyperparameters={}),
            mlflow=MLflowConfig(
                tracking_uri="http://localhost:5000",
                artifact_location=Path(),
                experiment_name="test",
            ),
            output=OutputConfig(
                plots_dir=Path(),
                predictions_dir=Path(),
                cache_dir=Path(),
            ),
        )

        df = pd.DataFrame({"dtv": [30, 50, 150, 80]})
        filtered = _filter_by_dtv(df, "dtv", config)

        assert len(filtered) == 3
        assert (filtered["dtv"] <= 100).all()


class TestGetFeatureColumns:
    """Tests for _get_feature_columns function."""

    def test_uses_config_features_when_available(
        self, sample_etl_output: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Should use config model_features when they exist in data."""
        df = _rename_columns(sample_etl_output)
        features = _get_feature_columns(df, "dtv", test_config)

        # Config specifies these features
        assert "infra_category" in features
        assert "participation_rate" in features
        assert "route_intensity" in features

    def test_excludes_metadata_columns(
        self, sample_etl_output: pd.DataFrame, test_config: PipelineConfig
    ) -> None:
        """Should exclude metadata columns like id, lat, lon."""
        df = _rename_columns(sample_etl_output)

        # Override config to empty features for default behavior
        config = test_config.model_copy()
        object.__setattr__(
            config.features,
            "_model_features",
            [],
        )

        features = _get_feature_columns(df, "dtv", test_config)

        # Metadata should be excluded
        assert "id" not in features
        assert "base_id" not in features
        assert "lat" not in features
        assert "lon" not in features
        assert "dtv" not in features  # target should be excluded


class TestComputeTargetStats:
    """Tests for _compute_target_stats function."""

    def test_computes_all_statistics(self) -> None:
        """Should compute all required statistics."""
        y = pd.Series([100, 150, 200, 250, 300])
        stats = _compute_target_stats(y)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        assert "q25" in stats
        assert "q75" in stats

    def test_statistics_are_accurate(self) -> None:
        """Statistics should be mathematically correct."""
        y = pd.Series([10, 20, 30, 40, 50])
        stats = _compute_target_stats(y)

        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["median"] == 30.0


class TestLoadTrainingData:
    """Tests for load_training_data function."""

    def test_loads_and_splits_data(
        self,
        sample_etl_output: pd.DataFrame,
        test_config: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """Should load data and create train/test splits."""
        # Save sample data to temp file
        data_path = tmp_path / "training_data.csv"
        sample_etl_output.to_csv(data_path, index=False)

        result = load_training_data(data_path, test_config)

        assert isinstance(result, TrainingData)
        assert len(result.X_train) + len(result.X_test) == result.n_samples
        assert len(result.y_train) == len(result.X_train)
        assert len(result.y_test) == len(result.X_test)

    def test_raises_on_missing_file(self, test_config: PipelineConfig) -> None:
        """Should raise FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_training_data(Path("/nonexistent/data.csv"), test_config)

    def test_respects_test_size(
        self,
        sample_etl_output: pd.DataFrame,
        test_config: PipelineConfig,
        tmp_path: Path,
    ) -> None:
        """Should respect test_size from config."""
        # Create larger sample for meaningful split
        df = pd.concat([sample_etl_output] * 10, ignore_index=True)
        data_path = tmp_path / "training_data.csv"
        df.to_csv(data_path, index=False)

        result = load_training_data(data_path, test_config)

        # test_size=0.2 means ~20% test
        # With 50 samples (5*10) and min_dtv filtering, expect roughly 80/20 split
        total_after_filter = result.n_samples
        expected_test = int(total_after_filter * 0.2)
        assert abs(len(result.X_test) - expected_test) <= 2  # Allow small variance


class TestAutoDetectDataPath:
    """Tests for auto_detect_data_path function."""

    def test_finds_year_specific_file(
        self, test_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Should find training_data_{year}.csv in cache dir."""
        # Create cache dir and file
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        expected_file = cache_dir / "training_data_2023.csv"
        expected_file.write_text("test")

        # Update config to use temp cache dir
        config = test_config.model_copy()
        object.__setattr__(config.output, "cache_dir", cache_dir)

        result = auto_detect_data_path(config)
        assert result == expected_file

    def test_raises_when_not_found(
        self, test_config: PipelineConfig, tmp_path: Path
    ) -> None:
        """Should raise FileNotFoundError when no data file exists."""
        # Create empty cache dir
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        config = test_config.model_copy()
        object.__setattr__(config.output, "cache_dir", cache_dir)

        with pytest.raises(FileNotFoundError, match="No training data found"):
            auto_detect_data_path(config)
