"""
Typed configuration models using Pydantic.

All configuration is defined here with explicit typing and validation.
No hardcoded year/region logic in processing code.
"""

from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RegionConfig(BaseModel):
    """Geographic region configuration."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(description="Federal state code (e.g., '06' for Hessen)")
    name: str = Field(description="Human-readable region name")
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box as (min_lon, min_lat, max_lon, max_lat)"
    )

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Ensure code is a 2-digit string."""
        if not v.isdigit() or len(v) != 2:
            msg = f"Region code must be a 2-digit string, got: {v}"
            raise ValueError(msg)
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Ensure bbox is valid (min < max)."""
        min_lon, min_lat, max_lon, max_lat = v
        if min_lon >= max_lon:
            msg = f"min_lon ({min_lon}) must be < max_lon ({max_lon})"
            raise ValueError(msg)
        if min_lat >= max_lat:
            msg = f"min_lat ({min_lat}) must be < max_lat ({max_lat})"
            raise ValueError(msg)
        return v


class TemporalConfig(BaseModel):
    """Time-related configuration."""

    model_config = ConfigDict(frozen=True)

    year: int = Field(ge=2018, le=2030, description="Analysis year")
    campaign_start: date = Field(description="STADTRADELN campaign start date")
    campaign_end: date = Field(description="STADTRADELN campaign end date")
    counter_period_start: date = Field(description="Counter measurement period start")
    counter_period_end: date = Field(description="Counter measurement period end")
    holiday_start: date | None = Field(default=None, description="School holiday start")
    holiday_end: date | None = Field(default=None, description="School holiday end")

    @field_validator("campaign_end")
    @classmethod
    def validate_campaign_dates(cls, v: date, info: Any) -> date:
        """Ensure campaign_end is after campaign_start."""
        if "campaign_start" in info.data and v <= info.data["campaign_start"]:
            msg = "campaign_end must be after campaign_start"
            raise ValueError(msg)
        return v


class DataPathsConfig(BaseModel):
    """Data file paths configuration."""

    model_config = ConfigDict(frozen=True)

    data_root: Path = Field(description="Root directory for all data files")

    # Counter data
    counter_locations: Path = Field(description="Path to counter location CSV")
    counter_measurements: Path = Field(description="Path to counter measurements CSV")

    # Traffic volumes
    traffic_volumes: Path = Field(description="Path to STADTRADELN traffic volumes FGB")

    # Structural data
    municipalities: Path = Field(description="Path to VG250 municipalities GPKG")
    regiostar: Path = Field(description="Path to RegioStaR classification CSV")
    city_centroids: Path = Field(description="Path to city centroids GPKG")

    # STADTRADELN participation
    kommunen_stats: Path = Field(description="Path to kommunen statistics shapefile")
    campaign_stats: Path = Field(description="Path to campaign statistics CSV")

    def resolve(self, path_attr: str) -> Path:
        """Resolve a relative path against data_root."""
        rel_path = getattr(self, path_attr)
        return self.data_root / rel_path


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    model_config = ConfigDict(frozen=True)

    raw_columns: list[str] = Field(description="Raw columns from data sources")
    derived: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Derived feature definitions"
    )
    model_features: list[str] = Field(description="Final feature list for model")


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    model_config = ConfigDict(frozen=True)

    infrastructure_mapping: dict[str, str] = Field(
        description="Regex pattern -> category mapping for OSM infrastructure"
    )
    valid_infrastructure_categories: list[str] = Field(
        description="Valid infrastructure category values"
    )
    target_column: str = Field(default="dtv_value", description="Target column name")
    target_transformation: str = Field(
        default="log1p", description="Target transformation method"
    )


class TrainingConfig(BaseModel):
    """Model training configuration."""

    model_config = ConfigDict(frozen=True)

    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    cv_folds: int = Field(default=5, ge=2, le=20)
    random_state: int = Field(default=1337)
    min_dtv: int = Field(default=25, ge=0, description="Minimum DTV for training")
    max_dtv: int | None = Field(default=None, description="Maximum DTV for training")
    metrics: list[str] = Field(default_factory=lambda: ["r2", "rmse", "mae", "mape"])


class ModelConfig(BaseModel):
    """Model selection and hyperparameter configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: list[str] = Field(description="List of enabled model names")
    hyperparameters: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Hyperparameter grids per model"
    )


class MLflowConfig(BaseModel):
    """MLflow experiment tracking configuration."""

    model_config = ConfigDict(frozen=True)

    tracking_uri: str = Field(default="http://127.0.0.1:5000")
    artifact_location: Path = Field(default=Path("./mlartifacts"))
    experiment_name: str = Field(description="MLflow experiment name")


class OutputConfig(BaseModel):
    """Output paths configuration."""

    model_config = ConfigDict(frozen=True)

    plots_dir: Path = Field(default=Path("./plots"))
    predictions_dir: Path = Field(default=Path("./predictions"))
    cache_dir: Path = Field(default=Path("./cache"))


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    project_name: str = Field(default="Bicycle Traffic Estimation")
    project_version: str = Field(default="0.1.0")

    region: RegionConfig
    temporal: TemporalConfig
    data_paths: DataPathsConfig
    features: FeatureConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    models: ModelConfig
    mlflow: MLflowConfig
    output: OutputConfig

    @property
    def year(self) -> int:
        """Convenience accessor for the analysis year."""
        return self.temporal.year

    @property
    def region_code(self) -> str:
        """Convenience accessor for the region code."""
        return self.region.code
