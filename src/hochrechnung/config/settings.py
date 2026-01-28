"""
Typed configuration models using Pydantic.

All configuration is defined here with explicit typing and validation.
No hardcoded year/region logic in processing code.
"""

from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StatisticsApproach(str, Enum):
    """Approach for loading STADTRADELN statistics."""

    LEGACY = "legacy"  # kommunen_stats shapefiles + VG250
    GEBIETSEINHEITEN = "gebietseinheiten"  # DE_Gebietseinheiten + aggregated JSON


class StatsConfig(BaseModel):
    """Configuration for STADTRADELN statistics loading."""

    model_config = ConfigDict(frozen=True)

    approach: StatisticsApproach = Field(
        default=StatisticsApproach.LEGACY,
        description="Approach for loading statistics",
    )
    admin_level: str = Field(
        default="Verwaltungsgemeinschaft",
        description="Administrative level for aggregation (Verwaltungsgemeinschaft, Kreis, Land)",
    )


class RegionConfig(BaseModel):
    """Geographic region configuration using ARS (Amtlicher Regionalschlüssel)."""

    model_config = ConfigDict(frozen=True)

    ars: str = Field(description="12-digit Amtlicher Regionalschlüssel")
    name: str | None = Field(default=None, description="Human-readable region name")

    @field_validator("ars")
    @classmethod
    def validate_ars(cls, v: str) -> str:
        """Ensure ARS is a 12-digit string."""
        if not v.isdigit() or len(v) != 12:
            msg = f"ARS must be a 12-digit string, got: {v!r} (length {len(v)})"
            raise ValueError(msg)
        return v

    @property
    def land_code(self) -> str:
        """Federal state code (first 2 digits of ARS)."""
        return self.ars[:2]

    @property
    def regierungsbezirk_code(self) -> str:
        """Regierungsbezirk code (digits 1-3 of ARS)."""
        return self.ars[:3]

    @property
    def kreis_code(self) -> str:
        """Kreis/county code (digits 1-5 of ARS)."""
        return self.ars[:5]

    @property
    def gemeinde_code(self) -> str:
        """Gemeinde/municipality code (full 12 digits)."""
        return self.ars


class TemporalConfig(BaseModel):
    """Time-related configuration with single analysis period."""

    model_config = ConfigDict(frozen=True)

    year: int = Field(ge=2018, le=2030, description="Analysis year")
    period_start: date = Field(description="Analysis period start date")
    period_end: date = Field(description="Analysis period end date")

    @field_validator("period_end")
    @classmethod
    def validate_period_dates(cls, v: date, info: Any) -> date:
        """Ensure period_end is after period_start."""
        if "period_start" in info.data and v <= info.data["period_start"]:
            msg = "period_end must be after period_start"
            raise ValueError(msg)
        return v

    @property
    def period_days(self) -> int:
        """Number of days in the analysis period."""
        return (self.period_end - self.period_start).days + 1

    # Backwards compatibility aliases
    @property
    def campaign_start(self) -> date:
        """Alias for period_start (backwards compatibility)."""
        return self.period_start

    @property
    def campaign_end(self) -> date:
        """Alias for period_end (backwards compatibility)."""
        return self.period_end


class DataPathsConfig(BaseModel):
    """Data file paths configuration.

    All paths are relative to data_root. Use resolve() to get absolute paths.

    Required for all modes:
        - traffic_volumes: STADTRADELN GPS trace volumes

    Required for training only:
        - counter_locations: Counter station locations
        - counter_measurements: Counter measurement data

    Germany-wide defaults (can be overridden):
        - osm_pbf, municipalities, regiostar, city_centroids
        - kommunen_stats, campaign_stats
    """

    model_config = ConfigDict(frozen=True)

    data_root: Path = Field(
        default=Path("./data"), description="Root directory for all data files"
    )

    # Always required
    traffic_volumes: Path = Field(description="Path to STADTRADELN traffic volumes FGB")

    # Required for training, optional for prediction
    counter_locations: Path | None = Field(
        default=None, description="Path to counter location CSV (required for training)"
    )
    counter_measurements: Path | None = Field(
        default=None,
        description="Path to counter measurements CSV (required for training)",
    )

    # Germany-wide defaults (can be overridden per project)
    # TODO: make the years from germany year-agnostic
    osm_pbf: Path = Field(
        default=Path("osm-data/germany-230101.osm.pbf"),
        description="Path to OpenStreetMap PBF file",
    )
    municipalities: Path = Field(
        default=Path("structural-data/DE_VG250.gpkg"),
        description="Path to VG250 municipalities GPKG",
    )
    regiostar: Path = Field(
        default=Path("structural-data/regiostar_2022.csv"),
        description="Path to RegioStaR classification CSV",
    )
    city_centroids: Path = Field(
        default=Path("structural-data/places.gpkg"),
        description="Path to city centroids GPKG",
    )
    kommunen_stats: Path = Field(
        default=Path("kommunen-stats/kommunen_stats.shp"),
        description="Path to kommunen statistics shapefile",
    )
    campaign_stats: Path = Field(
        default=Path("campaign/SR_TeilnehmendeKommunen.csv"),
        description="Path to campaign statistics CSV",
    )
    gebietseinheiten: Path = Field(
        default=Path("structural-data/DE_Gebietseinheiten.gpkg"),
        description="Path to DE_Gebietseinheiten GPKG",
    )

    def resolve(self, path_attr: str) -> Path:
        """Resolve a relative path against data_root."""
        rel_path = getattr(self, path_attr)
        if rel_path is None:
            msg = f"Path '{path_attr}' is not configured"
            raise ValueError(msg)
        return self.data_root / rel_path

    def validate_for_training(self) -> None:
        """Validate that all required paths for training are configured."""
        missing = []
        if self.counter_locations is None:
            missing.append("counter_locations")
        if self.counter_measurements is None:
            missing.append("counter_measurements")
        if missing:
            msg = f"Training requires: {', '.join(missing)}"
            raise ValueError(msg)


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
    cv_folds: int = Field(default=10, ge=2, le=20)  # Paper methodology (Section 3.5)
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
    # experiment_name is optional; derived from project_name if not set
    experiment_name: str | None = Field(
        default=None, description="MLflow experiment name (defaults to project name)"
    )


class OutputConfig(BaseModel):
    """Output paths configuration.

    Paths are derived from output_root and project name.
    Structure: ./output/{project}/predictions, ./output/{project}/plots, etc.
    """

    model_config = ConfigDict(frozen=True)

    output_root: Path = Field(
        default=Path("./output"), description="Root directory for all outputs"
    )


class CuratedConfig(BaseModel):
    """Curated Germany-wide data configuration."""

    model_config = ConfigDict(frozen=True)

    path: Path | None = Field(
        default=None,
        description="Path to curated counter data CSV (relative to data_root)",
    )
    city_centroids: Path | None = Field(
        default=None,
        description="Path to city centroids for distance calculation",
    )


class PipelineConfig(BaseModel):
    """Complete pipeline configuration.

    The project name drives:
    - MLflow experiment name (if not explicitly set)
    - Output directory structure: ./output/{project}/
    """

    model_config = ConfigDict(frozen=True)

    # Project identifier - used for MLflow experiment and output paths
    project: str = Field(description="Project identifier (e.g., 'hessen-2023')")

    region: RegionConfig
    temporal: TemporalConfig
    data_paths: DataPathsConfig
    features: FeatureConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    models: ModelConfig
    mlflow: MLflowConfig
    output: OutputConfig
    curated: CuratedConfig = Field(default_factory=CuratedConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)

    @property
    def year(self) -> int:
        """Convenience accessor for the analysis year."""
        return self.temporal.year

    @property
    def ars(self) -> str:
        """Convenience accessor for the ARS."""
        return self.region.ars

    @property
    def land_code(self) -> str:
        """Federal state code (first 2 digits of ARS)."""
        return self.region.land_code

    # Backwards compatibility
    @property
    def region_code(self) -> str:
        """Alias for land_code (backwards compatibility)."""
        return self.land_code

    @property
    def experiment_name(self) -> str:
        """MLflow experiment name (derived from project if not set)."""
        return self.mlflow.experiment_name or self.project

    # Output path helpers
    @property
    def predictions_dir(self) -> Path:
        """Path to predictions output directory."""
        return self.output.output_root / self.project / "predictions"

    @property
    def plots_dir(self) -> Path:
        """Path to plots output directory."""
        return self.output.output_root / self.project / "plots"

    @property
    def cache_dir(self) -> Path:
        """Path to cache directory."""
        return self.output.output_root / self.project / "cache"

    @property
    def artifacts_dir(self) -> Path:
        """Path to MLflow artifacts directory."""
        return self.output.output_root / self.project / "mlartifacts"
