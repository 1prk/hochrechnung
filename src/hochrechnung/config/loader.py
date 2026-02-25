"""
Configuration loading utilities.

Supports environment variable interpolation and config inheritance.
Minimal configs only need: project, ars, year, period, data.traffic_volumes
"""

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from hochrechnung.config.settings import (
    CalibrationConfig,
    CalibratorType,
    CuratedConfig,
    DataPathsConfig,
    FeatureConfig,
    MLflowConfig,
    ModelConfig,
    OutputConfig,
    PipelineConfig,
    PreprocessingConfig,
    RegionConfig,
    StatisticsApproach,
    StatsConfig,
    TemporalConfig,
    TrainingConfig,
)


def _interpolate_env_vars(value: str) -> str:
    """
    Interpolate environment variables in string values.

    Supports ${VAR} and ${VAR:default} syntax.
    """
    pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(2)
        return os.environ.get(var_name, default if default is not None else "")

    return re.sub(pattern, replacer, value)


def _process_config_values(obj: Any) -> Any:
    """Recursively process config values for env var interpolation."""
    if isinstance(obj, str):
        return _interpolate_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    return obj


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_date(value: Any) -> date:
    """Parse a date from string or return as-is if already a date."""
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    msg = f"Cannot parse date from {type(value)}: {value}"
    raise ValueError(msg)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and process environment variables."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _process_config_values(data) if data else {}


def load_config(
    config_path: Path,
    base_path: Path | None = None,
) -> PipelineConfig:
    """
    Load pipeline configuration from YAML file(s).

    Minimal config requires only:
        - project: str
        - ars: str (12-digit)
        - year: int
        - period.start, period.end: date strings
        - data.traffic_volumes: path

    Args:
        config_path: Path to the main configuration file.
        base_path: Optional path to base configuration for inheritance.

    Returns:
        Fully validated PipelineConfig instance.
    """
    # Load base config if provided
    if base_path is not None:
        base_data = load_yaml(base_path)
    else:
        # Try to find base.yaml in same directory
        potential_base = config_path.parent / "base.yaml"
        base_data = load_yaml(potential_base) if potential_base.exists() else {}

    # Load main config
    main_data = load_yaml(config_path)

    # Merge configs (main overrides base)
    merged = _deep_merge(base_data, main_data)

    # Extract project name (required)
    project = merged.get("project")
    if not project:
        msg = "Config must specify 'project' name"
        raise ValueError(msg)

    # Build region config from ARS
    ars = merged.get("ars")
    if not ars:
        msg = "Config must specify 'ars' (12-digit Amtlicher Regionalschlüssel)"
        raise ValueError(msg)
    region = RegionConfig(
        ars=str(ars),
        name=merged.get("region_name"),
    )

    # Build temporal config
    year = merged.get("year")
    if not year:
        msg = "Config must specify 'year'"
        raise ValueError(msg)

    period_data = merged.get("period", {})
    temporal = TemporalConfig(
        year=year,
        period_start=_parse_date(period_data.get("start", f"{year}-05-01")),
        period_end=_parse_date(period_data.get("end", f"{year}-09-30")),
    )

    # Build data paths config
    data_data = merged.get("data", {})
    traffic_volumes = data_data.get("traffic_volumes")
    if not traffic_volumes:
        msg = "Config must specify 'data.traffic_volumes'"
        raise ValueError(msg)

    data_paths = DataPathsConfig(
        data_root=Path(data_data.get("root", "./data")),
        traffic_volumes=Path(traffic_volumes),
        # Optional: counter data (required for training)
        counter_locations=Path(data_data["counter_locations"])
        if data_data.get("counter_locations")
        else None,
        counter_measurements=Path(data_data["counter_measurements"])
        if data_data.get("counter_measurements")
        else None,
        # Optional overrides for Germany-wide defaults
        # TODO: make that file year-agnostic
        # (we must parse the germany-{YY}0101.osm.pbf somehow)
        osm_pbf=Path(data_data["osm_pbf"])
        if data_data.get("osm_pbf")
        else Path("osm-data/germany-230101.osm.pbf"),
        municipalities=Path(data_data["municipalities"])
        if data_data.get("municipalities")
        else Path("structural-data/DE_VG250.gpkg"),
        regiostar=Path(data_data["regiostar"])
        if data_data.get("regiostar")
        else Path("structural-data/regiostar_2022.csv"),
        city_centroids=Path(data_data["city_centroids"])
        if data_data.get("city_centroids")
        else Path("structural-data/places.gpkg"),
        kommunen_stats=Path(data_data["kommunen_stats"])
        if data_data.get("kommunen_stats")
        else Path("kommunen-stats/kommunen_stats.shp"),
        campaign_stats=Path(data_data["campaign_stats"])
        if data_data.get("campaign_stats")
        else Path("campaign/SR_TeilnehmendeKommunen.csv"),
        gebietseinheiten=Path(data_data["gebietseinheiten"])
        if data_data.get("gebietseinheiten")
        else Path("structural-data/DE_Gebietseinheiten.gpkg"),
        images_db=Path(data_data["images_db"])
        if data_data.get("images_db")
        else None,
    )

    # Features config (from base.yaml defaults)
    features_data = merged.get("features", {})
    features = FeatureConfig(
        raw_columns=features_data.get("raw_columns", []),
        derived=features_data.get("derived", {}),
        model_features=features_data.get("model_features", []),
    )

    # Preprocessing config (from base.yaml defaults)
    preprocessing_data = merged.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        infrastructure_mapping=preprocessing_data.get("infrastructure_mapping", {}),
        valid_infrastructure_categories=preprocessing_data.get(
            "valid_infrastructure_categories", []
        ),
        target_column=preprocessing_data.get("target", {}).get("column", "dtv_value"),
        target_transformation=preprocessing_data.get("target", {}).get(
            "transformation", "log1p"
        ),
    )

    # Training config (can be overridden)
    training_data = merged.get("training", {})
    training = TrainingConfig(
        test_size=training_data.get("test_size", 0.2),
        cv_folds=training_data.get("cv_folds", 10),
        random_state=training_data.get("random_state", 1337),
        min_dtv=training_data.get("min_dtv", 25),
        max_dtv=training_data.get("max_dtv"),
        metrics=training_data.get("metrics", ["r2", "rmse", "mae", "mape"]),
        deduplicate_edges=training_data.get("deduplicate_edges", False),
        min_volume_ratio=training_data.get("min_volume_ratio"),
        max_volume_ratio=training_data.get("max_volume_ratio"),
    )

    # Models config (from base.yaml defaults)
    models_data = merged.get("models", {})
    models = ModelConfig(
        enabled=models_data.get("enabled", []),
        hyperparameters=models_data.get("hyperparameters", {}),
    )

    # MLflow config (experiment_name derived from project if not set)
    mlflow_data = merged.get("mlflow", {})
    mlflow = MLflowConfig(
        tracking_uri=mlflow_data.get("tracking_uri", "http://127.0.0.1:5000"),
        experiment_name=mlflow_data.get("experiment_name"),  # None = use project
    )

    # Output config (paths derived from project name)
    output_data = merged.get("output", {})
    output = OutputConfig(
        output_root=Path(output_data.get("root", "./output")),
    )

    # Curated config (optional)
    curated_data = merged.get("curated", {})
    curated = CuratedConfig(
        path=Path(curated_data["path"]) if curated_data.get("path") else None,
        city_centroids=Path(curated_data["city_centroids"])
        if curated_data.get("city_centroids")
        else None,
    )

    # Stats config (optional - controls statistics loading approach)
    stats_data = merged.get("stats", {})
    stats_approach_str = stats_data.get("approach", "legacy")
    stats = StatsConfig(
        approach=StatisticsApproach(stats_approach_str),
        admin_level=stats_data.get("admin_level", "Verwaltungsgemeinschaft"),
    )

    # Calibration config (optional - for transfer to new regions)
    calibration_data = merged.get("calibration", {})
    calibrator_str = calibration_data.get("calibrator", "log_linear")
    calibration = CalibrationConfig(
        counter_locations=Path(calibration_data["counter_locations"])
        if calibration_data.get("counter_locations")
        else None,
        calibrator=CalibratorType(calibrator_str),
        stratify_by=calibration_data.get("stratify_by"),
        min_stations_per_stratum=calibration_data.get("min_stations_per_stratum", 3),
        random_state=calibration_data.get("random_state", 1337),
    )

    return PipelineConfig(
        project=project,
        region=region,
        temporal=temporal,
        data_paths=data_paths,
        features=features,
        preprocessing=preprocessing,
        training=training,
        models=models,
        mlflow=mlflow,
        output=output,
        curated=curated,
        stats=stats,
        calibration=calibration,
    )
