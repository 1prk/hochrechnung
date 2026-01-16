"""
Configuration loading utilities.

Supports environment variable interpolation and config inheritance.
"""

import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

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

    # Build configuration objects
    region_data = merged.get("region", {})
    region = RegionConfig(
        code=region_data.get("code", "06"),
        name=region_data.get("name", "Unknown"),
        bbox=tuple(region_data.get("bbox", [0, 0, 1, 1])),
    )

    temporal_data = merged.get("temporal", {})
    temporal = TemporalConfig(
        year=temporal_data.get("year", 2024),
        campaign_start=_parse_date(temporal_data.get("campaign_start", "2024-05-01")),
        campaign_end=_parse_date(temporal_data.get("campaign_end", "2024-09-30")),
        counter_period_start=_parse_date(
            temporal_data.get("counter_period_start", "2024-05-01")
        ),
        counter_period_end=_parse_date(
            temporal_data.get("counter_period_end", "2024-09-30")
        ),
        holiday_start=_parse_date(temporal_data["holiday_start"])
        if temporal_data.get("holiday_start")
        else None,
        holiday_end=_parse_date(temporal_data["holiday_end"])
        if temporal_data.get("holiday_end")
        else None,
    )

    paths_data = merged.get("data_paths", {})
    data_paths = DataPathsConfig(
        data_root=Path(paths_data.get("data_root", "./data")),
        counter_locations=Path(
            paths_data.get("counter_locations", "counter-locations/default.csv")
        ),
        counter_measurements=Path(
            paths_data.get("counter_measurements", "counts/default.csv")
        ),
        osm_pbf=Path(paths_data["osm_pbf"]) if paths_data.get("osm_pbf") else None,
        traffic_volumes=Path(
            paths_data.get("traffic_volumes", "trafficvolumes/default.fgb")
        ),
        municipalities=Path(
            paths_data.get("municipalities", "structural-data/DE_VG250.gpkg")
        ),
        regiostar=Path(paths_data.get("regiostar", "structural-data/regiostar.csv")),
        city_centroids=Path(
            paths_data.get("city_centroids", "structural-data/centroids.gpkg")
        ),
        kommunen_stats=Path(
            paths_data.get("kommunen_stats", "kommunen-stats/default.shp")
        ),
        campaign_stats=Path(paths_data.get("campaign_stats", "campaign/default.csv")),
    )

    features_data = merged.get("features", {})
    features = FeatureConfig(
        raw_columns=features_data.get("raw_columns", []),
        derived=features_data.get("derived", {}),
        model_features=features_data.get("model_features", []),
    )

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

    training_data = merged.get("training", {})
    training = TrainingConfig(
        test_size=training_data.get("test_size", 0.2),
        cv_folds=training_data.get("cv_folds", 5),
        random_state=training_data.get("random_state", 1337),
        min_dtv=training_data.get("min_dtv", 25),
        max_dtv=training_data.get("max_dtv"),
        metrics=training_data.get("metrics", ["r2", "rmse", "mae", "mape"]),
    )

    models_data = merged.get("models", {})
    models = ModelConfig(
        enabled=models_data.get("enabled", []),
        hyperparameters=models_data.get("hyperparameters", {}),
    )

    mlflow_data = merged.get("mlflow", {})
    mlflow = MLflowConfig(
        tracking_uri=mlflow_data.get("tracking_uri", "http://127.0.0.1:5000"),
        artifact_location=Path(mlflow_data.get("artifact_location", "./mlartifacts")),
        experiment_name=mlflow_data.get("experiment_name", "default"),
    )

    output_data = merged.get("output", {})
    output = OutputConfig(
        plots_dir=Path(output_data.get("plots_dir", "./plots")),
        predictions_dir=Path(output_data.get("predictions_dir", "./predictions")),
        cache_dir=Path(output_data.get("cache_dir", "./cache")),
    )

    project_data = merged.get("project", {})

    return PipelineConfig(
        project_name=project_data.get("name", "Bicycle Traffic Estimation"),
        project_version=project_data.get("version", "0.1.0"),
        region=region,
        temporal=temporal,
        data_paths=data_paths,
        features=features,
        preprocessing=preprocessing,
        training=training,
        models=models,
        mlflow=mlflow,
        output=output,
    )
