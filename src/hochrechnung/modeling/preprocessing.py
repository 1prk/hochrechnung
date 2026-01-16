"""
Preprocessing pipeline construction.

Builds sklearn ColumnTransformer from configuration.
"""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

# Transformer class mapping
TRANSFORMER_MAP: dict[str, type] = {
    "StandardScaler": StandardScaler,
    "RobustScaler": RobustScaler,
    "PowerTransformer": PowerTransformer,
    "OneHotEncoder": OneHotEncoder,
    "OrdinalEncoder": OrdinalEncoder,
    "FunctionTransformer": FunctionTransformer,
}


def build_preprocessor(
    config: PipelineConfig,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build preprocessing ColumnTransformer from configuration.

    Args:
        config: Pipeline configuration.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.

    Returns:
        Configured ColumnTransformer.
    """
    transformers: list[tuple[str, Any, list[str]]] = []

    # Default feature lists if not provided
    if numeric_features is None:
        numeric_features = [
            "stadtradeln_volume",
            "population",
            "total_km",
            "n_trips",
            "n_users",
            "dist_to_center_m",
            "participation_rate",
            "route_intensity",
            "volume_per_trip",
        ]

    if categorical_features is None:
        categorical_features = ["infra_category", "regiostar5"]

    # Numeric transformations
    if numeric_features:
        # Standard scaling for most numeric features
        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_transformer, numeric_features))

    # Categorical transformations
    if categorical_features:
        # Ordinal encoding for infrastructure categories
        infra_categories = [
            [
                "no",
                "mixed_way",
                "mit_road",
                "bicycle_lane",
                "bicycle_road",
                "bicycle_way",
            ]
        ]

        categorical_transformer = OrdinalEncoder(
            categories=infra_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        if "infra_category" in categorical_features:
            transformers.append(("infra", categorical_transformer, ["infra_category"]))

        # RegioStaR is already numeric (1-5), just pass through
        if "regiostar5" in categorical_features:
            transformers.append(("regiostar", "passthrough", ["regiostar5"]))

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop columns not explicitly handled
    )

    log.info(
        "Built preprocessor",
        n_transformers=len(transformers),
        numeric_features=len(numeric_features) if numeric_features else 0,
        categorical_features=len(categorical_features) if categorical_features else 0,
    )

    return preprocessor


def build_target_transformer(
    method: str = "log1p",
) -> tuple[Any, Any]:
    """
    Build target variable transformer functions.

    Args:
        method: Transformation method ('log1p', 'boxcox', 'none').

    Returns:
        Tuple of (transform_func, inverse_func).
    """
    if method == "log1p":
        return np.log1p, np.expm1
    if method == "none":
        return None, None

    # Default to log1p
    log.warning("Unknown transform method, using log1p", method=method)
    return np.log1p, np.expm1


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
    input_features: list[str] | None = None,
) -> list[str]:
    """
    Get output feature names from fitted preprocessor.

    Args:
        preprocessor: Fitted ColumnTransformer.
        input_features: Original input feature names.

    Returns:
        List of output feature names.
    """
    try:
        return list(preprocessor.get_feature_names_out(input_features))
    except Exception:
        # Fallback: generate generic names
        n_features = preprocessor.transform(
            np.zeros((1, len(input_features or [])))
        ).shape[1]
        return [f"feature_{i}" for i in range(n_features)]
