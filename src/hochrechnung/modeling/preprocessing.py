"""
Preprocessing pipeline construction.

Builds sklearn ColumnTransformer from configuration.
"""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
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
    _config: PipelineConfig,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    _use_power_transform: bool = False,
) -> ColumnTransformer:
    """
    Build preprocessing ColumnTransformer with feature-specific transformers.

    Follows the old repo's approach: different transformers for different feature types.
    This is critical for model performance - applying the same transformation to all
    features destroys the relationships between them.

    Feature groups:
    - stadtradeln_volume: Box-Cox (PowerTransformer) - main predictor, highly skewed
    - dist_to_center_m: StandardScaler - continuous spatial feature
    - participation_rate, route_intensity: RobustScaler - derived ratios with outliers
    - infra_category: OrdinalEncoder - categorical with natural ordering
    - regiostar7: passthrough - already ordinal encoded (71-77)

    Args:
        config: Pipeline configuration.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
        use_power_transform: Ignored (kept for API compatibility).
            Feature-specific transformers are always used.

    Returns:
        Configured ColumnTransformer.
    """
    transformers: list[tuple[str, Any, list[str]]] = []

    # Default feature lists if not provided
    if numeric_features is None:
        numeric_features = [
            "stadtradeln_volume",
            "dist_to_center_m",
            "participation_rate",
            "route_intensity",
        ]

    if categorical_features is None:
        categorical_features = ["infra_category", "regiostar7"]

    # Feature-specific transformers (matching old repo's config.yaml)
    # 1. Box-Cox ONLY for stadtradeln_volume - the main predictor, highly skewed
    if "stadtradeln_volume" in numeric_features:
        transformers.append((
            "boxcox_stadtradeln",
            PowerTransformer(method="box-cox", standardize=True),
            ["stadtradeln_volume"],
        ))

    # 2. StandardScaler for continuous spatial/demographic features
    standard_features = [
        f for f in numeric_features
        if f in {"dist_to_center_m", "population", "total_km"}
    ]
    if standard_features:
        transformers.append((
            "standard_numeric",
            StandardScaler(),
            standard_features,
        ))

    # 3. RobustScaler for derived ratio features (participation_rate, route_intensity)
    # These features have outliers, so RobustScaler is more appropriate
    robust_features = [
        f for f in numeric_features
        if f in {"participation_rate", "route_intensity", "volume_per_trip"}
    ]
    if robust_features:
        transformers.append((
            "robust_ratios",
            RobustScaler(),
            robust_features,
        ))

    # 4. OrdinalEncoder for infrastructure categories
    if "infra_category" in categorical_features:
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
        transformers.append((
            "infra",
            OrdinalEncoder(
                categories=infra_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            ["infra_category"],
        ))

    # 5. RegioStaR passthrough - already ordinal encoded
    # Values are 71-77 for regiostar7, 51-55 for regiostar5
    if "regiostar7" in categorical_features:
        transformers.append(("regiostar", "passthrough", ["regiostar7"]))
    elif "regiostar5" in categorical_features:
        transformers.append(("regiostar", "passthrough", ["regiostar5"]))

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Drop columns not explicitly handled
    )

    # Log transformer details
    transformer_summary = {name: cols for name, _, cols in transformers}
    log.info(
        "Built preprocessor with feature-specific transformers",
        n_transformers=len(transformers),
        transformers=transformer_summary,
    )

    return preprocessor


class ClippedExpm1:
    """
    Picklable callable for clipped expm1 inverse transformation.

    Prevents extreme extrapolation by clipping predictions to reasonable bounds.
    """

    def __init__(self, clip_max: float) -> None:
        """
        Initialize with clipping bound.

        Args:
            clip_max: Maximum value for predictions.
        """
        self.clip_max = clip_max

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply clipped expm1 transformation."""
        # Clip input to prevent overflow in expm1 (max safe ~709 for float64)
        # Use log1p(clip_max) as upper bound since expm1 is inverse of log1p
        max_input = np.log1p(self.clip_max)
        x_clipped = np.clip(x, None, max_input)
        result = np.expm1(x_clipped)
        return np.clip(result, 0, self.clip_max)


def build_target_transformer(
    method: str = "log1p",
    *,
    clip_max: float | None = None,
) -> tuple[Any, Any]:
    """
    Build target variable transformer functions.

    Args:
        method: Transformation method ('log1p', 'boxcox', 'none').
        clip_max: Maximum value for inverse transform clipping.
            Prevents extreme extrapolation from linear models.
            If None, no clipping is applied.

    Returns:
        Tuple of (transform_func, inverse_func).
    """
    if method == "log1p":
        if clip_max is not None:
            # Use picklable class for clipped inverse
            return np.log1p, ClippedExpm1(clip_max)
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
