"""
Model registry and factory.

Provides registry of supported models with their default configurations.
Hyperparameter grids are aligned with paper methodology (Appendix).
"""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


# Model configurations: name -> (class, default_kwargs)
# Default params follow paper's overfitting prevention measures
MODEL_REGISTRY: dict[str, tuple[type[BaseEstimator], dict[str, Any]]] = {
    "Linear Regression": (LinearRegression, {}),
    "Poisson": (PoissonRegressor, {"max_iter": 100, "alpha": 1.0, "tol": 0.01}),
    "Poisson Regression": (PoissonRegressor, {"max_iter": 1000}),  # Legacy name
    "Lasso Regression": (Lasso, {"alpha": 1.0}),
    "ElasticNet Regression": (ElasticNet, {"alpha": 1.0, "l1_ratio": 0.5}),
    "Random Forest": (
        RandomForestRegressor,
        {
            "n_estimators": 128,  # Paper: max 128 trees
            "min_samples_split": 32,  # Paper: min 16-32 samples
            "n_jobs": -1,
        },
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor,
        {
            "n_estimators": 128,  # Paper: max 128 trees
            "n_iter_no_change": 10,  # Paper: early stopping after 10 iterations
            "min_samples_leaf": 16,  # Paper: min 16 samples per leaf
        },
    ),
    "AdaBoost Regression": (AdaBoostRegressor, {"n_estimators": 50}),
    "SVR": (SVR, {"kernel": "rbf", "C": 2.0}),  # Paper: C limited to max 4
    "Support Vector Machine": (SVR, {"kernel": "rbf", "C": 1.0}),  # Legacy name
    "MLP": (
        MLPRegressor,
        {
            "hidden_layer_sizes": (150, 100, 50),  # Paper: 3 hidden layers
            "learning_rate": "adaptive",
            "early_stopping": True,  # Paper: early stopping enabled
            "max_iter": 1000,
        },
    ),
}


# Hyperparameter grids for tuning (aligned with paper Appendix)
# Paper's overfitting prevention:
# - Tree-based: min 16 samples per leaf, max 128 trees
# - GBR: early stopping after 10 iterations without improvement
# - SVR: regularization C limited to max 4
# - MLP: only 3 hidden layers, early stopping enabled
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Random Forest": {
        "regressor__model__max_features": ["sqrt", "log2"],
        "regressor__model__criterion": ["squared_error", "poisson"],
        "regressor__model__min_samples_split": [32, 64],
        "regressor__model__n_estimators": [128],
    },
    "Gradient Boosting": {
        "regressor__model__n_estimators": [128],
        "regressor__model__n_iter_no_change": [10],
        "regressor__model__learning_rate": [0.3, 0.25, 0.2, 0.15, 0.1],
        "regressor__model__loss": ["squared_error", "absolute_error", "huber"],
        "regressor__model__max_features": ["sqrt", "log2"],
        "regressor__model__min_samples_leaf": [16, 32, 64],
    },
    "SVR": {
        "regressor__model__C": [1, 2, 3, 4],
        "regressor__model__gamma": ["auto", "scale", 0.1, 0.01, 0.001],
        "regressor__model__epsilon": [10, 1, 0.1, 0.01, 0.001],
        "regressor__model__kernel": ["poly", "rbf", "sigmoid"],
    },
    "Support Vector Machine": {  # Legacy name
        "regressor__model__C": [0.1, 1.0, 10.0],
        "regressor__model__kernel": ["rbf", "linear"],
    },
    "MLP": {
        "regressor__model__hidden_layer_sizes": [
            (150, 100, 50),
            (120, 80, 40),
            (100, 50, 30),
        ],
        "regressor__model__learning_rate": ["adaptive", "invscaling", "constant"],
        "regressor__model__activation": ["logistic", "tanh"],
        "regressor__model__alpha": [0.0001, 0.05],
        "regressor__model__learning_rate_init": [0.001, 0.01, 0.1],
    },
    "Poisson": {
        "regressor__model__alpha": [0.1, 1, 10],
        "regressor__model__tol": [0.1, 0.01, 0.001],
        "regressor__model__max_iter": [100],
    },
}


def get_model(name: str, **kwargs: Any) -> BaseEstimator:
    """
    Get a model instance by name.

    Args:
        name: Model name from registry.
        **kwargs: Override default parameters.

    Returns:
        Model instance.

    Raises:
        KeyError: If model not found.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        msg = f"Unknown model '{name}'. Available: {available}"
        raise KeyError(msg)

    model_class, default_kwargs = MODEL_REGISTRY[name]
    params = {**default_kwargs, **kwargs}

    log.debug("Creating model", name=name, params=params)
    return model_class(**params)


def get_param_grid(name: str) -> dict[str, list[Any]] | None:
    """
    Get hyperparameter grid for a model.

    Args:
        name: Model name.

    Returns:
        Parameter grid or None if not defined.
    """
    return PARAM_GRIDS.get(name)


def list_models() -> list[str]:
    """List all available model names."""
    return list(MODEL_REGISTRY.keys())


def create_ensemble(
    models: list[str],
    ensemble_type: str = "voting",
) -> BaseEstimator:
    """
    Create an ensemble model from multiple base models.

    Args:
        models: List of model names.
        ensemble_type: 'voting' or 'stacking'.

    Returns:
        Ensemble model.
    """
    estimators = [(name, get_model(name)) for name in models]

    if ensemble_type == "stacking":
        return StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
        )

    return VotingRegressor(estimators=estimators)


# Add XGBoost if available
try:
    from xgboost import XGBRegressor

    MODEL_REGISTRY["XGBoost"] = (
        XGBRegressor,
        {
            "n_estimators": 128,  # Paper: max 128 trees
            "learning_rate": 0.1,
            # Note: early_stopping_rounds requires eval_set, incompatible with
            # sklearn CV pipeline. Use min_child_weight for regularization instead.
            "min_child_weight": 16,
            "n_jobs": -1,
        },
    )
    PARAM_GRIDS["XGBoost"] = {
        "regressor__model__n_estimators": [128],
        "regressor__model__learning_rate": [0.3, 0.25, 0.2, 0.15, 0.1],
        "regressor__model__max_depth": [3, 5, 7],
        "regressor__model__min_child_weight": [16, 32],  # Similar to min_samples_leaf
    }
except ImportError:
    log.debug("XGBoost not available")
