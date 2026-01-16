"""
Model registry and factory.

Provides registry of supported models with their default configurations.
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
from sklearn.svm import SVR

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


# Model configurations: name -> (class, default_kwargs)
MODEL_REGISTRY: dict[str, tuple[type[BaseEstimator], dict[str, Any]]] = {
    "Linear Regression": (LinearRegression, {}),
    "Poisson Regression": (PoissonRegressor, {"max_iter": 1000}),
    "Lasso Regression": (Lasso, {"alpha": 1.0}),
    "ElasticNet Regression": (ElasticNet, {"alpha": 1.0, "l1_ratio": 0.5}),
    "Random Forest": (RandomForestRegressor, {"n_estimators": 100, "n_jobs": -1}),
    "Gradient Boosting": (
        GradientBoostingRegressor,
        {"n_estimators": 128, "n_iter_no_change": 10},
    ),
    "AdaBoost Regression": (AdaBoostRegressor, {"n_estimators": 50}),
    "Support Vector Machine": (SVR, {"kernel": "rbf", "C": 1.0}),
}


# Hyperparameter grids for tuning
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Random Forest": {
        "regressor__model__n_estimators": [50, 100, 200],
        "regressor__model__max_depth": [10, 20, None],
        "regressor__model__min_samples_split": [2, 5, 10],
    },
    "Gradient Boosting": {
        "regressor__model__n_estimators": [50, 100, 200],
        "regressor__model__learning_rate": [0.01, 0.1, 0.2],
        "regressor__model__max_depth": [3, 5, 7],
    },
    "Support Vector Machine": {
        "regressor__model__C": [0.1, 1.0, 10.0],
        "regressor__model__kernel": ["rbf", "linear"],
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
        {"n_estimators": 100, "learning_rate": 0.1, "n_jobs": -1},
    )
    PARAM_GRIDS["XGBoost"] = {
        "regressor__model__n_estimators": [50, 100, 200],
        "regressor__model__learning_rate": [0.01, 0.1, 0.2],
        "regressor__model__max_depth": [3, 5, 7],
    }
except ImportError:
    log.debug("XGBoost not available")
