"""
Model training functionality.

Provides training pipeline with proper preprocessing and MLflow integration.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer, max_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.evaluation.metrics import SQV_SCALING_FACTOR, compute_sqv
from hochrechnung.modeling.models import MODEL_REGISTRY, PARAM_GRIDS, get_model
from hochrechnung.modeling.preprocessing import (
    build_preprocessor,
    build_target_transformer,
)
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainedModel:
    """
    Container for a trained model with metadata.

    Attributes:
        name: Model name.
        pipeline: Fitted sklearn pipeline.
        cv_scores: Cross-validation scores including overfitting metrics.
        best_params: Best hyperparameters (if tuned).
        feature_names: Input feature names.
        training_time_s: Total training time in seconds.
    """

    name: str
    pipeline: BaseEstimator
    cv_scores: dict[str, float] = field(default_factory=dict)
    best_params: dict[str, Any] | None = None
    feature_names: list[str] = field(default_factory=list)
    training_time_s: float = 0.0


class ModelTrainer:
    """
    Trainer for bicycle traffic prediction models.

    Handles preprocessing, model fitting, and hyperparameter tuning.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize trainer.

        Args:
            config: Pipeline configuration.
        """
        self.config = config
        self.models: dict[str, TrainedModel] = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_names: list[str] | None = None,
        *,
        tune_hyperparameters: bool = True,
    ) -> dict[str, TrainedModel]:
        """
        Train models on data.

        Args:
            X: Feature matrix.
            y: Target variable.
            model_names: Models to train (default: from config).
            tune_hyperparameters: Whether to tune hyperparameters.

        Returns:
            Dictionary of trained models.
        """
        if model_names is None:
            model_names = self.config.models.enabled

        log.info(
            "Starting training",
            n_samples=len(X),
            n_features=X.shape[1],
            models=model_names,
        )

        # Identify feature types
        # RegioStaR columns are ordinal categorical despite numeric dtype
        ordinal_categorical = {"regiostar5", "regiostar7"}
        numeric_features = [
            col
            for col in X.columns
            if X[col].dtype in ["int64", "float64"] and col not in ordinal_categorical
        ]
        categorical_features = [col for col in X.columns if col not in numeric_features]

        # Get target transformer with clipping to prevent extreme extrapolation
        # Use max observed value * 2 as upper bound
        clip_max = float(y.max()) * 2.0
        transform_func, inverse_func = build_target_transformer(
            self.config.preprocessing.target_transformation,
            clip_max=clip_max,
        )

        # Build preprocessor with feature-specific transformers
        # This is shared across all model types since each feature gets
        # its appropriate transformation (Box-Cox for stadtradeln_volume,
        # RobustScaler for ratios, etc.)
        preprocessor = build_preprocessor(
            self.config,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        # Train each model
        for name in model_names:
            if name not in MODEL_REGISTRY:
                log.warning("Unknown model, skipping", name=name)
                continue

            log.info(
                "Training model",
                name=name,
            )
            trained = self._train_single_model(
                X,
                y,
                name,
                preprocessor,
                transform_func,
                inverse_func,
                tune=tune_hyperparameters,
            )
            self.models[name] = trained

        log.info("Training complete", n_models=len(self.models))
        return self.models

    def _train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str,
        preprocessor: Any,
        transform_func: Any,
        inverse_func: Any,
        *,
        tune: bool = True,
    ) -> TrainedModel:
        """Train a single model."""
        training_start = time.perf_counter()

        # Get base model
        model = get_model(name)

        # Build pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        # Wrap with target transformation
        if transform_func is not None:
            pipeline = TransformedTargetRegressor(
                regressor=pipeline,
                func=transform_func,
                inverse_func=inverse_func,
            )

        # Setup cross-validation
        cv = KFold(
            n_splits=self.config.training.cv_folds,
            shuffle=True,
            random_state=self.config.training.random_state,
        )

        best_params = None

        # Hyperparameter tuning if requested and grid exists
        if tune and name in PARAM_GRIDS:
            grid = PARAM_GRIDS[name]
            gs = GridSearchCV(
                pipeline,
                param_grid=grid,
                cv=cv,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X, y)
            pipeline = gs.best_estimator_
            best_params = gs.best_params_
            log.info("Hyperparameter tuning complete", best_params=best_params)
        else:
            pipeline.fit(X, y)

        training_time_s = time.perf_counter() - training_start

        # Calculate CV scores (includes R² no CV for overfitting detection)
        cv_scores = self._calculate_cv_scores(pipeline, X, y, cv, training_time_s)

        return TrainedModel(
            name=name,
            pipeline=pipeline,
            cv_scores=cv_scores,
            best_params=best_params,
            feature_names=list(X.columns),
            training_time_s=training_time_s,
        )

    def _calculate_cv_scores(
        self,
        pipeline: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Any,
        training_time_s: float = 0.0,
    ) -> dict[str, float]:
        """
        Calculate cross-validation scores with overfitting detection.

        Computes R² both with and without CV to detect overfitting:
        - r2_cv: Average R² across CV folds (generalization performance)
        - r2_no_cv: R² on full training set (in-sample performance)
        - r2_gap: Difference indicating overfitting risk
        """
        scores: dict[str, float] = {}

        # R² score (CV)
        r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2", n_jobs=-1)
        scores["r2_cv"] = float(np.mean(r2_scores))
        scores["r2_std"] = float(np.std(r2_scores))

        # R² without CV (fit on full training set for overfitting detection)
        # Use the already-fitted pipeline to predict on training data
        y_pred_train = pipeline.predict(X)
        scores["r2_no_cv"] = float(r2_score(y, y_pred_train))

        # Overfitting gap (higher = more overfitting)
        scores["r2_gap"] = scores["r2_no_cv"] - scores["r2_cv"]

        # RMSE (CV)
        mse_scores = -cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
        )
        scores["rmse"] = float(np.mean(np.sqrt(mse_scores)))

        # MAE (CV)
        mae_scores = -cross_val_score(
            pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
        )
        scores["mae"] = float(np.mean(mae_scores))

        # MAPE (CV)
        mape_scores = -cross_val_score(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
        )
        scores["mape"] = float(np.mean(mape_scores))

        # Max Error (CV) - using custom scorer
        max_error_scorer = make_scorer(max_error, greater_is_better=False)
        max_error_scores = -cross_val_score(
            pipeline, X, y, cv=cv, scoring=max_error_scorer, n_jobs=-1
        )
        scores["max_error"] = float(np.mean(max_error_scores))

        # SQV (CV) - using custom scorer
        def sqv_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return compute_sqv(y_true, y_pred, SQV_SCALING_FACTOR)

        sqv_scorer = make_scorer(sqv_score, greater_is_better=True)
        sqv_scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring=sqv_scorer, n_jobs=-1
        )
        scores["sqv"] = float(np.mean(sqv_scores))

        # Timing metrics
        scores["training_time_s"] = training_time_s

        # Prediction time (per sample)
        pred_start = time.perf_counter()
        _ = pipeline.predict(X)
        pred_time = time.perf_counter() - pred_start
        scores["prediction_time_s"] = pred_time / len(X)

        # Legacy compatibility: keep 'r2' as alias for 'r2_cv'
        scores["r2"] = scores["r2_cv"]

        # Determine overfitting risk level
        if scores["r2_gap"] < 0.05:
            risk = "low"
        elif scores["r2_gap"] < 0.10:
            risk = "moderate"
        else:
            risk = "high"

        log.info(
            "CV scores calculated",
            r2_cv=f"{scores['r2_cv']:.4f}",
            r2_no_cv=f"{scores['r2_no_cv']:.4f}",
            r2_gap=f"{scores['r2_gap']:.3f}",
            overfitting_risk=risk,
            rmse=f"{scores['rmse']:.2f}",
            sqv=f"{scores['sqv']:.4f}",
        )

        return scores


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
    model_name: str | None = None,
) -> TrainedModel:
    """
    Convenience function to train a single model.

    Args:
        X: Feature matrix.
        y: Target variable.
        config: Pipeline configuration.
        model_name: Model to train (default: first in config).

    Returns:
        Trained model.
    """
    trainer = ModelTrainer(config)

    if model_name is None:
        model_name = (
            config.models.enabled[0] if config.models.enabled else "Random Forest"
        )

    models = trainer.train(X, y, [model_name])
    return models[model_name]
