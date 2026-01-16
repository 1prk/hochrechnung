"""
MLflow experiment management.

Provides structured experiment tracking with required metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.evaluation.metrics import RegressionMetrics, compute_metrics
from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration for an MLflow experiment.

    Attributes:
        name: Experiment name.
        question: Single question this experiment answers.
        experiment_type: Category of experiment.
        region: Region being analyzed.
        year: Analysis year.
    """

    name: str
    question: str
    experiment_type: str  # feature_selection, model_comparison, hyperparameter, etc.
    region: str
    year: int
    tags: dict[str, str] = field(default_factory=dict)


class Experiment:
    """
    Base class for MLflow experiments.

    Each experiment answers a single clear question.
    """

    def __init__(
        self,
        config: PipelineConfig,
        experiment_config: ExperimentConfig,
    ) -> None:
        """
        Initialize experiment.

        Args:
            config: Pipeline configuration.
            experiment_config: Experiment-specific configuration.
        """
        self.config = config
        self.experiment_config = experiment_config
        self._run_id: str | None = None

    def setup(self) -> None:
        """Setup MLflow experiment."""
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.experiment_config.name)

        log.info(
            "Experiment setup",
            name=self.experiment_config.name,
            tracking_uri=self.config.mlflow.tracking_uri,
        )

    def start_run(self, run_name: str | None = None) -> str:
        """
        Start an MLflow run.

        Args:
            run_name: Optional run name.

        Returns:
            Run ID.
        """
        self.setup()

        # Build tags
        tags = {
            "experiment_type": self.experiment_config.experiment_type,
            "region": self.experiment_config.region,
            "year": str(self.experiment_config.year),
            "question": self.experiment_config.question,
            "pipeline_version": self.config.project_version,
            **self.experiment_config.tags,
        }

        run = mlflow.start_run(run_name=run_name, tags=tags)
        self._run_id = run.info.run_id

        log.info("Started MLflow run", run_id=self._run_id)
        return self._run_id

    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
        log.info("Ended MLflow run", run_id=self._run_id)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: RegressionMetrics | dict[str, float]) -> None:
        """Log metrics."""
        if isinstance(metrics, RegressionMetrics):
            metrics = metrics.to_dict()
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        """Log an artifact."""
        mlflow.log_artifact(str(path), artifact_path)

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_name: str | None = None,
    ) -> None:
        """Log a model."""
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_name,
        )


class ModelComparisonExperiment(Experiment):
    """
    Experiment to compare different model architectures.

    Question: Which model architecture works best for this data?
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize model comparison experiment."""
        exp_config = ExperimentConfig(
            name=f"{config.region.name.lower()}-model-comparison-{config.temporal.year}",
            question="Which model architecture achieves the best R² on validation data?",
            experiment_type="model_comparison",
            region=config.region.name,
            year=config.temporal.year,
        )
        super().__init__(config, exp_config)

    def run(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        models: dict[str, Any],
    ) -> dict[str, RegressionMetrics]:
        """
        Run model comparison experiment.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test: Test features.
            y_test: Test target.
            models: Dictionary of model name -> fitted model.

        Returns:
            Dictionary of model name -> test metrics.
        """
        self.start_run(f"model-comparison-{datetime.now():%Y%m%d-%H%M}")

        # Log common parameters
        self.log_params(
            {
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
                "n_features": X_train.shape[1],
                "n_models": len(models),
            }
        )

        results = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=f"eval-{name}", nested=True):
                mlflow.set_tag("model_name", name)

                # Predict and compute metrics
                y_pred = model.predict(X_test)
                metrics = compute_metrics(y_test.values, y_pred)

                self.log_metrics(metrics)
                results[name] = metrics

                log.info(f"Model {name}: {metrics}")

        self.end_run()
        return results


class FeatureSelectionExperiment(Experiment):
    """
    Experiment to identify important features.

    Question: Which features contribute most to prediction accuracy?
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize feature selection experiment."""
        exp_config = ExperimentConfig(
            name=f"{config.region.name.lower()}-feature-selection-{config.temporal.year}",
            question="Which features have the highest importance for DTV prediction?",
            experiment_type="feature_selection",
            region=config.region.name,
            year=config.temporal.year,
        )
        super().__init__(config, exp_config)
