"""
Model comparison experiment.

Question: Which model architecture works best for DTV prediction?

This experiment compares multiple regression models on the same
train/test split to determine optimal model architecture.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from hochrechnung.config import load_config
from hochrechnung.evaluation.experiment import ModelComparisonExperiment
from hochrechnung.evaluation.metrics import compute_accuracy_bands
from hochrechnung.modeling.training import ModelTrainer


def run_model_comparison(config_path: Path) -> None:
    """
    Run model comparison experiment.

    Args:
        config_path: Path to configuration file.
    """
    # Load configuration
    config = load_config(config_path)

    # TODO: Load and prepare data
    # This would use the ingestion and feature pipelines
    # For now, create placeholder

    print(f"Running model comparison for {config.region.name} {config.temporal.year}")
    print(f"Models to compare: {config.models.enabled}")

    # Initialize experiment
    experiment = ModelComparisonExperiment(config)

    # Example workflow (requires actual data):
    # 1. Load prepared training data
    # X, y = load_training_data(config)
    #
    # 2. Split data
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y,
    #     test_size=config.training.test_size,
    #     random_state=config.training.random_state
    # )
    #
    # 3. Train models
    # trainer = ModelTrainer(config)
    # trained_models = trainer.train(X_train, y_train)
    #
    # 4. Run comparison experiment
    # results = experiment.run(
    #     X_train, y_train, X_test, y_test,
    #     {name: m.pipeline for name, m in trained_models.items()}
    # )
    #
    # 5. Print results
    # for name, metrics in results.items():
    #     print(f"{name}: {metrics}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_comparison.py <config_path>")
        sys.exit(1)

    run_model_comparison(Path(sys.argv[1]))
