"""
Temporal stability experiment.

Question: Does the model generalize across different years?

This experiment trains on one year and evaluates on another
to assess temporal transferability.
"""

from pathlib import Path

from hochrechnung.config import load_config
from hochrechnung.evaluation.experiment import Experiment, ExperimentConfig


def run_temporal_stability(
    train_config_path: Path,
    test_config_path: Path,
) -> None:
    """
    Run temporal stability experiment.

    Args:
        train_config_path: Config for training year.
        test_config_path: Config for test year.
    """
    train_config = load_config(train_config_path)
    test_config = load_config(test_config_path)

    print(f"Temporal stability experiment:")
    print(f"  Training on: {train_config.temporal.year}")
    print(f"  Testing on: {test_config.temporal.year}")

    # Create experiment
    exp_config = ExperimentConfig(
        name=f"temporal-stability-{train_config.temporal.year}-{test_config.temporal.year}",
        question=f"Does model trained on {train_config.temporal.year} generalize to {test_config.temporal.year}?",
        experiment_type="temporal_stability",
        region=train_config.region.name,
        year=train_config.temporal.year,
        tags={
            "train_year": str(train_config.temporal.year),
            "test_year": str(test_config.temporal.year),
        },
    )

    experiment = Experiment(train_config, exp_config)

    # Example workflow:
    # 1. Load training data for train year
    # X_train, y_train = load_training_data(train_config)
    #
    # 2. Load test data for test year
    # X_test, y_test = load_training_data(test_config)
    #
    # 3. Train best model on training year
    # trainer = ModelTrainer(train_config)
    # models = trainer.train(X_train, y_train, ["Random Forest"])
    # model = models["Random Forest"].pipeline
    #
    # 4. Evaluate on test year
    # experiment.start_run("temporal-stability")
    # y_pred = model.predict(X_test)
    # metrics = compute_metrics(y_test, y_pred)
    # experiment.log_metrics(metrics)
    # experiment.end_run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python temporal_stability.py <train_config> <test_config>")
        sys.exit(1)

    run_temporal_stability(Path(sys.argv[1]), Path(sys.argv[2]))
