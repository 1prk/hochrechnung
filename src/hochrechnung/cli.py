"""Command-line interface for the hochrechnung pipeline."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="hochrechnung",
    help="Bicycle traffic estimation pipeline for DTV prediction.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def etl(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path for training data CSV.",
        ),
    ] = None,
) -> None:
    """Run the ETL pipeline to prepare training data."""
    from hochrechnung.config.loader import load_config
    from hochrechnung.etl import run_etl

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    # Set default output path if not specified
    if output is None:
        output = Path(f"./cache/training_data_{pipeline_config.year}.csv")

    console.print(f"[blue]Running ETL pipeline for year {pipeline_config.year}[/blue]")
    console.print(f"[dim]Output: {output}[/dim]")

    try:
        result = run_etl(pipeline_config, output_path=output)

        # Display results
        console.print()
        table = Table(title="ETL Pipeline Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Counters with valid DTV", str(result.n_counters))
        table.add_row("Traffic edges loaded", str(result.n_edges))
        table.add_row("Counters matched to edges", str(result.n_matched))
        table.add_row("Training samples", str(len(result.training_data)))
        table.add_row("Output columns", str(len(result.training_data.columns)))

        console.print(table)

        if result.output_path:
            console.print(f"\n[green]Saved to: {result.output_path}[/green]")

        # Show column summary
        console.print("\n[blue]Output columns:[/blue]")
        for col in result.training_data.columns:
            non_null = result.training_data[col].notna().sum()
            console.print(f"  {col}: {non_null}/{len(result.training_data)} non-null")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]ETL failed: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    data: Annotated[
        Path | None,
        typer.Option(
            "--data",
            "-d",
            help="Path to training CSV. Auto-detects cache/training_data_{year}.csv if not specified.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Single model name to train. Trains all from config if not specified.",
        ),
    ] = None,
    no_mlflow: Annotated[
        bool,
        typer.Option(
            "--no-mlflow",
            help="Skip MLflow logging.",
        ),
    ] = False,
    no_tune: Annotated[
        bool,
        typer.Option(
            "--no-tune",
            help="Skip hyperparameter tuning.",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Directory to save trained models.",
        ),
    ] = None,
) -> None:
    """Train models using the specified configuration."""
    import joblib

    from hochrechnung.config.loader import load_config
    from hochrechnung.evaluation.experiment import Experiment, ExperimentConfig
    from hochrechnung.evaluation.metrics import compute_metrics
    from hochrechnung.modeling.data import auto_detect_data_path, load_training_data
    from hochrechnung.modeling.training import ModelTrainer

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    # Determine data path
    if data is None:
        try:
            data = auto_detect_data_path(pipeline_config)
            console.print(f"[dim]Auto-detected data: {data}[/dim]")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1) from e
    else:
        console.print(f"[dim]Using specified data: {data}[/dim]")

    # Load training data
    try:
        training_data = load_training_data(data, pipeline_config)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Print data summary
    console.print()
    data_table = Table(title="Training Data Summary")
    data_table.add_column("Metric", style="cyan")
    data_table.add_column("Value", style="green")

    data_table.add_row("Total samples", str(training_data.n_samples))
    data_table.add_row("Training samples", str(len(training_data.X_train)))
    data_table.add_row("Test samples", str(len(training_data.X_test)))
    data_table.add_row("Features", str(len(training_data.feature_names)))
    data_table.add_row("Target mean", f"{training_data.target_stats['mean']:.2f}")
    data_table.add_row("Target std", f"{training_data.target_stats['std']:.2f}")
    data_table.add_row(
        "Target range",
        f"{training_data.target_stats['min']:.0f} - {training_data.target_stats['max']:.0f}",
    )

    console.print(data_table)

    # Print feature list
    console.print("\n[blue]Features:[/blue]")
    for feat in training_data.feature_names:
        dtype = training_data.X_train[feat].dtype
        console.print(f"  {feat} ({dtype})")

    # Determine models to train
    model_names = [model] if model is not None else pipeline_config.models.enabled

    console.print(f"\n[blue]Training models: {', '.join(model_names)}[/blue]")
    if not no_tune:
        console.print("[dim]Hyperparameter tuning enabled[/dim]")

    # Train models
    trainer = ModelTrainer(pipeline_config)
    try:
        trained_models = trainer.train(
            training_data.X_train,
            training_data.y_train,
            model_names=model_names,
            tune_hyperparameters=not no_tune,
        )
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Evaluate on test set
    console.print("\n[blue]Evaluating on test set...[/blue]")
    results_table = Table(title="Model Results")
    results_table.add_column("Model", style="cyan")
    results_table.add_column("R²", style="green")
    results_table.add_column("RMSE", style="yellow")
    results_table.add_column("MAE", style="yellow")
    results_table.add_column("MAPE", style="yellow")
    results_table.add_column("Best", style="magenta")

    test_results: dict[str, dict[str, float]] = {}
    best_r2 = -float("inf")
    best_model_name = ""

    for name, trained_model in trained_models.items():
        y_pred = trained_model.pipeline.predict(training_data.X_test)
        metrics = compute_metrics(training_data.y_test.values, y_pred)

        test_results[name] = metrics.to_dict()

        if metrics.r2 > best_r2:
            best_r2 = metrics.r2
            best_model_name = name

    # Add rows to table (mark best model)
    for name, result in test_results.items():
        is_best = "✓" if name == best_model_name else ""
        results_table.add_row(
            name,
            f"{result['r2']:.4f}",
            f"{result['rmse']:.2f}",
            f"{result['mae']:.2f}",
            f"{result['mape']:.2%}",
            is_best,
        )

    console.print(results_table)

    # MLflow logging
    if not no_mlflow:
        console.print("\n[blue]Logging to MLflow...[/blue]")
        try:
            exp_config = ExperimentConfig(
                name=pipeline_config.mlflow.experiment_name,
                question="Which model achieves best R² on test data?",
                experiment_type="model_training",
                region=pipeline_config.region.name,
                year=pipeline_config.year,
            )
            experiment = Experiment(pipeline_config, exp_config)
            run_id = experiment.start_run(f"train-{pipeline_config.year}")

            # Log parameters
            experiment.log_params(
                {
                    "n_train": len(training_data.X_train),
                    "n_test": len(training_data.X_test),
                    "n_features": len(training_data.feature_names),
                    "models": ",".join(model_names),
                    "tune_hyperparameters": not no_tune,
                }
            )

            # Log best model metrics
            experiment.log_metrics(test_results[best_model_name])

            # Log best model
            experiment.log_model(
                trained_models[best_model_name].pipeline,
                artifact_path="best_model",
            )

            experiment.end_run()
            console.print(f"[green]Logged to MLflow run: {run_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]MLflow logging failed: {e}[/yellow]")

    # Save model
    if output is None:
        output = pipeline_config.output.cache_dir / "models"

    output.mkdir(parents=True, exist_ok=True)

    for name, trained_model in trained_models.items():
        # Sanitize model name for filename
        safe_name = name.lower().replace(" ", "_")
        model_path = output / f"{safe_name}_{pipeline_config.year}.joblib"
        joblib.dump(trained_model.pipeline, model_path)
        console.print(f"[green]Saved: {model_path}[/green]")

    console.print(f"\n[green]Training complete. Best model: {best_model_name}[/green]")


@app.command()
def predict(
    config: Annotated[  # noqa: ARG001
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    model_uri: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="MLflow model URI or path to model artifact.",
        ),
    ],
    output: Annotated[  # noqa: ARG001
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for predictions.",
        ),
    ],
) -> None:
    """Generate predictions using a trained model."""
    console.print(f"[blue]Generating predictions with model: {model_uri}[/blue]")
    # TODO: Implement prediction pipeline
    console.print("[yellow]Prediction not yet implemented[/yellow]")


@app.command()
def validate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Validate data against defined schemas."""
    from hochrechnung.config.loader import load_config
    from hochrechnung.validation import ConsoleReporter, ValidationRunner

    console.print("[blue]Running schema validation...[/blue]")

    # Load config
    pipeline_config = load_config(config)

    # Run validation
    runner = ValidationRunner(pipeline_config)
    results = runner.run()

    # Report results
    reporter = ConsoleReporter(console)
    reporter.print_results(results)

    # Exit with appropriate code
    has_failures = any(r.schema_valid is False for r in results)
    if has_failures:
        raise typer.Exit(code=1)


@app.command()
def assess(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    etl_output: Annotated[
        Path | None,
        typer.Option(
            "--etl-output",
            "-e",
            help="Path to ETL output CSV. Auto-detects cache/training_data_{year}.csv if not specified.",
        ),
    ] = None,
) -> None:
    """
    Assess ETL output values against original source data.

    Automatically loads cached ETL output (cache/training_data_{year}.csv) if available.
    No need to re-run ETL if the file already exists.

    Compares the ETL output with source data to verify that all
    transformations produced correct values.
    """
    from hochrechnung.assessment import AssessmentReporter, AssessmentRunner
    from hochrechnung.config.loader import load_config

    console.print("[blue]Running ETL output assessment...[/blue]")

    # Load config
    pipeline_config = load_config(config)

    # Determine ETL output path
    if etl_output is None:
        etl_output = (
            pipeline_config.output.cache_dir
            / f"training_data_{pipeline_config.year}.csv"
        )
        console.print(f"[dim]Auto-detected ETL output: {etl_output}[/dim]")
    else:
        console.print(f"[dim]Using specified ETL output: {etl_output}[/dim]")

    # Check if file exists
    if not etl_output.exists():
        console.print(f"[red]Error: ETL output file not found: {etl_output}[/red]")
        console.print(
            f"[yellow]Run ETL first: uv run hochrechnung etl --config {config}[/yellow]"
        )
        raise typer.Exit(code=1)

    # Verify this is Hessen 2023/2024 (assessment designed for these years)
    if (
        pipeline_config.year not in [2023, 2024]
        or pipeline_config.region.name != "Hessen"
    ):
        console.print(
            "[yellow]Warning: Assessment is designed for Hessen 2023/2024 data. "
            f"Current config: {pipeline_config.region.name} {pipeline_config.year}[/yellow]"
        )

    # Run assessment
    runner = AssessmentRunner(pipeline_config, etl_output_path=etl_output)
    result = runner.run()

    # Report results
    reporter = AssessmentReporter(console)
    reporter.print_results(result)

    # Exit with appropriate code
    from hochrechnung.assessment import CheckStatus

    if result.overall_status == CheckStatus.FAIL:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Show version information."""
    from hochrechnung import __version__

    console.print(f"hochrechnung version {__version__}")


if __name__ == "__main__":
    app()
