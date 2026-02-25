"""Command-line interface for the hochrechnung pipeline."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from hochrechnung.config.settings import PipelineConfig
    from hochrechnung.modeling.curated import CuratedTrainingData

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
    mode: Annotated[
        str,
        typer.Option(
            "--mode",
            "-m",
            help="ETL mode: 'production' (uses verified counters) or 'verification' (creates them).",
        ),
    ] = "production",
) -> None:
    """Run the ETL pipeline to prepare training data."""
    from hochrechnung.config.loader import load_config
    from hochrechnung.etl import run_etl

    if mode not in ["production", "verification"]:
        console.print(
            f"[red]Error: Invalid mode '{mode}'. Use 'production' or 'verification'.[/red]"
        )
        raise typer.Exit(code=1)

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    # Set default output path if not specified
    if output is None:
        output = pipeline_config.cache_dir / f"training_data_{pipeline_config.year}.csv"

    console.print(f"[blue]Running ETL pipeline for year {pipeline_config.year}[/blue]")
    console.print(f"[dim]Mode: {mode}[/dim]")
    console.print(f"[dim]Output: {output}[/dim]")

    try:
        result = run_etl(pipeline_config, output_path=output, mode=mode)

        # Display results
        console.print()
        table = Table(title=f"ETL Pipeline Results ({result.mode} mode)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Counters with valid DTV", str(result.n_counters))
        table.add_row("Traffic edges loaded", str(result.n_edges))
        table.add_row("Counters matched to edges", str(result.n_matched))
        table.add_row("Training samples", str(len(result.training_data)))
        table.add_row("Output columns", str(len(result.training_data.columns)))

        if result.mode == "production" and result.used_verified_counters:
            table.add_row("Used verified counters", "Yes ✓")
        elif result.mode == "verification" and result.n_flagged_outliers is not None:
            table.add_row("Flagged outliers", str(result.n_flagged_outliers))

        console.print(table)

        if result.output_path:
            console.print(f"\n[green]Saved to: {result.output_path}[/green]")

        # In verification mode, prompt to run verification UI
        if (
            result.mode == "verification"
            and result.n_flagged_outliers
            and result.n_flagged_outliers > 0
        ):
            console.print(
                f"\n[yellow]⚠ Found {result.n_flagged_outliers} flagged counters that need verification[/yellow]"
            )
            console.print(
                f"[blue]Run verification UI: uv run hochrechnung verify --config {config}[/blue]"
            )

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
    curated: Annotated[
        bool,
        typer.Option(
            "--curated",
            help="Use pre-curated Germany-wide counter data instead of ETL output.",
        ),
    ] = False,
    year: Annotated[
        int | None,
        typer.Option(
            "--year",
            "-y",
            help="Filter curated data by year. Required when using --curated.",
        ),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            "-r",
            help="Override path for HTML training report. Default: saved in output folder.",
        ),
    ] = None,
) -> None:
    """
    Train models using the specified configuration.

    Outputs two model variants for each configured model:
    - baseline: Single predictor (stadtradeln_volume only) - y ~ x
    - enhanced: All configured features - y ~ x1, x2, ...

    Use --curated to train with pre-curated Germany-wide counter data.
    """
    import joblib
    import numpy as np

    from hochrechnung.config.loader import load_config
    from hochrechnung.evaluation.experiment import Experiment, ExperimentConfig
    from hochrechnung.evaluation.metrics import compute_metrics
    from hochrechnung.evaluation.report import save_prediction_tables
    from hochrechnung.modeling.data import (
        ModelVariant,
        auto_detect_data_path,
        load_training_data,
    )
    from hochrechnung.modeling.training import ModelTrainer, TrainedModel

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    # Handle curated data mode
    if curated:
        from hochrechnung.modeling.curated import load_curated_data

        # Validate curated config
        if pipeline_config.curated.path is None:
            console.print("[red]Error: curated.path not configured[/red]")
            raise typer.Exit(code=1)
        if pipeline_config.curated.city_centroids is None:
            console.print("[red]Error: curated.city_centroids not configured[/red]")
            raise typer.Exit(code=1)

        # Resolve paths relative to data_root
        curated_path = (
            pipeline_config.data_paths.data_root / pipeline_config.curated.path
        )
        centroids_path = (
            pipeline_config.data_paths.data_root
            / pipeline_config.curated.city_centroids
        )

        # Year is required for curated mode
        if year is None:
            console.print("[red]Error: --year is required when using --curated[/red]")
            raise typer.Exit(code=1)

        console.print(f"[blue]Using curated data: {curated_path}[/blue]")
        console.print(f"[dim]Year filter: {year}[/dim]")
        console.print(f"[dim]Centroids: {centroids_path}[/dim]")

        try:
            curated_data = load_curated_data(
                curated_path, pipeline_config, centroids_path, year=year
            )
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error loading curated data: {e}[/red]")
            raise typer.Exit(code=1) from e

        # Print curated data summary
        console.print()
        data_table = Table(title=f"Curated Data Summary (Year {year})")
        data_table.add_column("Metric", style="cyan")
        data_table.add_column("Value", style="green")
        data_table.add_row("Total samples", str(curated_data.n_samples))
        data_table.add_row("Training samples", str(len(curated_data.X_train)))
        data_table.add_row("Test samples", str(len(curated_data.X_test)))
        data_table.add_row("Features", str(len(curated_data.feature_names)))
        data_table.add_row("Target mean", f"{curated_data.target_stats['mean']:.2f}")
        data_table.add_row("Target std", f"{curated_data.target_stats['std']:.2f}")
        data_table.add_row(
            "Target range",
            f"{curated_data.target_stats['min']:.0f} - {curated_data.target_stats['max']:.0f}",
        )
        console.print(data_table)

        console.print("\n[blue]Curated features:[/blue]")
        for feat in curated_data.feature_names:
            dtype = curated_data.X_train[feat].dtype
            console.print(f"  {feat} ({dtype})")

        # Use curated data directly - skip variant loop, train enhanced only
        _train_curated_models(
            curated_data,
            pipeline_config,
            model,
            no_mlflow,
            no_tune,
            output,
            console,
            report_path=report,
        )
        return

    # Standard ETL data flow
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

    # Determine models to train
    model_names = [model] if model is not None else pipeline_config.models.enabled

    # Set output directory
    if output is None:
        output = pipeline_config.cache_dir / "models"
    output.mkdir(parents=True, exist_ok=True)

    # Track all trained models across variants
    all_trained_models: dict[str, tuple[TrainedModel, dict[str, float]]] = {}
    variant_results: dict[ModelVariant, dict[str, dict[str, float]]] = {}

    # Train both baseline and enhanced variants
    for variant in [ModelVariant.BASELINE, ModelVariant.ENHANCED]:
        console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
        console.print(f"[bold blue]Training {variant.value.upper()} models[/bold blue]")
        console.print(f"[bold blue]{'=' * 60}[/bold blue]")

        # Load training data for this variant
        try:
            training_data = load_training_data(data, pipeline_config, variant=variant)
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error loading data for {variant.value}: {e}[/red]")
            raise typer.Exit(code=1) from e

        # Print data summary
        console.print()
        data_table = Table(title=f"{variant.value.capitalize()} Data Summary")
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
        console.print(f"\n[blue]{variant.value.capitalize()} features:[/blue]")
        for feat in training_data.feature_names:
            dtype = training_data.X_train[feat].dtype
            console.print(f"  {feat} ({dtype})")

        console.print(f"\n[blue]Training models: {', '.join(model_names)}[/blue]")
        if not no_tune:
            console.print("[dim]Hyperparameter tuning enabled[/dim]")

        # Train models for this variant
        trainer = ModelTrainer(pipeline_config)
        try:
            trained_models = trainer.train(
                training_data.X_train,
                training_data.y_train,
                model_names=model_names,
                tune_hyperparameters=not no_tune,
            )
        except Exception as e:
            console.print(f"[red]Training failed for {variant.value}: {e}[/red]")
            raise typer.Exit(code=1) from e

        has_test_set = len(training_data.X_test) > 0

        # Display results
        if has_test_set:
            console.print(
                f"\n[blue]Evaluating {variant.value} models on test set...[/blue]"
            )
        else:
            console.print(
                f"\n[blue]{variant.value.capitalize()} model CV results (no holdout)[/blue]"
            )

        results_table = Table(title=f"{variant.value.capitalize()} Model Results")
        results_table.add_column("Model", style="cyan")
        results_table.add_column("R² CV", style="green")
        results_table.add_column("R² No CV", style="green")
        results_table.add_column("Gap", style="yellow")
        results_table.add_column("RMSE", style="yellow")
        results_table.add_column("MAE", style="yellow")
        results_table.add_column("Max Err", style="yellow")
        results_table.add_column("SQV", style="magenta")
        results_table.add_column("Train(s)", style="dim")

        test_results: dict[str, dict[str, float]] = {}
        for name, trained_model in trained_models.items():
            if has_test_set:
                y_pred = trained_model.pipeline.predict(training_data.X_test)
                metrics = compute_metrics(training_data.y_test.values, y_pred)
                test_results[name] = {
                    **metrics.to_dict(),
                    **trained_model.cv_scores,
                }
            else:
                # CV-only: use CV scores directly
                test_results[name] = {**trained_model.cv_scores}

            # Store for cross-variant comparison
            variant_key = f"{variant.value}_{name}"
            all_trained_models[variant_key] = (trained_model, test_results[name])

            # Color-code overfitting gap: green <0.05, yellow 0.05-0.10, red >0.10
            r2_gap = trained_model.cv_scores.get("r2_gap", 0.0)
            if r2_gap < 0.05:
                gap_style = "[green]"
            elif r2_gap < 0.10:
                gap_style = "[yellow]"
            else:
                gap_style = "[red]"

            results_table.add_row(
                name,
                f"{trained_model.cv_scores.get('r2_cv', 0):.4f}",
                f"{trained_model.cv_scores.get('r2_no_cv', 0):.4f}",
                f"{gap_style}{r2_gap:.3f}[/]",
                f"{trained_model.cv_scores.get('rmse', 0):.1f}",
                f"{trained_model.cv_scores.get('mae', 0):.1f}",
                f"{trained_model.cv_scores.get('max_error', 0):.0f}",
                f"{trained_model.cv_scores.get('sqv', 0):.4f}",
                f"{trained_model.training_time_s:.1f}",
            )

        console.print(results_table)
        console.print(
            "[dim]Gap Legend: [green]<0.05[/green] low | "
            "[yellow]0.05-0.10[/yellow] moderate | "
            "[red]>0.10[/red] high overfitting risk[/dim]"
        )
        variant_results[variant] = test_results

        # Save prediction tables for this variant
        predictions_dir = output / "predictions"
        console.print(
            f"\n[blue]Saving {variant.value} prediction tables to {predictions_dir}[/blue]"
        )
        for name, trained_model in trained_models.items():
            # Get train predictions (always available)
            y_train_pred = trained_model.pipeline.predict(training_data.X_train)

            if has_test_set:
                y_test_pred = trained_model.pipeline.predict(training_data.X_test)
            else:
                y_test_pred = np.array([])

            train_path, test_path = save_prediction_tables(
                model_name=name,
                y_train=np.array(training_data.y_train),
                y_train_pred=np.array(y_train_pred),
                y_test=np.array(training_data.y_test),
                y_test_pred=np.array(y_test_pred),
                output_dir=predictions_dir,
                experiment_name=f"{variant.value}_{pipeline_config.year}",
            )
            console.print(f"[dim]  {name}: {train_path.name}, {test_path.name}[/dim]")

        # Save models for this variant
        for name, trained_model in trained_models.items():
            safe_name = name.lower().replace(" ", "_")
            model_path = (
                output / f"{safe_name}_{variant.value}_{pipeline_config.year}.joblib"
            )
            joblib.dump(trained_model.pipeline, model_path)

            # Save metadata sidecar file with feature names
            import json
            metadata = {
                "feature_names": trained_model.feature_names,
                "model_name": name,
                "variant": variant.value,
                "year": pipeline_config.year,
                "cv_scores": trained_model.cv_scores,
            }
            metadata_path = model_path.with_suffix(".meta.json")
            with metadata_path.open("w") as f:
                json.dump(metadata, f, indent=2)

            console.print(f"[green]Saved: {model_path}[/green]")

    # Print comparison summary
    console.print(f"\n[bold blue]{'=' * 60}[/bold blue]")
    console.print("[bold blue]Model Comparison Summary[/bold blue]")
    console.print(f"[bold blue]{'=' * 60}[/bold blue]")

    comparison_table = Table(title="Baseline vs Enhanced Comparison")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("Variant", style="magenta")
    comparison_table.add_column("R² CV", style="green")
    comparison_table.add_column("Gap", style="yellow")
    comparison_table.add_column("RMSE", style="yellow")
    comparison_table.add_column("SQV", style="magenta")

    best_r2 = -float("inf")
    best_model_key = ""

    for variant in [ModelVariant.BASELINE, ModelVariant.ENHANCED]:
        for name, result in variant_results[variant].items():
            variant_key = f"{variant.value}_{name}"

            # Color-code overfitting gap
            r2_gap = result.get("r2_gap", 0.0)
            if r2_gap < 0.05:
                gap_style = "[green]"
            elif r2_gap < 0.10:
                gap_style = "[yellow]"
            else:
                gap_style = "[red]"

            comparison_table.add_row(
                name,
                variant.value,
                f"{result.get('r2_cv', result.get('r2', 0)):.4f}",
                f"{gap_style}{r2_gap:.3f}[/]",
                f"{result['rmse']:.1f}",
                f"{result.get('sqv', 0):.4f}",
            )
            r2_cv = result.get("r2_cv", result.get("r2", 0))
            if r2_cv > best_r2:
                best_r2 = r2_cv
                best_model_key = variant_key

    console.print(comparison_table)

    # MLflow logging
    if not no_mlflow:
        console.print("\n[blue]Logging to MLflow...[/blue]")
        try:
            exp_config = ExperimentConfig(
                name=pipeline_config.mlflow.experiment_name,
                question="Which model variant achieves best R² on test data?",
                experiment_type="model_training",
                region=pipeline_config.region.name,
                year=pipeline_config.year,
            )
            experiment = Experiment(pipeline_config, exp_config)
            run_id = experiment.start_run(f"train-{pipeline_config.year}")

            # Log parameters
            experiment.log_params(
                {
                    "models": ",".join(model_names),
                    "tune_hyperparameters": not no_tune,
                    "variants": "baseline,enhanced",
                }
            )

            # Log metrics for best model
            best_model, best_metrics = all_trained_models[best_model_key]
            experiment.log_metrics(best_metrics)
            experiment.log_params({"best_model": best_model_key})

            # Log best model
            experiment.log_model(
                best_model.pipeline,
                artifact_path="best_model",
            )

            experiment.end_run()
            console.print(f"[green]Logged to MLflow run: {run_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]MLflow logging failed: {e}[/yellow]")

    console.print(f"\n[green]Training complete. Best model: {best_model_key}[/green]")


def _train_curated_models(
    curated_data: "CuratedTrainingData",
    pipeline_config: "PipelineConfig",
    model: str | None,
    no_mlflow: bool,
    no_tune: bool,
    output: Path | None,
    console: "Console",
    report_path: Path | None = None,
) -> None:
    """Train models using curated Germany-wide data."""
    import joblib

    from hochrechnung.evaluation.experiment import Experiment, ExperimentConfig
    from hochrechnung.evaluation.metrics import compute_metrics
    from hochrechnung.evaluation.report import (
        create_report_data,
        generate_html_report,
        save_prediction_tables,
    )
    from hochrechnung.modeling.training import ModelTrainer

    # Determine models to train
    model_names = [model] if model is not None else pipeline_config.models.enabled

    # Set output directory
    if output is None:
        output = pipeline_config.cache_dir / "models"
    output.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[blue]Training models: {', '.join(model_names)}[/blue]")
    if not no_tune:
        console.print("[dim]Hyperparameter tuning enabled[/dim]")

    # Train models
    trainer = ModelTrainer(pipeline_config)
    try:
        trained_models = trainer.train(
            curated_data.X_train,
            curated_data.y_train,
            model_names=model_names,
            tune_hyperparameters=not no_tune,
        )
    except Exception as e:
        console.print(f"[red]Training failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Evaluate models
    has_test_set = len(curated_data.X_test) > 0

    if has_test_set:
        console.print("\n[blue]Evaluating models on test set...[/blue]")
    else:
        console.print("\n[blue]Curated model CV results (no holdout)[/blue]")

    results_table = Table(title="Curated Model Results")
    results_table.add_column("Model", style="cyan")
    results_table.add_column("R² CV", style="green")
    results_table.add_column("R² No CV", style="green")
    results_table.add_column("Gap", style="yellow")
    results_table.add_column("RMSE", style="yellow")
    results_table.add_column("MAE", style="yellow")
    results_table.add_column("SQV", style="magenta")

    test_results: dict[str, dict[str, float]] = {}
    best_r2 = -float("inf")
    best_model_key = ""

    for name, trained_model in trained_models.items():
        import numpy as np

        if has_test_set:
            # Predict on test set
            y_pred = trained_model.pipeline.predict(curated_data.X_test)

            # Compute metrics
            metrics = compute_metrics(
                np.array(curated_data.y_test),
                np.array(y_pred),
            )

            # Add CV metrics from training - convert to dict first
            metrics_dict = metrics.to_dict()
            metrics_dict.update(trained_model.cv_scores)
        else:
            # CV-only: use CV scores directly
            metrics_dict = {**trained_model.cv_scores}

        test_results[name] = metrics_dict

        # Color-code overfitting gap
        r2_gap = metrics_dict.get("r2_gap", 0.0)
        if r2_gap < 0.05:
            gap_style = "[green]"
        elif r2_gap < 0.10:
            gap_style = "[yellow]"
        else:
            gap_style = "[red]"

        results_table.add_row(
            name,
            f"{metrics_dict.get('r2_cv', metrics_dict.get('r2', 0)):.4f}",
            f"{metrics_dict.get('r2_no_cv', metrics_dict.get('r2', 0)):.4f}",
            f"{gap_style}{r2_gap:.3f}[/]",
            f"{metrics_dict.get('rmse', 0):.1f}",
            f"{metrics_dict.get('mae', 0):.1f}",
            f"{metrics_dict.get('sqv', 0):.4f}",
        )

        r2_cv = metrics_dict.get("r2_cv", metrics_dict.get("r2", 0))
        if r2_cv > best_r2:
            best_r2 = r2_cv
            best_model_key = name

    console.print(results_table)
    console.print(
        "[dim]Gap Legend: [green]<0.05[/green] low | "
        "[yellow]0.05-0.10[/yellow] moderate | "
        "[red]>0.10[/red] high overfitting risk[/dim]"
    )

    # Save prediction tables
    import numpy as np

    predictions_dir = output / "predictions"
    console.print(f"\n[blue]Saving prediction tables to {predictions_dir}[/blue]")
    for name, trained_model in trained_models.items():
        # Get train predictions (always available)
        y_train_pred = trained_model.pipeline.predict(curated_data.X_train)

        if has_test_set:
            y_test_pred = trained_model.pipeline.predict(curated_data.X_test)
        else:
            y_test_pred = np.array([])

        train_path, test_path = save_prediction_tables(
            model_name=name,
            y_train=np.array(curated_data.y_train),
            y_train_pred=np.array(y_train_pred),
            y_test=np.array(curated_data.y_test),
            y_test_pred=np.array(y_test_pred),
            output_dir=predictions_dir,
            experiment_name=f"curated_{curated_data.year}",
        )
        console.print(f"[dim]  {name}: {train_path.name}, {test_path.name}[/dim]")

    # Save models with feature metadata
    for name, trained_model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        model_path = output / f"{safe_name}_curated_{curated_data.year}.joblib"
        joblib.dump(trained_model.pipeline, model_path)
        console.print(f"[green]Saved: {model_path}[/green]")

        # Save feature metadata as sidecar JSON
        import json

        metadata_path = model_path.with_suffix(".meta.json")
        metadata = {
            "feature_names": trained_model.feature_names,
            "model_name": name,
            "year": curated_data.year,
            "data_source": "curated",
            "cv_scores": trained_model.cv_scores,
        }
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        console.print(f"[dim]Metadata: {metadata_path}[/dim]")

    # MLflow logging
    if not no_mlflow:
        console.print("\n[blue]Logging to MLflow...[/blue]")
        try:
            exp_config = ExperimentConfig(
                name=f"{pipeline_config.mlflow.experiment_name}_curated",
                question="Curated Germany-wide model performance",
                experiment_type="curated_training",
                region="Germany",
                year=curated_data.year,
            )
            experiment = Experiment(pipeline_config, exp_config)
            run_id = experiment.start_run(f"curated-{curated_data.year}")

            # Log parameters
            experiment.log_params(
                {
                    "models": ",".join(model_names),
                    "tune_hyperparameters": not no_tune,
                    "data_source": "curated",
                    "year": curated_data.year,
                    "features": ",".join(curated_data.feature_names),
                }
            )

            # Log metrics for best model
            best_metrics = test_results[best_model_key]
            experiment.log_metrics(best_metrics)
            experiment.log_params({"best_model": best_model_key})

            # Log best model
            best_model = trained_models[best_model_key]
            experiment.log_model(
                best_model.pipeline,
                artifact_path="best_model",
            )

            experiment.end_run()
            console.print(f"[green]Logged to MLflow run: {run_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]MLflow logging failed: {e}[/yellow]")

    # Generate HTML report (default: in output folder)
    actual_report_path = (
        report_path
        if report_path is not None
        else output / f"training_report_{curated_data.year}.html"
    )
    console.print("\n[blue]Generating training report...[/blue]")
    try:
        report_data = create_report_data(
            curated_data.X_train,
            curated_data.X_test if has_test_set else curated_data.X_train,
            curated_data.y_train,
            curated_data.y_test if has_test_set else curated_data.y_train,
            trained_models,
            curated_data.year,
        )
        generate_html_report(report_data, actual_report_path, console)
        console.print(f"[green]Report saved: {actual_report_path}[/green]")
    except Exception as e:
        console.print(f"[yellow]Report generation failed: {e}[/yellow]")

    console.print(f"\n[green]Training complete. Best model: {best_model_key}[/green]")


@app.command()
def predict(
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
    model_uri: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="MLflow model URI (models:/name/version) or path to .joblib file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path for predictions (without extension).",
        ),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            "-f",
            help="Output format: fgb (FlatGeobuf), gpkg, parquet, or csv.",
        ),
    ] = "fgb",
) -> None:
    """Generate predictions using a trained model."""
    from hochrechnung.config.loader import load_config
    from hochrechnung.modeling.inference import load_model, run_prediction

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    console.print(f"[blue]Loading model: {model_uri}[/blue]")
    try:
        loaded_model = load_model(model_uri)
    except FileNotFoundError as e:
        console.print(f"[red]Model not found: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Show model metadata if available
    if loaded_model.feature_names:
        console.print(
            f"[dim]Model features: {', '.join(loaded_model.feature_names)}[/dim]"
        )

    console.print(
        f"[blue]Running prediction pipeline for year {pipeline_config.year}[/blue]"
    )
    console.print(f"[dim]Output: {output}.{format}[/dim]")

    try:
        result = run_prediction(
            pipeline_config,
            loaded_model.model,
            output_path=output,
            output_format=format,
            feature_names=loaded_model.feature_names,
        )

        # Display results
        console.print()
        table = Table(title="Prediction Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", result.model_name)
        table.add_row("Traffic edges processed", str(result.n_edges))
        table.add_row("Successful predictions", str(result.n_predictions))
        table.add_row("Failed predictions", str(result.n_failures))

        # Add prediction statistics
        predictions = result.predictions["predicted_dtv"]
        valid_preds = predictions.dropna()
        if len(valid_preds) > 0:
            table.add_row("Mean predicted DTV", f"{valid_preds.mean():.1f}")
            table.add_row("Median predicted DTV", f"{valid_preds.median():.1f}")
            table.add_row(
                "DTV range",
                f"{valid_preds.min():.0f} - {valid_preds.max():.0f}",
            )

        console.print(table)

        if result.output_path:
            console.print(
                f"\n[green]Saved predictions to: {result.output_path}[/green]"
            )

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def calibrate(
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
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Path to trained model (.joblib) or MLflow URI.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output path for calibrator (default: alongside model).",
        ),
    ] = None,
    calibrator_type: Annotated[
        str,
        typer.Option(
            "--calibrator",
            help="Calibrator type: global_multiplicative, log_linear, stratified.",
        ),
    ] = "log_linear",
    no_loocv: Annotated[
        bool,
        typer.Option(
            "--no-loocv",
            help="Skip Leave-One-Out Cross-Validation.",
        ),
    ] = False,
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="Launch verification UI before calibration.",
        ),
    ] = False,
    verify_port: Annotated[
        int,
        typer.Option(
            "--verify-port",
            help="HTTP port for verification UI.",
        ),
    ] = 8000,
) -> None:
    """
    Calibrate model predictions using independent counting stations.

    Requires calibration.counter_locations in config YAML pointing to
    verified calibration station CSV with pre-calculated DTV values.

    Example:
        uv run hochrechnung calibrate \\
            --config configs/hessen_2023.yaml \\
            --model output/hessen-2023/models/xgboost_curated_2023.joblib \\
            --calibrator log_linear
    """
    from hochrechnung.calibration import (
        CalibratorType,
        calibrate_and_evaluate,
        create_calibrator,
        save_calibrator,
    )
    from hochrechnung.calibration.loader import load_verified_calibration_counters
    from hochrechnung.config.loader import load_config
    from hochrechnung.modeling.inference import load_model

    console.print(f"[blue]Loading configuration from {config}[/blue]")
    pipeline_config = load_config(config)

    # Validate calibration config
    if pipeline_config.calibration.counter_locations is None:
        console.print(
            "[red]Error: calibration.counter_locations not set in config[/red]"
        )
        console.print(
            "[dim]Add to config YAML:[/dim]\n"
            "[dim]calibration:[/dim]\n"
            "[dim]  counter_locations: 'path/to/calibration_counters.csv'[/dim]"
        )
        raise typer.Exit(code=1)

    # Load model
    console.print(f"[blue]Loading model from {model}[/blue]")
    try:
        loaded_model = load_model(model)
    except FileNotFoundError as e:
        console.print(f"[red]Model not found: {e}[/red]")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(code=1) from e

    if loaded_model.feature_names:
        console.print(
            f"[dim]Model features: {', '.join(loaded_model.feature_names)}[/dim]"
        )

    # Handle --verify flag: run verification UI before calibration
    if verify:
        from hochrechnung.calibration import (
            export_calibration_data,
            flag_calibration_stations,
            start_calibration_verification_server,
        )
        from hochrechnung.modeling.inference import PredictionPipeline

        console.print("\n[bold]Running calibration station verification[/bold]")

        # Load raw calibration data
        cal_path = (
            pipeline_config.data_paths.data_root
            / pipeline_config.calibration.counter_locations
        )
        console.print(f"[dim]Loading raw calibration data from: {cal_path}[/dim]")

        import pandas as pd

        try:
            try:
                raw_stations = pd.read_csv(cal_path, sep=";")
                if len(raw_stations.columns) == 1:
                    raw_stations = pd.read_csv(cal_path, sep=",")
            except Exception:
                raw_stations = pd.read_csv(cal_path, sep=",")
        except FileNotFoundError as e:
            console.print(f"[red]Calibration data not found: {cal_path}[/red]")
            raise typer.Exit(code=1) from e

        console.print(f"[dim]Loaded {len(raw_stations)} raw calibration stations[/dim]")

        # Detect DTV column
        dtv_col_candidates = ["dtv", "DZS_mean_SR", "DZS_mean_year", "dtv_value", "DTV"]
        dtv_column = None
        for col in dtv_col_candidates:
            if col in raw_stations.columns:
                dtv_column = col
                break
        if dtv_column is None:
            console.print(f"[red]No DTV column found. Expected: {dtv_col_candidates}[/red]")
            raise typer.Exit(code=1)

        # Detect ID column
        id_col_candidates = ["id", "counter_id", "station_id", "DZS_id", "Zaehlst_id"]
        id_column = None
        for col in id_col_candidates:
            if col in raw_stations.columns:
                id_column = col
                break
        if id_column is None:
            raw_stations = raw_stations.copy()
            raw_stations["id"] = raw_stations.index.astype(str)
            id_column = "id"

        # Run predictions for calibration stations
        console.print("[blue]Running predictions for calibration stations...[/blue]")
        try:
            pipeline = PredictionPipeline(pipeline_config)
            predicted_dtv = pipeline.predict_at_locations(
                raw_stations,
                loaded_model.model,
                feature_names=loaded_model.feature_names,
            )
            if hasattr(predicted_dtv, "values"):
                predicted_dtv = pd.Series(predicted_dtv.values.ravel())
        except Exception as e:
            console.print(f"[yellow]Warning: Could not run predictions: {e}[/yellow]")
            predicted_dtv = None

        # Flag stations for issues
        console.print("[blue]Flagging stations for issues...[/blue]")
        flagged_stations = flag_calibration_stations(
            raw_stations,
            predicted_dtv=predicted_dtv,
            dtv_column=dtv_column,
        )

        # Count flagged
        n_flagged = len(flagged_stations[~flagged_stations["flag_severity"].isin(["ok", "verified"])])
        console.print(f"[dim]Found {n_flagged} stations with potential issues[/dim]")

        # Export to JSON
        verification_dir = pipeline_config.cache_dir / "calibration_verification" / str(pipeline_config.year)
        verification_dir.mkdir(parents=True, exist_ok=True)

        region = pipeline_config.region.name or pipeline_config.project
        verification_data_path = export_calibration_data(
            flagged_stations,
            verification_dir,
            pipeline_config.year,
            region,
            dtv_column=dtv_column,
            id_column=id_column,
        )

        console.print(f"[green]Exported verification data to: {verification_data_path}[/green]")

        # Generate MBTiles with traffic volumes around calibration stations
        console.print("[blue]Generating MBTiles for map visualization...[/blue]")
        mbtiles_path = None
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            from hochrechnung.ingestion.traffic import load_traffic_volumes
            from hochrechnung.verification.tiles import generate_verification_mbtiles

            # Load traffic volumes
            traffic_gdf = load_traffic_volumes(pipeline_config, validate=False)

            # Detect coordinate columns
            lon_col = "longitude" if "longitude" in flagged_stations.columns else "lon"
            lat_col = "latitude" if "latitude" in flagged_stations.columns else "lat"

            # Create GeoDataFrame from calibration stations
            stations_gdf = gpd.GeoDataFrame(
                flagged_stations,
                geometry=[
                    Point(lon, lat)
                    for lon, lat in zip(
                        flagged_stations[lon_col], flagged_stations[lat_col]
                    )
                ],
                crs="EPSG:4326",
            )

            # Add required columns for MBTiles generation
            if "is_outlier" not in stations_gdf.columns:
                stations_gdf["is_outlier"] = stations_gdf["flag_severity"] != "ok"
            if "counter_id" not in stations_gdf.columns:
                stations_gdf["counter_id"] = stations_gdf[id_column].astype(str)

            mbtiles_path = verification_dir / "volumes.mbtiles"
            generate_verification_mbtiles(
                stations_gdf,
                traffic_gdf,
                mbtiles_path,
                buffer_m=500.0,  # Larger buffer for calibration context
                only_flagged=False,  # Include all stations
            )
            console.print(f"[green]Generated MBTiles: {mbtiles_path}[/green]")

        except FileNotFoundError as e:
            console.print(f"[yellow]Warning: Could not generate MBTiles: {e}[/yellow]")
            console.print("[yellow]Map will show stations without traffic volume overlay[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: MBTiles generation failed: {e}[/yellow]")
            console.print("[yellow]Map will show stations without traffic volume overlay[/yellow]")

        # Start verification server
        console.print("\n[bold]Launching verification UI[/bold]")
        console.print("[dim]Verify calibration stations, then press Ctrl+C to continue[/dim]")

        start_calibration_verification_server(
            verification_data_path=verification_data_path,
            mbtiles_path=mbtiles_path,
            data_root=pipeline_config.data_paths.data_root,
            year=pipeline_config.year,
            region=region,
            stations_df=flagged_stations,
            dtv_column=dtv_column,
            id_column=id_column,
            port=verify_port,
        )

        console.print("\n[green]Verification complete. Continuing with calibration...[/green]")

    # Load calibration stations
    console.print("[blue]Loading calibration stations[/blue]")
    try:
        cal_stations = load_verified_calibration_counters(
            pipeline_config.data_paths.data_root,
            pipeline_config.region.name or pipeline_config.project,
            pipeline_config.year,
            must_exist=True,
        )
    except FileNotFoundError:
        # Try loading from counter_locations path directly
        cal_path = (
            pipeline_config.data_paths.data_root
            / pipeline_config.calibration.counter_locations
        )
        console.print(f"[dim]Loading from: {cal_path}[/dim]")

        import pandas as pd

        try:
            # Try semicolon delimiter first (German CSV format), fall back to comma
            try:
                cal_stations = pd.read_csv(cal_path, sep=";")
                # Verify we got multiple columns (if only 1, wrong delimiter)
                if len(cal_stations.columns) == 1:
                    cal_stations = pd.read_csv(cal_path, sep=",")
            except Exception:
                cal_stations = pd.read_csv(cal_path, sep=",")
        except FileNotFoundError as e:
            console.print(f"[red]Calibration data not found: {cal_path}[/red]")
            raise typer.Exit(code=1) from e

    if cal_stations is None or len(cal_stations) == 0:
        console.print("[red]Error: No calibration stations found[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Loaded {len(cal_stations)} calibration stations[/dim]")

    # Run predictions for calibration stations
    console.print("[blue]Running predictions for calibration stations[/blue]")

    from hochrechnung.modeling.inference import PredictionPipeline

    try:
        pipeline = PredictionPipeline(pipeline_config)
        station_predictions = pipeline.predict_at_locations(
            cal_stations,
            loaded_model.model,
            feature_names=loaded_model.feature_names,
        )
    except Exception as e:
        console.print(f"[red]Prediction failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Handle case where prediction returns DataFrame
    if hasattr(station_predictions, "values"):
        station_predictions = station_predictions.values.ravel()

    console.print(f"[dim]Generated {len(station_predictions)} predictions[/dim]")

    # Create and fit calibrator
    console.print(f"[blue]Fitting {calibrator_type} calibrator[/blue]")

    # Get stratify_by from config if using stratified calibrator
    stratify_by = pipeline_config.calibration.stratify_by

    try:
        calibrator = create_calibrator(
            calibrator_type,
            stratify_by=stratify_by,
            min_stations_per_stratum=pipeline_config.calibration.min_stations_per_stratum,
        )
    except (ValueError, NotImplementedError) as e:
        console.print(f"[red]Error creating calibrator: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Prepare meta DataFrame for stratified calibrators
    meta = None
    if calibrator_type == "stratified" and stratify_by:
        missing_cols = set(stratify_by) - set(cal_stations.columns)
        if missing_cols:
            console.print(
                f"[red]Error: Calibration data missing columns for stratification: {missing_cols}[/red]"
            )
            raise typer.Exit(code=1)
        meta = cal_stations[stratify_by]

    # Resolve DTV column name (handle different naming conventions)
    dtv_column_candidates = ["dtv", "DZS_mean_SR", "DZS_mean_year", "dtv_value", "DTV"]
    dtv_column = None
    for col in dtv_column_candidates:
        if col in cal_stations.columns:
            dtv_column = col
            break
    if dtv_column is None:
        console.print(
            f"[red]Error: No DTV column found in calibration data. "
            f"Expected one of: {dtv_column_candidates}[/red]"
        )
        console.print(f"[dim]Available columns: {list(cal_stations.columns)[:20]}...[/dim]")
        raise typer.Exit(code=1)
    console.print(f"[dim]Using DTV column: {dtv_column}[/dim]")

    # Run calibration and evaluation
    try:
        result = calibrate_and_evaluate(
            calibrator,
            y_pred=station_predictions,
            y_true=cal_stations[dtv_column].values,
            meta=meta,
            run_loocv_validation=not no_loocv,
        )
    except Exception as e:
        console.print(f"[red]Calibration failed: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Display results table
    table = Table(title="Calibration Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Uncalibrated", style="yellow")
    table.add_column("Calibrated", style="green")
    table.add_column("Improvement", style="magenta")

    # Add metrics rows
    metrics_to_show = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("r2", "R²"),
        ("sqv", "SQV"),
    ]

    for metric_key, metric_name in metrics_to_show:
        uncal = getattr(result.uncalibrated_metrics, metric_key)
        cal = getattr(result.calibrated_metrics, metric_key)

        if metric_key == "r2":
            imp_val = result.improvement.get("r2_gain", 0)
            imp_str = f"{imp_val:+.4f}"
        elif metric_key == "sqv":
            imp_val = result.improvement.get("sqv_gain", 0)
            imp_str = f"{imp_val:+.4f}"
        else:
            imp_val = result.improvement.get(f"{metric_key}_reduction", 0)
            imp_str = f"{imp_val * 100:+.1f}%"

        table.add_row(metric_name, f"{uncal:.2f}", f"{cal:.2f}", imp_str)

    console.print(table)

    # LOOCV results
    if result.loocv_metrics:
        console.print(
            f"\n[blue]LOOCV MAE:[/blue] "
            f"{result.loocv_metrics.mae_mean:.2f} ± {result.loocv_metrics.mae_std:.2f}"
        )
        console.print(
            f"[blue]LOOCV RMSE:[/blue] "
            f"{result.loocv_metrics.rmse_mean:.2f} ± {result.loocv_metrics.rmse_std:.2f}"
        )

    # Save calibrator
    if output is None:
        output = Path(model).with_suffix("")

    try:
        cal_path, meta_path = save_calibrator(calibrator, output, result)
        console.print(f"\n[green]Calibrator saved to: {cal_path}[/green]")
        console.print(f"[green]Metadata saved to: {meta_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving calibrator: {e}[/red]")
        raise typer.Exit(code=1) from e

    # Print calibration params
    params = calibrator.get_params()
    console.print(f"\n[blue]Calibration parameters ({params.calibrator_type}):[/blue]")
    for k, v in params.params.items():
        if isinstance(v, float):
            console.print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            console.print(f"  {k}: {len(v)} items")
        else:
            console.print(f"  {k}: {v}")

    # Run full prediction pipeline and apply calibration
    console.print("\n[blue]Running predictions for all edges...[/blue]")
    try:
        pred_result = pipeline.run(
            loaded_model.model,
            feature_names=loaded_model.feature_names,
        )
        console.print(
            f"[dim]Generated {pred_result.n_predictions} predictions[/dim]"
        )

        # Apply calibration to predictions
        console.print("[blue]Applying calibration to predictions...[/blue]")
        uncalibrated_dtv = pred_result.predictions["predicted_dtv"].values
        calibrated_dtv = calibrator.predict(uncalibrated_dtv)

        # Add calibrated predictions to output
        pred_result.predictions["predicted_dtv_uncalibrated"] = uncalibrated_dtv
        pred_result.predictions["predicted_dtv"] = calibrated_dtv

        # Save to predictions folder
        predictions_dir = pipeline_config.predictions_dir
        predictions_dir.mkdir(parents=True, exist_ok=True)

        # Build output filename from model and calibrator
        model_name = Path(model).stem
        output_file = predictions_dir / f"{model_name}_calibrated_{pipeline_config.year}.fgb"

        pred_result.predictions.to_file(output_file, driver="FlatGeobuf")
        console.print(f"\n[green]Calibrated predictions saved to: {output_file}[/green]")

        # Print prediction statistics
        valid_preds = pred_result.predictions["predicted_dtv"].dropna()
        console.print(f"[dim]  Mean DTV: {valid_preds.mean():.1f}[/dim]")
        console.print(f"[dim]  Median DTV: {valid_preds.median():.1f}[/dim]")
        console.print(
            f"[dim]  Range: {valid_preds.min():.0f} - {valid_preds.max():.0f}[/dim]"
        )

    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate calibrated predictions: {e}[/yellow]")


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
            pipeline_config.cache_dir
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
def verify(
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
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="HTTP port for verification UI.",
        ),
    ] = 8000,
    prepare_only: Annotated[
        bool,
        typer.Option(
            "--prepare-only",
            help="Only prepare verification data, don't start UI.",
        ),
    ] = False,
) -> None:
    """
    Run counter verification workflow.

    This command:
    1. Runs ETL in verification mode (if not already done)
    2. Generates MBTiles with traffic volumes
    3. Exports verification data
    4. Launches interactive web UI for manual verification

    After verifying counters in the UI, the verified dataset is saved
    to data/verified/counters_verified_{year}.csv
    """
    from hochrechnung.config.loader import load_config
    from hochrechnung.etl import run_etl
    from hochrechnung.verification.export import export_verification_data
    from hochrechnung.verification.server import start_verification_server
    from hochrechnung.verification.tiles import (
        generate_counter_geojson,
        generate_verification_mbtiles,
    )

    console.print("[blue]Starting counter verification workflow[/blue]")

    # Load config
    pipeline_config = load_config(config)
    year = pipeline_config.year

    # Resolve images database path
    images_db_path = None
    if pipeline_config.data_paths.images_db is not None:
        images_db_path = (
            pipeline_config.data_paths.data_root
            / pipeline_config.data_paths.images_db
        )
        if images_db_path.exists():
            console.print(f"[dim]Images DB: {images_db_path}[/dim]")
        else:
            console.print(f"[yellow]Images DB not found: {images_db_path}[/yellow]")
            images_db_path = None

    # Step 1: Run ETL in verification mode
    console.print("\n[bold]Step 1: Running ETL in verification mode[/bold]")
    result = run_etl(pipeline_config, mode="verification")

    if result.verification_data is None:
        console.print("[red]Error: No verification data generated by ETL[/red]")
        raise typer.Exit(code=1)

    if result.n_flagged_outliers == 0:
        console.print(
            "[green]No flagged counters found. All counters look good![/green]"
        )
        if not prepare_only:
            console.print(
                "[dim]You can still launch the UI to verify unflagged counters[/dim]"
            )

    # Step 2: Prepare verification output directory
    verification_dir = pipeline_config.cache_dir / "verification" / str(year)
    verification_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold]Step 2: Generating verification assets[/bold]")
    console.print(f"[dim]Output: {verification_dir}[/dim]")

    # Step 3: Generate MBTiles
    console.print("\n[cyan]Generating MBTiles...[/cyan]")

    # Need to load traffic volumes for MBTiles generation
    from hochrechnung.ingestion.traffic import load_traffic_volumes

    traffic_gdf = load_traffic_volumes(pipeline_config, validate=False)

    mbtiles_path = verification_dir / "volumes.mbtiles"

    try:
        import geopandas as gpd

        # Convert verification_data to GeoDataFrame
        verification_gdf = gpd.GeoDataFrame(
            result.verification_data,
            geometry=gpd.points_from_xy(
                result.verification_data["longitude"],
                result.verification_data["latitude"],
            ),
            crs="EPSG:4326",
        )

        generate_verification_mbtiles(
            verification_gdf,
            traffic_gdf,
            mbtiles_path,
            buffer_m=250.0,
            only_flagged=True,
        )

        console.print(f"[green]✓ Generated MBTiles: {mbtiles_path}[/green]")

    except FileNotFoundError as e:
        console.print(f"[yellow]⚠ Warning: {e}[/yellow]")
        console.print(
            "[yellow]  Verification UI will work without MBTiles, but tiles won't be visible[/yellow]"
        )
        mbtiles_path = None
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: Failed to generate MBTiles: {e}[/yellow]")
        console.print(
            "[yellow]  Verification UI will work without MBTiles, but tiles won't be visible[/yellow]"
        )
        mbtiles_path = None

    # Step 4: Export verification data
    console.print("\n[cyan]Exporting verification data...[/cyan]")

    # Get outlier thresholds from verification_data
    outlier_result = (
        result.verification_data.get("dtv_volume_ratio").describe()
        if "dtv_volume_ratio" in result.verification_data
        else None
    )

    export_verification_data(
        result.verification_data,
        verification_dir,
        year,
        outlier_threshold_lower=0.0,  # Will be calculated from data
        outlier_threshold_upper=10.0,  # Will be calculated from data
        median_ratio=result.verification_data.get("dtv_volume_ratio").median()
        if "dtv_volume_ratio" in result.verification_data
        else 1.0,
        images_db_path=images_db_path,
    )

    verification_data_path = verification_dir / "verification_data.json"
    console.print(f"[green]✓ Exported data: {verification_data_path}[/green]")

    # Step 5: Generate counter GeoJSON
    console.print("\n[cyan]Generating counter GeoJSON...[/cyan]")

    counter_geojson_path = verification_dir / "counters.geojson"
    generate_counter_geojson(verification_gdf, counter_geojson_path)

    console.print(f"[green]✓ Generated GeoJSON: {counter_geojson_path}[/green]")

    if prepare_only:
        console.print("\n[green]✓ Verification assets prepared successfully[/green]")
        console.print(f"[dim]Data: {verification_dir}[/dim]")
        return

    # Step 6: Start verification server
    console.print("\n[bold]Step 3: Launching verification UI[/bold]")

    if mbtiles_path is None or not mbtiles_path.exists():
        console.print(
            "[yellow]⚠ Warning: MBTiles not available. Map tiles won't be visible.[/yellow]"
        )
        # Create empty placeholder
        mbtiles_path = verification_dir / "volumes.mbtiles"

    start_verification_server(
        verification_data_path,
        mbtiles_path,
        pipeline_config.data_paths.data_root,
        year,
        result.verification_data,
        port=port,
        images_db_path=images_db_path,
        project=pipeline_config.project,
    )


@app.command()
def version() -> None:
    """Show version information."""
    from hochrechnung import __version__

    console.print(f"hochrechnung version {__version__}")


if __name__ == "__main__":
    app()
