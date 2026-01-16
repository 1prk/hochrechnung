"""Command-line interface for the hochrechnung pipeline."""

from pathlib import Path
from typing import Annotated, Optional

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
    experiment_name: Annotated[
        Optional[str],
        typer.Option(
            "--experiment",
            "-e",
            help="MLflow experiment name. Overrides config value.",
        ),
    ] = None,
) -> None:
    """Train models using the specified configuration."""
    console.print(f"[blue]Loading configuration from {config}[/blue]")
    # TODO: Implement training pipeline
    console.print("[yellow]Training not yet implemented[/yellow]")


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
            help="MLflow model URI or path to model artifact.",
        ),
    ],
    output: Annotated[
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
            help="Path to ETL output CSV. Defaults to cache/training_data_{year}.csv",
        ),
    ] = None,
) -> None:
    """
    Assess ETL output values against original source data.

    Compares the ETL output with source data to verify that all
    transformations produced correct values. This is specific to
    the Hessen 2024 dataset.
    """
    from hochrechnung.assessment import AssessmentReporter, AssessmentRunner
    from hochrechnung.config.loader import load_config

    console.print("[blue]Running ETL output assessment...[/blue]")

    # Load config
    pipeline_config = load_config(config)

    # Verify this is Hessen 2024
    if pipeline_config.year != 2024 or pipeline_config.region.name != "Hessen":
        console.print(
            "[yellow]Warning: Assessment is designed for Hessen 2024 data. "
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
