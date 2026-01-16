"""Command-line interface for the hochrechnung pipeline."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="hochrechnung",
    help="Bicycle traffic estimation pipeline for DTV prediction.",
    no_args_is_help=True,
)

console = Console()


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
    console.print("[blue]Running schema validation...[/blue]")
    # TODO: Implement validation
    console.print("[yellow]Validation not yet implemented[/yellow]")


@app.command()
def version() -> None:
    """Show version information."""
    from hochrechnung import __version__

    console.print(f"hochrechnung version {__version__}")


if __name__ == "__main__":
    app()
