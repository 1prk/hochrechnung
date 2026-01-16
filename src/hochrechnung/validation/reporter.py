"""
Console reporter for validation results.

Formats validation results using Rich for clear, colored output.
"""

from rich.console import Console
from rich.table import Table

from hochrechnung.validation.core import ValidationResult


class ConsoleReporter:
    """Formats and displays validation results to the console."""

    def __init__(self, console: Console) -> None:
        """
        Initialize console reporter.

        Args:
            console: Rich Console instance for output.
        """
        self.console = console

    def print_results(self, results: list[ValidationResult]) -> None:
        """
        Print validation results as a formatted table.

        Args:
            results: List of validation results to display.
        """
        # Create results table
        table = Table(title="Schema Validation Results", show_header=True)
        table.add_column("Dataset", style="cyan", no_wrap=True)
        table.add_column("Schema", style="blue")
        table.add_column("Status", justify="center")
        table.add_column("Rows", justify="right")
        table.add_column("Details", style="dim")

        # Add rows
        for result in results:
            status = self._format_status(result)
            row_count = str(result.row_count) if result.row_count is not None else "-"
            details = self._format_details(result)

            table.add_row(
                result.dataset_name,
                result.schema_name or "-",
                status,
                row_count,
                details,
            )

        self.console.print(table)

        # Print summary
        self._print_summary(results)

        # Print detailed errors for failed validations
        self._print_detailed_errors(results)

    def _format_status(self, result: ValidationResult) -> str:
        """
        Format validation status with color.

        Args:
            result: Validation result.

        Returns:
            Formatted status string with color markup.
        """
        if not result.exists:
            return "[yellow]Missing[/yellow]"
        if result.schema_valid is None:
            return "[yellow]Skipped[/yellow]"
        if result.schema_valid:
            return "[green]Pass[/green]"
        return "[red]Fail[/red]"

    def _format_details(self, result: ValidationResult) -> str:
        """
        Format details/error message.

        Args:
            result: Validation result.

        Returns:
            Short details string (errors shown separately).
        """
        if not result.exists:
            return "File not found"
        if result.schema_valid is None:
            return result.error_message or "No schema"
        if result.schema_valid:
            return "OK"
        # For failures, just indicate there are errors (details printed below)
        return "See errors below"

    def _print_summary(self, results: list[ValidationResult]) -> None:
        """
        Print summary statistics.

        Args:
            results: List of validation results.
        """
        total = len(results)
        passed = sum(1 for r in results if r.schema_valid is True)
        failed = sum(1 for r in results if r.schema_valid is False)
        skipped = sum(1 for r in results if r.schema_valid is None)

        self.console.print()
        self.console.print("[bold]Summary:[/bold]")
        self.console.print(f"  Total datasets: {total}")
        self.console.print(f"  [green]Passed: {passed}[/green]")
        self.console.print(f"  [red]Failed: {failed}[/red]")
        self.console.print(f"  [yellow]Skipped: {skipped}[/yellow]")

    def _print_detailed_errors(self, results: list[ValidationResult]) -> None:
        """
        Print detailed error messages for failed validations.

        Args:
            results: List of validation results.
        """
        failed = [r for r in results if r.schema_valid is False]

        if not failed:
            return

        self.console.print()
        self.console.print("[bold red]Validation Errors:[/bold red]")

        for result in failed:
            self.console.print()
            self.console.print(
                f"[bold]{result.dataset_name}[/bold] "
                f"(schema: {result.schema_name}):"
            )
            self.console.print(f"  File: {result.file_path}")
            if result.error_message:
                # Print error with indentation
                for line in result.error_message.split("\n"):
                    self.console.print(f"  {line}")
