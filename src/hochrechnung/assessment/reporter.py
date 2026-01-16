"""
Reporter for assessment results.

Formats assessment results for console output using Rich.
"""

from typing import ClassVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hochrechnung.assessment.core import AssessmentResult, CheckResult, CheckStatus


class AssessmentReporter:
    """
    Formats and displays assessment results.

    Uses Rich for formatted console output.
    """

    STATUS_STYLES: ClassVar[dict[CheckStatus, tuple[str, str]]] = {
        CheckStatus.PASS: ("PASS", "green"),
        CheckStatus.WARN: ("WARN", "yellow"),
        CheckStatus.FAIL: ("FAIL", "red"),
        CheckStatus.SKIP: ("SKIP", "dim"),
    }

    def __init__(self, console: Console | None = None) -> None:
        """
        Initialize reporter.

        Args:
            console: Rich console for output. Creates new if not provided.
        """
        self.console = console or Console()

    def print_results(self, result: AssessmentResult) -> None:
        """
        Print full assessment results.

        Args:
            result: AssessmentResult to display.
        """
        # Header
        self.console.print()
        self._print_header(result)

        # Results table
        self.console.print()
        self._print_checks_table(result)

        # Summary
        self.console.print()
        self._print_summary(result)

        # Failure details if any
        failures = [c for c in result.checks if c.status == CheckStatus.FAIL]
        if failures:
            self.console.print()
            self._print_failure_details(failures)

    def _print_header(self, result: AssessmentResult) -> None:
        """Print assessment header."""
        _, status_style = self.STATUS_STYLES[result.overall_status]

        title = Text("ETL Assessment Results", style="bold")
        subtitle = Text(f"File: {result.etl_path}", style="dim")

        self.console.print(
            Panel.fit(
                Text.assemble(title, "\n", subtitle),
                border_style=status_style,
            )
        )

    def _print_checks_table(self, result: AssessmentResult) -> None:
        """Print table of individual check results."""
        table = Table(
            title="Assessment Checks",
            show_header=True,
            header_style="bold",
        )
        table.add_column("Check", style="cyan", min_width=25)
        table.add_column("Status", justify="center", min_width=8)
        table.add_column("Result", min_width=40)
        table.add_column("Checked", justify="right", min_width=8)
        table.add_column("Passed", justify="right", min_width=8)

        for check in result.checks:
            status_text, status_style = self.STATUS_STYLES[check.status]

            table.add_row(
                check.name,
                Text(status_text, style=status_style),
                check.message,
                str(check.n_checked) if check.n_checked > 0 else "-",
                str(check.n_passed) if check.n_passed > 0 else "-",
            )

        self.console.print(table)

    def _print_summary(self, result: AssessmentResult) -> None:
        """Print summary statistics."""
        status_text, status_style = self.STATUS_STYLES[result.overall_status]

        summary = Table(show_header=False, box=None)
        summary.add_column("Label", style="bold")
        summary.add_column("Value")

        summary.add_row("Overall Status:", Text(status_text, style=status_style))
        summary.add_row("Checks Passed:", Text(str(result.n_passed), style="green"))
        summary.add_row(
            "Checks Failed:",
            Text(str(result.n_failed), style="red" if result.n_failed > 0 else "dim"),
        )
        summary.add_row(
            "Checks Warned:",
            Text(
                str(result.n_warned), style="yellow" if result.n_warned > 0 else "dim"
            ),
        )

        self.console.print(Panel(summary, title="Summary", border_style=status_style))

    def _print_failure_details(self, failures: list[CheckResult]) -> None:
        """Print details for failed checks."""
        self.console.print(Text("Failure Details", style="bold red"))

        for check in failures:
            self.console.print(f"\n[red]{check.name}[/red]: {check.message}")

            if check.details:
                self.console.print(f"  [dim]{check.details}[/dim]")

            if check.sample_failures is not None and not check.sample_failures.empty:
                self.console.print("  [dim]Sample failures (showing up to 10):[/dim]")

                # Create a Rich table for better formatting
                sample_table = Table(
                    show_header=True,
                    header_style="bold dim",
                    box=None,
                    padding=(0, 1),
                )

                # Add columns from DataFrame
                for col in check.sample_failures.columns:
                    sample_table.add_column(str(col), overflow="fold")

                # Add rows (up to 10)
                for _, row in check.sample_failures.head(10).iterrows():
                    sample_table.add_row(*[str(v) for v in row])

                self.console.print(sample_table)

    def print_check(self, check: CheckResult) -> None:
        """
        Print a single check result.

        Args:
            check: CheckResult to display.
        """
        status_text, status_style = self.STATUS_STYLES[check.status]

        self.console.print(
            f"[{status_style}]{status_text}[/{status_style}] "
            f"[cyan]{check.name}[/cyan]: {check.message}"
        )

        if check.details:
            self.console.print(f"  [dim]{check.details}[/dim]")
