"""
Training report generation.

Generates HTML reports with scatterplots, metrics tables, and model comparison.
Also provides prediction table export for model evaluation.
"""

import base64
import contextlib
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table

from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    from hochrechnung.modeling.training import TrainedModel

log = get_logger(__name__)


def save_prediction_tables(
    model_name: str,
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    output_dir: Path,
    *,
    experiment_name: str | None = None,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """
    Save prediction tables for train and test splits.

    Creates CSV files with Actual and Predicted columns, compatible with
    the legacy prediction format used for model evaluation.

    Args:
        model_name: Name of the model (e.g., "Random Forest").
        y_train: Actual training target values.
        y_train_pred: Predicted training target values.
        y_test: Actual test target values.
        y_test_pred: Predicted test target values.
        output_dir: Directory to save prediction tables.
        experiment_name: Optional experiment prefix for filenames.
        timestamp: Optional timestamp string. If None, uses current time.

    Returns:
        Tuple of (train_predictions_path, test_predictions_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize model name for filename
    safe_model_name = model_name.replace(" ", "_").lower()

    # Build filename prefix
    if experiment_name:
        safe_experiment = experiment_name.replace(" ", "_").lower()
        prefix = f"{safe_experiment}_{safe_model_name}"
    else:
        prefix = safe_model_name

    # Save train predictions
    train_df = pd.DataFrame({"Actual": y_train, "Predicted": y_train_pred})
    train_path = output_dir / f"{prefix}_train_predictions_{timestamp}.csv"
    train_df.to_csv(train_path, index=False)

    # Save test predictions
    test_df = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
    test_path = output_dir / f"{prefix}_test_predictions_{timestamp}.csv"
    test_df.to_csv(test_path, index=False)

    log.info(
        "Saved prediction tables",
        model=model_name,
        train_path=str(train_path),
        test_path=str(test_path),
        n_train=len(y_train),
        n_test=len(y_test),
    )

    return train_path, test_path


@dataclass
class FeatureImportance:
    """Feature importance data for a model."""

    feature_names: list[str]
    importances: np.ndarray
    importance_type: str  # 'gini', 'permutation', 'coefficients', etc.


@dataclass
class ModelResult:
    """Results for a single model."""

    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    training_time_s: float
    prediction_time_s: float
    feature_importance: FeatureImportance | None = None


@dataclass
class ReportData:
    """Data for generating a training report."""

    # Descriptive data
    dtv_actual: Any  # DTV_DZS - actual counter values (np.ndarray)
    stadtradeln_volume: Any  # VM_SR - STADTRADELN volumes (np.ndarray)

    # Model results
    model_results: list[ModelResult]

    # Metadata
    year: int
    n_samples: int
    n_train: int
    n_test: int
    feature_names: list[str]


def create_report_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    trained_models: dict[str, "TrainedModel"],
    year: int,
) -> ReportData:
    """
    Create report data from training results.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training targets.
        y_test: Test targets.
        trained_models: Dictionary of trained models.
        year: Data year.

    Returns:
        ReportData object for report generation.
    """
    from hochrechnung.evaluation.metrics import compute_metrics

    # Combine train and test for descriptive stats
    y_all = pd.concat([y_train, y_test])

    # Get stadtradeln_volume if available
    stadtradeln_col = None
    for col in ["stadtradeln_volume", "count", "volume"]:
        if col in X_train.columns:
            stadtradeln_col = col
            break

    if stadtradeln_col is not None:
        stadtradeln_all = pd.concat([X_train[stadtradeln_col], X_test[stadtradeln_col]])
        stadtradeln_volume = np.asarray(stadtradeln_all.values)
    else:
        stadtradeln_volume = np.zeros_like(np.asarray(y_all.values))
        log.warning("No stadtradeln_volume column found for descriptive plot")

    # Generate predictions and metrics for each model
    model_results = []
    for name, trained_model in trained_models.items():
        # Pipeline has predict method (TransformedTargetRegressor or sklearn Pipeline)
        y_pred = trained_model.pipeline.predict(X_test)  # type: ignore[union-attr]

        # Compute metrics on test set
        metrics = compute_metrics(np.array(y_test), np.array(y_pred))

        # Merge with CV scores
        result_metrics = {
            **metrics.to_dict(),
            **trained_model.cv_scores,
        }

        # Extract feature importance
        feature_importance = extract_feature_importance(
            trained_model.pipeline,
            list(X_train.columns),
        )

        model_results.append(
            ModelResult(
                name=name,
                y_true=np.array(y_test),
                y_pred=np.array(y_pred),
                metrics=result_metrics,
                training_time_s=trained_model.training_time_s,
                prediction_time_s=trained_model.cv_scores.get("prediction_time_s", 0.0),
                feature_importance=feature_importance,
            )
        )

    return ReportData(
        dtv_actual=np.array(y_all),
        stadtradeln_volume=stadtradeln_volume,
        model_results=model_results,
        year=year,
        n_samples=len(y_train) + len(y_test),
        n_train=len(y_train),
        n_test=len(y_test),
        feature_names=list(X_train.columns),
    )


def generate_metrics_table(
    model_results: list[ModelResult],
    console: Console | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Generate metrics table for all models.

    Returns both a DataFrame and HTML string.
    """
    rows = []
    for result in model_results:
        m = result.metrics
        rows.append(
            {
                "Model": result.name,
                "R²": m.get("r2_cv", m.get("r2", 0.0)),
                "NMAPE": -m.get("mape", 0.0),  # Negative MAPE (as per paper)
                "NMAE": -m.get("mae", 0.0),  # Negative MAE (as per paper)
                "Max Error": m.get("max_error", 0.0),
                "SQV": m.get("sqv", 0.0),
                "R² w/o CV": m.get("r2_no_cv", 0.0),
                "Train Time (s)": result.training_time_s,
                "Pred Time (ms)": result.prediction_time_s * 1000,  # Convert to ms
            }
        )

    df = pd.DataFrame(rows)

    # Print to console if provided
    if console is not None:
        table = Table(title="Model Metrics Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("R²", style="green")
        table.add_column("NMAPE", style="yellow")
        table.add_column("NMAE", style="yellow")
        table.add_column("Max Error", style="yellow")
        table.add_column("SQV", style="magenta")
        table.add_column("R² w/o CV", style="green")
        table.add_column("Train (s)", style="dim")
        table.add_column("Pred (ms)", style="dim")

        for row in rows:
            table.add_row(
                row["Model"],
                f"{row['R²']:.4f}",
                f"{row['NMAPE']:.4f}",
                f"{row['NMAE']:.1f}",
                f"{row['Max Error']:.0f}",
                f"{row['SQV']:.4f}",
                f"{row['R² w/o CV']:.4f}",
                f"{row['Train Time (s)']:.1f}",
                f"{row['Pred Time (ms)']:.3f}",
            )

        console.print(table)

    # Generate HTML
    html = df.to_html(
        index=False,
        float_format=lambda x: f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}",
        classes="metrics-table",
    )

    return df, html


def generate_dtv_range_table(
    model_results: list[ModelResult],
    console: Console | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Generate table with min/max DTV predictions per model.

    Returns both a DataFrame and HTML string.
    """
    rows = []
    for result in model_results:
        rows.append(
            {
                "Model": result.name,
                "Min DTV (predicted)": result.y_pred.min(),
                "Max DTV (predicted)": result.y_pred.max(),
                "Mean DTV (predicted)": result.y_pred.mean(),
                "Min DTV (actual)": result.y_true.min(),
                "Max DTV (actual)": result.y_true.max(),
            }
        )

    df = pd.DataFrame(rows)

    # Print to console if provided
    if console is not None:
        table = Table(title="DTV Range of Estimates")
        table.add_column("Model", style="cyan")
        table.add_column("Min DTV (pred)", style="green")
        table.add_column("Max DTV (pred)", style="green")
        table.add_column("Mean DTV (pred)", style="green")
        table.add_column("Min DTV (actual)", style="dim")
        table.add_column("Max DTV (actual)", style="dim")

        for row in rows:
            table.add_row(
                row["Model"],
                f"{row['Min DTV (predicted)']:.0f}",
                f"{row['Max DTV (predicted)']:.0f}",
                f"{row['Mean DTV (predicted)']:.0f}",
                f"{row['Min DTV (actual)']:.0f}",
                f"{row['Max DTV (actual)']:.0f}",
            )

        console.print(table)

    # Generate HTML
    html = df.to_html(
        index=False,
        float_format=lambda x: f"{x:.0f}",
        classes="range-table",
    )

    return df, html


def _fig_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64 PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def extract_feature_importance(
    model: Any,
    feature_names: list[str],
) -> FeatureImportance | None:
    """
    Extract feature importances from a trained model.

    Handles different model types:
    - Tree-based models (RandomForest, GradientBoosting, XGBoost): use feature_importances_
    - Linear models (LinearRegression, Lasso, ElasticNet): use coefficients
    - Other models: return None

    Args:
        model: Trained sklearn model or pipeline.
        feature_names: List of feature names.

    Returns:
        FeatureImportance object or None if not available.
    """
    # Navigate through TransformedTargetRegressor -> Pipeline -> model
    inner_model = model
    preprocessor = None

    if hasattr(model, "regressor"):
        inner_model = model.regressor

    if hasattr(inner_model, "named_steps"):
        preprocessor = inner_model.named_steps.get("preprocessor")
        inner_model = inner_model.named_steps.get("model", inner_model)

    # Get transformed feature names from preprocessor if available
    transformed_names = feature_names
    if preprocessor is not None:
        with contextlib.suppress(Exception):
            transformed_names = list(preprocessor.get_feature_names_out())

    # Extract importances based on model type
    importances = None
    importance_type = "unknown"

    # Tree-based models (RandomForest, GradientBoosting, XGBoost, etc.)
    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
        importance_type = "gini_importance"

    # Linear models (coefficients)
    elif hasattr(inner_model, "coef_"):
        coef = inner_model.coef_
        if coef.ndim > 1:
            coef = coef.flatten()
        importances = np.abs(coef)  # Use absolute value for importance
        importance_type = "coefficient_magnitude"

    if importances is None:
        return None

    # Ensure lengths match
    if len(importances) != len(transformed_names):
        log.warning(
            "Feature importance length mismatch",
            n_importances=len(importances),
            n_features=len(transformed_names),
        )
        # Try to use original feature names
        if len(importances) == len(feature_names):
            transformed_names = feature_names
        else:
            transformed_names = [f"feature_{i}" for i in range(len(importances))]

    return FeatureImportance(
        feature_names=transformed_names,
        importances=importances,
        importance_type=importance_type,
    )


def generate_feature_importance_plot(
    feature_importance: FeatureImportance,
    model_name: str,
    top_n: int = 15,
) -> str:
    """
    Generate horizontal bar plot of feature importances.

    Args:
        feature_importance: FeatureImportance object.
        model_name: Name of the model.
        top_n: Number of top features to show.

    Returns:
        Base64 encoded PNG image.
    """
    # Sort by importance
    indices = np.argsort(feature_importance.importances)[::-1]
    sorted_names = [feature_importance.feature_names[i] for i in indices]
    sorted_importances = feature_importance.importances[indices]

    # Take top N
    n_show = min(top_n, len(sorted_names))
    show_names = sorted_names[:n_show][::-1]  # Reverse for horizontal bar
    show_importances = sorted_importances[:n_show][::-1]

    # Normalize to percentages
    total = sorted_importances.sum()
    if total > 0:
        show_importances_pct = (show_importances / total) * 100
    else:
        show_importances_pct = show_importances

    fig, ax = plt.subplots(figsize=(10, max(6, n_show * 0.4)))

    # Create horizontal bar chart
    bars = ax.barh(
        range(n_show), show_importances_pct, color="steelblue", edgecolor="none"
    )

    # Add value labels on bars
    for bar, pct in zip(bars, show_importances_pct):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_yticks(range(n_show))
    ax.set_yticklabels(show_names, fontsize=10)
    ax.set_xlabel(f"Importance ({feature_importance.importance_type})", fontsize=11)
    ax.set_title(f"{model_name}: Feature Importance", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # Adjust x-axis to accommodate labels
    max_val = max(show_importances_pct)
    ax.set_xlim(0, max_val * 1.15)

    plt.tight_layout()

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def print_feature_importance_table(
    feature_importance: FeatureImportance,
    model_name: str,
    console: Console | None = None,
) -> None:
    """
    Print feature importance table to console.

    Args:
        feature_importance: FeatureImportance object.
        model_name: Name of the model.
        console: Rich console for output.
    """
    if console is None:
        return

    # Sort by importance
    indices = np.argsort(feature_importance.importances)[::-1]

    # Normalize to percentages
    total = feature_importance.importances.sum()

    table = Table(
        title=f"{model_name} Feature Importance ({feature_importance.importance_type})"
    )
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Feature", style="cyan")
    table.add_column("Importance", style="green", justify="right")
    table.add_column("% Total", style="yellow", justify="right")

    for rank, idx in enumerate(indices, 1):
        imp = feature_importance.importances[idx]
        pct = (imp / total * 100) if total > 0 else 0
        table.add_row(
            str(rank),
            feature_importance.feature_names[idx],
            f"{imp:.4f}",
            f"{pct:.1f}%",
        )

    console.print(table)


def generate_descriptive_scatterplot(
    dtv_actual: np.ndarray,
    stadtradeln_volume: np.ndarray,
) -> str:
    """
    Generate scatterplot of DTV_DZS ~ VM_SR (descriptive data).

    Returns base64 encoded PNG image.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        stadtradeln_volume,
        dtv_actual,
        alpha=0.5,
        s=30,
        c="steelblue",
        edgecolors="none",
    )

    # Add regression line
    if len(stadtradeln_volume) > 1:
        z = np.polyfit(stadtradeln_volume, dtv_actual, 1)
        p = np.poly1d(z)
        x_line = np.linspace(stadtradeln_volume.min(), stadtradeln_volume.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit")

        # Calculate R² for the linear fit
        correlation = np.corrcoef(stadtradeln_volume, dtv_actual)[0, 1]
        r2 = correlation**2
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_xlabel("STADTRADELN Volume (VM_SR)", fontsize=11)
    ax.set_ylabel("Counter DTV (DTV_DZS)", fontsize=11)
    ax.set_title("Descriptive: Counter DTV vs STADTRADELN Volume", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    result = _fig_to_base64(fig)
    plt.close(fig)
    return result


def generate_prediction_scatterplots(
    model_results: list[ModelResult],
) -> dict[str, str]:
    """
    Generate scatterplots of DTV_DZS ~ DTV_predicted for each model.

    Returns dict of model_name -> base64 encoded PNG image.
    """
    plots = {}

    for result in model_results:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(
            result.y_pred,
            result.y_true,
            alpha=0.5,
            s=30,
            c="steelblue",
            edgecolors="none",
        )

        # Add identity line (y=x)
        min_val = min(result.y_true.min(), result.y_pred.min())
        max_val = max(result.y_true.max(), result.y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Identity (y=x)",
        )

        # Add metrics annotation
        r2 = result.metrics.get("r2_cv", result.metrics.get("r2", 0.0))
        rmse = result.metrics.get("rmse", 0.0)
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.4f}\nRMSE = {rmse:.1f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("Predicted DTV", fontsize=11)
        ax.set_ylabel("Actual DTV (DTV_DZS)", fontsize=11)
        ax.set_title(f"{result.name}: Actual vs Predicted DTV", fontsize=12)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plots[result.name] = _fig_to_base64(fig)
        plt.close(fig)

    return plots


def generate_html_report(
    report_data: ReportData,
    output_path: Path,
    console: Console | None = None,
) -> Path:
    """
    Generate complete HTML training report.

    Args:
        report_data: Report data containing all results.
        output_path: Path to save the HTML report.
        console: Optional console for printing tables.

    Returns:
        Path to the generated report.
    """
    log.info("Generating training report", output=str(output_path))

    # Generate all components
    _metrics_df, metrics_html = generate_metrics_table(
        report_data.model_results, console
    )
    _range_df, range_html = generate_dtv_range_table(report_data.model_results, console)

    descriptive_plot = generate_descriptive_scatterplot(
        report_data.dtv_actual, report_data.stadtradeln_volume
    )
    prediction_plots = generate_prediction_scatterplots(report_data.model_results)

    # Generate feature importance plots and print to console
    feature_importance_plots: dict[str, str] = {}
    for result in report_data.model_results:
        if result.feature_importance is not None:
            # Print to console for debug output
            print_feature_importance_table(
                result.feature_importance, result.name, console
            )

            # Generate plot for report
            feature_importance_plots[result.name] = generate_feature_importance_plot(
                result.feature_importance, result.name
            )

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Report - Year {report_data.year}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4a90a4;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #4a90a4;
            margin-top: 30px;
        }}
        .section {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-radius: 4px;
        }}
        .metadata-item strong {{
            display: block;
            color: #666;
            font-size: 0.85em;
            margin-bottom: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4a90a4;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .model-plots {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        .model-plot {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .model-plot img {{
            max-width: 100%;
            height: auto;
        }}
        .model-plot h3 {{
            margin-top: 0;
            color: #333;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <h1>🚲 Bicycle Traffic Model Training Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="section">
        <h2>Dataset Overview</h2>
        <div class="metadata">
            <div class="metadata-item">
                <strong>Year</strong>
                {report_data.year}
            </div>
            <div class="metadata-item">
                <strong>Total Samples</strong>
                {report_data.n_samples}
            </div>
            <div class="metadata-item">
                <strong>Training Samples</strong>
                {report_data.n_train}
            </div>
            <div class="metadata-item">
                <strong>Test Samples</strong>
                {report_data.n_test}
            </div>
            <div class="metadata-item">
                <strong>Features</strong>
                {len(report_data.feature_names)}
            </div>
            <div class="metadata-item">
                <strong>Models Trained</strong>
                {len(report_data.model_results)}
            </div>
        </div>
        <p><strong>Features used:</strong> {", ".join(report_data.feature_names)}</p>
    </div>

    <div class="section">
        <h2>Descriptive Analysis</h2>
        <p>Scatterplot showing the relationship between STADTRADELN volumes (VM_SR) and actual counter DTV values (DTV_DZS).</p>
        <div class="plot-container">
            <img src="data:image/png;base64,{descriptive_plot}" alt="DTV vs STADTRADELN Volume">
        </div>
    </div>

    <div class="section">
        <h2>Model Metrics Comparison</h2>
        <p>Performance metrics for all trained models. R² is from 10-fold cross-validation.</p>
        {metrics_html}
        <p><em>NMAPE = Negative Mean Absolute Percentage Error, NMAE = Negative Mean Absolute Error, SQV = skalierbarer Qualitätswert</em></p>
    </div>

    <div class="section">
        <h2>DTV Range of Estimates</h2>
        <p>Minimum and maximum predicted DTV values for each model.</p>
        {range_html}
    </div>

    <div class="section">
        <h2>Model Predictions</h2>
        <p>Scatterplots showing actual DTV (DTV_DZS) vs predicted DTV for each model. Points on the red dashed line indicate perfect predictions.</p>
        <div class="model-plots">
"""

    # Add prediction plots for each model
    for model_name, plot_b64 in prediction_plots.items():
        html_content += f"""
            <div class="model-plot">
                <h3>{model_name}</h3>
                <img src="data:image/png;base64,{plot_b64}" alt="{model_name} predictions">
            </div>
"""

    html_content += """
        </div>
    </div>
"""

    # Add feature importance section if we have any
    if feature_importance_plots:
        html_content += """
    <div class="section">
        <h2>Feature Importance</h2>
        <p>Feature importance shows which variables contribute most to predictions.
        Tree-based models use Gini importance (mean decrease in impurity),
        while linear models show coefficient magnitudes.</p>
        <div class="model-plots">
"""
        for model_name, plot_b64 in feature_importance_plots.items():
            html_content += f"""
            <div class="model-plot">
                <h3>{model_name}</h3>
                <img src="data:image/png;base64,{plot_b64}" alt="{model_name} feature importance">
            </div>
"""
        html_content += """
        </div>
    </div>
"""

    html_content += """
    <div class="section">
        <h2>Notes</h2>
        <ul>
            <li><strong>R² (CV)</strong>: Coefficient of determination from 10-fold cross-validation (higher is better)</li>
            <li><strong>R² w/o CV</strong>: R² on full training set (higher than CV indicates overfitting)</li>
            <li><strong>NMAPE</strong>: Negative Mean Absolute Percentage Error (higher/closer to 0 is better)</li>
            <li><strong>NMAE</strong>: Negative Mean Absolute Error in DTV units (higher/closer to 0 is better)</li>
            <li><strong>Max Error</strong>: Maximum absolute prediction error (lower is better)</li>
            <li><strong>SQV</strong>: Scalable Quality Value (Friedrich et al., 2019) - &ge;0.90 very good, &ge;0.85 good, &ge;0.80 acceptable</li>
            <li><strong>Feature Importance (Gini)</strong>: Mean decrease in impurity when splitting on this feature (tree-based models)</li>
            <li><strong>Feature Importance (Coefficient)</strong>: Absolute value of regression coefficient (linear models)</li>
        </ul>
    </div>
</body>
</html>
"""

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    log.info("Report generated", path=str(output_path))
    return output_path
