"""
Inference pipeline for generating predictions.

Handles prediction generation with validated output schema.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Container for prediction results.

    Attributes:
        predictions: DataFrame with predictions.
        model_name: Name of model used.
        feature_names: Features used for prediction.
        n_predictions: Number of predictions made.
        n_failures: Number of failed predictions.
    """

    predictions: pd.DataFrame
    model_name: str
    feature_names: list[str]
    n_predictions: int
    n_failures: int


def predict_traffic_volumes(
    model: Any,
    features: pd.DataFrame,
    original_data: pd.DataFrame | None = None,
    *,
    batch_size: int | None = None,
) -> PredictionResult:
    """
    Generate predictions for traffic volumes.

    Args:
        model: Fitted model (sklearn pipeline).
        features: Feature DataFrame.
        original_data: Original data to merge with predictions.
        batch_size: Optional batch size for large datasets.

    Returns:
        PredictionResult with predictions.
    """
    log.info("Generating predictions", n_samples=len(features))

    # Handle missing values
    n_missing = features.isna().any(axis=1).sum()
    if n_missing > 0:
        log.warning("Features contain missing values", n_missing=int(n_missing))

    # Generate predictions
    if batch_size and len(features) > batch_size:
        predictions = _predict_in_batches(model, features, batch_size)
    else:
        try:
            predictions = model.predict(features)
        except Exception as e:
            log.error("Prediction failed", error=str(e))
            predictions = np.full(len(features), np.nan)

    # Build result DataFrame
    result_df = pd.DataFrame({"predicted_dtv": predictions})

    # Add original data if provided
    if original_data is not None:
        # Align indices
        result_df.index = features.index
        for col in original_data.columns:
            if col in features.index.names or col == features.index.name:
                continue
            result_df[col] = original_data[col].values

    n_failures = int(np.isnan(predictions).sum())

    result = PredictionResult(
        predictions=result_df,
        model_name=_get_model_name(model),
        feature_names=list(features.columns),
        n_predictions=len(predictions) - n_failures,
        n_failures=n_failures,
    )

    log.info(
        "Predictions complete",
        n_predictions=result.n_predictions,
        n_failures=result.n_failures,
    )

    return result


def _predict_in_batches(
    model: Any,
    features: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """Generate predictions in batches for memory efficiency."""
    predictions = []

    for i in range(0, len(features), batch_size):
        batch = features.iloc[i : i + batch_size]
        try:
            batch_pred = model.predict(batch)
            predictions.append(batch_pred)
        except Exception as e:
            log.warning(f"Batch {i} failed: {e}")
            predictions.append(np.full(len(batch), np.nan))

        log.debug(f"Processed batch {i // batch_size + 1}")

    return np.concatenate(predictions)


def _get_model_name(model: Any) -> str:
    """Extract model name from pipeline."""
    if hasattr(model, "regressor"):
        # TransformedTargetRegressor
        inner = model.regressor
        if hasattr(inner, "named_steps"):
            return type(inner.named_steps.get("model", inner)).__name__
        return type(inner).__name__
    if hasattr(model, "named_steps"):
        return type(model.named_steps.get("model", model)).__name__
    return type(model).__name__


def save_predictions(
    result: PredictionResult,
    output_path: Path,
    format: str = "csv",
) -> Path:
    """
    Save predictions to file.

    Args:
        result: Prediction result.
        output_path: Output file path.
        format: Output format ('csv', 'parquet', 'fgb').

    Returns:
        Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.predictions

    if format == "csv":
        output_file = output_path.with_suffix(".csv")
        df.to_csv(output_file, index=False)
    elif format == "parquet":
        output_file = output_path.with_suffix(".parquet")
        df.to_parquet(output_file)
    elif format == "fgb":
        import geopandas as gpd

        output_file = output_path.with_suffix(".fgb")
        if "geometry" in df.columns or hasattr(df, "geometry"):
            gdf = gpd.GeoDataFrame(df)
            gdf.to_file(output_file, driver="FlatGeobuf")
        else:
            # Fall back to CSV if no geometry
            output_file = output_path.with_suffix(".csv")
            df.to_csv(output_file, index=False)
    else:
        output_file = output_path.with_suffix(".csv")
        df.to_csv(output_file, index=False)

    log.info("Saved predictions", path=str(output_file), n_rows=len(df))
    return output_file


def load_model(model_path: Path | str) -> Any:
    """
    Load a saved model.

    Args:
        model_path: Path to model file or MLflow URI.

    Returns:
        Loaded model.
    """
    model_path_str = str(model_path)

    # Check if it's an MLflow URI
    if model_path_str.startswith("runs:/") or model_path_str.startswith("models:/"):
        import mlflow

        return mlflow.sklearn.load_model(model_path_str)

    # Load from file
    import joblib

    return joblib.load(model_path)
