"""
Pandera schemas for model output data.
"""

from typing import Optional

import pandera as pa
from pandera.typing import Series


class PredictionOutputSchema(pa.DataFrameModel):
    """
    Schema for model prediction output.

    Validates the structure of prediction results.
    """

    edge_id: Optional[Series[int]] = pa.Field(
        nullable=True,
        description="Edge identifier (if available)",
    )
    predicted_dtv: Series[float] = pa.Field(
        ge=0,
        description="Predicted daily traffic volume",
    )
    prediction_lower: Optional[Series[float]] = pa.Field(
        nullable=True,
        description="Lower bound of prediction interval",
    )
    prediction_upper: Optional[Series[float]] = pa.Field(
        nullable=True,
        description="Upper bound of prediction interval",
    )

    class Config:
        """Schema configuration."""

        name = "PredictionOutputSchema"
        strict = False  # Allow extra columns
        coerce = True


class TrainingOutputSchema(pa.DataFrameModel):
    """
    Schema for training data output (features + target).

    Validates data ready for model training.
    """

    # Required features
    stadtradeln_volume: Series[float] = pa.Field(
        ge=0,
        description="STADTRADELN traffic volume",
    )
    population: Series[int] = pa.Field(
        ge=0,
        description="Municipality population",
    )

    # Target variable
    dtv: Series[float] = pa.Field(
        ge=0,
        description="Daily traffic volume (target)",
    )

    class Config:
        """Schema configuration."""

        name = "TrainingOutputSchema"
        strict = False
        coerce = True
