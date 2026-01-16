"""
Base classes and utilities for data ingestion.

Provides common functionality for all data loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import pandas as pd
import pandera as pa

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

T = TypeVar("T", bound=pa.DataFrameModel)


class DataLoader(ABC, Generic[T]):
    """
    Abstract base class for data loaders.

    All data loaders inherit from this class to ensure consistent
    schema validation at system boundaries.
    """

    def __init__(self, config: PipelineConfig, schema: type[T]) -> None:
        """
        Initialize data loader.

        Args:
            config: Pipeline configuration.
            schema: Pandera schema for validation.
        """
        self.config = config
        self.schema = schema
        self._cached_data: pd.DataFrame | None = None

    @abstractmethod
    def _load_raw(self) -> pd.DataFrame:
        """Load raw data from source. Implemented by subclasses."""
        ...

    def load(self, *, validate: bool = True) -> pd.DataFrame:
        """
        Load and optionally validate data.

        Args:
            validate: Whether to validate against schema.

        Returns:
            Loaded (and optionally validated) DataFrame.

        Raises:
            FileNotFoundError: If data file not found.
            pandera.errors.SchemaError: If validation fails.
        """
        log.info("Loading data", loader=self.__class__.__name__)

        df = self._load_raw()
        log.info("Loaded raw data", rows=len(df), columns=list(df.columns))

        if validate:
            df = self._validate(df)
            log.info("Schema validation passed")

        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate.

        Returns:
            Validated DataFrame.
        """
        return self.schema.validate(df)

    def resolve_path(self, relative_path: Path) -> Path:
        """
        Resolve a relative path against data root.

        Args:
            relative_path: Path relative to data root.

        Returns:
            Absolute path.
        """
        return self.config.data_paths.data_root / relative_path


class GeoDataLoader(DataLoader[T]):
    """Base class for geospatial data loaders."""

    @abstractmethod
    def _load_raw_geo(self) -> "pd.DataFrame":
        """Load raw geospatial data. Returns GeoDataFrame."""
        ...

    def _load_raw(self) -> pd.DataFrame:
        """Load raw data from geospatial source."""
        return self._load_raw_geo()
