"""
Traffic volume data ingestion.

Loads STADTRADELN GPS-derived traffic volumes from FlatGeoBuf files.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyogrio

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import GeoDataLoader
from hochrechnung.schemas.traffic import TrafficVolumeRawSchema, TrafficVolumeSchema
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)


class TrafficVolumeLoader(GeoDataLoader[TrafficVolumeRawSchema]):
    """
    Loader for traffic volume data.

    Loads from FlatGeoBuf format. Files are region-specific (no bbox filtering needed).
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        chunk_size: int | None = None,
    ) -> None:
        """
        Initialize traffic volume loader.

        Args:
            config: Pipeline configuration.
            chunk_size: Optional chunk size for memory-efficient loading.
        """
        super().__init__(config, TrafficVolumeRawSchema)
        self.chunk_size = chunk_size

    def _load_raw_geo(self) -> pd.DataFrame:
        """
        Load traffic volumes from FlatGeoBuf.

        Note: Files are pre-filtered by region (e.g., SR23_Hessen_VM.fgb only contains
        Hessen data). No bbox filtering needed during load.
        """
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.traffic_volumes)

        if not path.exists():
            msg = f"Traffic volumes file not found: {path}"
            raise FileNotFoundError(msg)

        log.info("Loading traffic volumes", path=str(path))

        # Load entire file (already region-specific)
        if self.chunk_size:
            gdf = self._load_chunked(path)
        else:
            gdf = gpd.read_file(path)

        log.info("Loaded traffic data", rows=len(gdf))

        # Validate expected columns
        expected_cols = ["base_id", "edge_id", "count", "bicycle_infrastructure"]
        missing = [col for col in expected_cols if col not in gdf.columns]
        if missing:
            log.warning("Missing expected columns", missing=missing)

        return gdf

    def _load_chunked(self, path: Path) -> "gpd.GeoDataFrame":
        """Load data in chunks for memory efficiency."""
        import geopandas as gpd

        chunks = []
        skip = 0

        while True:
            chunk = pyogrio.read_dataframe(
                path,
                skip_features=skip,
                max_features=self.chunk_size,
            )

            if chunk.empty:
                break

            chunks.append(chunk)
            skip += len(chunk)

            log.debug("Loaded chunk", rows=len(chunk), total=skip)

        if not chunks:
            return gpd.GeoDataFrame()

        return gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True))


def load_traffic_volumes(
    config: PipelineConfig,
    *,
    validate: bool = True,
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load traffic volumes.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.

    Returns:
        GeoDataFrame with traffic volumes.
    """
    loader = TrafficVolumeLoader(config)
    return loader.load(validate=validate)
