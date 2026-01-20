"""
Traffic volume data ingestion.

Loads STADTRADELN GPS-derived traffic volumes from FlatGeoBuf files.

Performance optimizations:
- Chunked loading for memory efficiency with large files
- Column selection to reduce memory footprint
- Lazy loading support via pyogrio
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyogrio

from hochrechnung.config.settings import PipelineConfig
from hochrechnung.ingestion.base import GeoDataLoader
from hochrechnung.schemas.traffic import TrafficVolumeRawSchema
from hochrechnung.utils.logging import get_logger

if TYPE_CHECKING:
    import geopandas as gpd

log = get_logger(__name__)

# Default chunk size for large file loading (rows per chunk)
DEFAULT_CHUNK_SIZE = 50_000

# =============================================================================
# Column name sets - supports both German and English naming
# =============================================================================
# German (STADTRADELN delivery) or English (internal) column names.
# The loader auto-detects which format is used.

# German column names (STADTRADELN delivery format)
COLUMNS_GERMAN = [
    "KantenId",  # edge_id
    "GrundlagenId",  # base_id
    "Verkehrsmenge",  # count
    "VerkehrsmengeIn",  # count_forward
    "VerkehrsmengeGegen",  # count_backward
    "GeschwIn",  # speed_forward_kmh
    "GeschwGegen",  # speed_backward_kmh
    "geometry",
]

# English column names (short internal names)
COLUMNS_ENGLISH = [
    "edge_id",
    "base_id",
    "count",
    "count_forward",
    "count_backward",
    "speed_forward_kmh",
    "speed_backward_kmh",
    "bicycle_infrastructure",
    "geometry",
]

# Required columns for ETL pipeline (either naming convention)
REQUIRED_COLUMNS_GERMAN = ["KantenId", "GrundlagenId", "Verkehrsmenge"]
REQUIRED_COLUMNS_ENGLISH = ["edge_id", "base_id", "count"]

# German -> English column mapping for automatic renaming
GERMAN_TO_ENGLISH: dict[str, str] = {
    "KantenId": "edge_id",
    "GrundlagenId": "base_id",
    "Verkehrsmenge": "count",
    "VerkehrsmengeIn": "count_forward",
    "VerkehrsmengeGegen": "count_backward",
    "GeschwIn": "speed_forward_kmh",
    "GeschwGegen": "speed_backward_kmh",
}

# Legacy column names (for backwards compatibility)
REQUIRED_COLUMNS = COLUMNS_ENGLISH  # Default to English


class TrafficVolumeLoader(GeoDataLoader[TrafficVolumeRawSchema]):
    """
    Loader for traffic volume data.

    Loads from FlatGeoBuf format. Files are region-specific (no bbox filtering needed).

    Performance features:
    - Chunked loading enabled by default for large files (>100k rows)
    - Column selection to load only required columns
    - Memory-efficient concatenation
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
        columns: list[str] | None = None,
    ) -> None:
        """
        Initialize traffic volume loader.

        Args:
            config: Pipeline configuration.
            chunk_size: Chunk size for memory-efficient loading. Set to None
                to load entire file at once. Default is 50,000 rows.
            columns: Columns to load. Default loads only required columns.
        """
        super().__init__(config, TrafficVolumeRawSchema)
        self.chunk_size = chunk_size
        self.columns = columns

    def _load_raw_geo(self) -> pd.DataFrame:
        """
        Load traffic volumes from FlatGeoBuf.

        Supports both German (STADTRADELN delivery) and English column names.
        Auto-detects which format is used based on file contents.

        Note: Files are pre-filtered by region (e.g., SR23_Hessen_VM.fgb only contains
        Hessen data). No bbox filtering needed during load.

        Performance optimizations:
        - Uses pyogrio for fast reads
        - Column selection to reduce memory
        - Chunked loading for large files
        """
        import geopandas as gpd

        path = self.resolve_path(self.config.data_paths.traffic_volumes)

        if not path.exists():
            msg = f"Traffic volumes file not found: {path}"
            raise FileNotFoundError(msg)

        # Detect column format and determine columns to load
        columns_to_load = self.columns
        column_format = "unknown"

        if columns_to_load is None:
            try:
                file_info = pyogrio.read_info(path)
                available_cols = file_info.get("fields", [])

                # Detect format: German or English
                # German: check for KantenId and Verkehrsmenge
                if (
                    "KantenId" in available_cols and "Verkehrsmenge" in available_cols
                ):
                    column_format = "german"
                    columns_to_load = [
                        c
                        for c in COLUMNS_GERMAN
                        if c in available_cols or c == "geometry"
                    ]
                elif "edge_id" in available_cols and "count" in available_cols:
                    column_format = "english"
                    columns_to_load = [
                        c
                        for c in COLUMNS_ENGLISH
                        if c in available_cols or c == "geometry"
                    ]
                else:
                    # Fall back to loading all columns
                    columns_to_load = None
                    log.warning(
                        "Could not detect column format",
                        available=available_cols[:10],
                    )
            except Exception:
                columns_to_load = None

        log.info(
            "Loading traffic volumes",
            path=str(path),
            format=column_format,
            chunked=self.chunk_size is not None,
            columns=columns_to_load,
        )

        # Load with chunking for large files or all at once
        if self.chunk_size is not None:
            gdf = self._load_chunked(path, columns=columns_to_load)
        else:
            # Load entire file at once (use for smaller files)
            read_kwargs: dict = {}
            if columns_to_load:
                # pyogrio uses 'columns' parameter (without geometry)
                non_geom_cols = [c for c in columns_to_load if c != "geometry"]
                if non_geom_cols:
                    read_kwargs["columns"] = non_geom_cols
            gdf = gpd.read_file(path, engine="pyogrio", **read_kwargs)

        log.info("Loaded traffic data", rows=len(gdf), columns=list(gdf.columns))

        # Rename German columns to English for internal processing
        if column_format == "german":
            rename_map = {k: v for k, v in GERMAN_TO_ENGLISH.items() if k in gdf.columns}
            if rename_map:
                gdf = gdf.rename(columns=rename_map)
                log.info(
                    "Renamed German columns to English",
                    renamed=list(rename_map.keys()),
                )
                # Update format after renaming
                column_format = "english"

        # Validate required columns based on detected format
        if column_format == "german":
            missing = [c for c in REQUIRED_COLUMNS_GERMAN if c not in gdf.columns]
        elif column_format == "english":
            missing = [c for c in REQUIRED_COLUMNS_ENGLISH if c not in gdf.columns]
        else:
            # Check for either format
            has_german = all(c in gdf.columns for c in REQUIRED_COLUMNS_GERMAN)
            has_english = all(c in gdf.columns for c in REQUIRED_COLUMNS_ENGLISH)
            if not has_german and not has_english:
                missing = REQUIRED_COLUMNS_GERMAN  # Report German as expected
            else:
                missing = []

        if missing:
            log.warning(
                "Missing required columns", missing=missing, format=column_format
            )

        return gdf

    def _load_chunked(
        self, path: Path, columns: list[str] | None = None
    ) -> "gpd.GeoDataFrame":
        """
        Load data in chunks for memory efficiency.

        Args:
            path: Path to the data file.
            columns: Columns to load (excluding geometry which is always loaded).

        Returns:
            GeoDataFrame with all chunks concatenated.
        """
        import geopandas as gpd

        chunks: list[gpd.GeoDataFrame] = []
        skip = 0
        chunk_num = 0

        # Prepare column filter for pyogrio
        read_columns = None
        if columns:
            read_columns = [c for c in columns if c != "geometry"]

        while True:
            read_kwargs: dict = {
                "skip_features": skip,
                "max_features": self.chunk_size,
            }
            if read_columns:
                read_kwargs["columns"] = read_columns

            chunk = pyogrio.read_dataframe(path, **read_kwargs)

            if chunk is None or len(chunk) == 0:
                break

            chunks.append(chunk)
            skip += len(chunk)
            chunk_num += 1

            if chunk_num % 10 == 0:
                log.debug("Loading progress", chunks=chunk_num, rows_loaded=skip)

        if not chunks:
            return gpd.GeoDataFrame()

        log.debug("Concatenating chunks", n_chunks=len(chunks), total_rows=skip)

        # Efficient concatenation - avoid intermediate copies
        result = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True, copy=False))

        # Explicitly set CRS from first chunk if available
        if chunks and chunks[0].crs is not None:
            result = result.set_crs(chunks[0].crs)

        return result


def load_traffic_volumes(
    config: PipelineConfig,
    *,
    validate: bool = True,
    chunk_size: int | None = DEFAULT_CHUNK_SIZE,
    columns: list[str] | None = None,
) -> "gpd.GeoDataFrame":
    """
    Convenience function to load traffic volumes.

    Args:
        config: Pipeline configuration.
        validate: Whether to validate against schema.
        chunk_size: Chunk size for memory-efficient loading.
            Default is 50,000 rows. Set to None to load all at once.
        columns: Columns to load. Default loads only required columns.

    Returns:
        GeoDataFrame with traffic volumes.
    """
    loader = TrafficVolumeLoader(config, chunk_size=chunk_size, columns=columns)
    return loader.load(validate=validate)
