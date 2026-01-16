"""
Caching utilities for large data files.

Provides content-addressable caching with automatic invalidation.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CacheManager:
    """
    Content-addressable cache for expensive computations.

    Cache keys are based on:
    - Function/operation name
    - Input file modification times
    - Configuration hash
    """

    def __init__(self, cache_dir: Path) -> None:
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self,
        key: str,
        version: str = "v1",
        extension: str = ".pkl",
    ) -> Path:
        """
        Get path for a cache file.

        Args:
            key: Cache key.
            version: Cache version.
            extension: File extension.

        Returns:
            Path to cache file.
        """
        # Hash the key for filesystem-safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{key_hash}_{version}{extension}"

    def compute_file_hash(self, path: Path) -> str:
        """
        Compute hash based on file modification time and size.

        Args:
            path: Path to file.

        Returns:
            Hash string.
        """
        if not path.exists():
            return "missing"

        stat = path.stat()
        content = f"{path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def compute_config_hash(self, config: Any) -> str:
        """
        Compute hash from configuration object.

        Args:
            config: Configuration object (must be serializable).

        Returns:
            Hash string.
        """
        try:
            config_str = str(config)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception:
            return "unknown"

    def get(
        self,
        key: str,
        source_files: list[Path] | None = None,
        config_hash: str | None = None,
    ) -> T | None:
        """
        Get cached value if valid.

        Args:
            key: Cache key.
            source_files: Source files to check for changes.
            config_hash: Configuration hash.

        Returns:
            Cached value or None if cache invalid/missing.
        """
        # Build cache key with dependencies
        full_key = key
        if source_files:
            file_hashes = [self.compute_file_hash(f) for f in source_files]
            full_key += "_" + "_".join(file_hashes)
        if config_hash:
            full_key += "_" + config_hash

        cache_path = self.get_cache_path(full_key)

        if not cache_path.exists():
            log.debug("Cache miss (not found)", key=key)
            return None

        # Check if cache is newer than source files
        if source_files:
            cache_mtime = cache_path.stat().st_mtime
            for source in source_files:
                if source.exists() and source.stat().st_mtime > cache_mtime:
                    log.debug("Cache miss (stale)", key=key, stale_source=str(source))
                    return None

        try:
            with cache_path.open("rb") as f:
                value = pickle.load(f)
            log.debug("Cache hit", key=key)
            return value
        except Exception as e:
            log.warning("Cache read error", key=key, error=str(e))
            return None

    def set(
        self,
        key: str,
        value: T,
        source_files: list[Path] | None = None,
        config_hash: str | None = None,
    ) -> Path:
        """
        Store value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            source_files: Source files (for key generation).
            config_hash: Configuration hash.

        Returns:
            Path to cache file.
        """
        full_key = key
        if source_files:
            file_hashes = [self.compute_file_hash(f) for f in source_files]
            full_key += "_" + "_".join(file_hashes)
        if config_hash:
            full_key += "_" + config_hash

        cache_path = self.get_cache_path(full_key)

        try:
            with cache_path.open("wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.debug("Cached value", key=key, path=str(cache_path))
        except Exception as e:
            log.warning("Cache write error", key=key, error=str(e))

        return cache_path

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if cache was invalidated.
        """
        cache_path = self.get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            log.debug("Cache invalidated", key=key)
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache files.

        Returns:
            Number of files removed.
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1

        log.info("Cache cleared", files_removed=count)
        return count


def cache_dataframe(
    df: pd.DataFrame,
    path: Path,
    format: str = "parquet",
) -> Path:
    """
    Cache a DataFrame to disk.

    Args:
        df: DataFrame to cache.
        path: Output path (without extension).
        format: Output format ('parquet', 'feather', 'pickle').

    Returns:
        Path to cached file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        output = path.with_suffix(".parquet")
        df.to_parquet(output)
    elif format == "feather":
        output = path.with_suffix(".feather")
        df.to_feather(output)
    else:
        output = path.with_suffix(".pkl")
        df.to_pickle(output)

    log.debug("Cached DataFrame", path=str(output), rows=len(df))
    return output


def load_cached_dataframe(
    path: Path,
    format: str | None = None,
) -> pd.DataFrame | None:
    """
    Load a cached DataFrame.

    Args:
        path: Path to cached file.
        format: File format (inferred from extension if None).

    Returns:
        DataFrame or None if not found.
    """
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if format is None:
        if suffix == ".parquet":
            format = "parquet"
        elif suffix == ".feather":
            format = "feather"
        else:
            format = "pickle"

    try:
        if format == "parquet":
            return pd.read_parquet(path)
        if format == "feather":
            return pd.read_feather(path)
        return pd.read_pickle(path)
    except Exception as e:
        log.warning("Error loading cached DataFrame", path=str(path), error=str(e))
        return None
