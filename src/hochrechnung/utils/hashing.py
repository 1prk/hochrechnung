"""
Deterministic hashing utilities.

Provides content-based hashing for DataFrames and configurations.
"""

import hashlib
from typing import Any

import pandas as pd

from hochrechnung.utils.logging import get_logger

log = get_logger(__name__)


def hash_dataframe(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    n_rows: int | None = None,
) -> str:
    """
    Compute deterministic hash of a DataFrame.

    Uses xxhash for speed when available, falls back to MD5.

    Args:
        df: DataFrame to hash.
        columns: Optional subset of columns to include.
        n_rows: Optional number of rows to sample (for large DataFrames).

    Returns:
        Hex digest string.
    """
    # Select columns
    if columns:
        df = df[columns]

    # Sample rows if specified
    if n_rows and len(df) > n_rows:
        # Use deterministic sampling
        df = df.sample(n=n_rows, random_state=42)

    # Convert to bytes
    try:
        # Try xxhash for speed
        import xxhash

        hasher = xxhash.xxh64()
        # Hash shape
        hasher.update(f"{df.shape}".encode())
        # Hash column names
        hasher.update(",".join(df.columns).encode())
        # Hash data
        for col in df.columns:
            hasher.update(df[col].to_numpy().tobytes())
        return hasher.hexdigest()
    except ImportError:
        pass

    # Fall back to MD5
    hasher = hashlib.md5()
    hasher.update(f"{df.shape}".encode())
    hasher.update(",".join(df.columns).encode())
    hasher.update(pd.util.hash_pandas_object(df).to_numpy().tobytes())
    return hasher.hexdigest()


def hash_config(config: Any) -> str:
    """
    Compute hash of a configuration object.

    Args:
        config: Configuration object.

    Returns:
        Hex digest string.
    """
    # Convert to string representation
    if hasattr(config, "model_dump"):
        # Pydantic v2
        config_str = str(config.model_dump())
    elif hasattr(config, "dict"):
        # Pydantic v1
        config_str = str(config.dict())
    else:
        config_str = str(config)

    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def hash_file(path: str | Any) -> str:
    """
    Compute hash based on file path and modification time.

    This is a fast hash that doesn't read file contents.

    Args:
        path: Path to file.

    Returns:
        Hex digest string.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "missing"

    stat = p.stat()
    content = f"{p.name}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def hash_file_content(path: str | Any, chunk_size: int = 8192) -> str:
    """
    Compute hash of file contents.

    Args:
        path: Path to file.
        chunk_size: Chunk size for reading.

    Returns:
        Hex digest string.
    """
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return "missing"

    try:
        import xxhash

        hasher = xxhash.xxh64()
    except ImportError:
        hasher = hashlib.md5()

    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()
