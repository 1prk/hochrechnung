"""
ETL pipeline for bicycle traffic data.

Orchestrates data ingestion, transformation, and training data preparation.
"""

from hochrechnung.etl.pipeline import ETLPipeline, run_etl

__all__ = ["ETLPipeline", "run_etl"]
