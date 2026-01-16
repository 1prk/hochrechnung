"""
Hochrechnung: Bicycle Traffic Estimation Pipeline.

This package provides ETL, feature engineering, and regression models
for estimating average daily bicycle traffic (DTV) per OSM edge.
"""

from importlib.metadata import version

__version__ = version("hochrechnung")

__all__ = ["__version__"]
