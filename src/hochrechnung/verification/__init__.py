"""
Counter verification system for OSM way assignment quality control.

Provides workflow for manually verifying and correcting counter-to-OSM-edge
assignments, with persistent verified datasets per campaign year.
"""

from hochrechnung.verification.outliers import (
    OutlierResult,
    calculate_ratios,
    flag_outliers,
)
from hochrechnung.verification.persistence import (
    VerifiedCounter,
    load_verified_counters,
    save_verified_counters,
)

__all__ = [
    "OutlierResult",
    "VerifiedCounter",
    "calculate_ratios",
    "flag_outliers",
    "load_verified_counters",
    "save_verified_counters",
]
