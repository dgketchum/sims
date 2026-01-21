"""
Ancillary data access for SIMS algorithm.

This module provides access to ancillary datasets required for the SIMS
algorithm:
- Reference ET from GridMET
- Crop type from USDA CDL via Planetary Computer
"""

from sims.ancillary.gridmet import (
    get_reference_et,
    get_daily_reference_et,
    get_reference_et_for_scene,
)
from sims.ancillary.cdl import (
    get_crop_type,
    get_crop_class_image,
    resample_to_landsat,
)

__all__ = [
    # GridMET
    "get_reference_et",
    "get_daily_reference_et",
    "get_reference_et_for_scene",
    # CDL
    "get_crop_type",
    "get_crop_class_image",
    "resample_to_landsat",
]
