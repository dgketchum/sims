"""
SIMS - Satellite Irrigation Management Support

A standalone Python package for computing ET fractions using the SIMS algorithm.

The SIMS algorithm computes crop coefficients (Kc) from satellite-derived NDVI
and crop type information, following the methodology of Melton et al. (2012)
and the OpenET implementation.

References
----------
Melton, F. S., Johnson, L. F., Lund, C. P., Pierce, L. L., Michaelis, A. R.,
Hiatt, S. H., ... & Nemani, R. R. (2012). Satellite Irrigation Management
Support with the Terrestrial Observation and Prediction System: A Framework
for Integration of Satellite and Surface Observations to Support Improvements
in Agricultural Water Resource Management. IEEE Journal of Selected Topics in
Applied Earth Observations and Remote Sensing, 5(6), 1709-1721.

Allen, R. G., & Pereira, L. S. (2009). Estimating crop coefficients from
fraction of ground cover and height. Irrigation Science, 28(1), 17-34.
"""

from sims.model import (
    et_fraction,
    compute_et,
    fraction_of_cover,
    kc_generic,
    kc_row_crop,
    kc_tree,
    kc_vine,
    kc_rice,
    kc_fallow,
    kc_grass_pasture,
)
from sims.image import LandsatImage
from sims.sources.base import DataSource
from sims.sources.local import LocalFileSource
from sims.sources.stac import STACSource
from sims.zonal import compute_zonal_stats
from sims.api import compute_etf_timeseries
from sims.run import process_scene, search_and_process

__version__ = "0.1.0"

__all__ = [
    # Algorithm functions
    "et_fraction",
    "compute_et",
    "fraction_of_cover",
    "kc_generic",
    "kc_row_crop",
    "kc_tree",
    "kc_vine",
    "kc_rice",
    "kc_fallow",
    "kc_grass_pasture",
    # Image class
    "LandsatImage",
    # Data sources
    "DataSource",
    "LocalFileSource",
    "STACSource",
    # Utilities
    "compute_zonal_stats",
    "compute_etf_timeseries",
    # Runners
    "process_scene",
    "search_and_process",
]
