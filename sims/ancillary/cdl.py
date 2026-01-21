"""
USDA Cropland Data Layer (CDL) access.

Provides access to crop type data from the USDA CDL via Microsoft
Planetary Computer STAC API.

The CDL provides annual 30m crop type classification for the CONUS.
"""

from datetime import datetime
from typing import Optional, Union
import warnings

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry

try:
    import pystac_client
    import planetary_computer
    import stackstac
    HAS_STAC = True
except ImportError:
    HAS_STAC = False

# Planetary Computer STAC endpoint
PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CDL_COLLECTION = "usda-cdl"


def get_crop_type(
    geometry: BaseGeometry,
    year: int,
    resolution: int = 30,
) -> xr.DataArray:
    """
    Get crop type data from USDA CDL for a geometry and year.

    Parameters
    ----------
    geometry : BaseGeometry
        Area of interest.
    year : int
        Year of CDL data (2008-present).
    resolution : int
        Output resolution in meters (default: 30).

    Returns
    -------
    xr.DataArray
        Crop type codes matching CDL classification.
        See sims.data.CDL for code definitions.

    Examples
    --------
    >>> from shapely.geometry import box
    >>> aoi = box(-120.0, 36.0, -119.5, 36.5)
    >>> crop_type = get_crop_type(aoi, 2023)
    """
    if not HAS_STAC:
        raise ImportError(
            "STAC dependencies required. Install with: "
            "pip install pystac-client planetary-computer stackstac"
        )

    # Open STAC catalog
    catalog = pystac_client.Client.open(PC_STAC_URL)

    # Search for CDL item
    search = catalog.search(
        collections=[CDL_COLLECTION],
        intersects=geometry.__geo_interface__,
        datetime=f"{year}-01-01/{year}-12-31",
    )

    items = list(search.items())
    if not items:
        raise ValueError(f"No CDL data found for year {year}")

    # Sign URLs for Planetary Computer
    item = planetary_computer.sign(items[0])

    # Load crop layer
    bounds = geometry.bounds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        da = stackstac.stack(
            [item],
            assets=['cropland'],
            bounds=bounds,
            resolution=resolution,
            resampling=0,  # Nearest neighbor for categorical data
        )

    # Squeeze dimensions
    da = da.squeeze(['time', 'band'], drop=True)

    return da


def get_crop_class_image(
    geometry: BaseGeometry,
    year: int,
    resolution: int = 30,
) -> xr.DataArray:
    """
    Get crop class (not type) image for SIMS algorithm.

    Converts CDL codes to SIMS crop classes (0-7).

    Parameters
    ----------
    geometry : BaseGeometry
        Area of interest.
    year : int
        Year of CDL data.
    resolution : int
        Output resolution in meters.

    Returns
    -------
    xr.DataArray
        Crop class codes:
        - 0: Non-agricultural
        - 1: Row crops
        - 2: Vines
        - 3: Trees
        - 5: Rice
        - 6: Fallow
        - 7: Grass/pasture
    """
    from sims.data import CDL as CDL_LOOKUP

    # Get raw CDL
    cdl = get_crop_type(geometry, year, resolution)

    # Create crop class array
    crop_class = xr.zeros_like(cdl)

    # Map CDL codes to crop classes
    for cdl_code, params in CDL_LOOKUP.items():
        mask = cdl == cdl_code
        crop_class = xr.where(mask, params.get('crop_class', 0), crop_class)

    return crop_class


def resample_to_landsat(
    cdl: xr.DataArray,
    landsat_template: xr.DataArray,
) -> xr.DataArray:
    """
    Resample CDL to match Landsat image grid.

    Parameters
    ----------
    cdl : xr.DataArray
        CDL data (30m resolution).
    landsat_template : xr.DataArray
        Landsat band to match (for grid alignment).

    Returns
    -------
    xr.DataArray
        CDL resampled to Landsat grid.
    """
    # Use rioxarray for reprojection
    if hasattr(cdl, 'rio') and hasattr(landsat_template, 'rio'):
        return cdl.rio.reproject_match(
            landsat_template,
            resampling=0,  # Nearest neighbor
        )
    else:
        raise ValueError("rioxarray required for reprojection")
