"""
GridMET reference ET data access.

Provides access to reference evapotranspiration (ETo/ETr) from GridMET
via THREDDS OPeNDAP or direct download.

GridMET provides daily 4km resolution meteorological data for the CONUS.
Reference ET is computed using the ASCE Penman-Monteith equation.

References
----------
Abatzoglou, J. T. (2013). Development of gridded surface meteorological data
for ecological applications and modelling. International Journal of
Climatology, 33(1), 121-131.
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Tuple
import warnings

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry

# GridMET THREDDS OPeNDAP base URL
GRIDMET_THREDDS_URL = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET"

# Variable names in GridMET
GRIDMET_VARS = {
    'etr': 'etr',          # ASCE grass reference ET (mm)
    'eto': 'eto',          # ASCE alfalfa reference ET (mm) - Note: GridMET uses 'etr' for both
    'pet': 'pet',          # Potential ET (mm)
    'tmmx': 'tmmx',        # Maximum temperature (K)
    'tmmn': 'tmmn',        # Minimum temperature (K)
    'srad': 'srad',        # Solar radiation (W/m^2)
    'vs': 'vs',            # Wind speed (m/s)
    'sph': 'sph',          # Specific humidity (kg/kg)
    'pr': 'pr',            # Precipitation (mm)
}


def get_reference_et(
    geometry: BaseGeometry,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    variable: str = 'etr',
) -> xr.DataArray:
    """
    Get reference ET from GridMET for a geometry and date range.

    Parameters
    ----------
    geometry : BaseGeometry
        Area of interest (point or polygon).
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    variable : str
        GridMET variable name ('etr', 'eto', 'pet'). Default 'etr'.

    Returns
    -------
    xr.DataArray
        Reference ET values with time, lat, lon dimensions.
        Units: mm/day

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> pt = Point(-119.5, 36.5)  # Central Valley, CA
    >>> etr = get_reference_et(pt, '2023-07-01', '2023-07-31')
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # Get bounds
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)

    # Build URL for the year(s)
    years = range(start_date.year, end_date.year + 1)

    datasets = []
    for year in years:
        url = f"{GRIDMET_THREDDS_URL}/{variable}_{year}.nc"

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(url, engine='netcdf4')

            # Select spatial subset
            ds = ds.sel(
                lon=slice(bounds[0] - 0.1, bounds[2] + 0.1),
                lat=slice(bounds[3] + 0.1, bounds[1] - 0.1),  # lat is descending
            )

            # Select time range for this year
            year_start = max(start_date, datetime(year, 1, 1))
            year_end = min(end_date, datetime(year, 12, 31))
            ds = ds.sel(day=slice(year_start, year_end))

            datasets.append(ds[variable])

        except Exception as e:
            warnings.warn(f"Failed to load GridMET for {year}: {e}")
            continue

    if not datasets:
        raise ValueError("No GridMET data could be loaded")

    # Combine years
    da = xr.concat(datasets, dim='day')
    da = da.rename({'day': 'time'})

    return da


def get_daily_reference_et(
    lon: float,
    lat: float,
    date: Union[str, datetime],
    variable: str = 'etr',
) -> float:
    """
    Get reference ET for a single point and date.

    Parameters
    ----------
    lon : float
        Longitude (degrees, negative for west).
    lat : float
        Latitude (degrees).
    date : str or datetime
        Date to retrieve.
    variable : str
        GridMET variable ('etr', 'eto', 'pet').

    Returns
    -------
    float
        Reference ET in mm/day.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    url = f"{GRIDMET_THREDDS_URL}/{variable}_{date.year}.nc"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(url, engine='netcdf4')

    # Select nearest point and date
    da = ds[variable].sel(
        lon=lon,
        lat=lat,
        day=date,
        method='nearest'
    )

    return float(da.values)


def get_reference_et_for_scene(
    geometry: BaseGeometry,
    date: Union[str, datetime],
    variable: str = 'etr',
    buffer_days: int = 0,
) -> Union[float, xr.DataArray]:
    """
    Get reference ET for a Landsat scene date.

    Parameters
    ----------
    geometry : BaseGeometry
        Scene geometry or field of interest.
    date : str or datetime
        Scene acquisition date.
    variable : str
        GridMET variable ('etr', 'eto', 'pet').
    buffer_days : int
        If > 0, return average of date ± buffer_days.

    Returns
    -------
    float or xr.DataArray
        Reference ET value(s) in mm/day.
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')

    if buffer_days > 0:
        start = date - timedelta(days=buffer_days)
        end = date + timedelta(days=buffer_days)
        da = get_reference_et(geometry, start, end, variable)
        return float(da.mean().values)
    else:
        # Get single day
        bounds = geometry.bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        return get_daily_reference_et(center_lon, center_lat, date, variable)
