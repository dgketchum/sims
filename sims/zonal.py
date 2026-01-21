"""
Zonal statistics module for extracting field-level statistics.

This module provides functions for computing zonal statistics from
raster ET fraction outputs using field polygon geometries.
"""

from typing import Optional, List, Union
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry.base import BaseGeometry

try:
    from rasterio.features import geometry_mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def compute_zonal_stats(
    image,  # LandsatImage
    etf: xr.DataArray,
    geometries: gpd.GeoDataFrame,
    field_id_column: str = 'fid',
    stats: Optional[List[str]] = None,
    apply_mask: bool = True,
) -> pd.DataFrame:
    """
    Extract zonal statistics for each geometry.

    Parameters
    ----------
    image : LandsatImage
        The Landsat image object (for metadata and cloud mask).
    etf : xr.DataArray
        ET fraction raster data.
    geometries : gpd.GeoDataFrame
        GeoDataFrame of field polygons.
    field_id_column : str
        Column name containing unique field identifiers.
    stats : list of str, optional
        Statistics to compute. Default: ['mean', 'std', 'count'].
        Options: 'mean', 'std', 'min', 'max', 'median', 'count', 'sum'
    apply_mask : bool
        If True, apply cloud mask before computing statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - field_id: Field identifier
        - date: Acquisition date
        - scene_id: Scene identifier
        - etf_mean, etf_std, etc.: Requested statistics
        - valid_count: Number of valid pixels
        - total_count: Total pixels in geometry
        - valid_fraction: Fraction of valid pixels
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for zonal statistics")

    if stats is None:
        stats = ['mean', 'std', 'count']

    # Get raster properties
    transform = etf.rio.transform() if hasattr(etf, 'rio') else None
    crs = etf.rio.crs if hasattr(etf, 'rio') else None

    # Ensure geometries are in the same CRS as the raster
    if crs is not None and geometries.crs is not None:
        geometries = geometries.to_crs(crs)

    # Get cloud mask if applying
    mask = None
    if apply_mask:
        try:
            mask = image.qa_mask.values
        except Exception:
            mask = None

    # Apply cloud mask to ETf
    if mask is not None:
        etf_values = np.where(mask, etf.values, np.nan)
    else:
        etf_values = etf.values

    # Handle different array shapes (may have extra dimensions)
    if etf_values.ndim > 2:
        etf_values = etf_values.squeeze()

    # Get raster shape
    height, width = etf_values.shape

    results = []

    for idx, row in geometries.iterrows():
        field_id = row[field_id_column]
        geom = row.geometry

        # Create mask for this geometry
        try:
            geom_mask = geometry_mask(
                [geom],
                out_shape=(height, width),
                transform=transform,
                invert=True  # True = inside polygon
            )
        except Exception:
            # Geometry may not overlap with raster
            continue

        # Extract values within geometry
        values = etf_values[geom_mask]

        # Count total and valid pixels
        total_count = len(values)
        valid_values = values[~np.isnan(values)]
        valid_count = len(valid_values)

        if valid_count == 0:
            # No valid data in this geometry
            result = {
                'field_id': field_id,
                'date': image.date,
                'scene_id': image.scene_id,
                'valid_count': 0,
                'total_count': total_count,
                'valid_fraction': 0.0,
            }
            for stat in stats:
                result[f'etf_{stat}'] = np.nan
            results.append(result)
            continue

        # Compute requested statistics
        result = {
            'field_id': field_id,
            'date': image.date,
            'scene_id': image.scene_id,
            'valid_count': valid_count,
            'total_count': total_count,
            'valid_fraction': valid_count / total_count if total_count > 0 else 0.0,
        }

        for stat in stats:
            if stat == 'mean':
                result['etf_mean'] = np.nanmean(valid_values)
            elif stat == 'std':
                result['etf_std'] = np.nanstd(valid_values)
            elif stat == 'min':
                result['etf_min'] = np.nanmin(valid_values)
            elif stat == 'max':
                result['etf_max'] = np.nanmax(valid_values)
            elif stat == 'median':
                result['etf_median'] = np.nanmedian(valid_values)
            elif stat == 'count':
                result['etf_count'] = valid_count
            elif stat == 'sum':
                result['etf_sum'] = np.nansum(valid_values)

        results.append(result)

    return pd.DataFrame(results)


def compute_zonal_stats_simple(
    data: Union[np.ndarray, xr.DataArray],
    geometries: gpd.GeoDataFrame,
    transform,
    field_id_column: str = 'fid',
    date: Optional[datetime] = None,
    scene_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Simplified zonal statistics without LandsatImage dependency.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        2D raster data (e.g., ET fraction).
    geometries : gpd.GeoDataFrame
        Field polygons.
    transform : affine.Affine
        Raster transform (from rasterio or rioxarray).
    field_id_column : str
        Field ID column name.
    date : datetime, optional
        Acquisition date.
    scene_id : str, optional
        Scene identifier.

    Returns
    -------
    pd.DataFrame
        Zonal statistics for each geometry.
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio is required for zonal statistics")

    if isinstance(data, xr.DataArray):
        values = data.values.squeeze()
        if transform is None and hasattr(data, 'rio'):
            transform = data.rio.transform()
    else:
        values = np.asarray(data).squeeze()

    height, width = values.shape

    results = []

    for idx, row in geometries.iterrows():
        field_id = row[field_id_column]
        geom = row.geometry

        try:
            geom_mask = geometry_mask(
                [geom],
                out_shape=(height, width),
                transform=transform,
                invert=True
            )
        except Exception:
            continue

        masked_values = values[geom_mask]
        valid_values = masked_values[~np.isnan(masked_values)]

        total_count = len(masked_values)
        valid_count = len(valid_values)

        result = {
            'field_id': field_id,
            'date': date,
            'scene_id': scene_id,
            'etf_mean': np.nanmean(valid_values) if valid_count > 0 else np.nan,
            'etf_std': np.nanstd(valid_values) if valid_count > 0 else np.nan,
            'valid_count': valid_count,
            'total_count': total_count,
            'valid_fraction': valid_count / total_count if total_count > 0 else 0.0,
        }

        results.append(result)

    return pd.DataFrame(results)
