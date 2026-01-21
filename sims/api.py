"""
High-level API for SIMS ET fraction computation.

This module provides a simple, high-level interface for common SIMS workflows.
"""

from typing import Optional, Union, List
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from sims.sources.base import DataSource
from sims.image import LandsatImage
from sims.zonal import compute_zonal_stats


def compute_etf_timeseries(
    source: DataSource,
    geometries: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    field_id_column: str = 'field_id',
    max_cloud_cover: float = 70.0,
    crop_type: Optional[Union[np.ndarray, xr.DataArray]] = None,
    use_crop_type_kc: bool = False,
    min_valid_fraction: float = 0.5,
    stats: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute ET fraction time series for multiple fields.

    This is the main entry point for processing multiple Landsat scenes
    and extracting field-level ET fraction statistics.

    Parameters
    ----------
    source : DataSource
        Data source for loading Landsat imagery (LocalFileSource or STACSource).
    geometries : gpd.GeoDataFrame
        GeoDataFrame of field polygons.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    field_id_column : str
        Column name containing unique field identifiers.
    max_cloud_cover : float
        Maximum cloud cover percentage for scene selection (0-100).
    crop_type : array-like, optional
        CDL crop type codes matching the geometry of the scenes.
        Required for SIMS parity; generate via sims.ancillary.get_crop_type().
    use_crop_type_kc : bool
        If True, use crop-type-specific Kc with height/density parameters.
    min_valid_fraction : float
        Minimum fraction of valid pixels required for a field observation.
        Observations with fewer valid pixels are excluded.
    stats : list of str, optional
        Statistics to compute. Default: ['mean', 'std', 'count'].

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - field_id: Field identifier
        - date: Acquisition date
        - scene_id: Landsat scene identifier
        - etf_mean: Mean ET fraction for the field
        - etf_std: Standard deviation of ET fraction
        - valid_fraction: Fraction of valid (cloud-free) pixels

    Examples
    --------
    >>> from sims import STACSource, compute_etf_timeseries
    >>> import geopandas as gpd
    >>>
    >>> source = STACSource(
    ...     catalog_url='https://planetarycomputer.microsoft.com/api/stac/v1',
    ...     collection='landsat-c2-l2'
    ... )
    >>> fields = gpd.read_file('fields.shp')
    >>>
    >>> result = compute_etf_timeseries(
    ...     source=source,
    ...     geometries=fields,
    ...     start_date='2023-01-01',
    ...     end_date='2023-12-31',
    ...     field_id_column='field_id',
    ... )
    """
    if stats is None:
        stats = ['mean', 'std', 'count']

    if crop_type is None:
        raise ValueError("crop_type is required for SIMS ETf. Provide CDL-derived crop_type matching the scenes.")

    # Get the bounding box of all geometries for scene search
    bounds = geometries.total_bounds
    from shapely.geometry import box
    search_geometry = box(*bounds)

    # Search for scenes
    scene_ids = source.search_scenes(
        geometry=search_geometry,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
    )

    if not scene_ids:
        return pd.DataFrame()

    all_results = []

    for scene_id in scene_ids:
        try:
            # Create LandsatImage for this scene
            image = LandsatImage(
                source=source,
                scene_id=scene_id,
                geometry=search_geometry,
                crop_type=crop_type,
                use_crop_type_kc=use_crop_type_kc,
            )

            # Compute ET fraction
            etf = image.et_fraction

            # Apply cloud mask
            etf_masked = image.apply_cloud_mask(etf)

            # Compute zonal statistics
            scene_results = compute_zonal_stats(
                image=image,
                etf=etf_masked,
                geometries=geometries,
                field_id_column=field_id_column,
                stats=stats,
                apply_mask=False,  # Already masked
            )

            all_results.append(scene_results)

        except Exception as e:
            # Log warning but continue processing other scenes
            print(f"Warning: Failed to process scene {scene_id}: {e}")
            continue

    if not all_results:
        return pd.DataFrame()

    # Combine results
    result = pd.concat(all_results, ignore_index=True)

    # Filter by minimum valid fraction
    result = result[result['valid_fraction'] >= min_valid_fraction]

    # Sort by field and date
    result = result.sort_values(['field_id', 'date']).reset_index(drop=True)

    return result


def process_scene(
    source: DataSource,
    scene_id: str,
    geometries: Optional[gpd.GeoDataFrame] = None,
    field_id_column: str = 'field_id',
    crop_type: Optional[Union[np.ndarray, xr.DataArray]] = None,
    use_crop_type_kc: bool = False,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Process a single Landsat scene.

    Parameters
    ----------
    source : DataSource
        Data source for loading imagery.
    scene_id : str
        Landsat scene ID.
    geometries : gpd.GeoDataFrame, optional
        If provided, compute zonal statistics for these geometries.
        If None, return the full raster dataset.
    field_id_column : str
        Column name for field identifiers (if geometries provided).
    crop_type : array-like, optional
        CDL crop type codes. Required for SIMS parity; generate via sims.ancillary.get_crop_type().
    use_crop_type_kc : bool
        Use crop-type-specific Kc calculations.

    Returns
    -------
        xr.Dataset or pd.DataFrame
        If geometries is None, returns Dataset with raster products.
        If geometries provided, returns DataFrame with zonal statistics.
    """
    if crop_type is None:
        raise ValueError("crop_type is required for SIMS ETf. Provide CDL-derived crop_type for the scene.")

    # Get geometry for clipping
    clip_geometry = None
    if geometries is not None:
        from shapely.geometry import box
        clip_geometry = box(*geometries.total_bounds)

    # Create LandsatImage
    image = LandsatImage(
        source=source,
        scene_id=scene_id,
        geometry=clip_geometry,
        crop_type=crop_type,
        use_crop_type_kc=use_crop_type_kc,
    )

    if geometries is None:
        # Return full raster dataset
        return image.calculate(variables=[
            'ndvi', 'fc', 'et_fraction', 'lai', 'albedo', 'lst', 'qa_mask'
        ])
    else:
        # Compute zonal statistics
        etf = image.apply_cloud_mask(image.et_fraction)
        return compute_zonal_stats(
            image=image,
            etf=etf,
            geometries=geometries,
            field_id_column=field_id_column,
            apply_mask=False,
        )


def compute_et(
    etf: Union[pd.DataFrame, xr.DataArray, np.ndarray],
    et_reference: Union[pd.Series, xr.DataArray, np.ndarray, float],
) -> Union[pd.DataFrame, xr.DataArray, np.ndarray]:
    """
    Compute actual ET from ET fraction and reference ET.

    Parameters
    ----------
    etf : DataFrame, DataArray, or array
        ET fraction values. If DataFrame, expects 'etf_mean' column.
    et_reference : Series, DataArray, array, or float
        Reference ET values. If DataFrame input, should be Series indexed by date.

    Returns
    -------
    Same type as input
        Actual ET = ETf × ET_reference

    Examples
    --------
    >>> # With DataFrame
    >>> result = compute_etf_timeseries(...)
    >>> et_ref = pd.Series({date: value, ...})  # Daily reference ET
    >>> et = compute_et(result, et_ref)
    >>>
    >>> # With arrays
    >>> etf = image.et_fraction.values
    >>> et = compute_et(etf, 5.0)  # Constant reference ET of 5 mm/day
    """
    if isinstance(etf, pd.DataFrame):
        result = etf.copy()

        if isinstance(et_reference, pd.Series):
            # Match by date
            result['et_reference'] = result['date'].map(et_reference)
        elif isinstance(et_reference, (int, float)):
            result['et_reference'] = et_reference
        else:
            raise ValueError("For DataFrame input, et_reference must be Series or scalar")

        result['et'] = result['etf_mean'] * result['et_reference']
        return result

    elif isinstance(etf, xr.DataArray):
        return etf * et_reference

    else:
        return np.asarray(etf) * np.asarray(et_reference)
