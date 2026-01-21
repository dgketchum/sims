"""
Convenience functions for running SIMS on Planetary Computer data.

This module provides high-level functions for common workflows.
"""

from typing import Optional, Union, List, Dict, Any
from datetime import datetime
import warnings

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box

from sims.sources.stac import STACSource
from sims.image import LandsatImage
from sims.model import compute_et


def process_scene(
    scene_id: str,
    geometry: BaseGeometry,
    et_reference: Optional[float] = None,
    crop_type: Optional[xr.DataArray] = None,
    use_crop_type_kc: bool = False,
) -> Dict[str, Any]:
    """
    Process a single Landsat scene and compute ET.

    Parameters
    ----------
    scene_id : str
        Landsat scene ID from Planetary Computer.
    geometry : BaseGeometry
        Area of interest to clip the scene.
    et_reference : float, optional
        Reference ET in mm/day. If provided, computes actual ET.
    crop_type : xr.DataArray, optional
        Crop type codes for crop-specific Kc.
    use_crop_type_kc : bool
        Use crop-type-specific Kc calculations.

    Returns
    -------
    dict
        Results containing:
        - scene_id: str
        - date: datetime
        - ndvi: xr.DataArray
        - et_fraction: xr.DataArray
        - et: xr.DataArray (if et_reference provided)
        - metadata: dict
    """
    source = STACSource()

    image = LandsatImage(
        source=source,
        scene_id=scene_id,
        geometry=geometry,
        crop_type=crop_type,
        use_crop_type_kc=use_crop_type_kc,
    )

    result = {
        'scene_id': scene_id,
        'date': image.date,
        'ndvi': image.ndvi,
        'et_fraction': image.apply_cloud_mask(image.et_fraction),
        'qa_mask': image.qa_mask,
        'metadata': image.metadata,
    }

    if et_reference is not None:
        result['et_reference'] = et_reference
        result['et'] = compute_et(result['et_fraction'].values, et_reference)
        result['et'] = xr.DataArray(
            result['et'],
            coords=result['et_fraction'].coords,
            dims=result['et_fraction'].dims,
        )

    return result


def search_and_process(
    geometry: BaseGeometry,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 30.0,
    et_reference: Optional[Union[float, Dict[str, float]]] = None,
    max_scenes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search for scenes and process all matching Landsat imagery.

    Parameters
    ----------
    geometry : BaseGeometry
        Area of interest.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    max_cloud_cover : float
        Maximum cloud cover percentage (0-100).
    et_reference : float or dict, optional
        Reference ET. If float, uses same value for all scenes.
        If dict, maps date strings to ET values.
    max_scenes : int, optional
        Maximum number of scenes to process.

    Returns
    -------
    list of dict
        Results for each processed scene.
    """
    source = STACSource()

    # Search for scenes
    scene_ids = source.search_scenes(
        geometry=geometry,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover,
    )

    if max_scenes:
        scene_ids = scene_ids[:max_scenes]

    results = []
    for scene_id in scene_ids:
        try:
            # Get ET reference for this scene
            etr = None
            if et_reference is not None:
                if isinstance(et_reference, (int, float)):
                    etr = float(et_reference)
                elif isinstance(et_reference, dict):
                    # Look up by date - get metadata first
                    meta = source.get_metadata(scene_id)
                    date_str = meta['acquisition_date'].strftime('%Y-%m-%d')
                    etr = et_reference.get(date_str)

            result = process_scene(
                scene_id=scene_id,
                geometry=geometry,
                et_reference=etr,
            )
            results.append(result)

        except Exception as e:
            warnings.warn(f"Failed to process {scene_id}: {e}")
            continue

    return results


def compute_field_mean(
    result: Dict[str, Any],
    geometry: BaseGeometry,
) -> Dict[str, float]:
    """
    Compute mean values for a field geometry from scene results.

    Parameters
    ----------
    result : dict
        Result from process_scene().
    geometry : BaseGeometry
        Field polygon.

    Returns
    -------
    dict
        Mean values for the field:
        - ndvi_mean
        - etf_mean
        - et_mean (if available)
        - valid_fraction
    """
    from rasterio.features import geometry_mask

    etf = result['et_fraction']

    # Create mask for geometry
    transform = etf.rio.transform() if hasattr(etf, 'rio') else None
    height, width = etf.shape[-2:]

    try:
        mask = geometry_mask(
            [geometry],
            out_shape=(height, width),
            transform=transform,
            invert=True,
        )
    except Exception:
        # Fallback: use all data
        mask = np.ones((height, width), dtype=bool)

    # Extract values
    etf_values = etf.values.squeeze()[mask]
    ndvi_values = result['ndvi'].values.squeeze()[mask]

    valid = ~np.isnan(etf_values)
    valid_fraction = valid.sum() / len(valid) if len(valid) > 0 else 0

    output = {
        'date': result['date'],
        'scene_id': result['scene_id'],
        'ndvi_mean': float(np.nanmean(ndvi_values)),
        'etf_mean': float(np.nanmean(etf_values)),
        'valid_fraction': float(valid_fraction),
    }

    if 'et' in result:
        et_values = result['et'].values.squeeze()[mask]
        output['et_mean'] = float(np.nanmean(et_values))
        output['et_reference'] = result['et_reference']

    return output
