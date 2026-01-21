"""
Local file data source for Landsat imagery.

Loads Landsat Collection 2 Level 2 imagery from local GeoTIFF files.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import re

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 - needed for rio accessor
from shapely.geometry.base import BaseGeometry

from sims.sources.base import DataSource


class LocalFileSource(DataSource):
    """
    Data source for loading Landsat imagery from local files.

    Expects Landsat Collection 2 Level 2 products organized in directories
    with standard naming conventions.

    Parameters
    ----------
    base_path : Path or str
        Base directory containing Landsat scene directories.
    file_pattern : str, optional
        Pattern for finding band files within scene directories.
        Default assumes standard Landsat naming: '{scene_id}_{band_name}.TIF'

    Examples
    --------
    >>> source = LocalFileSource('/data/landsat')
    >>> source.load_band('LC08_L2SP_042030_20200116_20200823_02_T1', 'SR_B4')
    """

    # Band name mappings for different Landsat sensors
    # Maps generic band names to sensor-specific names
    BAND_MAPPING = {
        'LANDSAT_5': {
            'blue': 'SR_B1',
            'green': 'SR_B2',
            'red': 'SR_B3',
            'nir': 'SR_B4',
            'swir1': 'SR_B5',
            'swir2': 'SR_B7',
            'thermal': 'ST_B6',
            'qa_pixel': 'QA_PIXEL',
            'qa_radsat': 'QA_RADSAT',
        },
        'LANDSAT_7': {
            'blue': 'SR_B1',
            'green': 'SR_B2',
            'red': 'SR_B3',
            'nir': 'SR_B4',
            'swir1': 'SR_B5',
            'swir2': 'SR_B7',
            'thermal': 'ST_B6',
            'qa_pixel': 'QA_PIXEL',
            'qa_radsat': 'QA_RADSAT',
        },
        'LANDSAT_8': {
            'blue': 'SR_B2',
            'green': 'SR_B3',
            'red': 'SR_B4',
            'nir': 'SR_B5',
            'swir1': 'SR_B6',
            'swir2': 'SR_B7',
            'thermal': 'ST_B10',
            'qa_pixel': 'QA_PIXEL',
            'qa_radsat': 'QA_RADSAT',
        },
        'LANDSAT_9': {
            'blue': 'SR_B2',
            'green': 'SR_B3',
            'red': 'SR_B4',
            'nir': 'SR_B5',
            'swir1': 'SR_B6',
            'swir2': 'SR_B7',
            'thermal': 'ST_B10',
            'qa_pixel': 'QA_PIXEL',
            'qa_radsat': 'QA_RADSAT',
        },
    }

    def __init__(
        self,
        base_path: Union[Path, str],
        file_pattern: str = '{scene_id}_{band_name}.TIF',
    ):
        self.base_path = Path(base_path)
        self.file_pattern = file_pattern

        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

    def _get_band_path(self, scene_id: str, band_name: str) -> Path:
        """Get the file path for a specific band."""
        filename = self.file_pattern.format(scene_id=scene_id, band_name=band_name)
        return self.base_path / scene_id / filename

    def _get_spacecraft_from_scene_id(self, scene_id: str) -> str:
        """Extract spacecraft ID from scene ID."""
        if scene_id.startswith('LC09') or scene_id.startswith('lc09'):
            return 'LANDSAT_9'
        elif scene_id.startswith('LC08') or scene_id.startswith('lc08'):
            return 'LANDSAT_8'
        elif scene_id.startswith('LE07') or scene_id.startswith('le07'):
            return 'LANDSAT_7'
        elif scene_id.startswith('LT05') or scene_id.startswith('lt05'):
            return 'LANDSAT_5'
        elif scene_id.startswith('LT04') or scene_id.startswith('lt04'):
            return 'LANDSAT_4'
        else:
            raise ValueError(f"Unknown spacecraft for scene ID: {scene_id}")

    def _resolve_band_name(self, scene_id: str, band_name: str) -> str:
        """Resolve generic band names to sensor-specific names."""
        spacecraft = self._get_spacecraft_from_scene_id(scene_id)
        band_lower = band_name.lower()

        if spacecraft in self.BAND_MAPPING:
            if band_lower in self.BAND_MAPPING[spacecraft]:
                return self.BAND_MAPPING[spacecraft][band_lower]

        # Return as-is if not a generic name
        return band_name

    def load_band(
        self,
        scene_id: str,
        band_name: str,
        geometry: Optional[BaseGeometry] = None,
    ) -> xr.DataArray:
        """
        Load a single band for a scene.

        Parameters
        ----------
        scene_id : str
            Landsat scene ID (e.g., 'LC08_L2SP_042030_20200116_20200823_02_T1').
        band_name : str
            Band name (e.g., 'SR_B4', 'red', 'nir').
        geometry : BaseGeometry, optional
            If provided, clip to geometry's bounding box.

        Returns
        -------
        xr.DataArray
            Band data with spatial coordinates and CRS.
        """
        resolved_band = self._resolve_band_name(scene_id, band_name)
        band_path = self._get_band_path(scene_id, resolved_band)

        if not band_path.exists():
            # Try alternative naming patterns
            alt_path = self.base_path / scene_id / f"{resolved_band}.TIF"
            if alt_path.exists():
                band_path = alt_path
            else:
                raise FileNotFoundError(f"Band file not found: {band_path}")

        # Load with rioxarray
        da = xr.open_dataarray(band_path, engine='rasterio')

        # Clip to geometry if provided
        if geometry is not None:
            da = da.rio.clip_box(*geometry.bounds)

        return da

    def load_bands(
        self,
        scene_id: str,
        band_names: List[str],
        geometry: Optional[BaseGeometry] = None,
    ) -> xr.Dataset:
        """
        Load multiple bands for a scene.

        Parameters
        ----------
        scene_id : str
            Landsat scene ID.
        band_names : list of str
            Band names to load.
        geometry : BaseGeometry, optional
            If provided, clip to geometry's bounding box.

        Returns
        -------
        xr.Dataset
            Dataset with each band as a variable.
        """
        data_vars = {}
        for band_name in band_names:
            resolved_name = self._resolve_band_name(scene_id, band_name)
            da = self.load_band(scene_id, band_name, geometry)
            # Use original band name as key
            data_vars[band_name] = da

        return xr.Dataset(data_vars)

    def search_scenes(
        self,
        geometry: BaseGeometry,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 70.0,
    ) -> List[str]:
        """
        Search for scenes in the local directory.

        Note: Local search is limited - it finds all scene directories
        but cannot filter by geometry or cloud cover without reading metadata.

        Parameters
        ----------
        geometry : BaseGeometry
            Area of interest (not used for local search).
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        max_cloud_cover : float
            Maximum cloud cover (not used for local search).

        Returns
        -------
        list of str
            Scene IDs found in the local directory.
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        scene_ids = []
        for scene_dir in self.base_path.iterdir():
            if not scene_dir.is_dir():
                continue

            scene_id = scene_dir.name

            # Extract date from scene ID
            try:
                # Standard Landsat C2 naming: LC08_L2SP_PPPRRR_YYYYMMDD_...
                date_match = re.search(r'_(\d{8})_', scene_id)
                if date_match:
                    scene_date = datetime.strptime(date_match.group(1), '%Y%m%d')
                    if start <= scene_date <= end:
                        scene_ids.append(scene_id)
            except (ValueError, AttributeError):
                continue

        # Sort by date
        scene_ids.sort(key=lambda x: re.search(r'_(\d{8})_', x).group(1))
        return scene_ids

    def get_metadata(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a scene.

        Extracts metadata from the scene ID and MTL file if available.

        Parameters
        ----------
        scene_id : str
            Landsat scene ID.

        Returns
        -------
        dict
            Scene metadata.
        """
        metadata = {
            'scene_id': scene_id,
            'spacecraft_id': self._get_spacecraft_from_scene_id(scene_id),
        }

        # Parse scene ID for path/row and date
        # Format: LC08_L2SP_PPPRRR_YYYYMMDD_YYYYMMDD_VV_TT
        parts = scene_id.split('_')
        if len(parts) >= 4:
            try:
                path_row = parts[2]
                metadata['path'] = int(path_row[:3])
                metadata['row'] = int(path_row[3:])

                acq_date = parts[3]
                metadata['acquisition_date'] = datetime.strptime(acq_date, '%Y%m%d')
                metadata['year'] = metadata['acquisition_date'].year
                metadata['doy'] = metadata['acquisition_date'].timetuple().tm_yday
            except (ValueError, IndexError):
                pass

        # Try to load MTL file for additional metadata
        mtl_path = self.base_path / scene_id / f"{scene_id}_MTL.txt"
        if mtl_path.exists():
            metadata.update(self._parse_mtl(mtl_path))

        return metadata

    def _parse_mtl(self, mtl_path: Path) -> Dict[str, Any]:
        """Parse Landsat MTL metadata file."""
        metadata = {}

        with open(mtl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"')

                    if key == 'CLOUD_COVER':
                        metadata['cloud_cover'] = float(value)
                    elif key == 'SUN_ELEVATION':
                        metadata['sun_elevation'] = float(value)
                    elif key == 'SUN_AZIMUTH':
                        metadata['sun_azimuth'] = float(value)

        return metadata
