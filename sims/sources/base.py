"""
Abstract base class for data sources.

Data sources abstract the loading of Landsat imagery from various backends
(local files, STAC APIs, cloud storage, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

import xarray as xr
from shapely.geometry.base import BaseGeometry


class DataSource(ABC):
    """
    Abstract base class for Landsat data sources.

    Data sources handle the retrieval of Landsat imagery and metadata
    from various backends. All implementations must provide methods for:
    - Loading individual bands
    - Searching for scenes by geometry and date
    - Retrieving scene metadata

    The data source abstraction allows the same LandsatImage class to work
    with local files, STAC APIs, or any other data backend.
    """

    @abstractmethod
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
            Unique identifier for the Landsat scene.
        band_name : str
            Name of the band to load (e.g., 'SR_B4', 'ST_B10', 'QA_PIXEL').
        geometry : BaseGeometry, optional
            If provided, clip the band to this geometry's bounding box.

        Returns
        -------
        xr.DataArray
            The band data as an xarray DataArray with CRS and transform metadata.
        """
        ...

    @abstractmethod
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
            Unique identifier for the Landsat scene.
        band_names : list of str
            Names of bands to load.
        geometry : BaseGeometry, optional
            If provided, clip bands to this geometry's bounding box.

        Returns
        -------
        xr.Dataset
            Dataset containing all requested bands.
        """
        ...

    @abstractmethod
    def search_scenes(
        self,
        geometry: BaseGeometry,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 70.0,
    ) -> List[str]:
        """
        Search for Landsat scenes matching the criteria.

        Parameters
        ----------
        geometry : BaseGeometry
            Area of interest (point or polygon).
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format.
        max_cloud_cover : float
            Maximum cloud cover percentage (0-100).

        Returns
        -------
        list of str
            List of scene IDs matching the criteria, sorted by date.
        """
        ...

    @abstractmethod
    def get_metadata(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a scene.

        Parameters
        ----------
        scene_id : str
            Unique identifier for the Landsat scene.

        Returns
        -------
        dict
            Scene metadata including:
            - acquisition_date: datetime
            - sensor: str (e.g., 'LC08', 'LE07')
            - path: int
            - row: int
            - cloud_cover: float
            - sun_elevation: float
            - sun_azimuth: float
            - crs: str or CRS object
        """
        ...

    def get_spacecraft_id(self, scene_id: str) -> str:
        """
        Extract spacecraft ID from scene ID.

        Parameters
        ----------
        scene_id : str
            Landsat scene ID.

        Returns
        -------
        str
            Spacecraft ID (e.g., 'LANDSAT_8', 'LANDSAT_9').
        """
        metadata = self.get_metadata(scene_id)
        return metadata.get('spacecraft_id', metadata.get('sensor', 'UNKNOWN'))
