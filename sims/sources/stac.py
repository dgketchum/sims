"""
STAC API data source for Landsat imagery.

Loads Landsat Collection 2 Level 2 imagery from STAC-compliant APIs
such as Microsoft Planetary Computer or Element 84.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
import xarray as xr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import mapping

import os
from sims.sources.base import DataSource

# Optional imports - will raise helpful errors if not installed
try:
    import pystac_client
    import planetary_computer
    import stackstac
except ImportError as e:
    _import_error = e
else:
    _import_error = None


class STACSource(DataSource):
    """
    Data source for loading Landsat imagery from STAC APIs.

    Supports any STAC-compliant API, with special handling for
    Microsoft Planetary Computer (automatic token signing).

    Parameters
    ----------
    catalog_url : str
        URL of the STAC catalog API endpoint.
    collection : str
        STAC collection ID (default: 'landsat-c2-l2').
    sign_urls : bool
        Whether to sign URLs for Planetary Computer (default: True for PC URLs).
    provider : str, optional
        Shortcut for common providers: 'mpc', 'aws'. Overrides catalog_url if set.

    Examples
    --------
    >>> # Microsoft Planetary Computer (default, recommended)
    >>> source = STACSource()
    >>> source = STACSource(provider='mpc')
    >>>
    >>> # AWS Earth Search (requires AWS credentials - requester pays bucket)
    >>> source = STACSource(provider='aws')
    >>>
    >>> # Custom STAC endpoint
    >>> source = STACSource(catalog_url='https://custom-stac.example.com/v1')

    Notes
    -----
    The AWS provider requires valid AWS credentials because the USGS Landsat
    bucket is configured as requester-pays. MPC is recommended for most users
    as it provides free access through the Planetary Computer platform.
    """

    # Provider presets
    PROVIDERS = {
        'mpc': {
            'catalog_url': 'https://planetarycomputer.microsoft.com/api/stac/v1',
            'collection': 'landsat-c2-l2',
            'sign_urls': True,
        },
        'aws': {
            'catalog_url': 'https://earth-search.aws.element84.com/v1',
            'collection': 'landsat-c2-l2',
            'sign_urls': False,
        },
    }

    # Band name mappings (same as LocalFileSource)
    BAND_MAPPING = {
        'LANDSAT_5': {
            'blue': 'blue', 'green': 'green', 'red': 'red',
            'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22',
            'thermal': 'lwir', 'qa_pixel': 'qa_pixel', 'qa_radsat': 'qa_radsat',
        },
        'LANDSAT_7': {
            'blue': 'blue', 'green': 'green', 'red': 'red',
            'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22',
            'thermal': 'lwir', 'qa_pixel': 'qa_pixel', 'qa_radsat': 'qa_radsat',
        },
        'LANDSAT_8': {
            'blue': 'blue', 'green': 'green', 'red': 'red',
            'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22',
            'thermal': 'lwir11', 'qa_pixel': 'qa_pixel', 'qa_radsat': 'qa_radsat',
        },
        'LANDSAT_9': {
            'blue': 'blue', 'green': 'green', 'red': 'red',
            'nir': 'nir08', 'swir1': 'swir16', 'swir2': 'swir22',
            'thermal': 'lwir11', 'qa_pixel': 'qa_pixel', 'qa_radsat': 'qa_radsat',
        },
    }

    # STAC asset name to common name mapping for Planetary Computer
    STAC_BAND_NAMES = {
        'blue': 'blue',
        'green': 'green',
        'red': 'red',
        'nir08': 'nir08',
        'swir16': 'swir16',
        'swir22': 'swir22',
        'lwir': 'lwir',
        'lwir11': 'lwir11',
        'qa_pixel': 'qa_pixel',
        'qa_radsat': 'qa_radsat',
    }

    def __init__(
        self,
        catalog_url: Optional[str] = None,
        collection: Optional[str] = None,
        sign_urls: Optional[bool] = None,
        provider: Optional[str] = None,
    ):
        if _import_error is not None:
            raise ImportError(
                "STAC dependencies not installed. Install with: "
                "pip install pystac-client planetary-computer stackstac"
            ) from _import_error

        # Handle provider presets
        if provider is not None:
            if provider not in self.PROVIDERS:
                raise ValueError(
                    f"Unknown provider: {provider}. "
                    f"Available: {list(self.PROVIDERS.keys())}"
                )
            preset = self.PROVIDERS[provider]
            catalog_url = catalog_url or preset['catalog_url']
            collection = collection or preset['collection']
            if sign_urls is None:
                sign_urls = preset['sign_urls']

        # Defaults (MPC)
        self.catalog_url = catalog_url or 'https://planetarycomputer.microsoft.com/api/stac/v1'
        self.collection = collection or 'landsat-c2-l2'

        # Auto-detect Planetary Computer
        self.is_planetary_computer = 'planetarycomputer' in self.catalog_url
        if sign_urls is None:
            self.sign_urls = self.is_planetary_computer
        else:
            self.sign_urls = sign_urls

        self.provider = provider or ('mpc' if self.is_planetary_computer else 'custom')

        # For AWS, enable unsigned requests for public buckets (USGS Landsat)
        # unless user has already configured AWS credentials
        if self.provider == 'aws':
            if 'AWS_ACCESS_KEY_ID' not in os.environ:
                os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

        # Initialize STAC client
        self.client = pystac_client.Client.open(self.catalog_url)

        # Cache for items to avoid repeated API calls
        self._item_cache: Dict[str, Any] = {}

    def _get_item(self, scene_id: str) -> Any:
        """Get STAC item by ID, with caching."""
        if scene_id not in self._item_cache:
            # Search for the item
            search = self.client.search(
                collections=[self.collection],
                ids=[scene_id],
            )
            items = list(search.items())
            if not items:
                raise ValueError(f"Scene not found: {scene_id}")

            item = items[0]

            # Sign URLs if needed
            if self.sign_urls:
                item = planetary_computer.sign(item)

            self._item_cache[scene_id] = item

        return self._item_cache[scene_id]

    def _get_spacecraft_from_item(self, item: Any) -> str:
        """Extract spacecraft ID from STAC item."""
        platform = item.properties.get('platform', '').upper()
        if 'LANDSAT-9' in platform or 'LANDSAT_9' in platform:
            return 'LANDSAT_9'
        elif 'LANDSAT-8' in platform or 'LANDSAT_8' in platform:
            return 'LANDSAT_8'
        elif 'LANDSAT-7' in platform or 'LANDSAT_7' in platform:
            return 'LANDSAT_7'
        elif 'LANDSAT-5' in platform or 'LANDSAT_5' in platform:
            return 'LANDSAT_5'
        else:
            return 'UNKNOWN'

    def _resolve_band_name(self, spacecraft: str, band_name: str) -> str:
        """Resolve generic band names to STAC asset names."""
        band_lower = band_name.lower()

        if spacecraft in self.BAND_MAPPING:
            if band_lower in self.BAND_MAPPING[spacecraft]:
                return self.BAND_MAPPING[spacecraft][band_lower]

        # Check if it's already a STAC band name
        if band_lower in self.STAC_BAND_NAMES:
            return self.STAC_BAND_NAMES[band_lower]

        return band_name

    def _get_epsg(self, item: Any) -> int:
        """Get EPSG code from STAC item."""
        # Try proj:epsg from item properties
        epsg = item.properties.get('proj:epsg')
        if epsg:
            return int(epsg)

        # Try from asset
        for asset in item.assets.values():
            if hasattr(asset, 'extra_fields'):
                epsg = asset.extra_fields.get('proj:epsg')
                if epsg:
                    return int(epsg)

        # Default to UTM zone based on centroid (rough estimate)
        # Or use WGS84 as fallback
        return 4326

    def load_band(
        self,
        scene_id: str,
        band_name: str,
        geometry: Optional[BaseGeometry] = None,
    ) -> xr.DataArray:
        """
        Load a single band for a scene from STAC.

        Parameters
        ----------
        scene_id : str
            STAC item ID.
        band_name : str
            Band name (generic or STAC asset name).
        geometry : BaseGeometry, optional
            If provided, clip to geometry's bounding box.

        Returns
        -------
        xr.DataArray
            Band data with spatial coordinates and CRS.
        """
        item = self._get_item(scene_id)
        spacecraft = self._get_spacecraft_from_item(item)
        resolved_band = self._resolve_band_name(spacecraft, band_name)

        # Get CRS from item
        epsg = self._get_epsg(item)

        # Get bounds for clipping
        bounds = geometry.bounds if geometry is not None else None

        # Use stackstac to load the band
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress stackstac warnings

            da = stackstac.stack(
                [item],
                assets=[resolved_band],
                bounds=bounds,
                epsg=4326,  # Output in WGS84 to match input bounds
                resolution=0.00027,  # ~30m in degrees at mid-latitudes
                chunksize=512,
            )

        # stackstac returns (time, band, y, x), squeeze to (y, x)
        da = da.squeeze(['time', 'band'], drop=True)

        # Compute to load data into memory (avoids dask overhead for small areas)
        da = da.compute()

        return da

    def load_bands(
        self,
        scene_id: str,
        band_names: List[str],
        geometry: Optional[BaseGeometry] = None,
    ) -> xr.Dataset:
        """
        Load multiple bands for a scene from STAC.

        Parameters
        ----------
        scene_id : str
            STAC item ID.
        band_names : list of str
            Band names to load.
        geometry : BaseGeometry, optional
            If provided, clip to geometry's bounding box.

        Returns
        -------
        xr.Dataset
            Dataset with each band as a variable.
        """
        item = self._get_item(scene_id)
        spacecraft = self._get_spacecraft_from_item(item)

        # Resolve all band names
        resolved_bands = [
            self._resolve_band_name(spacecraft, bn) for bn in band_names
        ]

        # Get CRS from item
        epsg = self._get_epsg(item)

        # Get bounds for clipping
        bounds = geometry.bounds if geometry is not None else None

        # Use stackstac to load all bands at once
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            da = stackstac.stack(
                [item],
                assets=resolved_bands,
                bounds=bounds,
                epsg=4326,  # Output in WGS84 to match input bounds
                resolution=0.00027,  # ~30m in degrees at mid-latitudes
                chunksize=512,
            )

        # Convert to dataset and compute
        da = da.squeeze('time', drop=True).compute()

        # Create dataset with original band names
        data_vars = {}
        for orig_name, resolved_name in zip(band_names, resolved_bands):
            band_da = da.sel(band=resolved_name)
            data_vars[orig_name] = band_da.drop_vars('band')

        return xr.Dataset(data_vars)

    def search_scenes(
        self,
        geometry: BaseGeometry,
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 70.0,
    ) -> List[str]:
        """
        Search for Landsat scenes in STAC catalog.

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

        Returns
        -------
        list of str
            STAC item IDs matching the criteria, sorted by date.
        """
        # Convert geometry to GeoJSON for STAC
        geojson = mapping(geometry)

        # Build date range
        datetime_range = f"{start_date}/{end_date}"

        # Search STAC catalog
        search = self.client.search(
            collections=[self.collection],
            intersects=geojson,
            datetime=datetime_range,
            query={
                'eo:cloud_cover': {'lt': max_cloud_cover},
            },
        )

        # Collect and sort results
        items = list(search.items())
        items.sort(key=lambda x: x.datetime)

        return [item.id for item in items]

    def get_metadata(self, scene_id: str) -> Dict[str, Any]:
        """
        Get metadata for a scene from STAC.

        Parameters
        ----------
        scene_id : str
            STAC item ID.

        Returns
        -------
        dict
            Scene metadata.
        """
        item = self._get_item(scene_id)

        metadata = {
            'scene_id': scene_id,
            'spacecraft_id': self._get_spacecraft_from_item(item),
            'acquisition_date': item.datetime,
            'cloud_cover': item.properties.get('eo:cloud_cover'),
            'sun_elevation': item.properties.get('view:sun_elevation'),
            'sun_azimuth': item.properties.get('view:sun_azimuth'),
            'path': item.properties.get('landsat:wrs_path'),
            'row': item.properties.get('landsat:wrs_row'),
        }

        if metadata['acquisition_date']:
            metadata['year'] = metadata['acquisition_date'].year
            metadata['doy'] = metadata['acquisition_date'].timetuple().tm_yday

        return metadata
