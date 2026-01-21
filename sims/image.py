"""
LandsatImage class for processing individual Landsat scenes.

This module provides the LandsatImage class which abstracts a single Landsat
scene and provides derived products (NDVI, LAI, albedo, LST, etc.) with
lazy computation and caching.
"""

from datetime import datetime
from typing import Optional, Dict, Any, Union
from functools import cached_property

import numpy as np
import xarray as xr

from sims.sources.base import DataSource
from sims import model
from sims.data import CDL, get_crop_params


class LandsatImage:
    """
    Represents a single Landsat scene with derived products.

    LandsatImage provides a high-level interface for working with Landsat
    Collection 2 Level 2 data. It handles:
    - Loading bands from any DataSource (local files, STAC, etc.)
    - Applying scaling factors and converting to physical units
    - Computing derived products (NDVI, albedo, LST, etc.)
    - Cloud masking using QA_PIXEL band
    - Computing ET fraction using the SIMS algorithm

    All derived products are computed lazily and cached.

    Parameters
    ----------
    source : DataSource
        Data source for loading bands.
    scene_id : str
        Unique identifier for the scene.
    geometry : BaseGeometry, optional
        If provided, clip all bands to this geometry's bounding box.
    crop_type : array-like, optional
        CDL crop type codes. If not provided, uses generic Kc.
    use_crop_type_kc : bool
        If True, use crop-type-specific Kc calculations.

    Examples
    --------
    >>> source = LocalFileSource('/data/landsat')
    >>> image = LandsatImage(source, 'LC08_L2SP_042030_20200116_...')
    >>> ndvi = image.ndvi
    >>> etf = image.et_fraction
    """

    # Scaling factors for Landsat Collection 2 Level 2
    # Surface reflectance: DN * 0.0000275 - 0.2
    SR_SCALE = 0.0000275
    SR_OFFSET = -0.2

    # Surface temperature: DN * 0.00341802 + 149.0 (Kelvin)
    ST_SCALE = 0.00341802
    ST_OFFSET = 149.0

    # QA_PIXEL bit positions for Landsat C2
    QA_BITS = {
        'fill': 0,           # Bit 0: Fill
        'dilated_cloud': 1,  # Bit 1: Dilated Cloud
        'cirrus': 2,         # Bit 2: Cirrus (L8/L9 only)
        'cloud': 3,          # Bit 3: Cloud
        'cloud_shadow': 4,   # Bit 4: Cloud Shadow
        'snow': 5,           # Bit 5: Snow
        'clear': 6,          # Bit 6: Clear
        'water': 7,          # Bit 7: Water
    }

    def __init__(
        self,
        source: DataSource,
        scene_id: str,
        geometry=None,
        crop_type: Optional[Union[np.ndarray, xr.DataArray]] = None,
        use_crop_type_kc: bool = False,
    ):
        self._source = source
        self._scene_id = scene_id
        self._geometry = geometry
        self._crop_type = crop_type
        self._use_crop_type_kc = use_crop_type_kc

        # Band cache
        self._bands: Dict[str, xr.DataArray] = {}

    @property
    def scene_id(self) -> str:
        """Scene identifier."""
        return self._scene_id

    @cached_property
    def metadata(self) -> Dict[str, Any]:
        """Scene metadata from the data source."""
        return self._source.get_metadata(self._scene_id)

    @property
    def date(self) -> datetime:
        """Acquisition date."""
        return self.metadata.get('acquisition_date')

    @property
    def year(self) -> int:
        """Acquisition year."""
        return self.metadata.get('year', self.date.year if self.date else None)

    @property
    def doy(self) -> int:
        """Day of year (1-366)."""
        return self.metadata.get('doy', self.date.timetuple().tm_yday if self.date else None)

    @property
    def spacecraft_id(self) -> str:
        """Spacecraft identifier (e.g., 'LANDSAT_8')."""
        return self.metadata.get('spacecraft_id')

    def _load_band(self, band_name: str) -> xr.DataArray:
        """Load a band with caching."""
        if band_name not in self._bands:
            self._bands[band_name] = self._source.load_band(
                self._scene_id, band_name, self._geometry
            )
        return self._bands[band_name]

    def _scale_sr_band(self, da: xr.DataArray) -> xr.DataArray:
        """Scale surface reflectance band to 0-1 range if needed."""
        # Check if data is already scaled (values in 0-2 range indicate scaled data)
        # Raw Landsat C2 L2 DNs are typically 7000-50000 range
        max_val = float(da.max())
        if max_val < 2.0:
            # Already scaled to reflectance
            return da
        else:
            # Apply scaling: DN * 0.0000275 - 0.2
            return da * self.SR_SCALE + self.SR_OFFSET

    def _scale_st_band(self, da: xr.DataArray) -> xr.DataArray:
        """Scale surface temperature band to Kelvin if needed."""
        # Check if data is already scaled (values 250-350 K range indicate scaled)
        # Raw Landsat C2 L2 ST DNs are typically 20000-50000 range
        max_val = float(da.max())
        if max_val < 400:
            # Already scaled to Kelvin
            return da
        else:
            # Apply scaling: DN * 0.00341802 + 149.0
            return da * self.ST_SCALE + self.ST_OFFSET

    @cached_property
    def red(self) -> xr.DataArray:
        """Red band in reflectance units (0-1)."""
        da = self._load_band('red')
        return self._scale_sr_band(da)

    @cached_property
    def nir(self) -> xr.DataArray:
        """NIR band in reflectance units (0-1)."""
        da = self._load_band('nir')
        return self._scale_sr_band(da)

    @cached_property
    def blue(self) -> xr.DataArray:
        """Blue band in reflectance units (0-1)."""
        da = self._load_band('blue')
        return self._scale_sr_band(da)

    @cached_property
    def green(self) -> xr.DataArray:
        """Green band in reflectance units (0-1)."""
        da = self._load_band('green')
        return self._scale_sr_band(da)

    @cached_property
    def swir1(self) -> xr.DataArray:
        """SWIR1 band in reflectance units (0-1)."""
        da = self._load_band('swir1')
        return self._scale_sr_band(da)

    @cached_property
    def swir2(self) -> xr.DataArray:
        """SWIR2 band in reflectance units (0-1)."""
        da = self._load_band('swir2')
        return self._scale_sr_band(da)

    @cached_property
    def thermal(self) -> xr.DataArray:
        """Thermal band (Land Surface Temperature in Kelvin)."""
        da = self._load_band('thermal')
        return self._scale_st_band(da)

    @cached_property
    def lst(self) -> xr.DataArray:
        """Land Surface Temperature in Kelvin (alias for thermal)."""
        return self.thermal

    @cached_property
    def qa_pixel(self) -> xr.DataArray:
        """QA_PIXEL band (raw values)."""
        return self._load_band('qa_pixel')

    @cached_property
    def ndvi(self) -> xr.DataArray:
        """
        Normalized Difference Vegetation Index.

        NDVI = (NIR - RED) / (NIR + RED)

        Returns values in range [-1, 1].
        """
        # Avoid divide by zero
        denom = self.nir + self.red
        ndvi = xr.where(denom != 0, (self.nir - self.red) / denom, 0)
        return ndvi.clip(-1, 1)

    @cached_property
    def evi(self) -> xr.DataArray:
        """
        Enhanced Vegetation Index.

        EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
        """
        denom = self.nir + 6 * self.red - 7.5 * self.blue + 1
        evi = xr.where(denom != 0, 2.5 * (self.nir - self.red) / denom, 0)
        return evi.clip(-1, 1)

    @cached_property
    def lai(self) -> xr.DataArray:
        """
        Leaf Area Index estimated from NDVI.

        Uses the relationship: LAI = -ln((1 - fc) / k) / k
        where fc is fraction of cover and k is extinction coefficient (~0.5).

        Simplified as: LAI = 3.618 * NDVI - 0.118
        """
        # Simple linear approximation valid for agricultural areas
        lai = 3.618 * self.ndvi - 0.118
        return lai.clip(0, 8)

    @cached_property
    def albedo(self) -> xr.DataArray:
        """
        Surface albedo using weighted band combination.

        Uses Liang (2001) coefficients for Landsat.
        """
        # Coefficients for broadband albedo from Liang (2001)
        albedo = (
            0.356 * self.blue +
            0.130 * self.red +
            0.373 * self.nir +
            0.085 * self.swir1 +
            0.072 * self.swir2 -
            0.0018
        )
        return albedo.clip(0, 1)

    @cached_property
    def emissivity(self) -> xr.DataArray:
        """
        Surface emissivity estimated from NDVI.

        Uses the NDVI-based emissivity method.
        - NDVI < 0.2: Bare soil (0.97)
        - NDVI > 0.5: Full vegetation (0.99)
        - Between: Linear interpolation
        """
        emis = xr.where(
            self.ndvi < 0.2,
            0.97,
            xr.where(
                self.ndvi > 0.5,
                0.99,
                0.97 + 0.02 * (self.ndvi - 0.2) / 0.3
            )
        )
        return emis

    @cached_property
    def fc(self) -> xr.DataArray:
        """
        Fraction of cover from NDVI.

        Uses the SIMS algorithm's NDVI-Fc relationship for surface reflectance.
        """
        fc = model.fraction_of_cover(self.ndvi.values, reflectance_type='SR')
        return xr.DataArray(fc, coords=self.ndvi.coords, dims=self.ndvi.dims)

    @cached_property
    def qa_mask(self) -> xr.DataArray:
        """
        Boolean mask where True = valid (clear) pixels.

        Masks clouds, cloud shadows, cirrus, snow, and fill values.
        """
        qa = self.qa_pixel.values.astype(np.uint16)

        # Create mask for bad pixels
        fill_mask = (qa >> self.QA_BITS['fill']) & 1
        cloud_mask = (qa >> self.QA_BITS['cloud']) & 1
        shadow_mask = (qa >> self.QA_BITS['cloud_shadow']) & 1
        cirrus_mask = (qa >> self.QA_BITS['cirrus']) & 1
        snow_mask = (qa >> self.QA_BITS['snow']) & 1
        dilated_mask = (qa >> self.QA_BITS['dilated_cloud']) & 1

        # Valid pixels have none of these flags set
        invalid = fill_mask | cloud_mask | shadow_mask | cirrus_mask | snow_mask | dilated_mask
        valid = ~invalid.astype(bool)

        return xr.DataArray(valid, coords=self.qa_pixel.coords, dims=self.qa_pixel.dims)

    def apply_cloud_mask(self, data: xr.DataArray) -> xr.DataArray:
        """
        Apply cloud mask to data, setting masked pixels to NaN.

        Parameters
        ----------
        data : xr.DataArray
            Data to mask.

        Returns
        -------
        xr.DataArray
            Masked data with NaN for cloudy/invalid pixels.
        """
        return data.where(self.qa_mask)

    @cached_property
    def crop_class(self) -> Optional[xr.DataArray]:
        """
        Crop class derived from crop type codes.

        Returns None if crop_type was not provided.
        """
        if self._crop_type is None:
            return None

        crop_type = np.asarray(self._crop_type)

        # Map CDL codes to crop classes
        crop_class = np.zeros_like(crop_type)
        for cdl_code, params in CDL.items():
            mask = crop_type == cdl_code
            crop_class = np.where(mask, params.get('crop_class', 0), crop_class)

        if isinstance(self._crop_type, xr.DataArray):
            return xr.DataArray(
                crop_class,
                coords=self._crop_type.coords,
                dims=self._crop_type.dims
            )
        return crop_class

    def _get_crop_param_array(self, param_name: str, default: float = 1.0) -> np.ndarray:
        """Get array of crop parameters from crop type codes."""
        if self._crop_type is None:
            return None

        crop_type = np.asarray(self._crop_type)
        param_array = np.full_like(crop_type, default, dtype=float)

        for cdl_code, params in CDL.items():
            mask = crop_type == cdl_code
            param_array = np.where(mask, params.get(param_name, default), param_array)

        return param_array

    @cached_property
    def et_fraction(self) -> xr.DataArray:
        """
        ET fraction (crop coefficient) using the SIMS algorithm.

        This is the main output of the SIMS algorithm.
        ETf = Kc where ET = Kc × ET_reference.
        """
        ndvi = self.ndvi.values

        if self._crop_type is None:
            # Use generic Kc if no crop type provided
            etf = model.kc_generic(ndvi)
        else:
            crop_class = self.crop_class
            if isinstance(crop_class, xr.DataArray):
                crop_class = crop_class.values

            # Get crop-specific parameters if using detailed Kc
            if self._use_crop_type_kc:
                h_max = self._get_crop_param_array('h_max', 1.0)
                m_l = self._get_crop_param_array('m_l', 2.0)
                fr_mid = self._get_crop_param_array('fr_mid', 1.0)
                fr_end = self._get_crop_param_array('fr_end', 1.0)
                ls_start = self._get_crop_param_array('ls_start', 1)
                ls_stop = self._get_crop_param_array('ls_stop', 365)

                etf = model.et_fraction(
                    ndvi=ndvi,
                    crop_class=crop_class,
                    doy=self.doy,
                    h_max=h_max,
                    m_l=m_l,
                    fr_mid=fr_mid,
                    fr_end=fr_end,
                    ls_start=ls_start,
                    ls_stop=ls_stop,
                    reflectance_type='SR',
                    use_crop_type_kc=True,
                )
            else:
                etf = model.et_fraction(
                    ndvi=ndvi,
                    crop_class=crop_class,
                    reflectance_type='SR',
                    use_crop_type_kc=False,
                )

        return xr.DataArray(etf, coords=self.ndvi.coords, dims=self.ndvi.dims)

    @cached_property
    def kc(self) -> xr.DataArray:
        """Crop coefficient (alias for et_fraction)."""
        return self.et_fraction

    def compute_et(self, et_reference: Union[float, np.ndarray, xr.DataArray]) -> xr.DataArray:
        """
        Compute actual ET from ET fraction and reference ET.

        Parameters
        ----------
        et_reference : float, array-like, or xr.DataArray
            Reference evapotranspiration (mm/day or mm/period).

        Returns
        -------
        xr.DataArray
            Actual ET in same units as et_reference.
        """
        if isinstance(et_reference, (int, float)):
            et = self.et_fraction * et_reference
        else:
            et = model.compute_et(self.et_fraction.values, np.asarray(et_reference))
            et = xr.DataArray(et, coords=self.et_fraction.coords, dims=self.et_fraction.dims)

        return et

    def calculate(self, variables: Optional[list] = None) -> xr.Dataset:
        """
        Calculate multiple variables and return as a Dataset.

        Parameters
        ----------
        variables : list of str, optional
            Variables to include. If None, includes common variables.
            Options: 'ndvi', 'fc', 'et_fraction', 'kc', 'lai', 'albedo',
                     'lst', 'qa_mask', 'emissivity', 'evi'

        Returns
        -------
        xr.Dataset
            Dataset containing requested variables.
        """
        if variables is None:
            variables = ['ndvi', 'fc', 'et_fraction', 'qa_mask']

        data_vars = {}
        for var in variables:
            if hasattr(self, var):
                data_vars[var] = getattr(self, var)

        ds = xr.Dataset(data_vars)
        ds.attrs['scene_id'] = self.scene_id
        ds.attrs['acquisition_date'] = str(self.date)
        ds.attrs['spacecraft_id'] = self.spacecraft_id

        return ds
