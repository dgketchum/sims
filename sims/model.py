"""
SIMS Algorithm Layer - Pure NumPy/XArray math operations.

This module contains the core SIMS algorithm for computing crop coefficients (Kc)
from NDVI and crop type information. All functions operate on numpy arrays or
xarray DataArrays and have no I/O dependencies.

The ET fraction (ETf) is equivalent to the crop coefficient (Kc) in the SIMS
framework: ETf = Kc = ET / ET_reference.

This is a NumPy port of the Google Earth Engine implementation from openet-sims:
https://github.com/Open-ET/openet-sims (Apache 2.0 license)

References
----------
Melton, F. S., et al. (2012). Satellite Irrigation Management Support with the
Terrestrial Observation and Prediction System. IEEE Journal of Selected Topics
in Applied Earth Observations and Remote Sensing, 5(6), 1709-1721.
https://doi.org/10.1109/JSTARS.2012.2214474

Pereira, L. S., et al. (2020). Prediction of basal crop coefficients from
fraction of ground cover and height. Agricultural Water Management.
https://doi.org/10.1016/j.agwat.2020.106197

Allen, R. G., & Pereira, L. S. (2009). Estimating crop coefficients from
fraction of ground cover and height. Irrigation Science, 28(1), 17-34.
"""

from typing import Union, Optional
import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray, float]


def et_fraction(
    ndvi: ArrayLike,
    crop_class: ArrayLike,
    doy: Optional[ArrayLike] = None,
    h_max: Optional[ArrayLike] = None,
    m_l: Optional[ArrayLike] = None,
    fr_mid: Optional[ArrayLike] = None,
    fr_end: Optional[ArrayLike] = None,
    ls_start: Optional[ArrayLike] = None,
    ls_stop: Optional[ArrayLike] = None,
    reflectance_type: str = 'SR',
    use_crop_type_kc: bool = False,
    mask_non_ag: bool = True,
    water_kc_flag: bool = True,
) -> ArrayLike:
    """
    Compute SIMS ET fraction (crop coefficient) from NDVI and crop type.

    This is the main entry point for the SIMS algorithm. It applies the appropriate
    Kc calculation based on crop class and returns the ET fraction.

    Parameters
    ----------
    ndvi : array-like
        Normalized Difference Vegetation Index (-1 to 1).
    crop_class : array-like
        Crop class code (0=non-ag, 1=row, 2=vine, 3=tree, 5=rice, 6=fallow, 7=grass).
    doy : array-like, optional
        Day of year (1-366). Required for vine/tree crops with seasonal adjustments.
    h_max : array-like, optional
        Maximum plant height in meters. Required for vine crops and tree/vine
        height-based seasonal adjustments.
    m_l : array-like, optional
        Density coefficient. Required for height-based row/tree adjustments.
    fr_mid : array-like, optional
        Mid-season reduction factor (default: 1.0).
    fr_end : array-like, optional
        End-season reduction factor.
    ls_start : array-like, optional
        Day of year when leaf senescence starts.
    ls_stop : array-like, optional
        Day of year when leaf senescence ends.
    reflectance_type : str
        Reflectance type: 'SR' (surface reflectance) or 'TOA' (top of atmosphere).
    use_crop_type_kc : bool
        If True, use crop-type-specific Kc with height/density parameters.
        If False, use generic crop-class Kc equations.
    mask_non_ag : bool
        If True, mask non-agricultural pixels (crop_class == 0) as NaN.
    water_kc_flag : bool
        If True, set Kc=1.05 for water pixels (ndvi < 0 and crop_class == 0)
        before optional masking.

    Returns
    -------
    array-like
        ET fraction (Kc) values, typically 0-1.2 range.

    Notes
    -----
    The function routes to different Kc calculations based on crop_class:
    - Class 0: Returns NaN (non-agricultural)
    - Class 1: Row crop equation (Melton et al. 2012)
    - Class 2: Vine equation with seasonal adjustment
    - Class 3: Tree equation (Ayars et al. 2003)
    - Class 5: Rice equation
    - Class 6: Fallow equation
    - Class 7: Grass/pasture equation
    """
    # Convert inputs to numpy arrays for consistent handling
    ndvi = np.asarray(ndvi)
    crop_class = np.asarray(crop_class)

    # Compute fraction of cover
    fc = fraction_of_cover(ndvi, reflectance_type)

    # Initialize output with generic Kc as default
    kc = kc_generic(ndvi)

    # Create masks for each crop class
    is_class_1 = crop_class == 1  # Row crops
    is_class_2 = crop_class == 2  # Vines
    is_class_3 = crop_class == 3  # Trees
    is_class_5 = crop_class == 5  # Rice
    is_class_6 = crop_class == 6  # Fallow
    is_class_7 = crop_class == 7  # Grass/pasture
    is_class_0 = crop_class == 0  # Non-agricultural

    # Apply crop-class-specific Kc calculations
    if np.any(is_class_1):
        row_generic = kc_row_crop(fc)
        if use_crop_type_kc and h_max is not None and m_l is not None:
            valid = (~np.isnan(h_max)) & (~np.isnan(m_l))
            kc_height = kc_row_crop_with_height(fc, h_max, m_l, fr_mid)
            kc = np.where(is_class_1 & valid, kc_height, row_generic)
        else:
            kc = np.where(is_class_1, row_generic, kc)

    if np.any(is_class_2):
        required = [h_max, doy, fr_mid, fr_end, ls_start, ls_stop]
        if any(v is None for v in required):
            raise ValueError("Vine crops require h_max, doy, fr_mid, fr_end, ls_start, and ls_stop.")
        vine_valid = (
            (~np.isnan(h_max)) &
            (~np.isnan(fr_mid)) &
            (~np.isnan(fr_end)) &
            (~np.isnan(ls_start)) &
            (~np.isnan(ls_stop))
        )
        kc_vine_val = kc_vine(fc, h_max, doy, fr_mid, fr_end, ls_start, ls_stop)
        kc = np.where(is_class_2 & vine_valid, kc_vine_val, kc)
        if mask_non_ag:
            kc = np.where(is_class_2 & ~vine_valid, np.nan, kc)

    if np.any(is_class_3):
        if use_crop_type_kc and all(v is not None for v in [h_max, m_l, doy, fr_mid, fr_end, ls_start, ls_stop]):
            tree_valid = (
                (~np.isnan(h_max)) &
                (~np.isnan(m_l)) &
                (~np.isnan(fr_mid)) &
                (~np.isnan(fr_end)) &
                (~np.isnan(ls_start)) &
                (~np.isnan(ls_stop))
            )
            kc_tree_height = kc_tree_with_height(fc, h_max, m_l, doy, fr_mid, fr_end, ls_start, ls_stop)
            kc = np.where(is_class_3 & tree_valid, kc_tree_height, kc_tree(fc))
        else:
            kc = np.where(is_class_3, kc_tree(fc), kc)

    if np.any(is_class_5):
        kc = np.where(is_class_5, kc_rice(fc, ndvi), kc)

    if np.any(is_class_6):
        kc = np.where(is_class_6, kc_fallow(fc, ndvi), kc)

    if np.any(is_class_7):
        kc = np.where(is_class_7, kc_grass_pasture(fc, ndvi), kc)

    if water_kc_flag:
        kc = np.where(is_class_0 & (ndvi < 0), 1.05, kc)

    # Mask non-agricultural pixels
    if mask_non_ag:
        kc = np.where(is_class_0, np.nan, kc)

    return kc


def fraction_of_cover(ndvi: ArrayLike, reflectance_type: str = 'SR') -> ArrayLike:
    """
    Convert NDVI to fraction of cover (Fc).

    Parameters
    ----------
    ndvi : array-like
        Normalized Difference Vegetation Index (-1 to 1).
    reflectance_type : str
        'SR' for surface reflectance or 'TOA' for top of atmosphere.

    Returns
    -------
    array-like
        Fraction of cover, clamped to [0, 1].

    Notes
    -----
    The conversion coefficients differ based on reflectance type:
    - Surface Reflectance: Fc = 1.26 * NDVI - 0.18
    - Top of Atmosphere: Fc = 1.465 * NDVI - 0.139
    """
    ndvi = np.asarray(ndvi)

    if reflectance_type.upper() == 'SR':
        fc = 1.26 * ndvi - 0.18
    elif reflectance_type.upper() == 'TOA':
        fc = 1.465 * ndvi - 0.139
    else:
        raise ValueError(f"Unknown reflectance_type: {reflectance_type}. Use 'SR' or 'TOA'.")

    # Clamp to [0, 1]
    return np.clip(fc, 0, 1)


def kc_generic(ndvi: ArrayLike) -> ArrayLike:
    """
    Compute generic crop coefficient from NDVI.

    This is the fallback Kc calculation when crop-specific equations are not available.

    Parameters
    ----------
    ndvi : array-like
        Normalized Difference Vegetation Index (-1 to 1).

    Returns
    -------
    array-like
        Crop coefficient (Kc), minimum 0.

    Notes
    -----
    Equation: Kc = 1.25 * NDVI + 0.2
    """
    ndvi = np.asarray(ndvi)
    return np.maximum(1.25 * ndvi + 0.2, 0)


def kc_row_crop(fc: ArrayLike) -> ArrayLike:
    """
    Compute crop coefficient for row crops (Class 1).

    Uses the quadratic Fc-Kc relationship from Melton et al. (2012).

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).

    Returns
    -------
    array-like
        Crop coefficient (Kc).

    Notes
    -----
    Equation: Kc = -0.4771 * Fc^2 + 1.4047 * Fc + 0.15

    Reference: Melton, F. S., et al. (2012). Satellite Irrigation Management
    Support. IEEE JSTARS, 5(6), 1709-1721.
    """
    fc = np.asarray(fc)
    return -0.4771 * fc**2 + 1.4047 * fc + 0.15


def kc_row_crop_with_height(
    fc: ArrayLike,
    h_max: ArrayLike,
    m_l: ArrayLike,
    fr_mid: Optional[ArrayLike] = None,
    kc_min: float = 0.15,
) -> ArrayLike:
    """
    Compute crop coefficient for row crops using plant height parameters.

    Uses the Allen & Pereira (2009) methodology with density coefficients.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    h_max : array-like
        Maximum plant height (meters).
    m_l : array-like
        Density coefficient.
    fr_mid : array-like, optional
        Mid-season reduction factor (default: 1.0).
    kc_min : float
        Minimum Kc value (default: 0.15).

    Returns
    -------
    array-like
        Crop coefficient (Kc).
    """
    fc = np.asarray(fc)
    h_max = np.asarray(h_max)
    m_l = np.asarray(m_l)
    fr_mid = np.asarray(fr_mid) if fr_mid is not None else np.ones_like(fc)

    # Compute density coefficient (Kd) for row crops
    kd = _kd_row_crop(fc, h_max, m_l)

    # Compute basal Kc at full cover
    kcb_full = np.minimum(0.1 * h_max + 1.0, 1.2) * fr_mid

    # Final Kcb
    return kd * (kcb_full - kc_min) + kc_min


def kc_tree(fc: ArrayLike) -> ArrayLike:
    """
    Compute crop coefficient for tree crops (Class 3).

    Uses the linear Fc-Kc relationship from Ayars et al. (2003).

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).

    Returns
    -------
    array-like
        Crop coefficient (Kc).

    Notes
    -----
    Equation: Kc = 1.48 * Fc + 0.007

    Reference: Ayars, J. E., et al. (2003). Using weighing lysimeters to develop
    evapotranspiration crop coefficients. Irrigation Science, 22(1), 1-9.
    """
    fc = np.asarray(fc)
    return 1.48 * fc + 0.007


def kc_tree_with_height(
    fc: ArrayLike,
    h_max: ArrayLike,
    m_l: ArrayLike,
    doy: ArrayLike,
    fr_mid: ArrayLike,
    fr_end: ArrayLike,
    ls_start: ArrayLike,
    ls_stop: ArrayLike,
    kc_min: float = 0.15,
) -> ArrayLike:
    """
    Compute crop coefficient for tree crops using seasonal adjustment.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    h_max : array-like
        Maximum plant height (meters).
    m_l : array-like
        Density coefficient.
    doy : array-like
        Day of year (1-366).
    fr_mid : array-like
        Mid-season reduction factor.
    fr_end : array-like
        End-season reduction factor.
    ls_start : array-like
        Day of year when leaf senescence starts.
    ls_stop : array-like
        Day of year when leaf senescence ends.
    kc_min : float
        Minimum Kc value (default: 0.15).

    Returns
    -------
    array-like
        Crop coefficient (Kc).
    """
    fc = np.asarray(fc)
    h_max = np.asarray(h_max)
    m_l = np.asarray(m_l)
    doy = np.asarray(doy)
    fr_mid = np.asarray(fr_mid)
    fr_end = np.asarray(fr_end)
    ls_start = np.asarray(ls_start)
    ls_stop = np.asarray(ls_stop)

    # Compute density coefficient for trees
    kd = _kd_tree(fc, h_max, m_l)

    # Compute seasonal reduction factor
    fr = _compute_fr(doy, fr_mid, fr_end, ls_start, ls_stop)

    # Compute basal Kc at full cover
    kcb_full = np.minimum(0.1 * h_max + 1.0, 1.2) * fr

    # Final Kcb
    return kd * (kcb_full - kc_min) + kc_min


def kc_vine(
    fc: ArrayLike,
    h_max: ArrayLike,
    doy: ArrayLike,
    fr_mid: ArrayLike,
    fr_end: ArrayLike,
    ls_start: ArrayLike,
    ls_stop: ArrayLike,
    kc_min: float = 0.15,
) -> ArrayLike:
    """
    Compute crop coefficient for vine crops (Class 2).

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    h_max : array-like
        Maximum plant height (meters).
    doy : array-like
        Day of year (1-366).
    fr_mid : array-like
        Mid-season reduction factor.
    fr_end : array-like
        End-season reduction factor.
    ls_start : array-like
        Day of year when leaf senescence starts.
    ls_stop : array-like
        Day of year when leaf senescence ends.
    kc_min : float
        Minimum Kc value (default: 0.15).

    Returns
    -------
    array-like
        Crop coefficient (Kc).
    """
    fc = np.asarray(fc)
    h_max = np.asarray(h_max)
    doy = np.asarray(doy)
    fr_mid = np.asarray(fr_mid)
    fr_end = np.asarray(fr_end)
    ls_start = np.asarray(ls_start)
    ls_stop = np.asarray(ls_stop)

    # Compute density coefficient for vines (Allen & Pereira vine formulation)
    kd = _kd_vine(fc)

    # Compute seasonal reduction factor
    fr = _compute_fr(doy, fr_mid, fr_end, ls_start, ls_stop)

    # Compute basal Kc at full cover
    kcb_full = np.minimum(0.1 * h_max + 1.0, 1.2) * fr

    # Final Kcb
    return kd * (kcb_full - kc_min) + kc_min


def kc_rice(fc: ArrayLike, ndvi: ArrayLike) -> ArrayLike:
    """
    Compute crop coefficient for rice (Class 5).

    Uses the row crop equation with special handling for low NDVI
    (flooded conditions).

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    ndvi : array-like
        Normalized Difference Vegetation Index.

    Returns
    -------
    array-like
        Crop coefficient (Kc).

    Notes
    -----
    For rice, low NDVI indicates flooded conditions where Kc should be set
    to 1.05 to account for open water evaporation.
    """
    fc = np.asarray(fc)
    ndvi = np.asarray(ndvi)

    # Base row crop equation
    kc = kc_row_crop(fc)

    # Adjust for flooded conditions (low NDVI but high Kc)
    # When NDVI <= 0.14, set Kc to 1.05 to account for standing water
    kc = np.where(ndvi <= 0.14, 1.05, kc)

    return kc


def kc_fallow(fc: ArrayLike, ndvi: ArrayLike) -> ArrayLike:
    """
    Compute crop coefficient for fallow land (Class 6).

    Uses the row crop equation with special handling for low vegetation.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    ndvi : array-like
        Normalized Difference Vegetation Index.

    Returns
    -------
    array-like
        Crop coefficient (Kc).

    Notes
    -----
    For fallow land with low NDVI, Kc is set to the fraction of cover (fc)
    to better represent sparse vegetation conditions.
    """
    fc = np.asarray(fc)
    ndvi = np.asarray(ndvi)

    # Base row crop equation
    kc = kc_row_crop(fc)

    # For low NDVI, set Kc to fc (with minimum of 0.01)
    kc = np.where(ndvi <= 0.35, np.maximum(fc, 0.01), kc)

    return kc


def kc_grass_pasture(fc: ArrayLike, ndvi: ArrayLike) -> ArrayLike:
    """
    Compute crop coefficient for grass/pasture (Class 7).

    Uses the row crop equation with special handling for low vegetation.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    ndvi : array-like
        Normalized Difference Vegetation Index.

    Returns
    -------
    array-like
        Crop coefficient (Kc).

    Notes
    -----
    For grass/pasture with low NDVI (dormant or sparse), Kc is set to the
    fraction of cover (fc) to better represent reduced transpiration.
    """
    fc = np.asarray(fc)
    ndvi = np.asarray(ndvi)

    # Base row crop equation
    kc = kc_row_crop(fc)

    # For low NDVI (dormant grass), set Kc to fc (with minimum of 0.01)
    kc = np.where(ndvi <= 0.35, np.maximum(fc, 0.01), kc)

    return kc


def _kd_row_crop(fc: ArrayLike, h_max: ArrayLike, m_l: ArrayLike) -> ArrayLike:
    """
    Compute density coefficient (Kd) for row crops.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    h_max : array-like
        Maximum plant height (meters).
    m_l : array-like
        Density coefficient.

    Returns
    -------
    array-like
        Density coefficient Kd (0-1).

    Notes
    -----
    From Allen & Pereira (2009), the density coefficient accounts for
    the relationship between canopy fraction and effective transpiration.
    """
    fc = np.asarray(fc)
    h_max = np.asarray(h_max)
    m_l = np.asarray(m_l)

    # Compute relative cover fraction
    fc_rel = fc / 0.7

    # Two cases based on relative cover fraction
    # Case 1: fc_rel <= 1
    kd_low = np.minimum(fc * m_l, fc ** (1 / (1 + h_max * fc_rel)))
    # Case 2: fc_rel > 1
    kd_high = np.minimum(fc * m_l, fc ** (1 / (1 + h_max)))

    kd = np.where(fc_rel > 1, kd_high, kd_low)

    return np.minimum(kd, 1.0)


def _kd_tree(fc: ArrayLike, h_max: ArrayLike, m_l: ArrayLike) -> ArrayLike:
    """
    Compute density coefficient (Kd) for tree crops.

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).
    h_max : array-like
        Maximum plant height (meters).
    m_l : array-like
        Density coefficient.

    Returns
    -------
    array-like
        Density coefficient Kd (0-1).
    """
    fc = np.asarray(fc)
    h_max = np.asarray(h_max)
    m_l = np.asarray(m_l)

    # Two cases based on fc threshold
    # Case 1: fc > 0.5
    kd_high = np.minimum(fc * m_l, fc ** (1 / (1 + h_max)))
    # Case 2: fc <= 0.5
    kd_low = np.minimum(fc * m_l, fc ** (1 / h_max))

    kd = np.where(fc > 0.5, kd_high, kd_low)

    return np.minimum(kd, 1.0)


def _kd_vine(fc: ArrayLike) -> ArrayLike:
    """
    Compute density coefficient (Kd) for vine crops.

    Uses the vine formulation from Williams & Ayars (2005).

    Parameters
    ----------
    fc : array-like
        Fraction of cover (0-1).

    Returns
    -------
    array-like
        Density coefficient Kd (0-1).
    """
    fc = np.asarray(fc)
    return np.minimum(fc * 1.5, np.minimum(fc ** (1 / (1 + 2)), 1.0))


def _compute_fr(
    doy: ArrayLike,
    fr_mid: ArrayLike,
    fr_end: ArrayLike,
    ls_start: ArrayLike,
    ls_stop: ArrayLike,
) -> ArrayLike:
    """
    Compute seasonal reduction factor (fr) for basal Kc.

    The reduction factor varies linearly from fr_mid to fr_end during the
    leaf senescence period (ls_start to ls_stop).

    Parameters
    ----------
    doy : array-like
        Day of year (1-366).
    fr_mid : array-like
        Mid-season reduction factor.
    fr_end : array-like
        End-season reduction factor.
    ls_start : array-like
        Day of year when leaf senescence starts.
    ls_stop : array-like
        Day of year when leaf senescence ends.

    Returns
    -------
    array-like
        Reduction factor fr, clamped between fr_end and fr_mid.
    """
    doy = np.asarray(doy)
    fr_mid = np.asarray(fr_mid)
    fr_end = np.asarray(fr_end)
    ls_start = np.asarray(ls_start)
    ls_stop = np.asarray(ls_stop)

    # Linear interpolation during senescence period
    # fr decreases from fr_mid to fr_end as DOY goes from ls_start to ls_stop
    fr = (ls_start - doy) * (fr_mid - fr_end) / (ls_stop - ls_start) + fr_mid

    # Clamp to valid range
    return np.clip(fr, fr_end, fr_mid)


def compute_et(kc: ArrayLike, et_reference: ArrayLike) -> ArrayLike:
    """
    Compute actual ET from crop coefficient and reference ET.

    Parameters
    ----------
    kc : array-like
        Crop coefficient (ET fraction).
    et_reference : array-like
        Reference evapotranspiration (mm/day or mm/period).

    Returns
    -------
    array-like
        Actual evapotranspiration in same units as et_reference.

    Notes
    -----
    Simple multiplication: ET = Kc × ET_reference
    """
    return np.asarray(kc) * np.asarray(et_reference)
