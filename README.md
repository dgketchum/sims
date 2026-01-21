# SIMS - Satellite Irrigation Management Support

A standalone Python package for computing ET fractions using the SIMS algorithm.

## Installation

```bash
pip install -e .
```

For Planetary Computer support:
```bash
pip install pystac-client planetary-computer stackstac rioxarray
```

## Quick Start

### Process a single Landsat scene from Planetary Computer

```python
from shapely.geometry import box
from sims import STACSource, LandsatImage

# Define area of interest
aoi = box(-120.5, 36.8, -120.4, 36.9)  # lon_min, lat_min, lon_max, lat_max

# Connect to Planetary Computer
source = STACSource()

# Search for scenes
scene_ids = source.search_scenes(
    geometry=aoi,
    start_date='2023-06-01',
    end_date='2023-06-30',
    max_cloud_cover=20.0
)

# Process first scene
image = LandsatImage(source, scene_ids[0], geometry=aoi)

# Get ET fraction
etf = image.et_fraction
print(f"Mean ETf: {float(etf.mean()):.3f}")

# Compute actual ET
from sims import compute_et
et = compute_et(etf.values, et_reference=6.0)  # 6 mm/day reference ET
```

### Convenience functions

```python
from shapely.geometry import box
from sims import process_scene, search_and_process

aoi = box(-120.5, 36.8, -120.4, 36.9)

# Process a single scene with reference ET
result = process_scene(
    scene_id='LC09_L2SP_042034_20230615_02_T1',
    geometry=aoi,
    et_reference=6.0,  # mm/day
)

print(f"Date: {result['date']}")
print(f"Mean ET: {float(result['et'].mean()):.2f} mm/day")

# Search and process multiple scenes
results = search_and_process(
    geometry=aoi,
    start_date='2023-06-01',
    end_date='2023-08-31',
    max_cloud_cover=30.0,
    et_reference=6.0,
    max_scenes=10,
)

for r in results:
    print(f"{r['date']}: ETf={float(r['et_fraction'].mean()):.3f}")
```

### Using crop type from CDL

```python
from sims import STACSource, LandsatImage
from sims.ancillary import get_crop_type

# Get crop type for the area
crop_type = get_crop_type(aoi, year=2023)

# Process with crop-specific coefficients
image = LandsatImage(
    source=STACSource(),
    scene_id=scene_ids[0],
    geometry=aoi,
    crop_type=crop_type,
    use_crop_type_kc=False,  # Use class-based Kc
)

etf = image.et_fraction
```

### Getting reference ET from GridMET

```python
from sims.ancillary import get_reference_et_for_scene

# Get reference ET for a scene date
etr = get_reference_et_for_scene(
    geometry=aoi,
    date='2023-06-15',
    variable='etr',  # ASCE grass reference ET
)
print(f"Reference ET: {etr:.2f} mm/day")
```

## Algorithm

SIMS computes crop coefficients (Kc) from NDVI using crop-type-specific equations:

1. **Fraction of Cover**: `Fc = 1.26 × NDVI - 0.18`
2. **Crop Coefficient by class**:
   - Row crops: `Kc = -0.4771×Fc² + 1.4047×Fc + 0.15`
   - Trees: `Kc = 1.48×Fc + 0.007`
   - Vines: Density-based with seasonal adjustment
3. **Actual ET**: `ET = Kc × ET_reference`

## Package Structure

```
sims/
├── model.py          # Core algorithm (pure numpy)
├── data.py           # CDL crop coefficients
├── image.py          # LandsatImage class
├── run.py            # Convenience runners
├── api.py            # High-level API
├── zonal.py          # Zonal statistics
├── sources/
│   ├── base.py       # DataSource ABC
│   ├── local.py      # Local file source
│   └── stac.py       # STAC/Planetary Computer
└── ancillary/
    ├── gridmet.py    # Reference ET
    └── cdl.py        # Crop type data
```

## References

- Melton, F. S., et al. (2012). Satellite Irrigation Management Support. IEEE JSTARS.
- Allen, R. G., & Pereira, L. S. (2009). Estimating crop coefficients. Irrigation Science.
