#!/usr/bin/env python3
"""
Crane Site Validation: Test SIMS algorithm against known EE results.

This script downloads Landsat imagery from a STAC provider (MPC or AWS) for
the Crane flux site (S2) and compares computed ETf values against known
OpenET results.

The Crane site is an irrigated alfalfa field in Oregon (Path 043, Row 030).

Usage:
    python test_crane_validation.py          # Use MPC (default)
    python test_crane_validation.py --aws    # Use AWS Earth Search
    python test_crane_validation.py --mpc    # Use MPC explicitly

Note: AWS requires valid AWS credentials with billing configured because the
USGS Landsat bucket is a requester-pays bucket. MPC (Planetary Computer) is
recommended for users without AWS accounts.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

from sims import STACSource, LandsatImage
from sims.model import et_fraction

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Test fixtures path
FIXTURES_DIR = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'crane'

# Known ETf values from OpenET/EE for mid-summer 2020
# Format: (short_scene_id, expected_etf)
VALIDATION_SCENES = [
    ('LC08_043030_20200701', 0.9396),
    ('LC08_043030_20200717', 0.7561),
    ('LC08_043030_20200802', 0.9987),
    ('LC08_043030_20200818', 1.0517),
]

# Crane site approximate location (from flux_fields.shp converted to WGS84)
CRANE_LON = -118.6141
CRANE_LAT = 43.4174

# CDL code for alfalfa (the crop at Crane S2)
ALFALFA_CDL = 36


def load_crane_geometry():
    """Load Crane site geometry from shapefile."""
    shp_path = FIXTURES_DIR / 'flux_fields.shp'
    if shp_path.exists():
        gdf = gpd.read_file(shp_path)
        # Convert to WGS84
        gdf = gdf.to_crs('EPSG:4326')
        # Get centroid and buffer for ~500m around point
        centroid = gdf.geometry.iloc[0]
        if centroid.geom_type == 'Point':
            # Buffer the point to create an area
            return centroid.buffer(0.005)  # ~500m buffer
        return centroid
    else:
        # Fallback to known coordinates
        print(f"Shapefile not found at {shp_path}, using known coordinates")
        return Point(CRANE_LON, CRANE_LAT).buffer(0.005)


def short_to_stac_id(short_id: str) -> str:
    """
    Convert short scene ID to STAC search pattern.

    Short: LC08_043030_20200701
    STAC:  LC08_L2SP_043030_20200701_*
    """
    parts = short_id.split('_')
    sensor = parts[0]
    pathrow = parts[1]
    date = parts[2]

    # Return pattern for searching
    return f"{sensor}_L2SP_{pathrow}_{date}"


def search_scene(source: STACSource, short_id: str, geometry) -> str:
    """Search for a specific scene on STAC provider and return full STAC ID."""
    parts = short_id.split('_')
    sensor = parts[0]
    date = parts[2]

    # Format date for search
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
    date_str = f"{year}-{month}-{day}"

    # Search for scenes on that date
    scene_ids = source.search_scenes(
        geometry=geometry,
        start_date=date_str,
        end_date=date_str,
        max_cloud_cover=100,  # Accept any cloud cover for validation
    )

    # Find matching sensor
    for sid in scene_ids:
        if sid.startswith(sensor):
            return sid

    return None


def run_validation(provider: str = 'mpc'):
    """Run validation against known ETf values.

    Parameters
    ----------
    provider : str
        STAC provider to use: 'mpc' (Planetary Computer) or 'aws' (Earth Search)
    """
    provider_names = {
        'mpc': 'Microsoft Planetary Computer',
        'aws': 'AWS Earth Search',
    }

    print("=" * 70)
    print("SIMS Crane Site Validation")
    print(f"Provider: {provider_names.get(provider, provider)}")
    print("=" * 70)

    # Load geometry
    print("\nLoading Crane site geometry...")
    geometry = load_crane_geometry()
    print(f"  Geometry bounds: {geometry.bounds}")

    # Connect to STAC provider
    print(f"\nConnecting to {provider_names.get(provider, provider)}...")
    source = STACSource(provider=provider)

    # Process each validation scene
    results = []

    for short_id, expected_etf in VALIDATION_SCENES:
        print(f"\n{'=' * 50}")
        print(f"Processing: {short_id}")
        print(f"Expected ETf: {expected_etf:.4f}")

        # Find scene on STAC provider
        scene_id = search_scene(source, short_id, geometry)

        if scene_id is None:
            print(f"  WARNING: Scene not found on {provider}")
            results.append({
                'short_id': short_id,
                'scene_id': None,
                'expected_etf': expected_etf,
                'computed_etf': None,
                'difference': None,
                'status': 'NOT_FOUND'
            })
            continue

        print(f"  STAC ID: {scene_id}")

        try:
            # Create LandsatImage with alfalfa crop type for correct Kc
            image = LandsatImage(
                source=source,
                scene_id=scene_id,
                geometry=geometry,
                crop_type=ALFALFA_CDL,  # Row crop formula for alfalfa
            )

            print(f"  Date: {image.date}")
            print(f"  Spacecraft: {image.spacecraft_id}")

            # Compute NDVI
            ndvi = image.ndvi
            ndvi_mean = float(ndvi.mean())
            print(f"  NDVI mean: {ndvi_mean:.4f}")

            # Compute ET fraction
            etf = image.et_fraction
            etf_masked = image.apply_cloud_mask(etf)

            # Get mean ETf for the site
            etf_mean = float(np.nanmean(etf_masked.values))
            print(f"  Computed ETf: {etf_mean:.4f}")

            # Compare
            diff = etf_mean - expected_etf
            pct_diff = (diff / expected_etf) * 100 if expected_etf != 0 else 0
            print(f"  Difference: {diff:+.4f} ({pct_diff:+.1f}%)")

            status = 'PASS' if abs(pct_diff) < 10 else 'FAIL'
            print(f"  Status: {status}")

            results.append({
                'short_id': short_id,
                'scene_id': scene_id,
                'expected_etf': expected_etf,
                'computed_etf': etf_mean,
                'difference': diff,
                'pct_diff': pct_diff,
                'ndvi_mean': ndvi_mean,
                'status': status
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'short_id': short_id,
                'scene_id': scene_id,
                'expected_etf': expected_etf,
                'computed_etf': None,
                'difference': None,
                'status': f'ERROR: {e}'
            })

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Statistics
    valid = df[df['computed_etf'].notna()]
    if len(valid) > 0:
        print(f"\nMean absolute difference: {valid['difference'].abs().mean():.4f}")
        print(f"Mean percent difference: {valid['pct_diff'].abs().mean():.1f}%")
        print(f"Pass rate: {(valid['status'] == 'PASS').sum()}/{len(valid)}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Validate SIMS ETf against known OpenET values for Crane site'
    )
    parser.add_argument(
        '--mpc', action='store_const', const='mpc', dest='provider',
        help='Use Microsoft Planetary Computer (default)'
    )
    parser.add_argument(
        '--aws', action='store_const', const='aws', dest='provider',
        help='Use AWS Earth Search'
    )
    parser.set_defaults(provider='mpc')

    args = parser.parse_args()
    run_validation(provider=args.provider)
