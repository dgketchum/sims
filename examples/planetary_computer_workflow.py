#!/usr/bin/env python3
"""
Example: SIMS ET Computation using Microsoft Planetary Computer

This script demonstrates the complete workflow for computing ET fractions
and actual ET using Landsat imagery from Planetary Computer.

Requirements:
    pip install pystac-client planetary-computer stackstac rioxarray geopandas

Usage:
    python planetary_computer_workflow.py
"""

import warnings
from datetime import datetime

import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point

# SIMS imports
from sims import STACSource, LandsatImage, compute_et
from sims.ancillary import get_crop_type, get_reference_et_for_scene


def run_single_scene_example():
    """
    Process a single Landsat scene and compute ET fraction.
    """
    print("=" * 60)
    print("Example 1: Single Scene Processing")
    print("=" * 60)

    # Define area of interest (Central Valley, CA - agricultural area)
    # This is a small area for demonstration
    aoi = box(-120.5, 36.8, -120.4, 36.9)
    print(f"\nArea of Interest: {aoi.bounds}")

    # Initialize Planetary Computer data source
    source = STACSource(
        catalog_url='https://planetarycomputer.microsoft.com/api/stac/v1',
        collection='landsat-c2-l2'
    )
    print("Connected to Planetary Computer")

    # Search for scenes
    print("\nSearching for Landsat scenes...")
    scene_ids = source.search_scenes(
        geometry=aoi,
        start_date='2023-06-01',
        end_date='2023-06-30',
        max_cloud_cover=70.0
    )

    if not scene_ids:
        print("No scenes found. Try adjusting date range or cloud cover threshold.")
        return

    print(f"Found {len(scene_ids)} scenes:")
    for sid in scene_ids[:5]:  # Show first 5
        print(f"  - {sid}")

    # Process first scene
    scene_id = scene_ids[0]
    print(f"\nProcessing scene: {scene_id}")

    # Create LandsatImage object
    image = LandsatImage(
        source=source,
        scene_id=scene_id,
        geometry=aoi,
    )

    print(f"  Acquisition date: {image.date}")
    print(f"  Spacecraft: {image.spacecraft_id}")
    print(f"  Day of year: {image.doy}")

    # Compute NDVI (triggers lazy load of bands)
    print("\nComputing NDVI...")
    ndvi = image.ndvi
    print(f"  NDVI shape: {ndvi.shape}")
    print(f"  NDVI range: {float(ndvi.min()):.3f} to {float(ndvi.max()):.3f}")

    # Compute ET fraction (crop coefficient)
    print("\nComputing ET fraction...")
    etf = image.et_fraction
    print(f"  ETf shape: {etf.shape}")
    print(f"  ETf range: {float(etf.min()):.3f} to {float(etf.max()):.3f}")
    print(f"  ETf mean: {float(etf.mean()):.3f}")

    # Apply cloud mask
    print("\nApplying cloud mask...")
    etf_masked = image.apply_cloud_mask(etf)
    valid_fraction = float((~np.isnan(etf_masked)).mean())
    print(f"  Valid pixel fraction: {valid_fraction:.1%}")

    # Get reference ET from GridMET
    print("\nGetting reference ET from GridMET...")
    try:
        etr = get_reference_et_for_scene(aoi, image.date, variable='etr')
        print(f"  Reference ET: {etr:.2f} mm/day")

        # Compute actual ET
        et = compute_et(etf_masked.values, etr)
        print(f"\nActual ET statistics:")
        print(f"  Mean ET: {np.nanmean(et):.2f} mm/day")
        print(f"  Max ET: {np.nanmax(et):.2f} mm/day")
    except Exception as e:
        print(f"  Could not get GridMET data: {e}")
        print("  (GridMET requires network access to THREDDS server)")

    print("\nSingle scene processing complete!")
    return image, etf_masked


def run_with_crop_type_example():
    """
    Process a scene using crop-type-specific Kc calculations.
    """
    print("\n" + "=" * 60)
    print("Example 2: Using Crop Type from CDL")
    print("=" * 60)

    # Define area of interest
    aoi = box(-120.5, 36.8, -120.4, 36.9)

    # Initialize data source
    source = STACSource()

    # Search for a scene
    scene_ids = source.search_scenes(
        geometry=aoi,
        start_date='2023-07-01',
        end_date='2023-07-31',
        max_cloud_cover=20.0
    )

    if not scene_ids:
        print("No scenes found.")
        return

    scene_id = scene_ids[0]
    print(f"\nProcessing scene: {scene_id}")

    # Get crop type from CDL
    print("\nGetting crop type from CDL...")
    try:
        crop_type = get_crop_type(aoi, year=2023)
        print(f"  Crop type shape: {crop_type.shape}")

        # Show unique crop types
        unique_types = np.unique(crop_type.values[~np.isnan(crop_type.values)])
        print(f"  Unique crop types: {len(unique_types)}")

    except Exception as e:
        print(f"  Could not get CDL data: {e}")
        crop_type = None

    # Create LandsatImage with crop type
    image = LandsatImage(
        source=source,
        scene_id=scene_id,
        geometry=aoi,
        crop_type=crop_type,
        use_crop_type_kc=False,  # Use generic class-based Kc
    )

    # Compute ET fraction with crop type awareness
    etf = image.et_fraction
    print(f"\nET fraction with crop type:")
    print(f"  Mean: {float(etf.mean()):.3f}")

    print("\nCrop type processing complete!")


def run_time_series_example():
    """
    Process multiple scenes to build a time series.
    """
    print("\n" + "=" * 60)
    print("Example 3: Time Series Processing")
    print("=" * 60)

    # Define area of interest - a single field location
    field_center = Point(-120.45, 36.85)
    aoi = field_center.buffer(0.01)  # ~1km buffer

    # Initialize data source
    source = STACSource()

    # Search for scenes over growing season
    print("\nSearching for scenes over growing season...")
    scene_ids = source.search_scenes(
        geometry=aoi,
        start_date='2023-04-01',
        end_date='2023-09-30',
        max_cloud_cover=30.0
    )

    print(f"Found {len(scene_ids)} scenes")

    # Process each scene
    results = []
    for i, scene_id in enumerate(scene_ids[:10]):  # Limit to 10 for demo
        try:
            image = LandsatImage(
                source=source,
                scene_id=scene_id,
                geometry=aoi,
            )

            # Get mean values for the AOI
            ndvi_mean = float(image.ndvi.mean())
            etf_mean = float(image.et_fraction.mean())

            results.append({
                'scene_id': scene_id,
                'date': image.date,
                'ndvi': ndvi_mean,
                'etf': etf_mean,
            })

            print(f"  {image.date.strftime('%Y-%m-%d')}: NDVI={ndvi_mean:.3f}, ETf={etf_mean:.3f}")

        except Exception as e:
            print(f"  Failed to process {scene_id}: {e}")
            continue

    print(f"\nProcessed {len(results)} scenes successfully")
    print("\nTime series processing complete!")

    return results


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SIMS Planetary Computer Workflow Examples")
    print("=" * 60)

    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    try:
        # Example 1: Single scene
        run_single_scene_example()

        # Example 2: With crop type (optional, requires CDL access)
        # run_with_crop_type_example()

        # Example 3: Time series
        # run_time_series_example()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have the required packages installed:")
        print("  pip install pystac-client planetary-computer stackstac rioxarray")
        raise


if __name__ == '__main__':
    main()
