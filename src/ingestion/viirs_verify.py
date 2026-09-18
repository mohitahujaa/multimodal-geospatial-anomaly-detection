"""
VIIRS Nighttime Lights Verification Script.
Phase 2 Component: Validates Earth Engine access, VIIRS collection availability,
March 2024 monthly image presence, required bands (avg_rad, cf_cvg), and study area intersection.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


def run_viirs_verification():
    print("=" * 65)
    print(" PHASE 2: VIIRS NIGHTTIME LIGHTS VERIFICATION ")
    print("=" * 65)

    # 1. Earth Engine Python Package Import
    try:
        import ee
        print(f"1. GEE package import:          PASS (ee version {getattr(ee, '__version__', 'unknown')})")
    except ImportError:
        print("1. GEE package import:          FAIL")
        print("   [ACTION] Run: pip install earthengine-api")
        return False

    # 2. Earth Engine Authentication & Initialization
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    try:
        if project_id:
            ee.Initialize(project=project_id)
            print(f"2. GEE authentication:          PASS (Project: {project_id})")
        else:
            ee.Initialize()
            print("2. GEE authentication:          PASS (Default credentials)")
    except Exception as exc:
        print(f"2. GEE authentication:          FAIL ({exc})")
        return False

    # 3. Load & Validate Study Area
    try:
        bbox = config.BBOX
        aoi = ee.Geometry.BBox(
            bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
        )
        coords = aoi.coordinates().getInfo()
        if coords and len(coords[0]) >= 4:
            print(f"3. Study area loaded:           PASS (BBOX: [{bbox['min_lon']}, {bbox['min_lat']}] to [{bbox['max_lon']}, {bbox['max_lat']}])")
        else:
            print("3. Study area loaded:           FAIL (Invalid coordinates)")
            return False
    except Exception as exc:
        print(f"3. Study area loaded:           FAIL ({exc})")
        return False

    # 4. VIIRS Collection Accessibility
    collection_id = config.GEE_DATASETS.get("viirs", "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    start_date = config.DEFAULT_VIIRS_START_DATE
    end_date = config.DEFAULT_VIIRS_END_DATE

    try:
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        collection_size = collection.size().getInfo()
        print(f"4. VIIRS collection access:     PASS (Collection: {collection_id})")
    except Exception as exc:
        print(f"4. VIIRS collection access:     FAIL ({exc})")
        return False

    # 5. March 2024 Image Exists
    print(f"   Date range checked:          [{start_date}, {end_date})")
    print(f"   Monthly images found:        {collection_size}")
    if collection_size >= 1:
        print("5. Monthly test image exists:   PASS")
    else:
        print(f"5. Monthly test image exists:   FAIL (No image found in {start_date} to {end_date})")
        return False

    # 6. Retrieve First Image & Check Required Bands
    try:
        viirs_img = collection.first()
        band_names = viirs_img.bandNames().getInfo()
        required_bands = ["avg_rad", "cf_cvg"]
        missing_bands = [b for b in required_bands if b not in band_names]
        if missing_bands:
            print(f"6. Required bands check:        FAIL (Missing: {missing_bands})")
            return False
        else:
            print(f"6. Required bands (avg_rad, cf_cvg): PASS (Available: {band_names})")
    except Exception as exc:
        print(f"6. Required bands check:        FAIL ({exc})")
        return False

    # 7. Read Image Metadata & Date
    try:
        date_millis = viirs_img.get("system:time_start").getInfo()
        date_str = ee.Date(date_millis).format("YYYY-MM-dd").getInfo()
        nominal_scale = viirs_img.select("avg_rad").projection().nominalScale().getInfo()
        print(f"7. Image metadata/timestamp:    PASS (Acquisition date: {date_str}, Nominal scale: {nominal_scale:.2f} m)")
    except Exception as exc:
        print(f"7. Image metadata/timestamp:    FAIL ({exc})")
        return False

    # 8. Spatial Intersection with Study Area
    try:
        # Check intersection by computing coverage over AOI
        pixel_count = viirs_img.select("avg_rad").reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi,
            scale=config.VIIRS_NATIVE_SCALE,
            crs=config.PROJECTED_CRS
        ).getInfo().get("avg_rad", 0)

        if pixel_count > 0:
            print(f"8. Study area intersection:     PASS ({pixel_count} valid raster pixels intersect AOI)")
        else:
            print("8. Study area intersection:     FAIL (Zero pixels in study area)")
            return False
    except Exception as exc:
        print(f"8. Study area intersection:     FAIL ({exc})")
        return False

    # 9. Basic Study Area Statistics
    try:
        stats = viirs_img.select(["avg_rad", "cf_cvg"]).reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
            geometry=aoi,
            scale=config.VIIRS_NATIVE_SCALE,
            crs=config.PROJECTED_CRS
        ).getInfo()

        rad_mean = stats.get("avg_rad_mean")
        rad_min = stats.get("avg_rad_min")
        rad_max = stats.get("avg_rad_max")
        cvg_mean = stats.get("cf_cvg_mean")

        print("9. Basic statistics computed:   PASS")
        print(f"   - avg_rad [min, mean, max]:   [{rad_min:.2f}, {rad_mean:.2f}, {rad_max:.2f}] nW/(cm^2 sr)")
        print(f"   - cf_cvg mean observations:  {cvg_mean:.1f}")
    except Exception as exc:
        print(f"9. Basic statistics computed:   FAIL ({exc})")
        return False

    print("=" * 65)
    print(" VIIRS VERIFICATION COMPLETE: ALL CHECKS PASSED ")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_viirs_verification()
    sys.exit(0 if success else 1)
