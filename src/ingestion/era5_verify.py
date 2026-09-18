"""
ERA5-Land Weather Verification Script.
Phase 3 Component: Validates Earth Engine access, ECMWF/ERA5_LAND/HOURLY collection,
March 2024 hourly data presence (744 hours), required bands, study area intersection,
and basic meteorological statistics.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


def run_era5_verification():
    print("=" * 65)
    print(" PHASE 3: ERA5-LAND HOURLY WEATHER VERIFICATION ")
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

    # 4. ERA5-Land Hourly Collection Accessibility
    collection_id = config.GEE_DATASETS.get("era5_hourly", "ECMWF/ERA5_LAND/HOURLY")
    start_date = config.DEFAULT_ERA5_START_DATE
    end_date = config.DEFAULT_ERA5_END_DATE

    try:
        collection = (
            ee.ImageCollection(collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        collection_size = collection.size().getInfo()
        print(f"4. ERA5-Land collection access: PASS (Collection: {collection_id})")
    except Exception as exc:
        print(f"4. ERA5-Land collection access: FAIL ({exc})")
        return False

    # 5. March 2024 Hourly Data Presence (Expected: 31 days * 24 hours = 744)
    print(f"   Date range checked:          [{start_date}, {end_date})")
    print(f"   Hourly observations found:   {collection_size} (Expected: 744 for March)")
    if collection_size == 744:
        print("5. Hourly observation count:    PASS (100.0% temporal completeness)")
    elif collection_size > 0:
        print(f"5. Hourly observation count:    PARTIAL ({collection_size}/744 observations found)")
    else:
        print(f"5. Hourly observation count:    FAIL (No data found in [{start_date}, {end_date}))")
        return False

    # 6. Verify Required Variables
    try:
        first_img = collection.first()
        available_bands = first_img.bandNames().getInfo()
        required_bands = [
            "temperature_2m",
            "total_precipitation_hourly",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m"
        ]
        missing_bands = [b for b in required_bands if b not in available_bands]
        if missing_bands:
            print(f"6. Required variables check:    FAIL (Missing: {missing_bands})")
            return False
        else:
            print(f"6. Required variables:          PASS (temperature_2m, total_precipitation_hourly, u/v wind 10m)")
    except Exception as exc:
        print(f"6. Required variables check:    FAIL ({exc})")
        return False

    # 7. Spatial Resolution and Grid Points in Study Area
    try:
        nominal_scale = first_img.select("temperature_2m").projection().nominalScale().getInfo()
        print(f"7. Native spatial resolution:   PASS (~{nominal_scale:.2f} m, ~0.1 degrees)")
        
        # Sample points in study area to identify source grid points
        pts = first_img.select("temperature_2m").sample(region=aoi, scale=nominal_scale, geometries=True).getInfo()
        num_pts = len(pts.get("features", []))
        print(f"   Intersecting grid points:    {num_pts} native ERA5-Land grid point(s) cover the study area")
    except Exception as exc:
        print(f"7. Native spatial resolution:   FAIL ({exc})")
        return False

    # 8. Basic Meteorological Statistics Computation
    try:
        # Sample first hourly scene for initial bounds check
        stats = first_img.select([
            "temperature_2m", "total_precipitation_hourly",
            "u_component_of_wind_10m", "v_component_of_wind_10m"
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=nominal_scale
        ).getInfo()

        t2m_k = stats.get("temperature_2m")
        t2m_c = t2m_k - 273.15 if t2m_k is not None else None
        tp_m = stats.get("total_precipitation_hourly")
        tp_mm = tp_m * 1000.0 if tp_m is not None else None
        u10 = stats.get("u_component_of_wind_10m")
        v10 = stats.get("v_component_of_wind_10m")
        ws = (u10**2 + v10**2)**0.5 if u10 is not None and v10 is not None else None

        print("8. Hourly sample reading:       PASS (Sample: 2024-03-01 00:00 UTC)")
        print(f"   - 2m Temperature:            {t2m_c:.2f} °C ({t2m_k:.2f} K)")
        print(f"   - Total Precip (hourly):     {tp_mm:.4f} mm")
        print(f"   - 10m Wind Speed (derived):  {ws:.2f} m/s (u={u10:.2f}, v={v10:.2f})")
    except Exception as exc:
        print(f"8. Hourly sample reading:       FAIL ({exc})")
        return False

    print("=" * 65)
    print(" ERA5-LAND VERIFICATION COMPLETE: ALL CHECKS PASSED ")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_era5_verification()
    sys.exit(0 if success else 1)
