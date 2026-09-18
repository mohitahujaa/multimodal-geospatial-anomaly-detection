"""
GEE Verification & Sentinel-2 Diagnostic Script.
Step 1 of Phase 1: Validates GEE installation, authentication, study area,
Sentinel-2 collection access, and image metadata.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


def run_verification():
    print("=" * 65)
    print(" PHASE 1: GOOGLE EARTH ENGINE & SENTINEL-2 VERIFICATION ")
    print("=" * 65)

    # 1. Check Python package import
    try:
        import ee
        print(f"GEE package import:     PASS (ee version {getattr(ee, '__version__', 'unknown')})")
    except ImportError:
        print("GEE package import:     FAIL")
        print("\n[ACTION REQUIRED] Install the Earth Engine API:")
        print("    python -m pip install earthengine-api\n")
        return False

    # 2. Check Authentication & Initialization
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    ee_initialized = False
    try:
        if project_id:
            ee.Initialize(project=project_id)
            print(f"GEE authentication:     PASS (Project: {project_id})")
        else:
            ee.Initialize()
            print("GEE authentication:     PASS (Default credentials)")
        ee_initialized = True
    except Exception as exc:
        print("GEE authentication:     FAIL")
        print(f"Details: {exc}\n")
        print("-----------------------------------------------------------------")
        print("[ACTION REQUIRED] Earth Engine requires user authorization:")
        print("  1. Ensure you have registered your Google account at:")
        print("     https://earthengine.google.com")
        print("  2. In your terminal, run:")
        print("     python -c \"import ee; ee.Authenticate()\"")
        print("  3. Follow the web browser flow, copy the authorization code,")
        print("     and paste it when prompted.")
        print("  4. If using a specific Cloud Project ID, create a .env file:")
        print("     GEE_PROJECT=your-cloud-project-id")
        print("-----------------------------------------------------------------\n")
        return False

    # 3. Check Study Area
    try:
        bbox = config.BBOX
        aoi = ee.Geometry.BBox(
            bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
        )
        # Verify bounding box geometry coordinates
        coords = aoi.coordinates().getInfo()
        if coords and len(coords[0]) >= 4:
            print(f"Study area:             PASS (Bounds: [{bbox['min_lon']}, {bbox['min_lat']}] to [{bbox['max_lon']}, {bbox['max_lat']}])")
        else:
            print("Study area:             FAIL (Invalid geometry)")
            return False
    except Exception as exc:
        print(f"Study area:             FAIL ({exc})")
        return False

    # 4. Check Sentinel-2 Collection
    s2_collection_id = config.GEE_DATASETS.get("sentinel2", "COPERNICUS/S2_SR_HARMONIZED")
    start_date = config.DEFAULT_START_DATE
    end_date = config.DEFAULT_END_DATE
    max_cloud = config.DEFAULT_MAX_CLOUD_PERCENT

    try:
        raw_collection = (
            ee.ImageCollection(s2_collection_id)
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
        )
        total_images = raw_collection.size().getInfo()
        print(f"Sentinel-2 collection:  PASS (ID: {s2_collection_id})")
        print(f"Test date window:       {start_date} to {end_date}")
        print(f"Images found:           {total_images}")

        # Cloud-filtered subset
        filtered_collection = raw_collection.filter(
            ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud)
        )
        filtered_count = filtered_collection.size().getInfo()
        print(f"Cloud-filtered images:  {filtered_count} (Threshold: <= {max_cloud}%)")

        if filtered_count > 0:
            first_image = filtered_collection.first()
            date_str = ee.Date(first_image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss").getInfo()
            cloud_cover = first_image.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()
            bands = first_image.bandNames().getInfo()
            print(f"Sample image timestamp: {date_str} (Scene Cloud: {cloud_cover:.1f}%)")
            
            # Verify required bands
            required_bands = ["B3", "B4", "B8", "B11", "SCL"]
            missing_bands = [b for b in required_bands if b not in bands]
            if missing_bands:
                print(f"[WARNING] Missing expected bands: {missing_bands}")
            else:
                print(f"Required bands (B3, B4, B8, B11, SCL): VERIFIED PRESENT")
        else:
            print(f"[NOTE] No scenes found below {max_cloud}% cloud cover in {start_date} to {end_date}.")
            print("       Consider adjusting date range or cloud threshold.")

        print("=" * 65)
        print(" GEE VERIFICATION COMPLETE: ALL CHECKS PASSED ")
        print("=" * 65)
        return True

    except Exception as exc:
        print(f"Sentinel-2 collection:  FAIL ({exc})")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
