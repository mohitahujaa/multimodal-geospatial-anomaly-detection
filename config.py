"""
Configuration module for Multimodal Geospatial Anomaly Intelligence System.
Defines constants, bounding box, spatial grid parameters, and directory paths.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

# Base project paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
STUDY_AREA_PATH = DATA_DIR / "study_area.geojson"
GRID_PATH = PROCESSED_DATA_DIR / "study_grid.geojson"

# Ensure runtime directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Study Area Bounding Box (Delhi-NCR ~10 km x 10 km)
# Min Lon, Min Lat, Max Lon, Max Lat
# Covering South East / East Delhi along the Yamuna river, Okhla, and surrounding urban/green areas
BBOX = {
    "min_lon": 77.25,
    "min_lat": 28.50,
    "max_lon": 77.35,
    "max_lat": 28.60,
}

# Spatial Grid Specifications
GRID_RESOLUTION_METERS = 500  # 500m x 500m cells
PROJECTED_CRS = "EPSG:32643"  # UTM zone 43N (Delhi-NCR meter-based coordinate reference system)
GEOGRAPHIC_CRS = "EPSG:4326"  # Standard WGS84 lat/lon

# Google Earth Engine Configuration
GEE_PROJECT = os.getenv("GEE_PROJECT", "")

# Datasets in Google Earth Engine
GEE_DATASETS = {
    "sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "viirs": "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG",
    "era5": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
}

# Phase 1: Sentinel-2 Ingestion Parameters (Configurable)
DEFAULT_START_DATE = os.getenv("S2_START_DATE", "2024-03-01")
DEFAULT_END_DATE = os.getenv("S2_END_DATE", "2024-03-31")
DEFAULT_MAX_CLOUD_PERCENT = int(os.getenv("S2_MAX_CLOUD_PERCENT", "20"))

# Phase 2: VIIRS Nighttime Lights Parameters (Configurable)
DEFAULT_VIIRS_START_DATE = os.getenv("VIIRS_START_DATE", "2024-03-01")
DEFAULT_VIIRS_END_DATE = os.getenv("VIIRS_END_DATE", "2024-04-01")
VIIRS_NATIVE_SCALE = 463.8312  # meters (approximate native pixel resolution of VIIRS 15 arc-second product)

