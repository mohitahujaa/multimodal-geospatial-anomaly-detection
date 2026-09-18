"""
Sentinel-2 Surface Reflectance Ingestion, Index Calculation, and Grid Aggregation Pipeline.
Phase 1 Core Implementation for Multimodal Geospatial Anomaly Intelligence System.

Pipeline Stages:
1. Spatial and Temporal Filtering (Study Area Bounding Box, Date Range)
2. Scene-Level Cloud Filtering (CLOUDY_PIXEL_PERCENTAGE <= threshold)
3. Pixel-Level Cloud and Shadow Masking via SCL (Scene Classification Layer)
4. Median Compositing across Clear Observations
5. Spectral Indices Computation (NDVI, NDBI, NDWI) with Safe Numerical Handling
6. Raster Sanity Checks (Min, Max, Mean, Valid Range)
7. Polygonal Spatial Aggregation over 460 500m Grid Cells (EPSG:32643 UTM 43N)
8. Missing Data Explicit Representation (NaN, not zero)
9. Output Generation: Parquet, CSV, Debug Quality Report, and Visual Validation PNG Maps
"""

import sys
import os
import json
import math
import datetime
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# Pixel-level Cloud Masking using Sentinel-2 L2A SCL (Scene Classification)
# ---------------------------------------------------------------------------
# SCL Class Definitions:
# 0: NO_DATA                -> MASKED
# 1: SATURATED_OR_DEFECTIVE -> MASKED
# 2: DARK_AREA_PIXELS       -> KEPT (asphalt / deep shadows)
# 3: CLOUD_SHADOWS          -> MASKED
# 4: VEGETATION             -> KEPT
# 5: NOT_VEGETATED          -> KEPT (bare soil, built-up)
# 6: WATER                  -> KEPT
# 7: UNCLASSIFIED           -> KEPT
# 8: CLOUD_MEDIUM_PROB      -> MASKED
# 9: CLOUD_HIGH_PROB        -> MASKED
# 10: THIN_CIRRUS           -> MASKED
# 11: SNOW                  -> MASKED
# ---------------------------------------------------------------------------
def mask_s2_clouds_scl(image):
    """
    Mask clouds, shadows, cirrus, saturated, and invalid pixels using Sentinel-2 L2A SCL.
    Also scales surface reflectance by 1/10000 to convert raw DN to physical reflectance [0, 1].
    """
    import ee
    scl = image.select("SCL")
    
    # Valid mask: Keep classes 2, 4, 5, 6, 7
    # Mask out: 0, 1, 3, 8, 9, 10, 11
    valid_mask = (
        scl.eq(2)
        .Or(scl.eq(4))
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )
    
    # Select reflective bands and scale to [0, 1]
    reflective_bands = image.select(["B2", "B3", "B4", "B8", "B11"]).divide(10000.0)
    
    # Return masked image with scaled reflective bands and original SCL band for diagnostics
    return (
        image
        .addBands(reflective_bands, overwrite=True)
        .updateMask(valid_mask)
    )


# ---------------------------------------------------------------------------
# Spectral Indices Computation with Safe Numerical Handling
# ---------------------------------------------------------------------------
def compute_indices(image):
    """
    Compute NDVI, NDBI, and NDWI:
      - ndvi: (B8 - B4) / (B8 + B4)   [Near Infrared, Red]
      - ndbi: (B11 - B8) / (B11 + B8) [SWIR1, Near Infrared]
      - ndwi: (B3 - B8) / (B3 + B8)   [Green, Near Infrared]
    
    Uses safe normalized difference handling. If the denominator (A + B) is zero,
    the resulting pixel is masked to avoid division by zero.
    """
    import ee
    b3 = image.select("B3")
    b4 = image.select("B4")
    b8 = image.select("B8")
    b11 = image.select("B11")

    # Safe NDVI
    denom_ndvi = b8.add(b4)
    mask_ndvi = denom_ndvi.abs().gt(1e-5)
    ndvi = b8.subtract(b4).divide(denom_ndvi).updateMask(mask_ndvi).rename("ndvi")

    # Safe NDBI
    denom_ndbi = b11.add(b8)
    mask_ndbi = denom_ndbi.abs().gt(1e-5)
    ndbi = b11.subtract(b8).divide(denom_ndbi).updateMask(mask_ndbi).rename("ndbi")

    # Safe NDWI
    denom_ndwi = b3.add(b8)
    mask_ndwi = denom_ndwi.abs().gt(1e-5)
    ndwi = b3.subtract(b8).divide(denom_ndwi).updateMask(mask_ndwi).rename("ndwi")

    return image.addBands([ndvi, ndbi, ndwi])


# ---------------------------------------------------------------------------
# Sentinel-2 Processing Pipeline
# ---------------------------------------------------------------------------
def process_sentinel2_pipeline(
    start_date=None,
    end_date=None,
    max_cloud_percent=None,
    grid_path=None,
    output_dir=None
):
    """
    Execute complete Phase 1 Sentinel-2 pipeline.
    """
    import ee

    # Configuration fallbacks
    start_date = start_date or config.DEFAULT_START_DATE
    end_date = end_date or config.DEFAULT_END_DATE
    max_cloud_percent = max_cloud_percent if max_cloud_percent is not None else config.DEFAULT_MAX_CLOUD_PERCENT
    grid_path = Path(grid_path or config.GRID_PATH)
    output_dir = Path(output_dir or config.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PIPELINE START] Sentinel-2 Processing ({start_date} to {end_date})")
    print(f"[CONFIG] Cloud Threshold: <= {max_cloud_percent}%")
    print(f"[CONFIG] Grid Source: {grid_path}")

    # 1. Initialize Earth Engine
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    if project_id:
        ee.Initialize(project=project_id)
    else:
        ee.Initialize()

    # 2. Load Grid
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found at {grid_path}. Run `python src/grid.py` first.")
    
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_geojson = json.load(f)

    total_cells = len(grid_geojson["features"])
    print(f"[INFO] Loaded {total_cells} grid cells from GeoJSON.")

    # 3. Define Spatial Extent
    bbox = config.BBOX
    aoi = ee.Geometry.BBox(
        bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
    )

    # 4. Query Sentinel-2 Collection
    s2_id = config.GEE_DATASETS["sentinel2"]
    base_col = ee.ImageCollection(s2_id).filterBounds(aoi).filterDate(start_date, end_date)
    raw_image_count = base_col.size().getInfo()
    print(f"[GEE] Found {raw_image_count} total Sentinel-2 scenes in window.")

    filtered_col = base_col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
    filtered_image_count = filtered_col.size().getInfo()
    print(f"[GEE] {filtered_image_count} scenes remaining after scene-level cloud filter (<= {max_cloud_percent}%).")

    if filtered_image_count == 0:
        raise ValueError(
            f"No scenes match cloud threshold <= {max_cloud_percent}% in {start_date} to {end_date}. "
            f"Increase threshold or adjust date range."
        )

    # 5. Apply Pixel-level Masking and Compute Indices
    processed_col = filtered_col.map(mask_s2_clouds_scl).map(compute_indices)

    # 6. Create Temporal Median Composite
    # Median composite selects the per-pixel median reflectance/index over time,
    # filtering out any remaining clouds, shadows, or transient anomalies.
    composite = processed_col.select(["ndvi", "ndbi", "ndwi"]).median()

    # 7. Raster Pre-Aggregation Sanity Check
    print("[CHECK] Performing raster sanity checks on median composite...")
    raster_stats = composite.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=aoi,
        scale=50,
        crs=config.PROJECTED_CRS,
        maxPixels=1e6
    ).getInfo()

    print("--- Study Area Raster Summary ---")
    for key, val in sorted(raster_stats.items()):
        if val is not None:
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: None")

    # Sanity check bounds
    for idx_name in ["ndvi", "ndbi", "ndwi"]:
        min_v = raster_stats.get(f"{idx_name}_min")
        max_v = raster_stats.get(f"{idx_name}_max")
        if min_v is not None and (min_v < -1.5 or max_v > 1.5):
            print(f"[WARNING] {idx_name} values span [{min_v:.2f}, {max_v:.2f}], outside expected [-1, 1] range!")

    # 8. Spatial Aggregation over 460 500m Grid Cells
    print(f"[AGGREGATION] Aggregating raster features over {total_cells} cells using {config.PROJECTED_CRS}...")

    # Build ee.FeatureCollection from grid polygons
    ee_features = []
    for feat in grid_geojson["features"]:
        cid = feat["properties"]["cell_id"]
        coords = feat["geometry"]["coordinates"]
        geom = ee.Geometry.Polygon(coords)
        ee_features.append(ee.Feature(geom, {"cell_id": cid}))

    ee_grid = ee.FeatureCollection(ee_features)

    # Reducers: mean, standard deviation, and count of valid pixels per cell
    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )

    # Perform polygonal spatial reduction at 20m scale (Sentinel-2 resolution)
    # in UTM Zone 43N (meters-based CRS)
    reduced_grid = composite.reduceRegions(
        collection=ee_grid,
        reducer=reducer,
        scale=20,
        crs=config.PROJECTED_CRS
    )

    print("[INFO] Fetching aggregated feature data from Earth Engine...")
    results = reduced_grid.getInfo()

    # 9. Format Tidy DataFrame with Quality & Missing Data Handling
    # At 20m pixel resolution, a 500m x 500m cell has approximately:
    # expected_pixels = (500 / 20) * (500 / 20) = 25 * 25 = 625 pixels
    EXPECTED_PIXELS_PER_CELL = 625

    records = []
    for feat in results["features"]:
        props = feat["properties"]
        cid = props.get("cell_id")

        # Counts can be checked from ndvi_count
        count = props.get("ndvi_count")
        if count is None or count == 0:
            v_count = 0
            v_fraction = 0.0
            n_mean, n_std = np.nan, np.nan
            b_mean, b_std = np.nan, np.nan
            w_mean, w_std = np.nan, np.nan
        else:
            v_count = int(count)
            v_fraction = round(min(1.0, v_count / EXPECTED_PIXELS_PER_CELL), 4)
            n_mean = props.get("ndvi_mean", np.nan)
            n_std = props.get("ndvi_stdDev", np.nan)
            b_mean = props.get("ndbi_mean", np.nan)
            b_std = props.get("ndbi_stdDev", np.nan)
            w_mean = props.get("ndwi_mean", np.nan)
            w_std = props.get("ndwi_stdDev", np.nan)

        records.append({
            "cell_id": cid,
            "start_date": start_date,
            "end_date": end_date,
            "ndvi_mean": n_mean,
            "ndbi_mean": b_mean,
            "ndwi_mean": w_mean,
            "ndvi_std": n_std,
            "ndbi_std": b_std,
            "ndwi_std": w_std,
            "valid_pixel_count": v_count,
            "valid_pixel_fraction": v_fraction
        })

    df = pd.DataFrame(records)

    # Sort deterministically by cell_id
    df = df.sort_values("cell_id").reset_index(drop=True)

    # 10. Sanity Checks on Aggregated Table
    print("\n[VALIDATION] Running sanity checks on feature table...")
    assert len(df) == total_cells, f"Expected {total_cells} rows, got {len(df)}"
    assert df["cell_id"].nunique() == total_cells, "Duplicate cell IDs detected!"
    
    missing_cells = df[df["valid_pixel_count"] == 0]
    valid_cells = df[df["valid_pixel_count"] > 0]
    num_missing = len(missing_cells)
    num_valid = len(valid_cells)
    
    print(f"  Total cells:             {len(df)}")
    print(f"  Cells with valid data:   {num_valid}")
    print(f"  Cells with missing data: {num_missing}")
    print(f"  Mean NDVI across cells:  {df['ndvi_mean'].mean():.4f}")
    print(f"  Mean NDBI across cells:  {df['ndbi_mean'].mean():.4f}")
    print(f"  Mean NDWI across cells:  {df['ndwi_mean'].mean():.4f}")

    # 11. Save Analytical Datasets
    parquet_path = output_dir / "s2_features.parquet"
    csv_path = output_dir / "s2_features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Parquet saved: {parquet_path}")
    print(f"[SAVE] CSV saved:     {csv_path}")

    # 12. Create Debug Quality Report
    report_path = output_dir / "s2_processing_report.txt"
    report_content = f"""=============================================================
 Sentinel-2 Phase 1 Processing Report
=============================================================
AOI:                      Delhi-NCR Study Area (BBOX: 77.25-77.35 E, 28.50-28.60 N)
Grid Source:              {grid_path.name}
CRS (Projected):          {config.PROJECTED_CRS} (UTM Zone 43N)
CRS (Geographic):         {config.GEOGRAPHIC_CRS}
Grid Cells:               {total_cells} (23 rows x 20 columns, ~500m resolution)
Date Range:               {start_date} -> {end_date}
Scenes before filtering:  {raw_image_count}
Scenes after filtering:   {filtered_image_count}
Scene Cloud Threshold:    <= {max_cloud_percent}%
Pixel Cloud Masking:      Sentinel-2 L2A SCL (Scene Classification Layer)
                          [Masked: 0, 1, 3, 8, 9, 10, 11 | Kept: 2, 4, 5, 6, 7]
Composite Method:         Temporal Median Composite
Scale for Reduction:      20 meters (polygonal reduceRegions)

Feature Statistics:
-------------------------------------------------------------
Cells with valid data:    {num_valid} ({num_valid / total_cells * 100:.1f}%)
Cells with missing data:  {num_missing} ({num_missing / total_cells * 100:.1f}%)

Metric           Min       Max       Mean      Std
NDVI (mean)      {df['ndvi_mean'].min():.4f}    {df['ndvi_mean'].max():.4f}    {df['ndvi_mean'].mean():.4f}    {df['ndvi_mean'].std():.4f}
NDBI (mean)      {df['ndbi_mean'].min():.4f}    {df['ndbi_mean'].max():.4f}    {df['ndbi_mean'].mean():.4f}    {df['ndbi_mean'].std():.4f}
NDWI (mean)      {df['ndwi_mean'].min():.4f}    {df['ndwi_mean'].max():.4f}    {df['ndwi_mean'].mean():.4f}    {df['ndwi_mean'].std():.4f}
Valid Pixels     {df['valid_pixel_count'].min()}       {df['valid_pixel_count'].max()}       {df['valid_pixel_count'].mean():.1f}     {df['valid_pixel_count'].std():.1f}

Outputs:
  - Feature Parquet:      {parquet_path}
  - Feature CSV:          {csv_path}
  - Processing Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=============================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SAVE] Report saved:  {report_path}")

    # 13. Create Visual Validation Maps
    generate_validation_maps(df, grid_geojson, output_dir, start_date, end_date)

    print("\n>>> Phase 1 Sentinel-2 Pipeline Completed Successfully! <<<\n")
    return df


# ---------------------------------------------------------------------------
# Visual Validation Outputs (Matplotlib Spatial Choropleths)
# ---------------------------------------------------------------------------
def generate_validation_maps(df, grid_geojson, output_dir, start_date, end_date):
    """
    Generate static visual verification maps for NDVI, NDBI, and NDWI
    overlaid with the 500m grid cells and study area boundary.
    """
    print("[VIZ] Generating spatial validation maps...")
    
    # Map cell_id to feature index and values
    val_lookup = {
        row["cell_id"]: {
            "ndvi": row["ndvi_mean"],
            "ndbi": row["ndbi_mean"],
            "ndwi": row["ndwi_mean"],
        }
        for _, row in df.iterrows()
    }

    metrics = [
        ("ndvi", "NDVI (Vegetation Index)", "YlGn", (-0.2, 0.6)),
        ("ndbi", "NDBI (Built-up Index)", "YlOrRd", (-0.4, 0.4)),
        ("ndwi", "NDWI (Water Index)", "Blues", (-0.5, 0.3)),
    ]

    for metric_key, metric_title, cmap_name, (vmin, vmax) in metrics:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        
        patches_list = []
        values_list = []

        for feat in grid_geojson["features"]:
            cid = feat["properties"]["cell_id"]
            coords = feat["geometry"]["coordinates"][0]  # exterior ring
            poly = patches.Polygon(coords, closed=True)
            patches_list.append(poly)
            val = val_lookup.get(cid, {}).get(metric_key, np.nan)
            values_list.append(val)

        values_arr = np.array(values_list)
        
        # Determine actual data bounds for robust colorbar if within range
        valid_vals = values_arr[~np.isnan(values_arr)]
        if len(valid_vals) > 0:
            c_min = max(vmin, float(np.percentile(valid_vals, 1)))
            c_max = min(vmax, float(np.percentile(valid_vals, 99)))
            if c_min >= c_max:
                c_min, c_max = vmin, vmax
        else:
            c_min, c_max = vmin, vmax

        p = PatchCollection(patches_list, cmap=cmap_name, edgecolor="black", linewidth=0.2, alpha=0.9)
        p.set_array(values_arr)
        p.set_clim(c_min, c_max)
        ax.add_collection(p)

        # Plot study area bounds
        bbox = config.BBOX
        ax.set_xlim(bbox["min_lon"] - 0.005, bbox["max_lon"] + 0.005)
        ax.set_ylim(bbox["min_lat"] - 0.005, bbox["max_lat"] + 0.005)
        ax.set_aspect("equal", "box")

        # Colorbar & Labels
        cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Cell Mean {metric_title}", fontsize=11)

        ax.set_title(
            f"Delhi-NCR 500m Grid — {metric_title}\n"
            f"Sentinel-2 Median Composite ({start_date} to {end_date})",
            fontsize=12,
            fontweight="bold",
            pad=12
        )
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        out_img = output_dir / f"map_{metric_key}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"[SAVE] Visual validation map: {out_img}")


if __name__ == "__main__":
    process_sentinel2_pipeline()
