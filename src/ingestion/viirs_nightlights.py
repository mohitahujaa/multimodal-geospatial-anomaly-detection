"""
VIIRS Nighttime Lights Ingestion, Spatial Aggregation, and Validation Pipeline.
Phase 2 Implementation for Multimodal Geospatial Anomaly Intelligence System.

Aggregates monthly VIIRS Day/Night Band radiance and cloud-free coverage
onto the canonical 460-cell 500m study grid (EPSG:32643 UTM Zone 43N).
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


def process_viirs_pipeline(
    start_date=None,
    end_date=None,
    grid_path=None,
    output_dir=None
):
    """
    Execute complete Phase 2 VIIRS Nighttime Lights processing pipeline.
    
    Args:
        start_date (str): 'YYYY-MM-DD' start of monthly window (inclusive)
        end_date (str): 'YYYY-MM-DD' end of monthly window (exclusive)
        grid_path (Path or str): Path to canonical study_grid.geojson
        output_dir (Path or str): Path to processed data output directory
    
    Returns:
        pd.DataFrame: Formatted VIIRS feature table with 460 rows
    """
    import ee

    # 1. Configuration & Paths
    start_date = start_date or config.DEFAULT_VIIRS_START_DATE
    end_date = end_date or config.DEFAULT_VIIRS_END_DATE
    grid_path = Path(grid_path or config.GRID_PATH)
    output_dir = Path(output_dir or config.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PIPELINE START] VIIRS Nighttime Lights Processing ({start_date} to {end_date})")
    print(f"[CONFIG] Grid Source: {grid_path}")
    print(f"[CONFIG] Projected CRS: {config.PROJECTED_CRS} (Native scale: ~{config.VIIRS_NATIVE_SCALE:.2f} m)")

    # 2. Authenticate Earth Engine
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    if project_id:
        ee.Initialize(project=project_id)
    else:
        ee.Initialize()

    # 3. Load Existing 460-Cell Grid
    if not grid_path.exists():
        raise FileNotFoundError(f"Canonical grid not found at {grid_path}. Run `python src/grid.py` first.")
    
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_geojson = json.load(f)

    total_cells = len(grid_geojson["features"])
    print(f"[INFO] Loaded {total_cells} canonical cells from {grid_path.name}.")

    # 4. Define Spatial Bounding Box
    bbox = config.BBOX
    aoi = ee.Geometry.BBox(
        bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]
    )

    # 5. Query VIIRS Stray Light Corrected Monthly Collection
    collection_id = config.GEE_DATASETS.get("viirs", "NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    image_count = collection.size().getInfo()
    print(f"[GEE] Monthly images matching [{start_date}, {end_date}): {image_count}")
    if image_count == 0:
        raise ValueError(f"No VIIRS monthly image found in date range [{start_date}, {end_date})")

    viirs_img = collection.first()
    acq_date = ee.Date(viirs_img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    print(f"[GEE] Source monthly image acquisition date: {acq_date}")

    # 6. Prepare Multi-Band Image for Polygonal Reduction
    # Bands:
    #   - avg_rad: DNB monthly average radiance (nW/(cm^2 sr))
    #   - cf_cvg: Cloud-free coverage count (observation quality)
    #   - valid_mask: Binary mask of unmasked pixels for spatial coverage fraction
    mask_layer = viirs_img.select("avg_rad").mask().rename("valid_mask")
    multi_band_img = viirs_img.select(["avg_rad", "cf_cvg"]).addBands(mask_layer)

    # 7. Convert Canonical GeoJSON Grid to Earth Engine FeatureCollection
    ee_features = []
    for feat in grid_geojson["features"]:
        cid = feat["properties"]["cell_id"]
        coords = feat["geometry"]["coordinates"]
        geom = ee.Geometry.Polygon(coords)
        ee_features.append(ee.Feature(geom, {"cell_id": cid}))

    ee_grid = ee.FeatureCollection(ee_features)

    # 8. Define Polygonal Reducers
    # For radiance: mean, stdDev, count
    # For coverage: mean, min, max
    # For mask: mean (spatial valid fraction across polygon)
    rad_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )
    cvg_reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.max(), sharedInputs=True)
    )
    mask_reducer = ee.Reducer.mean()

    print(f"[AGGREGATION] Aggregating VIIRS raster over all {total_cells} cells using {config.PROJECTED_CRS}...")
    
    # Run polygonal reduction for radiance
    reduced_rad = multi_band_img.select(["avg_rad"]).reduceRegions(
        collection=ee_grid,
        reducer=rad_reducer,
        scale=config.VIIRS_NATIVE_SCALE,
        crs=config.PROJECTED_CRS
    )

    # Run polygonal reduction for cloud-free coverage
    reduced_cvg = multi_band_img.select(["cf_cvg"]).reduceRegions(
        collection=ee_grid,
        reducer=cvg_reducer,
        scale=config.VIIRS_NATIVE_SCALE,
        crs=config.PROJECTED_CRS
    )

    # Run polygonal reduction for spatial mask coverage fraction
    reduced_mask = multi_band_img.select(["valid_mask"]).reduceRegions(
        collection=ee_grid,
        reducer=mask_reducer,
        scale=config.VIIRS_NATIVE_SCALE,
        crs=config.PROJECTED_CRS
    )

    print("[INFO] Fetching aggregated feature data from Earth Engine...")
    rad_results = reduced_rad.getInfo()["features"]
    cvg_results = reduced_cvg.getInfo()["features"]
    mask_results = reduced_mask.getInfo()["features"]

    # Index properties by cell_id
    rad_lookup = {f["properties"]["cell_id"]: f["properties"] for f in rad_results}
    cvg_lookup = {f["properties"]["cell_id"]: f["properties"] for f in cvg_results}
    mask_lookup = {f["properties"]["cell_id"]: f["properties"] for f in mask_results}

    # 9. Assemble Analytical Feature Table
    records = []
    for feat in grid_geojson["features"]:
        cid = feat["properties"]["cell_id"]
        r_props = rad_lookup.get(cid, {})
        c_props = cvg_lookup.get(cid, {})
        m_props = mask_lookup.get(cid, {})

        count = r_props.get("count")
        if count is None or count == 0:
            v_count = 0
            v_fraction = 0.0
            rad_mean = np.nan
            rad_std = np.nan
            cvg_mean = np.nan
            cvg_min = np.nan
            cvg_max = np.nan
        else:
            v_count = int(count)
            # Spatial valid fraction directly from mask mean [0.0, 1.0]
            raw_frac = m_props.get("mean", 1.0)
            v_fraction = round(float(np.clip(raw_frac, 0.0, 1.0)), 4) if raw_frac is not None else 1.0
            rad_mean = r_props.get("mean", np.nan)
            rad_std = r_props.get("stdDev", np.nan)
            cvg_mean = c_props.get("mean", np.nan)
            cvg_min = c_props.get("min", np.nan)
            cvg_max = c_props.get("max", np.nan)

        records.append({
            "cell_id": cid,
            "start_date": start_date,
            "end_date": end_date,
            "viirs_avg_rad_mean": rad_mean,
            "viirs_avg_rad_std": rad_std,
            "viirs_cf_cvg_mean": cvg_mean,
            "viirs_cf_cvg_min": cvg_min,
            "viirs_cf_cvg_max": cvg_max,
            "viirs_valid_pixel_count": v_count,
            "viirs_valid_pixel_fraction": v_fraction,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("cell_id").reset_index(drop=True)

    # 10. Run Sanity Checks
    print("\n[VALIDATION] Running sanity checks on VIIRS feature table...")
    assert len(df) == total_cells, f"Expected {total_cells} rows, got {len(df)}"
    assert df["cell_id"].nunique() == total_cells, "Duplicate cell IDs detected!"

    valid_cells = df[df["viirs_valid_pixel_count"] > 0]
    missing_cells = df[df["viirs_valid_pixel_count"] == 0]
    num_valid = len(valid_cells)
    num_missing = len(missing_cells)

    rad_mean = df["viirs_avg_rad_mean"].mean()
    rad_min = df["viirs_avg_rad_mean"].min()
    rad_max = df["viirs_avg_rad_mean"].max()

    cvg_mean = df["viirs_cf_cvg_mean"].mean()
    cvg_min = df["viirs_cf_cvg_min"].min()
    cvg_max = df["viirs_cf_cvg_max"].max()

    frac_mean = df["viirs_valid_pixel_fraction"].mean()

    print(f"  Total cells:                 {len(df)}")
    print(f"  Cells with valid VIIRS data: {num_valid} ({num_valid / total_cells * 100:.1f}%)")
    print(f"  Cells with missing data:     {num_missing} ({num_missing / total_cells * 100:.1f}%)")
    print(f"  Radiance [min, mean, max]:   [{rad_min:.2f}, {rad_mean:.2f}, {rad_max:.2f}] nW/(cm^2 sr)")
    print(f"  Coverage [min, mean, max]:   [{cvg_min:.1f}, {cvg_mean:.1f}, {cvg_max:.1f}] observations")
    print(f"  Mean valid pixel fraction:   {frac_mean:.4f}")

    # 11. Save Datasets
    parquet_path = output_dir / "viirs_features.parquet"
    csv_path = output_dir / "viirs_features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Parquet saved: {parquet_path}")
    print(f"[SAVE] CSV saved:     {csv_path}")

    # 12. Create Processing Report
    report_path = output_dir / "viirs_processing_report.txt"
    report_content = f"""=============================================================
 VIIRS Nighttime Lights Phase 2 Processing Report
=============================================================
Dataset ID:               {collection_id}
Dataset Name:             VIIRS Stray Light Corrected Day/Night Band Monthly (VCMSLCFG)
Bands Used:               avg_rad (DNB radiance), cf_cvg (cloud-free coverage count)
Native Pixel Resolution:  ~{config.VIIRS_NATIVE_SCALE:.2f} meters (15 arc-seconds)
Study Area Bounding Box:  [{bbox['min_lon']}, {bbox['min_lat']}] to [{bbox['max_lon']}, {bbox['max_lat']}] (WGS84)
Spatial Analysis Grid:    study_grid.geojson ({total_cells} cells, ~500m x 500m)
Aggregation CRS:          {config.PROJECTED_CRS} (UTM Zone 43N)
Date Range:               [{start_date}, {end_date})
Monthly Images Ingested:  {image_count} (Acquisition: {acq_date})

Aggregation Method:
-------------------------------------------------------------
- Polygonal reduceRegions over complete cell boundary
- Scale: {config.VIIRS_NATIVE_SCALE:.2f} m
- avg_rad: mean, stdDev, count
- cf_cvg: mean, min, max
- valid_pixel_fraction: mean of binary valid mask over polygon [0.0, 1.0]

Feature Statistics:
-------------------------------------------------------------
Grid Cells Processed:     {total_cells}
Cells with Valid Data:    {num_valid} ({num_valid / total_cells * 100:.1f}%)
Cells with Missing Data:  {num_missing} ({num_missing / total_cells * 100:.1f}%)

Metric                       Min       Max       Mean      Std
-------------------------------------------------------------
viirs_avg_rad_mean (nW/cm2)  {rad_min:.4f}   {rad_max:.4f}   {rad_mean:.4f}   {df['viirs_avg_rad_mean'].std():.4f}
viirs_cf_cvg_mean (obs)      {cvg_min:.1f}      {cvg_max:.1f}      {cvg_mean:.1f}      {df['viirs_cf_cvg_mean'].std():.1f}
viirs_valid_pixel_count      {df['viirs_valid_pixel_count'].min()}         {df['viirs_valid_pixel_count'].max()}         {df['viirs_valid_pixel_count'].mean():.1f}       {df['viirs_valid_pixel_count'].std():.1f}
viirs_valid_pixel_fraction   {df['viirs_valid_pixel_fraction'].min():.4f}    {df['viirs_valid_pixel_fraction'].max():.4f}    {frac_mean:.4f}    {df['viirs_valid_pixel_fraction'].std():.4f}

Data Representation & Masking Decisions:
-------------------------------------------------------------
- Zero radiance (avg_rad == 0) is NOT treated as no observations/missing data.
- Missing data is represented strictly as NaN (never zero-imputed).
- cf_cvg is retained as an observation quality indicator, not as a filtering cutoff.
- Native VIIRS pixel resolution is ~463.83m; aggregation over the ~500m grid represents
  spatial re-binning onto the project analysis framework, not sub-pixel resolution.

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
    generate_viirs_validation_maps(df, grid_geojson, output_dir, start_date, end_date)

    print("\n>>> Phase 2 VIIRS Pipeline Completed Successfully! <<<\n")
    return df


def generate_viirs_validation_maps(df, grid_geojson, output_dir, start_date, end_date):
    """
    Generate static visual verification choropleths for VIIRS nighttime radiance
    and cloud-free coverage quality over the canonical 460-cell grid.
    """
    print("[VIZ] Generating VIIRS spatial validation maps...")
    
    val_lookup = {
        row["cell_id"]: {
            "avg_rad": row["viirs_avg_rad_mean"],
            "cf_cvg": row["viirs_cf_cvg_mean"],
        }
        for _, row in df.iterrows()
    }

    metrics = [
        ("avg_rad", "Nighttime Radiance (nW/(cm² sr))", "magma", (0, 80)),
        ("cf_cvg", "Cloud-Free Coverage (Observation Count)", "viridis", (10, 20)),
    ]

    for metric_key, metric_title, cmap_name, (vmin, vmax) in metrics:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        
        patches_list = []
        values_list = []

        for feat in grid_geojson["features"]:
            cid = feat["properties"]["cell_id"]
            coords = feat["geometry"]["coordinates"][0]
            poly = patches.Polygon(coords, closed=True)
            patches_list.append(poly)
            val = val_lookup.get(cid, {}).get(metric_key, np.nan)
            values_list.append(val)

        values_arr = np.array(values_list)
        valid_vals = values_arr[~np.isnan(values_arr)]
        if len(valid_vals) > 0:
            c_min = float(np.percentile(valid_vals, 1))
            c_max = float(np.percentile(valid_vals, 99))
            if c_min >= c_max:
                c_min, c_max = vmin, vmax
        else:
            c_min, c_max = vmin, vmax

        p = PatchCollection(patches_list, cmap=cmap_name, edgecolor="black", linewidth=0.2, alpha=0.9)
        p.set_array(values_arr)
        p.set_clim(c_min, c_max)
        ax.add_collection(p)

        bbox = config.BBOX
        ax.set_xlim(bbox["min_lon"] - 0.005, bbox["max_lon"] + 0.005)
        ax.set_ylim(bbox["min_lat"] - 0.005, bbox["max_lat"] + 0.005)
        ax.set_aspect("equal", "box")

        cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_title, fontsize=11)

        ax.set_title(
            f"Delhi-NCR 500m Grid — VIIRS {metric_title}\n"
            f"Monthly Composite ({start_date} to {end_date})",
            fontsize=12,
            fontweight="bold",
            pad=12
        )
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        out_img = output_dir / f"map_viirs_{metric_key}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"[SAVE] Visual validation map: {out_img}")


if __name__ == "__main__":
    process_viirs_pipeline()
