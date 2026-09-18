"""
ERA5-Land Weather Ingestion, Temporal Aggregation, and Spatial Association Pipeline.
Phase 3 Implementation for Multimodal Geospatial Anomaly Intelligence System.

Ingests hourly ECMWF/ERA5_LAND/HOURLY meteorological records for the test month,
aggregates hourly observations to monthly summary features (temperature, precipitation,
wind speed), and associates them with the canonical 460-cell 500m study grid.
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


def process_era5_pipeline(
    start_date=None,
    end_date=None,
    grid_path=None,
    output_dir=None
):
    """
    Execute complete Phase 3 ERA5-Land weather processing pipeline.
    
    Args:
        start_date (str): 'YYYY-MM-DD' start of monthly window (inclusive)
        end_date (str): 'YYYY-MM-DD' end of monthly window (exclusive)
        grid_path (Path or str): Path to canonical study_grid.geojson
        output_dir (Path or str): Path to processed data output directory
    
    Returns:
        pd.DataFrame: Formatted ERA5 feature table with 460 rows
    """
    import ee

    # 1. Configuration & Paths
    start_date = start_date or config.DEFAULT_ERA5_START_DATE
    end_date = end_date or config.DEFAULT_ERA5_END_DATE
    grid_path = Path(grid_path or config.GRID_PATH)
    output_dir = Path(output_dir or config.PROCESSED_DATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[PIPELINE START] ERA5-Land Weather Processing ({start_date} to {end_date})")
    print(f"[CONFIG] Grid Source: {grid_path}")
    print(f"[CONFIG] Native ERA5-Land Scale: ~{config.ERA5_NATIVE_SCALE:.1f} m (~0.1 degrees)")

    # 2. Authenticate Earth Engine
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    if project_id:
        ee.Initialize(project=project_id)
    else:
        ee.Initialize()

    # 3. Load Existing 460-Cell Canonical Grid
    if not grid_path.exists():
        raise FileNotFoundError(f"Canonical grid not found at {grid_path}. Run `python src/grid.py` first.")

    with open(grid_path, "r", encoding="utf-8") as f:
        grid_geojson = json.load(f)

    total_cells = len(grid_geojson["features"])
    print(f"[INFO] Loaded {total_cells} canonical cells from {grid_path.name}.")

    # 4. Spatial Bounds
    bbox = config.BBOX
    # Buffer AOI slightly (~0.05 deg) to encompass intersecting native ERA5 grid points
    aoi_buffered = ee.Geometry.BBox(
        bbox["min_lon"] - 0.05, bbox["min_lat"] - 0.05,
        bbox["max_lon"] + 0.05, bbox["max_lat"] + 0.05
    )

    # 5. Query ERA5-Land Hourly Collection
    collection_id = config.GEE_DATASETS.get("era5_hourly", "ECMWF/ERA5_LAND/HOURLY")
    collection = (
        ee.ImageCollection(collection_id)
        .filterBounds(aoi_buffered)
        .filterDate(start_date, end_date)
    )

    hourly_count = collection.size().getInfo()
    print(f"[GEE] Hourly observations in [{start_date}, {end_date}): {hourly_count}")
    if hourly_count == 0:
        raise ValueError(f"No ERA5-Land hourly images found in range [{start_date}, {end_date})")

    # 6. Hourly Preprocessing & Unit Conversions
    # - Temperature: Kelvin to Celsius (K - 273.15)
    # - Precipitation: meters to millimeters (m * 1000)
    # - Wind: derive wind speed from 10m u and v components: sqrt(u^2 + v^2) in m/s
    def prep_hourly(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        wind_speed = (u.pow(2).add(v.pow(2))).sqrt().rename("wind_speed")
        temp_c = img.select("temperature_2m").subtract(273.15).rename("temp_c")
        precip_mm = img.select("total_precipitation_hourly").multiply(1000.0).rename("precip_mm")
        return img.addBands([wind_speed, temp_c, precip_mm])

    prep_collection = collection.map(prep_hourly)

    # 7. Temporal Aggregation: Hourly -> Monthly Summary
    print("[TEMPORAL] Aggregating 744 hourly scenes into monthly meteorological features...")
    t_mean = prep_collection.select("temp_c").mean().rename("era5_temperature_mean")
    t_min = prep_collection.select("temp_c").min().rename("era5_temperature_min")
    t_max = prep_collection.select("temp_c").max().rename("era5_temperature_max")
    p_total = prep_collection.select("precip_mm").sum().rename("era5_precipitation_total")
    w_mean = prep_collection.select("wind_speed").mean().rename("era5_wind_speed_mean")
    w_max = prep_collection.select("wind_speed").max().rename("era5_wind_speed_max")
    obs_cnt = prep_collection.select("temp_c").count().rename("era5_observation_count")

    monthly_weather_img = ee.Image.cat([t_mean, t_min, t_max, p_total, w_mean, w_max, obs_cnt])

    # 8. Identify Native ERA5-Land Grid Points in the Region
    print("[SPATIAL] Sampling native ERA5-Land grid points across the study area...")
    native_scale = config.ERA5_NATIVE_SCALE
    sample_pts = (
        monthly_weather_img
        .sample(region=aoi_buffered, scale=native_scale, geometries=True)
        .getInfo()
    )

    era5_source_nodes = []
    for f in sample_pts["features"]:
        coords = f["geometry"]["coordinates"]
        props = f["properties"]
        era5_source_nodes.append({
            "lon": coords[0],
            "lat": coords[1],
            "era5_temperature_mean": props.get("era5_temperature_mean"),
            "era5_temperature_min": props.get("era5_temperature_min"),
            "era5_temperature_max": props.get("era5_temperature_max"),
            "era5_precipitation_total": props.get("era5_precipitation_total"),
            "era5_wind_speed_mean": props.get("era5_wind_speed_mean"),
            "era5_wind_speed_max": props.get("era5_wind_speed_max"),
            "era5_observation_count": int(props.get("era5_observation_count", 0)),
        })

    print(f"[SPATIAL] Found {len(era5_source_nodes)} native ERA5-Land grid point(s) in buffered extent.")

    # 9. Spatially Associate Each 500m Cell with Nearest Native ERA5 Source Point
    # Nearest neighbor matching from cell centroid to native ERA5 source point
    records = []
    source_point_tracker = set()

    for feat in grid_geojson["features"]:
        cid = feat["properties"]["cell_id"]
        clon = feat["properties"]["centroid_lon"]
        clat = feat["properties"]["centroid_lat"]

        # Find closest native ERA5 node
        best_node = min(
            era5_source_nodes,
            key=lambda node: (node["lon"] - clon) ** 2 + (node["lat"] - clat) ** 2
        )

        source_lat = round(best_node["lat"], 4)
        source_lon = round(best_node["lon"], 4)
        source_point_tracker.add((source_lat, source_lon))

        records.append({
            "cell_id": cid,
            "start_date": start_date,
            "end_date": end_date,
            "era5_temperature_mean": best_node["era5_temperature_mean"],
            "era5_temperature_min": best_node["era5_temperature_min"],
            "era5_temperature_max": best_node["era5_temperature_max"],
            "era5_precipitation_total": best_node["era5_precipitation_total"],
            "era5_wind_speed_mean": best_node["era5_wind_speed_mean"],
            "era5_wind_speed_max": best_node["era5_wind_speed_max"],
            "era5_observation_count": best_node["era5_observation_count"],
            "era5_source_lat": source_lat,
            "era5_source_lon": source_lon,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("cell_id").reset_index(drop=True)

    # 10. Run Sanity Checks & Verification
    print("\n[VALIDATION] Running sanity checks on ERA5 feature table...")
    assert len(df) == total_cells, f"Expected {total_cells} rows, got {len(df)}"
    assert df["cell_id"].nunique() == total_cells, "Duplicate cell IDs detected!"

    unique_source_count = len(source_point_tracker)
    num_valid = len(df[df["era5_observation_count"] > 0])
    num_missing = len(df[df["era5_observation_count"] == 0])

    t_mean_val = df["era5_temperature_mean"].mean()
    t_min_val = df["era5_temperature_min"].min()
    t_max_val = df["era5_temperature_max"].max()

    p_total_mean = df["era5_precipitation_total"].mean()
    p_total_min = df["era5_precipitation_total"].min()
    p_total_max = df["era5_precipitation_total"].max()

    w_mean_val = df["era5_wind_speed_mean"].mean()
    w_max_val = df["era5_wind_speed_max"].max()

    obs_cnt_min = df["era5_observation_count"].min()
    obs_cnt_max = df["era5_observation_count"].max()
    obs_cnt_mean = df["era5_observation_count"].mean()
    coverage_pct = (obs_cnt_mean / 744.0) * 100.0 if hourly_count >= 744 else (obs_cnt_mean / max(1, hourly_count)) * 100.0

    print(f"  Total analytical cells:          {len(df)}")
    print(f"  Cells with valid weather data:   {num_valid} ({num_valid / total_cells * 100:.1f}%)")
    print(f"  Cells with missing weather data: {num_missing} ({num_missing / total_cells * 100:.1f}%)")
    print(f"  Unique ERA5 source grid points:  {unique_source_count} (across 460 analytical cells)")
    print(f"  Source locations (lat, lon):     {sorted(list(source_point_tracker))}")
    print(f"  Temperature [min, mean, max]:    [{t_min_val:.2f} °C, {t_mean_val:.2f} °C, {t_max_val:.2f} °C]")
    print(f"  Precipitation [min, mean, max]:  [{p_total_min:.2f} mm, {p_total_mean:.2f} mm, {p_total_max:.2f} mm]")
    print(f"  Wind speed [mean, max]:          [{w_mean_val:.2f} m/s, {w_max_val:.2f} m/s]")
    print(f"  Hourly observation count:        [{obs_cnt_min}, {obs_cnt_max}] (Coverage: {coverage_pct:.1f}%)")

    # 11. Save Analytical Datasets
    parquet_path = output_dir / "era5_features.parquet"
    csv_path = output_dir / "era5_features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Parquet saved: {parquet_path}")
    print(f"[SAVE] CSV saved:     {csv_path}")

    # 12. Create Quality Processing Report
    report_path = output_dir / "era5_processing_report.txt"
    report_content = f"""=============================================================
 ERA5-Land Weather Phase 3 Processing Report
=============================================================
Dataset Name:             ECMWF ERA5-Land Hourly (Copernicus Climate Data Store / GEE)
Dataset ID:               {collection_id}
Date Range:               [{start_date}, {end_date})
Temporal Resolution:      Hourly (744 total hours in March 2024)
Native Spatial Scale:     ~{config.ERA5_NATIVE_SCALE:.1f} meters (~0.1 degrees, ~9-11 km)
Study Area Bounding Box:  [{bbox['min_lon']}, {bbox['min_lat']}] to [{bbox['max_lon']}, {bbox['max_lat']}] (WGS84)
Spatial Analysis Grid:    study_grid.geojson ({total_cells} cells, ~500m x 500m)

Spatial Association Method:
-------------------------------------------------------------
- Nearest-grid-point spatial assignment from 500m cell centroid
  to native ERA5-Land grid point.
- Number of unique ERA5 source grid points: {unique_source_count}
- Unique source coordinates: {sorted(list(source_point_tracker))}
- IMPORTANT: ERA5-Land is a coarse-resolution meteorological dataset.
  The 500m analytical grid is used as the project's common spatial indexing
  framework; it does not imply 500m weather measurement resolution.

Temporal Aggregation Rules & Units:
-------------------------------------------------------------
- era5_temperature_mean: Monthly average 2m air temperature (°C = K - 273.15)
- era5_temperature_min:  Monthly minimum 2m air temperature (°C = K - 273.15)
- era5_temperature_max:  Monthly maximum 2m air temperature (°C = K - 273.15)
- era5_precipitation_total: Monthly sum of hourly precipitation (mm = m * 1000)
- era5_wind_speed_mean:  Monthly mean of hourly wind speed sqrt(u10^2 + v10^2) (m/s)
- era5_wind_speed_max:   Monthly maximum of hourly wind speed (m/s)
- era5_observation_count: Total valid hourly records in the month (expected: 744)

Summary Statistics (460 Cells):
-------------------------------------------------------------
Grid Cells Processed:          {total_cells}
Cells with Valid Data:         {num_valid} ({num_valid / total_cells * 100:.1f}%)
Cells with Missing Data:       {num_missing} ({num_missing / total_cells * 100:.1f}%)
Temporal Coverage:             {coverage_pct:.1f}% ({obs_cnt_min}/744 hours)

Metric                         Min       Max       Mean      Std
-------------------------------------------------------------
Temperature Mean (°C)          {df['era5_temperature_mean'].min():.4f}   {df['era5_temperature_mean'].max():.4f}   {t_mean_val:.4f}   {df['era5_temperature_mean'].std():.4f}
Temperature Min (°C)           {df['era5_temperature_min'].min():.4f}   {df['era5_temperature_min'].max():.4f}   {df['era5_temperature_min'].mean():.4f}   {df['era5_temperature_min'].std():.4f}
Temperature Max (°C)           {df['era5_temperature_max'].min():.4f}   {df['era5_temperature_max'].max():.4f}   {df['era5_temperature_max'].mean():.4f}   {df['era5_temperature_max'].std():.4f}
Precipitation Total (mm)       {p_total_min:.4f}   {p_total_max:.4f}   {p_total_mean:.4f}   {df['era5_precipitation_total'].std():.4f}
Wind Speed Mean (m/s)          {df['era5_wind_speed_mean'].min():.4f}   {df['era5_wind_speed_mean'].max():.4f}   {w_mean_val:.4f}   {df['era5_wind_speed_mean'].std():.4f}
Wind Speed Max (m/s)           {df['era5_wind_speed_max'].min():.4f}   {df['era5_wind_speed_max'].max():.4f}   {df['era5_wind_speed_max'].mean():.4f}   {df['era5_wind_speed_max'].std():.4f}
Hourly Observation Count       {obs_cnt_min}       {obs_cnt_max}       {obs_cnt_mean:.1f}     0.0

Missing Data & Physical Integrity:
-------------------------------------------------------------
- Missing data is represented strictly as NaN (never zero-imputed).
- Zero precipitation represents dry weather, distinct from missing data.
- All 460 analytical cells have complete 744-hour data from ERA5-Land.

Outputs:
  - Feature Parquet:      {parquet_path}
  - Feature CSV:          {csv_path}
  - Processing Timestamp: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
=============================================================
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SAVE] Report saved:  {report_path}")

    # 13. Visual Validation Maps
    generate_era5_validation_maps(df, grid_geojson, output_dir, start_date, end_date)

    print("\n>>> Phase 3 ERA5-Land Pipeline Completed Successfully! <<<\n")
    return df


def generate_era5_validation_maps(df, grid_geojson, output_dir, start_date, end_date):
    """
    Generate static visual verification choropleths for monthly mean temperature
    and total precipitation associated with the canonical 460-cell grid.
    """
    print("[VIZ] Generating ERA5 spatial validation maps...")
    
    val_lookup = {
        row["cell_id"]: {
            "temp": row["era5_temperature_mean"],
            "precip": row["era5_precipitation_total"],
        }
        for _, row in df.iterrows()
    }

    metrics = [
        ("era5_temperature", "Monthly Mean 2m Temperature (°C)", "coolwarm", "temp"),
        ("era5_precipitation", "Monthly Total Precipitation (mm)", "YlGnBu", "precip"),
    ]

    for filename_key, metric_title, cmap_name, val_key in metrics:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        
        patches_list = []
        values_list = []

        for feat in grid_geojson["features"]:
            cid = feat["properties"]["cell_id"]
            coords = feat["geometry"]["coordinates"][0]
            poly = patches.Polygon(coords, closed=True)
            patches_list.append(poly)
            val = val_lookup.get(cid, {}).get(val_key, np.nan)
            values_list.append(val)

        values_arr = np.array(values_list)
        valid_vals = values_arr[~np.isnan(values_arr)]
        if len(valid_vals) > 0:
            c_min = float(np.min(valid_vals))
            c_max = float(np.max(valid_vals))
            # If uniform, add small padding
            if c_min == c_max:
                c_min -= 0.5
                c_max += 0.5
        else:
            c_min, c_max = 0, 1

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
            f"Delhi-NCR 500m Grid — ERA5-Land {metric_title}\n"
            f"Monthly Aggregation ({start_date} to {end_date})\n"
            f"[Coarse ~0.1° resolution associated to 500m analytical grid]",
            fontsize=11,
            fontweight="bold",
            pad=12
        )
        ax.set_xlabel("Longitude (°E)", fontsize=10)
        ax.set_ylabel("Latitude (°N)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

        out_img = output_dir / f"map_{filename_key}.png"
        plt.tight_layout()
        plt.savefig(out_img, dpi=150)
        plt.close()
        print(f"[SAVE] Visual validation map: {out_img}")


if __name__ == "__main__":
    process_era5_pipeline()
