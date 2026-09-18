"""
Historical Temporal Expansion Pipeline for Multimodal Geospatial Anomaly Intelligence System.
Phase 4 Implementation: Builds longitudinal monthly datasets (cell_id x month x features)
covering 12 consecutive months across Sentinel-2, VIIRS Nighttime Lights, and ERA5-Land Weather.
"""

import sys
import os
import json
import calendar
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


# ---------------------------------------------------------------------------
# Reusable Calendar-Aware Month Range Generator
# ---------------------------------------------------------------------------
def get_month_ranges(start_date_str: str, end_date_str: str):
    """
    Generate calendar-aware monthly intervals [month_start, month_end).
    
    Returns:
        list of tuples: (month_key, start_str, end_str, days_in_month)
        e.g. ('2023-04', '2023-04-01', '2023-05-01', 30)
    """
    start = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()

    current = start
    ranges = []
    while current < end:
        year = current.year
        month = current.month
        days_in_month = calendar.monthrange(year, month)[1]
        m_start = datetime.date(year, month, 1)
        if month == 12:
            m_end = datetime.date(year + 1, 1, 1)
        else:
            m_end = datetime.date(year, month + 1, 1)

        if m_end > end:
            m_end = end

        month_key = f"{year}-{month:02d}"
        ranges.append((
            month_key,
            m_start.strftime("%Y-%m-%d"),
            m_end.strftime("%Y-%m-%d"),
            days_in_month
        ))
        current = m_end

    return ranges


# ---------------------------------------------------------------------------
# Sentinel-2 SCL Cloud Masking and Index Calculation (Reused from Phase 1)
# ---------------------------------------------------------------------------
def mask_s2_clouds_scl(image):
    """Mask clouds, shadows, and invalid pixels using Sentinel-2 L2A SCL."""
    import ee
    scl = image.select("SCL")
    valid_mask = (
        scl.eq(2)
        .Or(scl.eq(4))
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )
    reflective_bands = image.select(["B2", "B3", "B4", "B8", "B11"]).divide(10000.0)
    return image.addBands(reflective_bands, overwrite=True).updateMask(valid_mask)


def compute_s2_indices(image):
    """Compute safe NDVI, NDBI, and NDWI."""
    import ee
    b3 = image.select("B3")
    b4 = image.select("B4")
    b8 = image.select("B8")
    b11 = image.select("B11")

    denom_ndvi = b8.add(b4)
    mask_ndvi = denom_ndvi.abs().gt(1e-5)
    ndvi = b8.subtract(b4).divide(denom_ndvi).updateMask(mask_ndvi).rename("ndvi")

    denom_ndbi = b11.add(b8)
    mask_ndbi = denom_ndbi.abs().gt(1e-5)
    ndbi = b11.subtract(b8).divide(denom_ndbi).updateMask(mask_ndbi).rename("ndbi")

    denom_ndwi = b3.add(b8)
    mask_ndwi = denom_ndwi.abs().gt(1e-5)
    ndwi = b3.subtract(b8).divide(denom_ndwi).updateMask(mask_ndwi).rename("ndwi")

    return image.addBands([ndvi, ndbi, ndwi])


# ---------------------------------------------------------------------------
# Modality 1: Sentinel-2 Historical Processing
# ---------------------------------------------------------------------------
def process_historical_sentinel2(month_ranges, aoi, ee_grid, grid_geojson, max_cloud=20):
    import ee
    print("\n" + "=" * 65)
    print(" [MODALITY 1/3] Sentinel-2 Historical Processing")
    print("=" * 65)

    s2_id = config.GEE_DATASETS["sentinel2"]
    base_collection = ee.ImageCollection(s2_id).filterBounds(aoi)

    s2_records = []
    s2_metadata = {}
    EXPECTED_PIXELS_PER_CELL = 625

    reducer = (
        ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )

    for idx, (m_key, m_start, m_end, _) in enumerate(month_ranges, 1):
        print(f"  [{idx:02d}/{len(month_ranges):02d}] Sentinel-2: {m_key} ({m_start} to {m_end})")
        m_col = base_collection.filterDate(m_start, m_end)
        raw_cnt = m_col.size().getInfo()

        clean_col = m_col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        clean_cnt = clean_col.size().getInfo()

        s2_metadata[m_key] = {
            "raw_scenes": raw_cnt,
            "clean_scenes": clean_cnt,
            "used_scenes": clean_cnt,
        }

        # If zero scenes under threshold, handle explicitly
        if clean_cnt == 0:
            print(f"      [WARNING] 0 clean scenes (<= {max_cloud}%) in {m_key}. Candidate scenes: {raw_cnt}")
            # If candidate scenes exist, check if slightly higher threshold (e.g. <= 40%) has scenes with SCL masking
            fallback_col = m_col.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
            fb_cnt = fallback_col.size().getInfo()
            if fb_cnt > 0:
                print(f"      [INFO] Falling back to {fb_cnt} scenes with cloud <= 40% (protected by SCL pixel masking).")
                clean_col = fallback_col
                clean_cnt = fb_cnt
                s2_metadata[m_key]["used_scenes"] = clean_cnt
            else:
                print(f"      [MISSING] No usable scenes in {m_key}. Filling with NaN.")
                for feat in grid_geojson["features"]:
                    cid = feat["properties"]["cell_id"]
                    s2_records.append({
                        "cell_id": cid,
                        "month": m_key,
                        "start_date": m_start,
                        "end_date": m_end,
                        "ndvi_mean": np.nan,
                        "ndbi_mean": np.nan,
                        "ndwi_mean": np.nan,
                        "ndvi_std": np.nan,
                        "ndbi_std": np.nan,
                        "ndwi_std": np.nan,
                        "valid_pixel_count": 0,
                        "valid_pixel_fraction": 0.0,
                    })
                continue

        # Process valid scenes
        processed = clean_col.map(mask_s2_clouds_scl).map(compute_s2_indices)
        composite = processed.select(["ndvi", "ndbi", "ndwi"]).median()

        reduced = composite.reduceRegions(
            collection=ee_grid,
            reducer=reducer,
            scale=20,
            crs=config.PROJECTED_CRS
        )

        res_features = reduced.getInfo()["features"]
        for feat in res_features:
            props = feat["properties"]
            cid = props.get("cell_id")
            cnt = props.get("ndvi_count")
            if cnt is None or cnt == 0:
                v_cnt = 0
                v_frac = 0.0
                n_mean, n_std = np.nan, np.nan
                b_mean, b_std = np.nan, np.nan
                w_mean, w_std = np.nan, np.nan
            else:
                v_cnt = int(cnt)
                v_frac = round(min(1.0, v_cnt / EXPECTED_PIXELS_PER_CELL), 4)
                n_mean = props.get("ndvi_mean", np.nan)
                n_std = props.get("ndvi_stdDev", np.nan)
                b_mean = props.get("ndbi_mean", np.nan)
                b_std = props.get("ndbi_stdDev", np.nan)
                w_mean = props.get("ndwi_mean", np.nan)
                w_std = props.get("ndwi_stdDev", np.nan)

            s2_records.append({
                "cell_id": cid,
                "month": m_key,
                "start_date": m_start,
                "end_date": m_end,
                "ndvi_mean": n_mean,
                "ndbi_mean": b_mean,
                "ndwi_mean": w_mean,
                "ndvi_std": n_std,
                "ndbi_std": b_std,
                "ndwi_std": w_std,
                "valid_pixel_count": v_cnt,
                "valid_pixel_fraction": v_frac,
            })

    df_s2 = pd.DataFrame(s2_records).sort_values(["cell_id", "month"]).reset_index(drop=True)
    out_path = config.PROCESSED_DATA_DIR / "historical_s2_features.parquet"
    df_s2.to_parquet(out_path, index=False)
    print(f"[SAVE] Historical Sentinel-2 saved: {out_path} ({len(df_s2)} rows)")
    return df_s2, s2_metadata


# ---------------------------------------------------------------------------
# Modality 2: VIIRS Historical Processing
# ---------------------------------------------------------------------------
def process_historical_viirs(month_ranges, aoi, ee_grid, grid_geojson):
    import ee
    print("\n" + "=" * 65)
    print(" [MODALITY 2/3] VIIRS Nighttime Lights Historical Processing")
    print("=" * 65)

    collection_id = config.GEE_DATASETS["viirs"]
    base_col = ee.ImageCollection(collection_id).filterBounds(aoi)

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

    viirs_records = []
    viirs_metadata = {}

    for idx, (m_key, m_start, m_end, _) in enumerate(month_ranges, 1):
        print(f"  [{idx:02d}/{len(month_ranges):02d}] VIIRS: {m_key} ({m_start} to {m_end})")
        m_col = base_col.filterDate(m_start, m_end)
        img_cnt = m_col.size().getInfo()

        viirs_metadata[m_key] = {"image_count": img_cnt}

        if img_cnt == 0:
            print(f"      [MISSING] No VIIRS monthly composite in {m_key}. Filling with NaN.")
            for feat in grid_geojson["features"]:
                cid = feat["properties"]["cell_id"]
                viirs_records.append({
                    "cell_id": cid,
                    "month": m_key,
                    "start_date": m_start,
                    "end_date": m_end,
                    "viirs_avg_rad_mean": np.nan,
                    "viirs_avg_rad_std": np.nan,
                    "viirs_cf_cvg_mean": np.nan,
                    "viirs_cf_cvg_min": np.nan,
                    "viirs_cf_cvg_max": np.nan,
                    "viirs_valid_pixel_count": 0,
                    "viirs_valid_pixel_fraction": 0.0,
                })
            continue

        viirs_img = m_col.first()
        mask_layer = viirs_img.select("avg_rad").mask().rename("valid_mask")
        multi_img = viirs_img.select(["avg_rad", "cf_cvg"]).addBands(mask_layer)

        reduced_rad = multi_img.select(["avg_rad"]).reduceRegions(
            collection=ee_grid, reducer=rad_reducer, scale=config.VIIRS_NATIVE_SCALE, crs=config.PROJECTED_CRS
        ).getInfo()["features"]

        reduced_cvg = multi_img.select(["cf_cvg"]).reduceRegions(
            collection=ee_grid, reducer=cvg_reducer, scale=config.VIIRS_NATIVE_SCALE, crs=config.PROJECTED_CRS
        ).getInfo()["features"]

        reduced_mask = multi_img.select(["valid_mask"]).reduceRegions(
            collection=ee_grid, reducer=mask_reducer, scale=config.VIIRS_NATIVE_SCALE, crs=config.PROJECTED_CRS
        ).getInfo()["features"]

        rad_lookup = {f["properties"]["cell_id"]: f["properties"] for f in reduced_rad}
        cvg_lookup = {f["properties"]["cell_id"]: f["properties"] for f in reduced_cvg}
        mask_lookup = {f["properties"]["cell_id"]: f["properties"] for f in reduced_mask}

        for feat in grid_geojson["features"]:
            cid = feat["properties"]["cell_id"]
            r_props = rad_lookup.get(cid, {})
            c_props = cvg_lookup.get(cid, {})
            m_props = mask_lookup.get(cid, {})

            count = r_props.get("count")
            if count is None or count == 0:
                v_cnt = 0
                v_frac = 0.0
                rad_mean, rad_std = np.nan, np.nan
                c_mean, c_min, c_max = np.nan, np.nan, np.nan
            else:
                v_cnt = int(count)
                raw_frac = m_props.get("mean", 1.0)
                v_frac = round(float(np.clip(raw_frac, 0.0, 1.0)), 4) if raw_frac is not None else 1.0
                rad_mean = r_props.get("mean", np.nan)
                rad_std = r_props.get("stdDev", np.nan)
                c_mean = c_props.get("mean", np.nan)
                c_min = c_props.get("min", np.nan)
                c_max = c_props.get("max", np.nan)

            viirs_records.append({
                "cell_id": cid,
                "month": m_key,
                "start_date": m_start,
                "end_date": m_end,
                "viirs_avg_rad_mean": rad_mean,
                "viirs_avg_rad_std": rad_std,
                "viirs_cf_cvg_mean": c_mean,
                "viirs_cf_cvg_min": c_min,
                "viirs_cf_cvg_max": c_max,
                "viirs_valid_pixel_count": v_cnt,
                "viirs_valid_pixel_fraction": v_frac,
            })

    df_viirs = pd.DataFrame(viirs_records).sort_values(["cell_id", "month"]).reset_index(drop=True)
    out_path = config.PROCESSED_DATA_DIR / "historical_viirs_features.parquet"
    df_viirs.to_parquet(out_path, index=False)
    print(f"[SAVE] Historical VIIRS saved: {out_path} ({len(df_viirs)} rows)")
    return df_viirs, viirs_metadata


# ---------------------------------------------------------------------------
# Modality 3: ERA5-Land Historical Processing
# ---------------------------------------------------------------------------
def process_historical_era5(month_ranges, aoi_buffered, grid_geojson):
    import ee
    print("\n" + "=" * 65)
    print(" [MODALITY 3/3] ERA5-Land Weather Historical Processing")
    print("=" * 65)

    era5_id = config.GEE_DATASETS.get("era5_hourly", "ECMWF/ERA5_LAND/HOURLY")
    base_col = ee.ImageCollection(era5_id).filterBounds(aoi_buffered)

    def prep_hourly(img):
        u = img.select("u_component_of_wind_10m")
        v = img.select("v_component_of_wind_10m")
        ws = (u.pow(2).add(v.pow(2))).sqrt().rename("wind_speed")
        t_c = img.select("temperature_2m").subtract(273.15).rename("temp_c")
        p_mm = img.select("total_precipitation_hourly").multiply(1000.0).rename("precip_mm")
        return img.addBands([ws, t_c, p_mm])

    era5_records = []
    era5_metadata = {}

    for idx, (m_key, m_start, m_end, days_in_month) in enumerate(month_ranges, 1):
        expected_hours = days_in_month * 24
        print(f"  [{idx:02d}/{len(month_ranges):02d}] ERA5-Land: {m_key} ({m_start} to {m_end}, Expected: {expected_hours}h)")

        m_col = base_col.filterDate(m_start, m_end)
        act_cnt = m_col.size().getInfo()

        era5_metadata[m_key] = {
            "expected_hours": expected_hours,
            "actual_hours": act_cnt,
            "completeness_ratio": round(act_cnt / expected_hours, 4) if expected_hours > 0 else 0.0,
        }

        if act_cnt == 0:
            print(f"      [MISSING] No ERA5 hourly data for {m_key}. Filling with NaN.")
            for feat in grid_geojson["features"]:
                cid = feat["properties"]["cell_id"]
                era5_records.append({
                    "cell_id": cid,
                    "month": m_key,
                    "start_date": m_start,
                    "end_date": m_end,
                    "era5_temperature_mean": np.nan,
                    "era5_temperature_min": np.nan,
                    "era5_temperature_max": np.nan,
                    "era5_precipitation_total": np.nan,
                    "era5_wind_speed_mean": np.nan,
                    "era5_wind_speed_max": np.nan,
                    "era5_observation_count": 0,
                    "era5_source_lat": np.nan,
                    "era5_source_lon": np.nan,
                })
            continue

        prep_col = m_col.map(prep_hourly)
        t_mean = prep_col.select("temp_c").mean().rename("era5_temperature_mean")
        t_min = prep_col.select("temp_c").min().rename("era5_temperature_min")
        t_max = prep_col.select("temp_c").max().rename("era5_temperature_max")
        p_total = prep_col.select("precip_mm").sum().rename("era5_precipitation_total")
        w_mean = prep_col.select("wind_speed").mean().rename("era5_wind_speed_mean")
        w_max = prep_col.select("wind_speed").max().rename("era5_wind_speed_max")
        obs_cnt = prep_col.select("temp_c").count().rename("era5_observation_count")

        monthly_img = ee.Image.cat([t_mean, t_min, t_max, p_total, w_mean, w_max, obs_cnt])

        sample_pts = monthly_img.sample(
            region=aoi_buffered, scale=config.ERA5_NATIVE_SCALE, geometries=True
        ).getInfo()

        era5_nodes = []
        for f in sample_pts["features"]:
            coords = f["geometry"]["coordinates"]
            props = f["properties"]
            era5_nodes.append({
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

        for feat in grid_geojson["features"]:
            cid = feat["properties"]["cell_id"]
            clon = feat["properties"]["centroid_lon"]
            clat = feat["properties"]["centroid_lat"]

            best_node = min(
                era5_nodes,
                key=lambda node: (node["lon"] - clon) ** 2 + (node["lat"] - clat) ** 2
            )

            era5_records.append({
                "cell_id": cid,
                "month": m_key,
                "start_date": m_start,
                "end_date": m_end,
                "era5_temperature_mean": best_node["era5_temperature_mean"],
                "era5_temperature_min": best_node["era5_temperature_min"],
                "era5_temperature_max": best_node["era5_temperature_max"],
                "era5_precipitation_total": best_node["era5_precipitation_total"],
                "era5_wind_speed_mean": best_node["era5_wind_speed_mean"],
                "era5_wind_speed_max": best_node["era5_wind_speed_max"],
                "era5_observation_count": best_node["era5_observation_count"],
                "era5_source_lat": round(best_node["lat"], 4),
                "era5_source_lon": round(best_node["lon"], 4),
            })

    df_era5 = pd.DataFrame(era5_records).sort_values(["cell_id", "month"]).reset_index(drop=True)
    out_path = config.PROCESSED_DATA_DIR / "historical_era5_features.parquet"
    df_era5.to_parquet(out_path, index=False)
    print(f"[SAVE] Historical ERA5 saved: {out_path} ({len(df_era5)} rows)")
    return df_era5, era5_metadata


# ---------------------------------------------------------------------------
# Step 11: Join into Common Multimodal Dataset
# ---------------------------------------------------------------------------
def build_multimodal_dataset(df_s2, df_viirs, df_era5):
    print("\n" + "=" * 65)
    print(" [STEP 11] Joining Modalities into Common Multimodal Dataset")
    print("=" * 65)

    # Base on complete index: cell_id, month, start_date, end_date
    join_keys = ["cell_id", "month", "start_date", "end_date"]

    # Outer join to ensure no cells or months are dropped
    df_merged = df_s2.merge(df_viirs, on=join_keys, how="outer")
    df_merged = df_merged.merge(df_era5, on=join_keys, how="outer")

    df_merged = df_merged.sort_values(["cell_id", "month"]).reset_index(drop=True)

    parquet_path = config.PROCESSED_DATA_DIR / "historical_multimodal_features.parquet"
    csv_path = config.PROCESSED_DATA_DIR / "historical_multimodal_features.csv"

    df_merged.to_parquet(parquet_path, index=False)
    df_merged.to_csv(csv_path, index=False)

    print(f"[SAVE] Historical Multimodal Parquet: {parquet_path}")
    print(f"[SAVE] Historical Multimodal CSV:     {csv_path}")
    print(f"[INFO] Final Table Dimensions:        {df_merged.shape[0]} rows x {df_merged.shape[1]} columns")

    return df_merged


# ---------------------------------------------------------------------------
# Step 14: Diagnostic Temporal Time-Series Plots
# ---------------------------------------------------------------------------
def generate_temporal_diagnostic_plots(df_multi, output_dir):
    print("\n[STEP 14] Generating temporal diagnostic time-series plots...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by month for regional mean
    monthly_agg = df_multi.groupby("month").agg({
        "ndvi_mean": "mean",
        "ndbi_mean": "mean",
        "ndwi_mean": "mean",
        "viirs_avg_rad_mean": "mean",
        "era5_temperature_mean": "mean",
        "era5_precipitation_total": "mean",
        "era5_wind_speed_mean": "mean",
    }).reset_index()

    plots = [
        ("historical_ndvi_trend.png", "ndvi_mean", "Regional Mean NDVI (Vegetation Index)", "#2ca02c", (-0.1, 0.6)),
        ("historical_ndbi_trend.png", "ndbi_mean", "Regional Mean NDBI (Built-up Index)", "#d62728", (-0.3, 0.3)),
        ("historical_ndwi_trend.png", "ndwi_mean", "Regional Mean NDWI (Water Index)", "#1f77b4", (-0.5, 0.1)),
        ("historical_viirs_avg_rad_trend.png", "viirs_avg_rad_mean", "Regional Mean VIIRS Radiance (nW/(cm² sr))", "#9467bd", (0, 80)),
        ("historical_era5_temperature_trend.png", "era5_temperature_mean", "Regional Mean 2m Temperature (°C)", "#ff7f0e", (10, 40)),
        ("historical_era5_precipitation_trend.png", "era5_precipitation_total", "Regional Mean Monthly Precipitation (mm)", "#17becf", (0, None)),
        ("historical_era5_wind_speed_trend.png", "era5_wind_speed_mean", "Regional Mean 10m Wind Speed (m/s)", "#7f7f7f", (0, 6)),
    ]

    months = monthly_agg["month"].tolist()
    x = range(len(months))

    for filename, col, title, color, ylim in plots:
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        y = monthly_agg[col].values

        ax.plot(x, y, marker="o", linewidth=2, color=color, label=title)
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
        ax.set_title(f"12-Month Temporal Trend: {title}\n(Delhi-NCR 10km Study Area, 2023-04 to 2024-03)", fontsize=11, fontweight="bold")
        ax.set_ylabel(title.split("(")[0].strip(), fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        if ylim[0] is not None and ylim[1] is not None:
            ax.set_ylim(ylim[0], ylim[1])
        elif ylim[0] is not None:
            ax.set_ylim(bottom=ylim[0])

        plt.tight_layout()
        out_file = output_dir / filename
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"  [PLOT] Saved {out_file.name}")


# ---------------------------------------------------------------------------
# Step 18: March 2024 Regression Check against Phase 1, 2, 3 outputs
# ---------------------------------------------------------------------------
def run_regression_check(df_multi):
    print("\n" + "=" * 65)
    print(" [STEP 18] March 2024 Regression Check against Phase 1/2/3 Benchmarks")
    print("=" * 65)

    df_m03 = df_multi[df_multi["month"] == "2024-03"].sort_values("cell_id").reset_index(drop=True)

    s2_bench_path = config.PROCESSED_DATA_DIR / "s2_features.parquet"
    viirs_bench_path = config.PROCESSED_DATA_DIR / "viirs_features.parquet"
    era5_bench_path = config.PROCESSED_DATA_DIR / "era5_features.parquet"

    results = {}

    # S2 comparison
    if s2_bench_path.exists():
        df_s2_bench = pd.read_parquet(s2_bench_path).sort_values("cell_id").reset_index(drop=True)
        diff_ndvi = np.abs(df_m03["ndvi_mean"].values - df_s2_bench["ndvi_mean"].values)
        diff_ndbi = np.abs(df_m03["ndbi_mean"].values - df_s2_bench["ndbi_mean"].values)
        diff_ndwi = np.abs(df_m03["ndwi_mean"].values - df_s2_bench["ndwi_mean"].values)
        results["s2"] = {
            "max_abs_diff_ndvi": float(np.nanmax(diff_ndvi)),
            "mean_abs_diff_ndvi": float(np.nanmean(diff_ndvi)),
            "mismatch_count_ndvi": int(np.sum(diff_ndvi > 1e-4)),
        }
        print(f"  Sentinel-2 NDVI: Max Diff = {results['s2']['max_abs_diff_ndvi']:.6e}, Mean Diff = {results['s2']['mean_abs_diff_ndvi']:.6e}, Mismatches = {results['s2']['mismatch_count_ndvi']}")

    # VIIRS comparison
    if viirs_bench_path.exists():
        df_viirs_bench = pd.read_parquet(viirs_bench_path).sort_values("cell_id").reset_index(drop=True)
        diff_rad = np.abs(df_m03["viirs_avg_rad_mean"].values - df_viirs_bench["viirs_avg_rad_mean"].values)
        diff_cvg = np.abs(df_m03["viirs_cf_cvg_mean"].values - df_viirs_bench["viirs_cf_cvg_mean"].values)
        results["viirs"] = {
            "max_abs_diff_rad": float(np.nanmax(diff_rad)),
            "mean_abs_diff_rad": float(np.nanmean(diff_rad)),
            "mismatch_count_rad": int(np.sum(diff_rad > 1e-4)),
        }
        print(f"  VIIRS Radiance:  Max Diff = {results['viirs']['max_abs_diff_rad']:.6e}, Mean Diff = {results['viirs']['mean_abs_diff_rad']:.6e}, Mismatches = {results['viirs']['mismatch_count_rad']}")

    # ERA5 comparison
    if era5_bench_path.exists():
        df_era5_bench = pd.read_parquet(era5_bench_path).sort_values("cell_id").reset_index(drop=True)
        diff_temp = np.abs(df_m03["era5_temperature_mean"].values - df_era5_bench["era5_temperature_mean"].values)
        diff_precip = np.abs(df_m03["era5_precipitation_total"].values - df_era5_bench["era5_precipitation_total"].values)
        results["era5"] = {
            "max_abs_diff_temp": float(np.nanmax(diff_temp)),
            "mean_abs_diff_temp": float(np.nanmean(diff_temp)),
            "mismatch_count_temp": int(np.sum(diff_temp > 1e-4)),
            "max_abs_diff_precip": float(np.nanmax(diff_precip)),
            "mismatch_count_precip": int(np.sum(diff_precip > 1e-4)),
        }
        print(f"  ERA5 Temperature: Max Diff = {results['era5']['max_abs_diff_temp']:.6e}, Mean Diff = {results['era5']['mean_abs_diff_temp']:.6e}, Mismatches = {results['era5']['mismatch_count_temp']}")

    print("=" * 65)
    return results


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_historical_expansion(start_date=None, end_date=None):
    start_date = start_date or config.HISTORICAL_START_DATE
    end_date = end_date or config.HISTORICAL_END_DATE

    print("=" * 65)
    print(" PHASE 4: HISTORICAL TEMPORAL EXPANSION (MONTHLY)")
    print("=" * 65)
    print(f"Historical period:     {start_date} -> {end_date}")

    month_ranges = get_month_ranges(start_date, end_date)
    num_months = len(month_ranges)
    print(f"Number of months:      {num_months}")

    # Load canonical grid
    grid_path = config.GRID_PATH
    with open(grid_path, "r", encoding="utf-8") as f:
        grid_geojson = json.load(f)
    num_cells = len(grid_geojson["features"])
    print(f"Grid cells:            {num_cells}")
    print(f"Expected observations: {num_cells} x {num_months} = {num_cells * num_months}")

    # Initialize GEE
    import ee
    project_id = config.GEE_PROJECT or os.getenv("GEE_PROJECT", "")
    if project_id:
        ee.Initialize(project=project_id)
    else:
        ee.Initialize()

    bbox = config.BBOX
    aoi = ee.Geometry.BBox(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"])
    aoi_buffered = ee.Geometry.BBox(bbox["min_lon"] - 0.05, bbox["min_lat"] - 0.05, bbox["max_lon"] + 0.05, bbox["max_lat"] + 0.05)

    ee_features = [
        ee.Feature(ee.Geometry.Polygon(f["geometry"]["coordinates"]), {"cell_id": f["properties"]["cell_id"]})
        for f in grid_geojson["features"]
    ]
    ee_grid = ee.FeatureCollection(ee_features)

    # 1. Process Sentinel-2
    df_s2, s2_meta = process_historical_sentinel2(month_ranges, aoi, ee_grid, grid_geojson)

    # 2. Process VIIRS
    df_viirs, viirs_meta = process_historical_viirs(month_ranges, aoi, ee_grid, grid_geojson)

    # 3. Process ERA5
    df_era5, era5_meta = process_historical_era5(month_ranges, aoi_buffered, grid_geojson)

    # 4. Join Common Multimodal Dataset
    df_multimodal = build_multimodal_dataset(df_s2, df_viirs, df_era5)

    # 5. Diagnostic Plots
    generate_temporal_diagnostic_plots(df_multimodal, config.HISTORICAL_PLOTS_DIR)

    # 6. March 2024 Regression Check
    reg_results = run_regression_check(df_multimodal)

    # 7. Write Comprehensive Processing Report
    report_path = config.PROCESSED_DATA_DIR / "historical_expansion_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write(" Phase 4: Historical Temporal Expansion Report\n")
        f.write("=============================================================\n")
        f.write(f"Selected Historical Period:   {start_date} -> {end_date}\n")
        f.write(f"Number of Months:             {num_months}\n")
        f.write(f"Months List:                  {[m[0] for m in month_ranges]}\n")
        f.write(f"Number of Grid Cells:         {num_cells}\n")
        f.write(f"Expected Cell-Month Rows:     {num_cells * num_months}\n")
        f.write(f"Actual Multimodal Rows:       {len(df_multimodal)}\n")
        f.write(f"Unique (cell_id, month) Rows: {df_multimodal[['cell_id', 'month']].drop_duplicates().shape[0]}\n\n")

        f.write("Modality Ingestion & Completeness:\n")
        f.write("-------------------------------------------------------------\n")
        f.write("Month     S2 Raw/Clean/Used   VIIRS Imgs   ERA5 Hours (Actual/Expected)\n")
        for m_key, _, _, d_in_m in month_ranges:
            s_info = s2_meta.get(m_key, {})
            v_info = viirs_meta.get(m_key, {})
            e_info = era5_meta.get(m_key, {})
            f.write(
                f"{m_key:<9} "
                f"{s_info.get('raw_scenes', 0):>2}/{s_info.get('clean_scenes', 0):>2}/{s_info.get('used_scenes', 0):>2}              "
                f"{v_info.get('image_count', 0):>2}           "
                f"{e_info.get('actual_hours', 0):>3}/{e_info.get('expected_hours', 0):<3} ({e_info.get('completeness_ratio', 0)*100:.1f}%)\n"
            )

        f.write("\nMissing Data / NaN Counts across 5,520 Observations:\n")
        f.write("-------------------------------------------------------------\n")
        for col in df_multimodal.columns:
            n_nan = int(df_multimodal[col].isna().sum())
            pct = (n_nan / len(df_multimodal)) * 100.0
            f.write(f"  {col:<28}: {n_nan:>5} NaNs ({pct:>5.1f}%)\n")

        f.write("\nMarch 2024 Benchmark Regression Check:\n")
        f.write("-------------------------------------------------------------\n")
        if "s2" in reg_results:
            f.write(f"  S2 NDVI:  Max Abs Diff = {reg_results['s2']['max_abs_diff_ndvi']:.6e}, Mismatches = {reg_results['s2']['mismatch_count_ndvi']}\n")
        if "viirs" in reg_results:
            f.write(f"  VIIRS:    Max Abs Diff = {reg_results['viirs']['max_abs_diff_rad']:.6e}, Mismatches = {reg_results['viirs']['mismatch_count_rad']}\n")
        if "era5" in reg_results:
            f.write(f"  ERA5 T2m: Max Abs Diff = {reg_results['era5']['max_abs_diff_temp']:.6e}, Mismatches = {reg_results['era5']['mismatch_count_temp']}\n")

        f.write("\nFinal Multimodal Feature Columns:\n")
        f.write("-------------------------------------------------------------\n")
        for idx, c in enumerate(df_multimodal.columns, 1):
            f.write(f"  {idx:02d}. {c}\n")

        f.write(f"\nProcessing Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=============================================================\n")

    print(f"[SAVE] Historical processing report saved: {report_path}")
    print("\n>>> Phase 4 Historical Expansion Completed Successfully! <<<\n")
    return df_multimodal


if __name__ == "__main__":
    run_historical_expansion()
