# Multimodal Geospatial Anomaly Intelligence System — Phase 1: Sentinel-2 Pipeline

This repository implements **Phase 1** of the Multimodal Geospatial Anomaly Intelligence System for the Delhi-NCR region. In this phase, the pipeline ingests, masks, composites, and computes optical indices from Sentinel-2 Surface Reflectance data, then aggregates those indices over a 500m regular spatial grid.

---

## 📍 Study Area & Spatial Grid

- **Location:** South-East/East Delhi, India (covering the Yamuna riverfront, Okhla industrial area, and mixed urban/green zones).
- **Bounding Box (WGS84, `EPSG:4326`):**
  - West (`min_lon`): `77.25° E`
  - South (`min_lat`): `28.50° N`
  - East (`max_lon`): `77.35° E`
  - North (`max_lat`): `28.60° N`
  - Dimensions: ~10 km × 11 km
- **Projected Coordinate Reference System (CRS):** `EPSG:32643` (WGS 84 / UTM Zone 43N), ensuring metric accuracy for Delhi-NCR.
- **Grid Resolution:** Regular ~500 m × 500 m cells.
- **Grid File:** `data/processed/study_grid.geojson` (Contains exactly **460 polygon cells**, 23 rows × 20 columns, each with unique `cell_id`).

---

## 🛰️ Sentinel-2 Dataset & Preprocessing

- **Dataset Source:** Google Earth Engine (GEE)
- **Dataset ID:** `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2 Level-2A Surface Reflectance Harmonized)
- **Bands Used:**
  - `B3` (Green, 10m resolution)
  - `B4` (Red, 10m resolution)
  - `B8` (Near Infrared, 10m resolution)
  - `B11` (Shortwave Infrared 1, 20m resolution)
  - `SCL` (Scene Classification Layer, 20m resolution)
- **Surface Reflectance Scaling:**
  Raw Sentinel-2 L2A pixel digital numbers (DN) in Earth Engine are scaled by 10,000 (reflectance = DN / 10000.0). The pipeline explicitly scales reflectance values.

### Cloud Filtering Strategy
The pipeline separates scene-level filtering from pixel-level masking:
1. **Scene-level filtering:** Filters the image collection to scenes with `CLOUDY_PIXEL_PERCENTAGE <= max_cloud_percent` (configurable, default 20%).
2. **Pixel-level masking:** Employs the Sentinel-2 L2A Scene Classification Layer (`SCL`).
   - **Masked out:** Class 0 (No Data), Class 1 (Saturated/Defective), Class 3 (Cloud Shadows), Class 8 (Cloud Medium Probability), Class 9 (Cloud High Probability), Class 10 (Thin Cirrus), Class 11 (Snow).
   - **Kept:** Class 2 (Dark Area Pixels / Deep Shadows), Class 4 (Vegetation), Class 5 (Not-vegetated / Soil / Built-up), Class 6 (Water), Class 7 (Unclassified Land).

### Temporal Compositing
- **Method:** Temporal median composite over the configured date window.
- **Rationale:** A median composite selects the per-pixel median reflectance over clear-sky acquisitions, filtering out residual cloud edges, cloud shadows, and transient sensor artifacts while preserving typical land surface reflectance.

---

## 🧮 Spectral Indices

All indices are computed with safe division handling (zero denominators are masked):

1. **Normalized Difference Vegetation Index (NDVI):**
   $$\text{ndvi} = \frac{\text{B8} - \text{B4}}{\text{B8} + \text{B4}}$$
2. **Normalized Difference Built-up Index (NDBI):**
   $$\text{ndbi} = \frac{\text{B11} - \text{B8}}{\text{B11} + \text{B8}}$$
3. **Normalized Difference Water Index (NDWI):**
   $$\text{ndwi} = \frac{\text{B3} - \text{B8}}{\text{B3} + \text{B8}}$$

---

## 📐 Spatial Aggregation

- Aggregation uses the actual polygon boundary of each of the 460 cells via Earth Engine's `reduceRegions`.
- Aggregation is performed at a 20m pixel scale using the projected metric CRS `EPSG:32643` (UTM Zone 43N).
- Reducers calculated per cell:
  - Mean: `ndvi_mean`, `ndbi_mean`, `ndwi_mean`
  - Standard Deviation: `ndvi_std`, `ndbi_std`, `ndwi_std`
  - Valid Pixels: `valid_pixel_count`
  - Quality fraction: `valid_pixel_fraction = valid_pixel_count / expected_pixel_count` (where expected pixels at 20m resolution in 500m cell ≈ 625).
- **Missing Data Handling:** Cells with zero valid pixels are explicitly marked with `NaN` (not zero).

---

## 📁 Output Files

All outputs are saved to `data/processed/`:
- `data/processed/study_grid.geojson`: GeoJSON with all 460 grid cell polygon boundaries and metadata.
- `data/processed/s2_features.parquet`: Machine-readable feature dataset (Apache Parquet).
- `data/processed/s2_features.csv`: Machine-readable feature dataset (CSV).
- `data/processed/s2_processing_report.txt`: Automated debug and quality report with statistics.
- `data/processed/map_ndvi.png`: Static visual validation choropleth map of NDVI.
- `data/processed/map_ndbi.png`: Static visual validation choropleth map of NDBI.
- `data/processed/map_ndwi.png`: Static visual validation choropleth map of NDWI.

---

## 💻 How to Run

### 1. Configure Credentials
Authenticate your Google Earth Engine account:
```bash
python -c "import ee; ee.Authenticate()"
```
Set your Earth Engine Cloud project ID in `.env` (or environment variable `GEE_PROJECT`):
```env
GEE_PROJECT=your-cloud-project-id
```

### 2. Verify Earth Engine & Dataset
Run the diagnostic check script:
```bash
python src/ingestion/gee_verify.py
```

### 3. Generate 500m Grid (if not already present)
```bash
python src/grid.py
```

### 4. Execute the Sentinel-2 Ingestion Pipeline
```bash
python src/ingestion/s2_optical.py
```

### Configuration Options
The date window and cloud thresholds can be adjusted in `config.py` or via environment variables:
- `S2_START_DATE` (default: `2024-03-01`)
- `S2_END_DATE` (default: `2024-03-31`)
- `S2_MAX_CLOUD_PERCENT` (default: `20`)

---

## ⚠️ Known Limitations & Assumptions

1. **Optical Revisit Interval:** Sentinel-2 has a 5-day revisit cycle. Persistent monsoon clouds (July–August) can reduce the number of clear observations per cell.
2. **Fixed Grid Boundary Effects:** Grid cells intersecting the boundary of the study area bounding box are clipped cleanly to the box geometry.
3. **Resampling:** Bands with 10m native resolution (`B3`, `B4`, `B8`) and 20m native resolution (`B11`, `SCL`) are aggregated at 20m nominal scale during polygonal spatial reduction.
