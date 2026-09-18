# Project Kickoff Prompt — Multimodal Geospatial Anomaly Intelligence System

Use this as a working brief for yourself, or paste it directly into Claude Code / another AI coding
assistant as the starting task description. It's self-contained: objective, scope, datasets,
requirements, and a concrete first-week action plan.

---

## 1. Project Objective

Build a system that learns the **normal spatiotemporal behaviour** of a single bounded urban region
(~10 km × 10 km, within Delhi-NCR) using three independent data modalities — satellite optical
imagery, nighttime-lights radiance, and weather data — and detects **localized statistical anomalies**
without relying on pre-labelled event categories (i.e., it does not assume in advance whether an
anomaly is a flood, fire, construction event, or outage).

When an anomaly is flagged, an **LLM-based agent** investigates it by checking historical precedent,
weather context, and nearby infrastructure (OpenStreetMap), then produces a **ranked, evidence-backed
hypothesis report** explaining what likely happened. Results are shown on an **interactive map
dashboard**.

The core intellectual contribution is *not* a trained classifier for a specific event type — it's:
1. An unsupervised, per-location "normality model" that generalizes across unseen anomaly types.
2. A cross-modal reliability/disagreement mechanism that decides which sensors to trust in a given
   situation rather than naively averaging them.
3. An agentic reasoning layer that investigates *why* an anomaly occurred using real tool calls,
   not just a single LLM prompt.

---

## 2. Scope & Constraints (read this before building anything)

- **Solo project, one semester (~18 weeks).** Scope is deliberately narrow — do not add modalities
  or expand the study area without cutting something else first.
- **Study area:** one fixed bounding box (~10 km × 10 km). Pick it early and never change it —
  a stable area is what lets your historical baseline mean something.
- **Grid:** 500 m × 500 m cells over the study area only (not all of NCR).
- **Modalities (exactly three, no more):**
  1. Sentinel-2 (optical imagery → NDVI, NDBI, NDWI surface indices)
  2. VIIRS (nighttime-lights radiance → human activity proxy)
  3. ERA5-Land or IMD (weather → temperature, rainfall, wind)
- **Explicitly out of scope for the core system:** energy/DISCOM data, X/Twitter data. These are
  hard to obtain reliably and are mentioned only as "future work" in the report.
- **No labelled anomaly dataset exists.** Validation will be done against 2–3 hand-picked, real,
  documented events in your study area/timeframe (e.g. a known flooding day, a visible construction
  project you can confirm via Google Earth timelapse) — not against an accuracy/F1 metric on a
  labelled test set.

---

## 3. Datasets

All datasets below are free and accessible via **Google Earth Engine (GEE)**, which also handles
cloud masking and mosaicking — this matters a lot for you since you have no prior GIS experience.

| Modality | Source | GEE Dataset ID | Notes |
|---|---|---|---|
| Optical satellite | Sentinel-2 Surface Reflectance | `COPERNICUS/S2_SR_HARMONIZED` | ~5-day revisit; filter clouds using the `QA60` band or `CLOUD_PROBABILITY` |
| Nighttime lights | VIIRS Day/Night Band | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` | Monthly composite, ~500 m resolution |
| Weather | ERA5-Land | `ECMWF/ERA5_LAND/MONTHLY_AGGR` or hourly variant | Temperature, precipitation, wind |
| Infrastructure context | OpenStreetMap | via `osmnx` Python library (not GEE) | Roads, buildings, land use, POIs |

**Sign-up needed:** a free Google Earth Engine account (`earthengine.google.com`) — approval is
usually near-instant for student accounts.

**History depth to pull:** aim for 1–2 years of biweekly-to-monthly composites per modality. Even
6 months is workable as a fallback if storage/time becomes tight.

---

## 4. Functional Requirements

The system must be able to:

1. Ingest and align all three modalities onto a common 500 m spatiotemporal grid over the study area.
2. Build a per-cell historical feature vector time series (NDVI, NDBI, NDWI, nighttime radiance,
   temperature, rainfall, wind, etc.).
3. Learn an expected "normal" range of behaviour per cell and season (start with a statistical
   baseline — rolling mean/variance or seasonal z-score — before attempting anything more complex).
4. Score new/held-out observations against that baseline to produce a per-modality anomaly score.
5. Combine per-modality scores into one anomaly score, with a reliability weighting mechanism
   (e.g., down-weight a modality when cloud cover is high or data is missing).
6. Flag cells where the combined anomaly score crosses a threshold.
7. For flagged cells, invoke an LLM agent with tool access to: query historical baseline for that
   cell, query weather at that time, query nearby OSM infrastructure, and return a ranked list of
   hypotheses with supporting/contradicting evidence (JSON output).
8. Display results on an interactive map (colored by anomaly score) with a per-cell drill-down
   panel showing the agent's report.

---

## 5. Technical Requirements / Stack

- **Language:** Python 3.10+
- **Geospatial:** `earthengine-api` (GEE Python client), `geopandas`, `rasterio`, `osmnx`
- **Data/ML:** `numpy`, `pandas`, `scikit-learn`, `statsmodels`
- **Agent layer:** an LLM API with tool/function-calling support (Anthropic Claude API or OpenAI
  API) — implement a small set of tools: `get_historical_baseline(cell_id)`,
  `get_weather(cell_id, date)`, `get_nearby_infrastructure(cell_id)`
- **Dashboard:** Streamlit + `folium` or `pydeck` (fastest path for a solo dev); a React frontend
  is an option only if you're already comfortable with it
- **Version control:** Git + GitHub from day one
- **Hardware:** 8 GB RAM minimum (16 GB recommended), ~20 GB free disk space, stable internet

---

## 6. Getting Started — Concrete First Steps (Week 1)

Do these in order. Each should take at most a day or two; don't move on until the previous step
actually runs.

1. **Create a Google Earth Engine account** and verify Python API access:
   ```python
   import ee
   ee.Authenticate()
   ee.Initialize(project="your-gee-project-id")
   ```
2. **Define your study area** as a GeoJSON bounding box (use geojson.io to draw it visually — pick
   a mixed-use area with visible variety: some green space, some built-up, some industrial).
3. **Pull one Sentinel-2 image** over your box for a recent cloud-free date, compute NDVI, and plot
   it with `matplotlib`. This is your "hello world" — if this works, the hardest unfamiliar part
   (GEE + raster basics) is behind you.
4. **Pull one VIIRS composite** and one ERA5-Land monthly aggregate over the same box, and plot both.
5. **Build your 500 m grid** over the study area using `geopandas` (a simple `shapely` grid generator
   is enough — you don't need anything fancy here).
6. **Write one ingestion script per modality** that, given a date range, pulls and exports per-cell
   average values into a single tidy table (rows = cell × timestamp, columns = features). This
   table is the foundation everything else builds on — get it right before moving to modeling.

Once step 6 works end-to-end for a short date range (e.g. one month), you have validated the whole
pipeline shape and can scale up the history depth with confidence.

---

## 7. Suggested Repository Structure

```
project-root/
├── data/
│   ├── raw/              # cached GEE exports, per modality
│   └── processed/        # aligned grid feature table
├── src/
│   ├── ingestion/         # one script per modality
│   ├── grid.py            # grid construction
│   ├── features.py        # feature engineering (NDVI/NDBI/NDWI etc.)
│   ├── normality_model.py # baseline + anomaly scoring
│   ├── fusion.py           # cross-modal disagreement/reliability weighting
│   ├── agent/              # LLM agent + tool definitions
│   └── dashboard/          # Streamlit app
├── notebooks/              # exploration, one-off validation checks
├── validation/              # your 2-3 known-event validation cases
└── README.md
```

---

## 8. Milestone Checklist (maps to the 18-week plan)

- [ ] Weeks 1–3: GEE + grid + one-modality pipeline working end-to-end
- [ ] Weeks 4–6: all three modalities ingested and aligned into one feature table
- [ ] Weeks 7–9: per-cell normality baseline + per-modality anomaly scoring
- [ ] Weeks 10–11: cross-modal fusion + reliability weighting
- [ ] Weeks 12–14: agentic investigation layer with real tool calls
- [ ] Weeks 15–16: dashboard (map + drill-down report)
- [ ] Weeks 17–18: validation against known events, report/synopsis finalized, buffer for fixes

---

## 9. When You Get Stuck

If you paste this whole document into an AI coding assistant along with "help me implement step X,"
give it the specific step number and any error output — that will get you a much more useful answer
than a general "help me with my project" prompt.
