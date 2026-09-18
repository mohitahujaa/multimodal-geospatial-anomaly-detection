# Validation Cases

Since ground-truth anomaly datasets are unlabelled, validation is performed against 2–3 documented, real-world events within the Delhi-NCR study area and timeframe:

1. **Event 1: Yamuna River Monsoon Flooding (July 2023)**
   - *Characteristics:* High precipitation in ERA5, sharp spike in Sentinel-2 NDWI (water index), drop in NDVI along riverfront cells.
   - *Expected system output:* Elevated anomaly score in river corridor cells, agent hypothesizing flood inundation supported by ERA5 rainfall data.

2. **Event 2: Major Infrastructure / Construction Project (e.g. Expressway or Flyover)**
   - *Characteristics:* Sustained multi-month drop in NDVI and rise in NDBI (built-up index).
   - *Expected system output:* Flagged persistence anomaly, agent checking OSM road/land-use tags and attributing to construction.

3. **Event 3: Localized Industrial Activity or Nighttime Lights Outage/Spike**
   - *Characteristics:* Anomaly in VIIRS radiance channel without major optical or weather change.
