"""
Grid generation module for Multimodal Geospatial Anomaly Intelligence System.
Generates a regular ~500m x 500m spatial grid over the study area bounding box,
assigning unique cell IDs, spatial coordinates, and saving to GeoJSON.
"""

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


def generate_grid(bbox=None, cell_size_meters=500, output_path=None):
    """
    Generate a regular grid covering the bounding box with ~cell_size_meters resolution.
    
    Args:
        bbox (dict): {'min_lon': float, 'min_lat': float, 'max_lon': float, 'max_lat': float}
        cell_size_meters (int): Cell resolution in meters (default 500m)
        output_path (Path or str): File path to save output GeoJSON
    
    Returns:
        dict: GeoJSON FeatureCollection containing all grid cells
    """
    if bbox is None:
        bbox = config.BBOX
    if output_path is None:
        output_path = config.GRID_PATH

    min_lon = bbox["min_lon"]
    min_lat = bbox["min_lat"]
    max_lon = bbox["max_lon"]
    max_lat = bbox["max_lat"]

    # Earth radius and degree conversions
    earth_radius = 6371000.0  # meters
    meters_per_deg_lat = (math.pi / 180.0) * earth_radius  # ~111,195 m/deg
    
    mid_lat = (min_lat + max_lat) / 2.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(mid_lat))  # ~97,770 m/deg

    # Calculate step size in degrees for ~cell_size_meters
    d_lat = cell_size_meters / meters_per_deg_lat
    d_lon = cell_size_meters / meters_per_deg_lon

    # Compute number of rows and columns
    num_cols = max(1, math.ceil((max_lon - min_lon) / d_lon))
    num_rows = max(1, math.ceil((max_lat - min_lat) / d_lat))

    # Adjust exact delta to fit exactly between min and max bounds
    step_lon = (max_lon - min_lon) / num_cols
    step_lat = (max_lat - min_lat) / num_rows

    features = []
    cell_index = 0

    for r in range(num_rows):
        c_min_lat = min_lat + r * step_lat
        c_max_lat = c_min_lat + step_lat

        for c in range(num_cols):
            c_min_lon = min_lon + c * step_lon
            c_max_lon = c_min_lon + step_lon

            cell_id = f"cell_{r:02d}_{c:02d}"
            centroid_lon = round((c_min_lon + c_max_lon) / 2.0, 6)
            centroid_lat = round((c_min_lat + c_max_lat) / 2.0, 6)

            feature = {
                "type": "Feature",
                "id": cell_id,
                "properties": {
                    "cell_id": cell_id,
                    "row": r,
                    "col": c,
                    "index": cell_index,
                    "centroid_lon": centroid_lon,
                    "centroid_lat": centroid_lat,
                    "min_lon": round(c_min_lon, 6),
                    "min_lat": round(c_min_lat, 6),
                    "max_lon": round(c_max_lon, 6),
                    "max_lat": round(c_max_lat, 6),
                    "approx_width_m": round(step_lon * meters_per_deg_lon, 1),
                    "approx_height_m": round(step_lat * meters_per_deg_lat, 1),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [round(c_min_lon, 6), round(c_min_lat, 6)],
                            [round(c_max_lon, 6), round(c_min_lat, 6)],
                            [round(c_max_lon, 6), round(c_max_lat, 6)],
                            [round(c_min_lon, 6), round(c_max_lat, 6)],
                            [round(c_min_lon, 6), round(c_min_lat, 6)],
                        ]
                    ],
                },
            }
            features.append(feature)
            cell_index += 1

    geojson_data = {
        "type": "FeatureCollection",
        "name": "study_area_500m_grid",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "metadata": {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "total_cells": len(features),
            "cell_size_meters": cell_size_meters,
            "bbox": bbox,
        },
        "features": features,
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    print(
        f"[INFO] Successfully created grid: {len(features)} cells ({num_rows} rows x {num_cols} cols)"
    )
    print(f"[INFO] Saved to: {output_path}")

    return geojson_data


if __name__ == "__main__":
    generate_grid()
