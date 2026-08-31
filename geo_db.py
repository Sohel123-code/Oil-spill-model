"""
geo_db.py — Load JEETHU_BHAI.csv and build a filename → [object rows] lookup.

CSV column layout (0-indexed, after skipping 3 header rows):
  1  = IMAGE        (jpg filename, e.g. ow-0001.jpg)
  5  = start_time   (datetime string)
  8  = patch_width
  9  = patch_height
  18 = obj_ul_lon
  19 = obj_ul_lat
  20 = obj_ur_lon
  21 = obj_ur_lat
  22 = obj_br_lon
  23 = obj_br_lat
  24 = obj_bl_lon
  25 = obj_bl_lat
  26 = obj_patchloc_xmin
  27 = obj_patchloc_ymin
  28 = obj_patchloc_xmax
  29 = obj_patchloc_ymax
"""

import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "external", "JEETHU_BHAI.csv")

_GEO_COLS = [
    "obj_ul_lon", "obj_ul_lat",
    "obj_ur_lon", "obj_ur_lat",
    "obj_br_lon", "obj_br_lat",
    "obj_bl_lon", "obj_bl_lat",
]

_PIXEL_COLS = [
    "patch_width", "patch_height",
    "xmin", "ymin", "xmax", "ymax"
]

_USE_COLS = [1, 5, 8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


def load_geo_db() -> dict:
    """
    Returns:
        {filename: [ {datetime, obj_ul_lon, ..., xmin, ymin, xmax, ymax}, ... ]}
        Empty dict if CSV is missing or unreadable.
    """
    if not os.path.exists(CSV_PATH):
        return {}
    try:
        df = pd.read_csv(
            CSV_PATH,
            skiprows=3,
            header=None,
            usecols=_USE_COLS,
            dtype=str,
        )
        df.columns = ["IMAGE", "datetime", "patch_width", "patch_height"] + _GEO_COLS + ["xmin", "ymin", "xmax", "ymax"]

        numeric_cols = ["patch_width", "patch_height"] + _GEO_COLS + ["xmin", "ymin", "xmax", "ymax"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["IMAGE"] + _GEO_COLS + ["xmin", "ymin", "xmax", "ymax"])

        geo_db: dict = {}
        for _, row in df.iterrows():
            fname = str(row["IMAGE"]).strip()
            entry = {
                "datetime":     str(row["datetime"]).strip()[:10],  # YYYY-MM-DD
                "patch_width":  int(row["patch_width"]),
                "patch_height": int(row["patch_height"]),
                "obj_ul_lon":   round(float(row["obj_ul_lon"]), 6),
                "obj_ul_lat":   round(float(row["obj_ul_lat"]), 6),
                "obj_ur_lon":   round(float(row["obj_ur_lon"]), 6),
                "obj_ur_lat":   round(float(row["obj_ur_lat"]), 6),
                "obj_br_lon":   round(float(row["obj_br_lon"]), 6),
                "obj_br_lat":   round(float(row["obj_br_lat"]), 6),
                "obj_bl_lon":   round(float(row["obj_bl_lon"]), 6),
                "obj_bl_lat":   round(float(row["obj_bl_lat"]), 6),
                "xmin":         int(row["xmin"]),
                "ymin":         int(row["ymin"]),
                "xmax":         int(row["xmax"]),
                "ymax":         int(row["ymax"]),
            }
            geo_db.setdefault(fname, []).append(entry)

        return geo_db
    except Exception as e:
        print(f"Error loading geo db: {e}")
        return {}
