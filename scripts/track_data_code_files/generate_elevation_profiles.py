#!/usr/bin/env python3
"""
generate_elevation_profiles.py

Uses a geodetic sampler (via pyproj.Geod) to sample every 10m true distance
along each track centerline (in lon/lat), fetch elevation (with retry),
and compute deltas.
"""

import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pyproj import Geod
from requests.exceptions import ReadTimeout, RequestException
from shapely.geometry import LineString, Point

TARGET_TRACKS = [
    "brazilian_grand_prix",
    "canadian_grand_prix",
    "chinese_grand_prix",
    "dutch_grand_prix",
    "eifel_grand_prix",
    "french_grand_prix",
    "german_grand_prix",
    "hungarian_grand_prix",
    "italian_grand_prix",
    "japanese_grand_prix",
    "las_vegas_grand_prix",
    "mexican_grand_prix",
    "mexico_city_grand_prix",
    "miami_grand_prix",
    "portuguese_grand_prix",
    "qatar_grand_prix",
    "sakhir_grand_prix",
    "sao_paulo_grand_prix",
    "singapore_grand_prix",
    "tuscan_grand_prix",
    "united_states_grand_prix",
]

BASE_DIR = os.path.dirname(__file__)
dotenv_path = os.path.join(BASE_DIR, "config", ".env")
load_dotenv(dotenv_path)

API_KEY = os.getenv("GOOGLE_ELEVATION_API_KEY")
if not API_KEY:
    raise RuntimeError("Set GOOGLE_ELEVATION_API_KEY in config/.env")

CL_DIR = (
    "/Users/rishipatel/Desktop/f1-strategy-platform/"
    "data/track_data_csv_files/track_centerlines_gps_cords"
)
OUT_DIR = (
    "/Users/rishipatel/Desktop/f1-strategy-platform/"
    "data/track_data_csv_files/elevation_profiles"
)
os.makedirs(OUT_DIR, exist_ok=True)

geod = Geod(ellps="WGS84")


def get_elevation(lat: float, lon: float) -> float:
    """Fetch elevation in meters with up to 3 retries on timeout."""
    url = (
        "https://maps.googleapis.com/maps/api/elevation/json"
        f"?locations={lat},{lon}&key={API_KEY}"
    )
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=10).json()
            if resp.get("status") == "OK":
                return resp["results"][0]["elevation"]
            return np.nan
        except ReadTimeout:
            print(f"timeout (attempt {attempt}/3) for {lat},{lon}")
        except RequestException as e:
            print(f"request error: {e}")
        time.sleep(0.5 * attempt)
    print(f"giving up, returning NaN for {lat},{lon}")
    return np.nan


def sample_geodetic(line: LineString, interval_m: float = 10.0):
    """
    Sample a LineString (in lon/lat) every `interval_m` meters true distance.
    Returns (samples: List[Point], total_length_m).
    """
    coords = list(line.coords)
    cumdist = [0.0]
    for (lon0, lat0), (lon1, lat1) in zip(coords, coords[1:]):
        _, _, d = geod.inv(lon0, lat0, lon1, lat1)
        cumdist.append(cumdist[-1] + d)
    total_len = cumdist[-1]

    targets = list(np.arange(0, total_len, interval_m)) + [total_len]
    samples = []
    for td in targets:
        idx = next(i for i, d in enumerate(cumdist) if d >= td)
        lon0, lat0 = coords[idx - 1]
        lon1, lat1 = coords[idx]
        d0, d1 = cumdist[idx - 1], cumdist[idx]
        frac = (td - d0) / (d1 - d0) if d1 > d0 else 0.0
        samples.append(Point(lon0 + (lon1 - lon0) * frac, lat0 + (lat1 - lat0) * frac))
    return samples, total_len


master_records = []

for TARGET_TRACK in TARGET_TRACKS:
    geojson_path = os.path.join(CL_DIR, f"{TARGET_TRACK}.geojson")
    if not os.path.exists(geojson_path):
        print(f"Warning: cannot find {geojson_path}, skipping.")
        continue

    # load centerline in lon/lat
    gdf = gpd.read_file(geojson_path).to_crs(epsg=4326)
    centerline: LineString = gdf.geometry.iloc[0]

    # sample every 10 m true distance
    samples, lap_length_m = sample_geodetic(centerline, interval_m=10.0)

    # fetch elevations
    elevs = []
    for pt in samples:
        elevs.append(get_elevation(pt.y, pt.x))
        time.sleep(0.1)

    # build the per-sample profile DataFrame
    interval_m = 10.0
    df = pd.DataFrame(
        {
            "track": TARGET_TRACK,
            "distance_m": np.arange(len(samples)) * interval_m,
            "elevation_m": elevs,
        }
    )
    df["elev_delta_m"] = df["elevation_m"].diff().fillna(0)

    # compute summary stats
    total_change = df["elevation_m"].max() - df["elevation_m"].min()
    up_change = df.loc[df["elev_delta_m"] > 0, "elev_delta_m"].sum()
    down_change = -df.loc[df["elev_delta_m"] < 0, "elev_delta_m"].sum()

    master_records.append(
        {
            "track": TARGET_TRACK,
            "lap_length_m": lap_length_m,
            "total_elev_change": total_change,
            "total_up_m": up_change,
            "total_down_m": down_change,
        }
    )

    # save per-track profile CSV
    out_csv = os.path.join(OUT_DIR, f"{TARGET_TRACK}_elevation_profile.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote elevation profile: {out_csv}")
