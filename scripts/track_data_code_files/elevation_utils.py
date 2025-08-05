# elevation_utils.py
import os
import pickle

import requests

CACHE_PATH = "elevation_cache.pkl"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}


def get_elev_batch(coords, api_key):
    """
    coords: list of (lat, lon) tuples, up to 512 points per batch
    returns: list of floats in meters
    """
    # Use the tuple of coords as a cache key
    key = tuple(coords)
    if key in cache:
        return cache[key]

    # Build the pipe‑delimited locations param
    locs = "|".join(f"{lat},{lon}" for lat, lon in coords)
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/elevation/json",
        params={"locations": locs, "key": api_key},
    )
    resp.raise_for_status()
    data = resp.json()
    elevs = [pt["elevation"] for pt in data["results"]]

    # Save to cache and persist
    cache[key] = elevs
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    return elevs
