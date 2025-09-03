"""
scripts/track_data_code_files/elevation_utils.py

Provides a small helper to fetch elevations for many (lat, lon) points using the Google Elevation API.
Uses a simple on-disk pickle cache so repeated requests with the same coordinates avoid network calls.
Designed for track/segment geometry extraction where up to 512 points can be queried in one batch.
Returns elevations in meters, preserving the input order.
"""
import os
import pickle
import requests

# Path for the persisted elevation lookup cache (coordinates tuple -> list of elevations)
CACHE_PATH = "elevation_cache.pkl"

# Warm the cache from disk if present; otherwise start with an empty dict
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
else:
    cache = {}


def get_elev_batch(coords, api_key):
    # Use an immutable tuple of input coordinates as the cache key to enable memoization
    key = tuple(coords)
    if key in cache:
        return cache[key]

    # Build the locations parameter expected by the API: "lat1,lon1|lat2,lon2|..."
    locs = "|".join(f"{lat},{lon}" for lat, lon in coords)
    resp = requests.get(
        "https://maps.googleapis.com/maps/api/elevation/json",
        params={"locations": locs, "key": api_key},
    )
    resp.raise_for_status()
    data = resp.json()
    elevs = [pt["elevation"] for pt in data["results"]]

    # Record the fresh result and persist the entire cache back to disk
    cache[key] = elevs
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    return elevs
