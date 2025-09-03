"""
scripts/weather_code_files/fetch_weather_visualcrossing.py

Fetches minute-level weather from Visual Crossing for each F1 session (Q/S/R).
Uses FastF1 to enumerate events and obtain exact session dates.
Looks up track lat/lon from config/track_coords.csv, then saves raw JSON
under data/raw/visualcrossing_weather/{year}/{grand_prix_slug}/.
"""

import json
import os
import time
from pathlib import Path

import fastf1
import pandas as pd
import requests
from dotenv import load_dotenv


YEARS = [2025] # seasons to pull; adjust as needed to control API usage
MAX_RACES = 100 # simple guard to avoid hammering the API during tests


def get_vc_key():
    # Load VISUALCROSSING_API_KEY from .env or environment; fail fast if missing
    load_dotenv()
    key = os.getenv("VISUALCROSSING_API_KEY")
    if not key:
        raise RuntimeError("Please set the VISUALCROSSING_API_KEY environment variable")
    return key


def slugify(name: str) -> str:
    # Normalize GP names into filesystem-safe identifiers (e.g., "Bahrain Grand Prix" -> "bahrain_grand_prix")
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_track_coords(path="config/track_coords.csv") -> pd.DataFrame:
    # Read track coordinates and normalize potentially varied column headers into (gp, lat, lon, slug)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = next(
        (v for k, v in cols.items() if "grand" in k or "event" in k), df.columns[0]
    )
    lat_col = next(v for k, v in cols.items() if "lat" in k)
    lon_col = next(v for k, v in cols.items() if "lon" in k)
    df = df.rename(columns={name_col: "gp", lat_col: "lat", lon_col: "lon"})
    df["slug"] = df.gp.apply(slugify)
    return df[["gp", "lat", "lon", "slug"]]


def fetch_vc_minutely(lat, lon, date, api_key, retries=3):
    # Visual Crossing "timeline" endpoint for a specific lat/lon and ISO date; includes minute granularity
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{lat},{lon}/{date}"
    )
    params = {"unitGroup": "metric", "include": "minutes", "key": api_key}
    backoff = 1
    for _ in range(retries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code in (429, 403):
            print(f"rate/forbid ({r.status_code}), retrying in {backoff}s…")
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Failed to fetch VC for {date}")


def main():
    # Initialize API key and output layout once; subsequent loops just write files
    VC_KEY = get_vc_key()
    print("Using VisualCrossing key from VISUALCROSSING_API_KEY\n")

    coords = load_track_coords("config/track_coords.csv")
    out_base = Path("data/raw/visualcrossing_weather")
    out_base.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        print(f"=== Season {year} ===")
        schedule = fastf1.get_event_schedule(year)
        rounds = sorted(schedule.RoundNumber.unique())

        for idx, rnd in enumerate(rounds):
            if idx >= MAX_RACES:
                print(f"— reached {MAX_RACES} races, stopping test.")
                return

            ev = schedule[schedule.RoundNumber == rnd].iloc[0]
            gp_name = ev["EventName"]
            slug = slugify(gp_name)

            # Join against the coordinates table; skip if we don't have a lat/lon
            tr = coords[coords.slug == slug]
            if tr.empty:
                print(f"No coords for {gp_name}, skipping")
                continue
            lat, lon = tr.iloc[0][["lat", "lon"]]

            # Loop the sessions we care about and fetch weather for the exact session date
            for label, code in [("qualifying", "Q"), ("sprint", "S"), ("race", "R")]:
                try:
                    sess = fastf1.get_session(year, rnd, code)
                    sess.load()
                except Exception:
                    continue

                date_iso = sess.date.date().isoformat()
                print(f"> Fetching {label.upper():>10} for {gp_name} on {date_iso}")

                try:
                    data = fetch_vc_minutely(lat, lon, date_iso, VC_KEY)
                except Exception as e:
                    print(f"error: {e}")
                    continue

                # Persist raw JSON for later ETL; keep per-year/per-GP folders for organization
                dest = out_base / str(year) / slug
                dest.mkdir(parents=True, exist_ok=True)
                fn = f"{label}_{date_iso}.json"
                with open(dest / fn, "w") as f:
                    json.dump(data, f)
                print(f"saved → {dest/ fn}\n")


if __name__ == "__main__":
    main()
