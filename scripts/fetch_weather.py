#!/usr/bin/env python3
"""
scripts/fetch_weather.py

For each F1 seasons race, fetch weather via OpenWeatherMap One-Call API
and write a flattened Parquet with the key fields.
"""

import argparse
import os

import fastf1 as ff1
import pandas as pd
import requests

# 1) Auto-load .env before anything else
from dotenv import load_dotenv

load_dotenv("config/.env")  # ← reads OWM_API_KEY into os.environ


def sanitize(name: str) -> str:
    """Make a filesystem-safe lowercase name."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("'", "")
    )


def fetch_weather_for_season(
    season: int,
    api_key: str,
    track_coords_csv: str,
    output_root: str = "data/raw/race_weather",
):
    # 1) load schedule
    schedule = ff1.get_event_schedule(season)

    # 2) load your manually-maintained CSV mapping EventName → lat,lon
    coords = pd.read_csv(track_coords_csv)
    coords.set_index("EventName", inplace=True)

    for _, ev in schedule.iterrows():
        round_num = int(ev["RoundNumber"])
        gp_name = ev["EventName"]
        if round_num < 1:
            continue

        # 3) look up lat/lon
        if gp_name not in coords.index:
            print(f"⚠️  No coords for {gp_name}, skipping")
            continue
        lat = coords.loc[gp_name, "lat"]
        lon = coords.loc[gp_name, "lon"]

        print(f"→ Fetching weather for {gp_name} ({lat},{lon})")
        resp = requests.get(
            "https://api.openweathermap.org/data/3.0/onecall",
            params={
                "lat": lat,
                "lon": lon,
                "units": "metric",
                "appid": api_key,
                "exclude": "alerts",
            },
        )
        resp.raise_for_status()
        w = resp.json()

        # 4) flatten to one record
        rec = {
            "season": season,
            "event": gp_name,
            "lat": w.get("lat"),
            "lon": w.get("lon"),
            "timezone": w.get("timezone"),
            "timezone_offset": w.get("timezone_offset"),
            # current
            "current_temp": w["current"].get("temp"),
            "current_feels_like": w["current"].get("feels_like"),
            "current_humidity": w["current"].get("humidity"),
            "current_pressure": w["current"].get("pressure"),
            "current_visibility": w["current"].get("visibility"),
            "current_clouds": w["current"].get("clouds"),
            "current_wind_speed": w["current"].get("wind_speed"),
            "current_wind_gust": w["current"].get("wind_gust"),
            "current_wind_deg": w["current"].get("wind_deg"),
            # minute‐level
            "minutely_precip_max": (
                max(m.get("precipitation", 0) for m in w.get("minutely", []))
                if w.get("minutely")
                else None
            ),
            # first‐hour
            "hourly_0_temp": w.get("hourly", [{}])[0].get("temp"),
            "hourly_0_pop": w.get("hourly", [{}])[0].get("pop"),
            "hourly_0_weather": w.get("hourly", [{}])[0]
            .get("weather", [{}])[0]
            .get("main"),
            # day‐ahead
            "daily_0_temp_max": w.get("daily", [{}])[0].get("temp", {}).get("max"),
            "daily_0_temp_min": w.get("daily", [{}])[0].get("temp", {}).get("min"),
            "daily_0_pop": w.get("daily", [{}])[0].get("pop"),
        }

        # 5) write out
        out_dir = os.path.join(output_root, str(season))
        os.makedirs(out_dir, exist_ok=True)
        fn = sanitize(gp_name)
        out_path = os.path.join(out_dir, f"{fn}_{season}.parquet")
        pd.DataFrame([rec]).to_parquet(out_path, index=False)
        print(f"   ✔️  Wrote {out_path}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Fetch per-race weather via OpenWeather One-Call"
    )
    p.add_argument("--season", "-s", type=int, default=2022, help="F1 season year")
    p.add_argument(
        "--api_key",
        "-k",
        type=str,
        default=os.getenv("OWM_API_KEY"),
        help="OpenWeatherMap One-Call API key (defaults to OWM_API_KEY in config/.env)",
    )
    p.add_argument(
        "--tracks",
        "-t",
        type=str,
        default="config/track_coords.csv",
        help="CSV mapping EventName → lat,lon",
    )
    p.add_argument(
        "--out",
        "-o",
        type=str,
        default="data/raw/race_weather",
        help="Output directory for per-race weather Parquet",
    )
    args = p.parse_args()

    api_key = args.api_key
    if not api_key:
        p.error("No API key provided! Set OWM_API_KEY in config/.env or pass --api_key")

    fetch_weather_for_season(
        season=args.season,
        api_key=api_key,
        track_coords_csv=args.tracks,
        output_root=args.out,
    )
