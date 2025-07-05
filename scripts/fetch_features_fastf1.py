#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import fastf1 as ff1
import pandas as pd

# Enable FastF1 cache directory
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(str(CACHE_DIR))


def fetch_all_features(season: int, output_dir: str = "data/features"):
    """
    Fetch per-driver telemetry for each race in a season, aggregate per-lap,
    and write a single Parquet file per season.
    """
    print(f"Fetching features for season {season}")

    # Clear old data to prevent appending stale files
    out_path = Path(output_dir)
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the season schedule
    schedule = ff1.get_event_schedule(season)
    feature_frames = []

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        if round_num < 1:
            continue  # skip tests/pre-season

        gp_name = event["EventName"]
        print(f"  → Processing {gp_name} (Round {round_num})")
        try:
            session = ff1.get_session(season, round_num, "R")
            session.load()
        except Exception as e:
            print(f"    ⚠️ Could not load session for Round {round_num}: {e}")
            continue

        # Iterate over each driver
        for drv in session.laps["DriverNumber"].unique():
            laps_drv = session.laps.pick_driver(drv)
            # fetch telemetry; TelemetryLapNumber no longer available
            tel = laps_drv.get_telemetry()

            # Select only existing telemetry columns
            tel = tel[
                [
                    "Time",
                    "RPM",
                    "Speed",
                    "nGear",
                    "Throttle",
                    "Brake",
                    "DRS",
                    "Distance",
                    "RelativeDistance",
                ]
            ]
            tel = tel.rename(
                columns={
                    "Time": "time",
                    "RPM": "engine_rpm",
                    "Speed": "speed_kmh",
                    "nGear": "gear",
                    "Throttle": "throttle",
                    "Brake": "brake",
                    "DRS": "drs",
                    "Distance": "track_distance",
                    "RelativeDistance": "dist_to_car_ahead",
                }
            )

            # Add metadata but note we lack lap_number here
            tel["car_id"] = drv
            tel["grand_prix"] = gp_name
            tel["season"] = season

            feature_frames.append(tel)

    if not feature_frames:
        print("No features collected—check your session data.")
        return

    # Concatenate and write out
    all_feats = pd.concat(feature_frames, ignore_index=True)
    out_file = out_path / f"features_{season}.parquet"
    all_feats.to_parquet(out_file, index=False, compression="snappy")
    print(f"Wrote {out_file} with {len(all_feats)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch F1 telemetry features via FastF1 (races only)"
    )
    parser.add_argument(
        "--season",
        "-s",
        type=int,
        default=2022,
        help="F1 season year to process (only Round ≥ 1)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="data/features",
        help="Directory to write features Parquet",
    )
    args = parser.parse_args()

    fetch_all_features(args.season, args.out)
