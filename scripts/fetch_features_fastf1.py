#!/usr/bin/env python3
import argparse
import os

import fastf1 as ff1
import pandas as pd

# Enable FastF1 cache
ff1.Cache.enable_cache("data/cache")


def fetch_all_features(season: int, output_dir: str = "data/features"):
    """
    For a given F1 season, loads each race (Round ≥ 1),
    pulls driver telemetry, selects & renames the columns
    we actually have, and writes out a CSV of all features.
    """
    print(f"Fetching features for season {season}")

    # full season schedule (includes pre-season, etc.)
    schedule = ff1.get_event_schedule(season)

    feature_frames = []
    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        gp_name = event["EventName"]

        # skip any non-race rounds (e.g. Round 0: testing)
        if round_num < 1:
            print(f"  → Skipping non-race session: {gp_name} (Round {round_num})")
            continue

        print(f"  → Processing {gp_name} (Round {round_num})")

        try:
            session = ff1.get_session(season, round_num, "R")
            session.load()
        except ValueError as e:
            print(f"    ⚠️  Could not load race session for Round {round_num}: {e}")
            continue

        # iterate each driver in that race
        for drv in session.laps["DriverNumber"].unique():
            print(f"    • Driver {drv}")
            laps_drv = session.laps.pick_driver(drv)

            # get raw telemetry
            tel = laps_drv.get_telemetry()

            # select & rename only the columns that exist
            features = tel[
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
            ].rename(
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

            # add metadata columns
            features["driver"] = drv
            features["grand_prix"] = gp_name
            features["season"] = season

            feature_frames.append(features)

    # concatenate and write out
    if not feature_frames:
        print("No features collected—check schedule or session loading.")
        return

    all_feats = pd.concat(feature_frames, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"features_{season}.csv")
    all_feats.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}\n")


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
        help="Directory to write features CSV",
    )
    args = parser.parse_args()

    fetch_all_features(args.season, args.out)
