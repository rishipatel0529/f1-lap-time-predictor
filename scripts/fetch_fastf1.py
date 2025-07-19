#!/usr/bin/env python3
"""
scripts/fetch_fastf1.py

Fetches for each season:
  - Telemetry: driver_id, time, speed, RPM, gear, throttle, brake, DRS
  - Lap summaries: lap_time, sector_times, position, DRS_active

Writes out:
  data/raw/fastf1/{season}/telemetry_{season}.parquet
  data/raw/fastf1/{season}/telemetry_by_gp/{gp}_{season}.parquet
  data/raw/fastf1/{season}/{season}_laps.parquet

Usage:
    python scripts/collect_fastf1_data.py --seasons 2022 2023
"""
import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd

# enable FastF1 cache
CACHE = Path("data/cache")
fastf1.Cache.enable_cache(str(CACHE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def sanitize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def fetch_telemetry(season: int, out_base: Path):
    logging.info(f"Fetching telemetry for season {season}")
    schedule = fastf1.get_event_schedule(season)

    all_tel = []
    by_gp_dir = out_base / str(season) / "telemetry_by_gp"
    by_gp_dir.mkdir(parents=True, exist_ok=True)

    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd < 1:
            continue
        gp = ev["EventName"]
        gp_key = sanitize(gp)
        logging.info(f"  → {gp} (Round {rnd})")

        try:
            sess = fastf1.get_session(season, rnd, "R")
            sess.load()
        except Exception as e:
            logging.warning(f"     skipped {gp}: {e}")
            continue

        # collect lap summaries for merge
        laps = sess.laps[
            [
                "DriverNumber",
                "LapNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Position",
            ]
        ].copy()
        laps["driver_id"] = laps["DriverNumber"].astype(int)
        laps["lap_time"] = laps["LapTime"].dt.total_seconds()
        laps = laps.rename(
            columns={
                "LapNumber": "lap_number",
                "Sector1Time": "sector1_time",
                "Sector2Time": "sector2_time",
                "Sector3Time": "sector3_time",
                "Position": "position",
            }
        )
        laps = laps[
            [
                "driver_id",
                "lap_number",
                "lap_time",
                "sector1_time",
                "sector2_time",
                "sector3_time",
                "position",
            ]
        ]

        # for each driver, fetch telemetry and merge lap context
        for drv in laps["driver_id"].unique():
            tel = sess.laps.pick_driver(drv).get_telemetry()

            # select only the available fields
            df = tel[
                ["Time", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS"]
            ].copy()
            df = df.rename(
                columns={
                    "Time": "timestamp",
                    "Speed": "speed",
                    "RPM": "engine_rpm",
                    "nGear": "gear",
                    "Throttle": "throttle_pct",
                    "Brake": "brake_pressure",
                    "DRS": "drs_active",
                }
            )

            # annotate
            df["driver_id"] = int(drv)
            df["season"] = season
            df["grand_prix"] = gp

            # merge lap context (matches by index within each lap)
            # FastF1 sets df["LapIndex"] for merge:
            if "LapIndex" in df.columns:
                df = df.merge(
                    laps,
                    left_on=["driver_id", "LapIndex"],
                    right_on=["driver_id", "lap_number"],
                    how="left",
                )

            # collect
            all_tel.append(df)

        # write per-GP file
        gp_df = pd.concat(
            [d for d in all_tel if d.iloc[0]["grand_prix"] == gp], ignore_index=True
        )
        gp_file = by_gp_dir / f"{gp_key}_{season}.parquet"
        gp_df.to_parquet(gp_file, index=False, compression="snappy")
        logging.info(f"    • wrote {gp_file}")

    if not all_tel:
        raise RuntimeError(f"No telemetry collected for {season}")

    season_df = pd.concat(all_tel, ignore_index=True)
    season_file = out_base / str(season) / f"telemetry_{season}.parquet"
    season_df.to_parquet(season_file, index=False, compression="snappy")
    logging.info(f"season telemetry: {season_file}")
    return season_df


def fetch_laps(season: int, out_base: Path):
    logging.info(f"Fetching lap summaries for season {season}")
    schedule = fastf1.get_event_schedule(season)
    laps_all = []
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd < 1:
            continue
        gp = ev["EventName"]
        logging.info(f"  → {gp} (Round {rnd})")
        try:
            sess = fastf1.get_session(season, rnd, "R")
            sess.load()
        except Exception as e:
            logging.warning(f"     skipped {gp}: {e}")
            continue

        laps = sess.laps[
            [
                "DriverNumber",
                "LapNumber",
                "LapTime",
                "Sector1Time",
                "Sector2Time",
                "Sector3Time",
                "Position",
            ]
        ].copy()
        laps["driver_id"] = laps["DriverNumber"].astype(int)
        laps["lap_time"] = laps["LapTime"].dt.total_seconds()
        laps = laps.rename(
            columns={
                "LapNumber": "lap_number",
                "Sector1Time": "sector1_time",
                "Sector2Time": "sector2_time",
                "Sector3Time": "sector3_time",
                "Position": "position",
            }
        )
        laps["season"] = season
        laps["grand_prix"] = gp
        laps_all.append(
            laps[
                [
                    "season",
                    "grand_prix",
                    "driver_id",
                    "lap_number",
                    "lap_time",
                    "sector1_time",
                    "sector2_time",
                    "sector3_time",
                    "position",
                ]
            ]
        )

    if not laps_all:
        raise RuntimeError(f"No lap data for {season}")

    out_file = out_base / str(season) / f"{season}_laps.parquet"
    pd.concat(laps_all, ignore_index=True).to_parquet(
        out_file, index=False, compression="snappy"
    )
    logging.info(f"season laps: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect FastF1 telemetry + laps")
    p.add_argument(
        "--seasons",
        "-s",
        nargs="+",
        type=int,
        default=list(range(2018, 2026)),
        help="Seasons to process",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("data/raw/fastf1"),
        help="Base output directory",
    )
    args = p.parse_args()

    for yr in args.seasons:
        season_dir = args.out / str(yr)
        season_dir.mkdir(parents=True, exist_ok=True)
        # telemetry + partition
        fetch_telemetry(yr, args.out)
        # lap summaries
        fetch_laps(yr, args.out)
