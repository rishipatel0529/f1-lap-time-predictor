"""
scripts/driver_telemetry_code_files/fetch_fastf1.py

Fetches F1 telemetry and lap summaries via FastF1 for one or more seasons.
Produces Parquet outputs partitioned by season and by Grand Prix, plus a season-wide rollup.
Uses the on-disk FastF1 cache for faster re-runs and lower network usage.
Intended as a reproducible ETL step feeding model-ready datasets.
"""
import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd

# Configure the FastF1 local cache; repeated runs will reuse downloaded data
CACHE = Path("data/cache")
fastf1.Cache.enable_cache(str(CACHE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def sanitize(name: str) -> str:
    # Normalizes event names into safe file keys (e.g., "Bahrain Grand Prix" -> "bahrain_grand_prix")
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
        logging.info(f"{gp} (Round {rnd})")

        # Load the race session; .load() populates laps and allows telemetry extraction
        try:
            sess = fastf1.get_session(season, rnd, "R")
            sess.load()
        except Exception as e:
            logging.warning(f"skipped {gp}: {e}")
            continue

        # Extract a compact lap context table used to enrich per-driver telemetry
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

        # For each driver in this event, pull high-frequency telemetry and join lap context if available
        for drv in laps["driver_id"].unique():
            tel = sess.laps.pick_driver(drv).get_telemetry()

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

            df["driver_id"] = int(drv)
            df["season"] = season
            df["grand_prix"] = gp

            # If telemetry carries LapIndex, align each telemetry row to its lap-level context
            if "LapIndex" in df.columns:
                df = df.merge(
                    laps,
                    left_on=["driver_id", "LapIndex"],
                    right_on=["driver_id", "lap_number"],
                    how="left",
                )

            all_tel.append(df)

        # Persist a per-GP telemetry file to simplify downstream incremental processing
        gp_df = pd.concat(
            [d for d in all_tel if d.iloc[0]["grand_prix"] == gp], ignore_index=True
        )
        gp_file = by_gp_dir / f"{gp_key}_{season}.parquet"
        gp_df.to_parquet(gp_file, index=False, compression="snappy")
        logging.info(f"wrote {gp_file}")

    if not all_tel:
        raise RuntimeError(f"No telemetry collected for {season}")

    # Write a season-wide concatenated telemetry file for convenience
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
        logging.info(f"{gp} (Round {rnd})")
        # Load race session to expose the laps table with timing columns
        try:
            sess = fastf1.get_session(season, rnd, "R")
            sess.load()
        except Exception as e:
            logging.warning(f"skipped {gp}: {e}")
            continue

        # Normalize core lap columns into a consistent schema across events
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

    # Persist a season-level Parquet with lap summaries for all rounds
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
        default=list(range(2018, 2025)),
        help="Seasons to process",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("data/driver_telemetry_csv_files"),
        help="Base output directory",
    )
    args = p.parse_args()

    for yr in args.seasons:
        season_dir = args.out / str(yr)
        season_dir.mkdir(parents=True, exist_ok=True)
        # Generate per-GP and season-level telemetry artifacts
        fetch_telemetry(yr, args.out)
        # Generate season-level lap summary artifacts
        fetch_laps(yr, args.out)
