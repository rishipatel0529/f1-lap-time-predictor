"""
scripts/driver_telemetry_code_files/fetch_f1tv_data.py

This script pulls race sessions from F1TV via the FastF1 library and writes two Parquet files per event:
 (1) per-driver telemetry time series and (2) a laps table with normalized column names.
It iterates over the configured seasons, uses a local FastF1 cache for speed/offline use, and
organizes outputs under FASTF1_CACHE_PATH (or a default path) by season and Grand Prix.
"""

import os
import fastf1
import pandas as pd
from dotenv import load_dotenv

def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")

def main():
    # Read config from .env and set lightweight runtime knobs (cache dir, seasons to fetch)
    load_dotenv("config/.env")
    CACHE_DIR = os.getenv("FASTF1_CACHE_PATH", "data/driver_telemetry_csv_files")
    SEASONS = list(range(2019, 2025))

    # Ensure the on-disk FastF1 cache exists and is enabled so repeated runs avoid re-downloading
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    # Walk every event in each season; for each race session load both telemetry and lap timing
    for year in SEASONS:
        sched = fastf1.get_event_schedule(year)
        # Uncomment to get column names:
        # print(f"Schedule columns for {year}:", sched.columns)

        for _, ev in sched.iterrows():
            round_no = ev["RoundNumber"]
            gp_name = ev["EventName"]
            safe_gp = slugify(gp_name)

            print(f"\nFetching {year} · {gp_name} (round {round_no})")
            try:
                # Load the Race ("R") session with both telemetry streams and aggregated laps
                sess = fastf1.get_session(year, round_no, "R")
                sess.load(telemetry=True, laps=True)
            except Exception as e:
                print(f"skipped ({e})")
                continue

            # Flatten the per-driver telemetry dict (sess.car_data) into a single DataFrame with driver_id attached
            all_tel = []
            for drv, df in sess.car_data.items():
                df = df.reset_index()
                df["driver_id"] = drv
                all_tel.append(df)
            telemetry_df = pd.concat(all_tel, ignore_index=True)

            tel_outdir = os.path.join(CACHE_DIR, str(year), "telemetry_by_gp")
            os.makedirs(tel_outdir, exist_ok=True)
            tel_path = os.path.join(tel_outdir, f"{safe_gp}_{year}.parquet")
            telemetry_df.to_parquet(tel_path, index=False)
            print(f"telemetry: {tel_path}")

            # Normalize key lap columns into a stable schema for downstream merging and model features
            laps = sess.laps.reset_index().rename(
                columns={
                    "DriverNumber": "driver_id",
                    "LapNumber": "lap_number",
                    "LapTime": "lap_time",
                    "Sector1Time": "sector1_time",
                    "Sector2Time": "sector2_time",
                    "Sector3Time": "sector3_time",
                    "GridPosition": "grid_position",
                    "FinishPosition": "finish_position",
                }
            )

            # Derive pit-related fields (pit lap and pit service duration in seconds) from in/out timestamps
            laps["pit_stop_lap"] = laps["lap_number"].where(laps["PitInTime"].notna())
            laps["pit_stop_duration"] = (
                laps["PitOutTime"] - laps["PitInTime"]
            ).dt.total_seconds()

            # Persist both outputs next to the cache: telemetry per GP and a per-GP laps table
            laps_outdir = os.path.join(CACHE_DIR, str(year), "laps_by_gp")
            os.makedirs(laps_outdir, exist_ok=True)
            laps_path = os.path.join(laps_outdir, f"{safe_gp}_laps_{year}.parquet")
            laps.to_parquet(laps_path, index=False)
            print(f"laps: {laps_path}")


if __name__ == "__main__":
    main()
