#!/usr/bin/env python3
import os

import fastf1
import pandas as pd
from dotenv import load_dotenv


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


def main():
    # CONFIG
    load_dotenv("config/.env")
    CACHE_DIR = os.getenv("FASTF1_CACHE_PATH", "data/driver_telemetry_csv_files")
    SEASONS = list(range(2019, 2025))

    # CACHE SETUP
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    # FETCH LOOP
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
                sess = fastf1.get_session(year, round_no, "R")
                sess.load(telemetry=True, laps=True)
            except Exception as e:
                print(f"skipped ({e})")
                continue

            # Telemetry for all drivers via session.car_data
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

            # Laps (compound, pit events, lap & sector times, grid/finish)
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

            # derive pit fields
            laps["pit_stop_lap"] = laps["lap_number"].where(laps["PitInTime"].notna())
            laps["pit_stop_duration"] = (
                laps["PitOutTime"] - laps["PitInTime"]
            ).dt.total_seconds()

            laps_outdir = os.path.join(CACHE_DIR, str(year), "laps_by_gp")
            os.makedirs(laps_outdir, exist_ok=True)
            laps_path = os.path.join(laps_outdir, f"{safe_gp}_laps_{year}.parquet")
            laps.to_parquet(laps_path, index=False)
            print(f"laps: {laps_path}")


if __name__ == "__main__":
    main()
