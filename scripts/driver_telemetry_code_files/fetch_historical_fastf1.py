"""
scripts/driver_telemetry_code_files/fetch_historical_fastf1.py

Fetches historical race lap summaries with FastF1 and writes one Parquet per Grand Prix.
Enriches laps with grid/finish positions, tyre compound (if present), and session-end weather.
Uses an on-disk FastF1 cache so repeated runs are fast and network-light.
Intended as a reproducible ETL step to build model-ready lap-level datasets.
"""

import argparse
from pathlib import Path

import fastf1 as ff1

# Initialize and enable a persistent cache directory for FastF1 network responses
cache_dir = Path("data/cache/fastf1")
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(str(cache_dir))


def fetch_race_laps(season: int, rnd: int, out_dir: Path):
    ses = ff1.get_session(season, rnd, "R")
    ses.load() # Load timing/laps/results into memory for this race session

    # Select the lap timing columns we care about (lap/sector times and pit in/out markers)
    laps = ses.laps[
        [
            "DriverNumber",
            "LapNumber",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "PitInTime",
            "PitOutTime",
        ]
    ].copy()

    # Normalize column names and convert timedelta columns to seconds for modeling
    laps = laps.rename(
        columns={
            "DriverNumber": "driver_id",
            "LapNumber": "lap_number",
            "LapTime": "lap_time",
            "Sector1Time": "sector1_time",
            "Sector2Time": "sector2_time",
            "Sector3Time": "sector3_time",
        }
    )
    for col in ["lap_time", "sector1_time", "sector2_time", "sector3_time"]:
        laps[col] = laps[col].dt.total_seconds()

    # Pull classification info; include compound if available to enrich features
    res = ses.results.rename(
        columns={
            "DriverNumber": "driver_id",
            "GridPosition": "grid_position",
            "Position": "finish_position",
        }
    )
    merge_cols = ["driver_id", "grid_position", "finish_position"]
    if "Compound" in res.columns:
        res = res.rename(columns={"Compound": "compound"})
        merge_cols.append("compound")

    # Left-join per-driver results into the lap table
    laps = laps.merge(res[merge_cols], on="driver_id", how="left")

    # Attach last-known (session-end) weather snapshot to each row for quick context
    weather = ses.weather_data.iloc[-1]
    for col in ["Conditions", "Temperature", "Humidity", "TrackTemperature"]:
        laps[col.lower()] = weather.get(col, None)

    # Add season/round metadata and a file-safe Grand Prix key
    laps["season"] = season
    laps["race_round"] = rnd
    gp_name = ses.event["EventName"]
    safe_gp = gp_name.lower().replace(" ", "_")
    laps["grand_prix"] = safe_gp

    # Ensure output dir exists and write a Parquet file per event
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_gp}_{season}.parquet"
    laps.to_parquet(out_path, index=False, compression="snappy")
    print(f"Wrote {out_path} ({len(laps)} rows)")


def main():
    p = argparse.ArgumentParser("Fetch lap summaries via FastF1")
    p.add_argument("--season", "-s", type=int, required=True)
    args = p.parse_args()

    # Iterate the full season schedule and export each round’s lap table
    out_base = Path("data/driver_telemetry_csv_files") / str(args.season)
    schedule = ff1.get_event_schedule(args.season)
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd < 1:
            continue
        fetch_race_laps(args.season, rnd, out_base)


if __name__ == "__main__":
    main()
