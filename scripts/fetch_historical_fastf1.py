# scripts/fetch_historical_fastf1.py
import argparse
from pathlib import Path

import fastf1 as ff1

# Cache setup
cache_dir = Path("data/cache/fastf1")
cache_dir.mkdir(parents=True, exist_ok=True)
ff1.Cache.enable_cache(str(cache_dir))


def fetch_race_laps(season: int, rnd: int, out_dir: Path):
    ses = ff1.get_session(season, rnd, "R")
    ses.load()  # downloads telemetry & laps

    # 1) Pull lap summaries & timing
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

    # HERE’S THE FIX: rename DriverNumber → driver_id, LapNumber → lap_number, etc.
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
    # convert all the Timedelta columns to seconds
    for col in ["lap_time", "sector1_time", "sector2_time", "sector3_time"]:
        laps[col] = laps[col].dt.total_seconds()

    # 2) Pull result info (grid, finish, compound if present)
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

    # merge on driver_id (now present in both)
    laps = laps.merge(res[merge_cols], on="driver_id", how="left")

    # 3) Session‐end weather
    weather = ses.weather_data.iloc[-1]
    for col in ["Conditions", "Temperature", "Humidity", "TrackTemperature"]:
        laps[col.lower()] = weather.get(col, None)

    # 4) Add metadata
    laps["season"] = season
    laps["race_round"] = rnd
    gp_name = ses.event["EventName"]
    safe_gp = gp_name.lower().replace(" ", "_")
    laps["grand_prix"] = safe_gp

    # 5) Write out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{safe_gp}_{season}.parquet"
    laps.to_parquet(out_path, index=False, compression="snappy")
    print(f"Wrote {out_path} ({len(laps)} rows)")


def main():
    p = argparse.ArgumentParser("Fetch lap summaries via FastF1")
    p.add_argument("--season", "-s", type=int, required=True)
    args = p.parse_args()

    out_base = Path("data/raw/historical_fastf1") / str(args.season)
    schedule = ff1.get_event_schedule(args.season)
    for _, ev in schedule.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd < 1:
            continue
        fetch_race_laps(args.season, rnd, out_base)


if __name__ == "__main__":
    main()
