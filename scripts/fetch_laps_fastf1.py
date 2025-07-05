# scripts/fetch_laps_fastf1.py

from pathlib import Path

import fastf1

# enable FastF1’s cache (creates it if needed)
fastf1.Cache.enable_cache("data/cache")


def fetch_all_laps(season: int):
    schedule = fastf1.get_event_schedule(season)
    for _, row in schedule.iterrows():
        rnd = int(row["RoundNumber"])
        if rnd < 1:  # skip preseason entries (RoundNumber == 0)
            continue

        print(f"→ Season {season} — Round {rnd}")
        # Use the round number to load the race session
        ses = fastf1.get_session(season, rnd, "R")
        ses.load()

        # Extract and convert lap times
        laps = ses.laps[["DriverNumber", "LapNumber", "LapTime"]].copy()
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        df = laps.rename(
            columns={
                "DriverNumber": "car_number",
                "LapNumber": "lap_number",
                "LapTimeSec": "lap_time",
            }
        )

        # Write to data/historical/{season}/{season}_round_{rnd}.parquet
        out_dir = Path("data") / "historical" / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{season}_round_{rnd}.parquet"
        df.to_parquet(out_file, index=False)

        print(f"Wrote {out_file} ({len(df)} rows)\n")


if __name__ == "__main__":
    for yr in [2022, 2023]:
        fetch_all_laps(yr)
