# scripts/fetch_historical.py
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://ergast.com/api/f1"


def fetch_laps_for_race(season: int, round_: int):
    url = f"{BASE}/{season}/{round_}/laps.json?limit=2000"
    j = requests.get(url).json()
    records = []
    for lap in j["MRData"]["LapTable"]["Laps"]:
        lap_num = int(lap["number"])
        for timing in lap["Timings"]:
            # parse mm:ss.sss → seconds
            mm, ss = timing["time"].split(":")
            lap_time_s = int(mm) * 60 + float(ss)
            records.append(
                {
                    "season": season,
                    "race_round": round_,
                    "driver": timing["driverId"],
                    "car_number": (
                        int(timing["driverId"][-2:])
                        if timing["driverId"][-2:].isdigit()
                        else None
                    ),
                    "lap_number": lap_num,
                    "lap_time": lap_time_s,
                }
            )
    return pd.DataFrame(records)


def fetch_season(season: int):
    # first, get list of races in this season
    try:
        resp = requests.get(f"{BASE}/{season}/races.json?limit=100", timeout=10)
        resp.raise_for_status()
        j = resp.json()
    except Exception as e:
        print(f"⚠️  Failed to fetch season {season}: {e}")
        return

    races = j["MRData"]["RaceTable"]["Races"]
    out_dir = Path("data/historical") / str(season)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for r in races:
        rnd = int(r["round"])
        print(f"Fetching season {season} round {rnd} …")
        df = fetch_laps_for_race(season, rnd)
        all_dfs.append(df)
        time.sleep(1)

    season_df = pd.concat(all_dfs, ignore_index=True)
    out_file = out_dir / f"{season}_laps.parquet"
    season_df.to_parquet(out_file, index=False)
    print(f"Wrote {out_file} with {len(season_df)} rows")


if __name__ == "__main__":
    # pick the season(s) you want
    for year in [2022, 2023]:
        fetch_season(year)
