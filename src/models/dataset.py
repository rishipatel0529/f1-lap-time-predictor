#!/usr/bin/env python3
import argparse
import time
from pathlib import Path

import pandas as pd


def load_telemetry(season: int, telemetry_base: Path) -> pd.DataFrame:
    season_dir = telemetry_base / str(season)
    season_file = season_dir / f"telemetry_{season}.parquet"

    # 1) Try full-season file
    if season_file.exists():
        try:
            df = pd.read_parquet(season_file)
            if "grand_prix" not in df.columns:
                df["grand_prix"] = None
            return df
        except Exception as e:
            print(
                f"⚠️  Could not read {season_file}: {e}; falling back to per-GP slices."
            )

    # 2) Fallback: load per-GP slices
    gp_dir = season_dir / "telemetry_by_gp"
    if not gp_dir.exists():
        raise FileNotFoundError(f"No telemetry under {season_dir} or {gp_dir}")

    dfs = []
    for fp in sorted(gp_dir.glob("*.parquet")):
        # retry on read failures
        while True:
            try:
                df = pd.read_parquet(fp)
                break
            except Exception as e:
                print(f"⏳  Read error on {fp.name}: {e}; retrying in 1s…")
                time.sleep(1)
        gp_key = fp.stem.rsplit(f"_{season}", 1)[0]
        df["grand_prix"] = gp_key
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid telemetry slices in {gp_dir}")
    return pd.concat(dfs, ignore_index=True)


def load_laps(season: int, laps_base: Path) -> pd.DataFrame:
    season_dir = laps_base / str(season)
    files = sorted(season_dir.glob(f"*_{season}.parquet"))
    if not files:
        raise FileNotFoundError(f"No lap files under {season_dir}")

    dfs = []
    for fp in files:
        df = pd.read_parquet(fp)
        gp_key = fp.stem.rsplit(f"_{season}", 1)[0]
        df["grand_prix"] = gp_key

        # normalize driver column
        if "driver" in df.columns:
            df = df.rename(columns={"driver": "driver_id"})
        if "DriverNumber" in df.columns:
            df = df.rename(columns={"DriverNumber": "driver_id"})
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_weather(season: int, weather_base: Path) -> pd.DataFrame:
    season_dir = weather_base / str(season)
    files = sorted(season_dir.glob(f"*_{season}.parquet"))
    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"⚠️  Could not read weather slice {fp.name}: {e}")
            continue
        # extract grand_prix key from filename
        gp_key = fp.stem.rsplit(f"_{season}", 1)[0]
        df["grand_prix"] = gp_key
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def build_master_dataset(
    season: int,
    telemetry_base: Path,
    laps_base: Path,
    weather_base: Path,
    output_path: Path,
):
    # load each data source
    tel = load_telemetry(season, telemetry_base)
    laps = load_laps(season, laps_base)
    weather = load_weather(season, weather_base)

    # 1) Merge lap summaries if telemetry has LapIndex
    if "LapIndex" in tel.columns:
        laps_cols = [
            "driver_id",
            "lap_number",
            "lap_time",
            "sector1_time",
            "sector2_time",
            "sector3_time",
            "position",
        ]
        tel = tel.merge(
            laps[laps_cols],
            left_on=["driver_id", "LapIndex"],
            right_on=["driver_id", "lap_number"],
            how="left",
        )
    else:
        print("⚠️  No LapIndex in telemetry; skipped lap-level merge.")

    # 2) Merge weather on season + grand_prix
    if "grand_prix" in tel.columns and not weather.empty:
        tel = tel.merge(weather, on=["season", "grand_prix"], how="left")
    else:
        print("⚠️  Missing grand_prix or no weather data; skipped weather merge.")

    # 3) Sort for window operations
    sort_keys = ["driver_id", "season"]
    if "grand_prix" in tel.columns:
        sort_keys.append("grand_prix")
    if "lap_number" in tel.columns:
        sort_keys.append("lap_number")
    if "timestamp" in tel.columns:
        sort_keys.append("timestamp")
    tel = tel.sort_values(sort_keys)

    # 4) Rolling stats
    for feat in ["speed", "engine_rpm", "throttle_pct", "brake_pressure"]:
        if feat in tel.columns:
            grp = tel.groupby("driver_id")[feat]
            tel[f"{feat}_roll5_mean"] = (
                grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            tel[f"{feat}_roll5_std"] = (
                grp.rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
            )

    # 5) Acceleration proxy
    if "speed" in tel.columns:
        tel["accel"] = tel.groupby("driver_id")["speed"].diff().fillna(0)
        grp = tel.groupby("driver_id")["accel"]
        tel["accel_roll5_max"] = (
            grp.rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
        )
        tel["accel_roll5_min"] = (
            grp.rolling(5, min_periods=1).min().reset_index(level=0, drop=True)
        )

    # 6) Stint-based features
    if "PitInTime" in tel.columns and "track_distance" in tel.columns:
        tel["in_pit"] = ~tel["PitInTime"].isna()
        tel["stint_id"] = tel.groupby("driver_id")["in_pit"].cumsum()
        tel["time_since_last_pit"] = (
            tel.groupby(["driver_id", "stint_id"])["timestamp"].diff().fillna(0)
        )
        tel["dist_since_last_pit"] = (
            tel.groupby(["driver_id", "stint_id"])["track_distance"].diff().fillna(0)
        )

    # 7) Write final parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tel.to_parquet(output_path, index=False, compression="snappy")
    print(f"Master dataset for season {season} → {output_path}")


def main():
    p = argparse.ArgumentParser("Build master training dataset")
    p.add_argument("--season", "-s", type=int, required=True)
    p.add_argument("--telemetry-dir", type=Path, default=Path("data/raw/fastf1"))
    p.add_argument("--laps-dir", type=Path, default=Path("data/raw/historical"))
    p.add_argument("--weather-dir", type=Path, default=Path("data/raw/race_weather"))
    p.add_argument(
        "--output", "-o", type=Path, default=Path("data/train_dataset_{season}.parquet")
    )
    args = p.parse_args()

    out = Path(str(args.output).format(season=args.season))
    build_master_dataset(
        season=args.season,
        telemetry_base=args.telemetry_dir,
        laps_base=args.laps_dir,
        weather_base=args.weather_dir,
        output_path=out,
    )


if __name__ == "__main__":
    main()
