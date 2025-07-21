# src/models/data_loader.py

from pathlib import Path

import numpy as np
import pandas as pd


def load_all_races(root: str = "data/raw/three_merged_files") -> pd.DataFrame:
    """Reads, aligns, and concatenates all per-season CSVs."""
    paths = list(Path(root).rglob("*.csv"))
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["season"] = int(p.parts[-2])  # e.g. ".../2019/..." → 2019
        df["grand_prix"] = p.stem.replace(f"_{df['season'][0]}", "")
        dfs.append(df)
    all_cols = set().union(*(df.columns for df in dfs))
    aligned = [df.reindex(columns=all_cols) for df in dfs]
    return pd.concat(aligned, ignore_index=True)


def preprocess_times(df: pd.DataFrame) -> pd.DataFrame:
    # List of timedelta‑like columns to convert to seconds
    td_cols = [
        "Time",
        "PitInTime",
        "PitOutTime",
        "pit_stop_duration",
        "Sector1SessionTime",
        "Sector2SessionTime",
        "Sector3SessionTime",
    ]
    for c in td_cols:
        if c in df:
            df[c] = pd.to_timedelta(df[c]).dt.total_seconds().fillna(0.0)

    # sector*_time are already in seconds
    for c in ["sector1_time", "sector2_time", "sector3_time"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # LapStartDate → day of year
    if "LapStartDate" in df:
        dt = pd.to_datetime(df["LapStartDate"], errors="coerce")
        df["LapStartDayOfYear"] = dt.dt.dayofyear.fillna(0).astype(int)
        df = df.drop(columns=["LapStartDate"])

    # LapStartTime → seconds since midnight, then cyclical encode
    if "LapStartTime" in df:
        t = pd.to_timedelta(df["LapStartTime"].fillna("0 days 00:00:00"))
        secs = t.dt.total_seconds()
        day_secs = 24 * 3600
        ang = 2 * np.pi * (secs / day_secs)
        df["LapStart_sin"] = np.sin(ang)
        df["LapStart_cos"] = np.cos(ang)
        df = df.drop(columns=["LapStartTime"])

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    # Columns to one-hot encode
    cats = ["grand_prix", "Team", "Driver", "Compound"]
    present = [c for c in cats if c in df]
    return pd.get_dummies(df, columns=present, drop_first=False)


def load_raw_df():
    """
    Returns the full preprocessed DataFrame (with all features & lap_time target),
    but before train/test split.
    """
    df = load_all_races()

    # --- target & basic clean ---
    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df = df.dropna(subset=["lap_time"])

    # --- convert all times to seconds / encode cyclic fields ---
    df = preprocess_times(df)

    # --- now synthesize PitDuration and drop the old columns ---
    if "PitInTime" in df.columns and "PitOutTime" in df.columns:
        df["PitDuration"] = df["PitOutTime"] - df["PitInTime"]
        df = df.drop(
            columns=["PitInTime", "PitOutTime", "pit_stop_duration"],
            errors="ignore",
        )

    # --- drop a few other unused bits ---
    df = df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"], errors="ignore"
    )

    # --- one-hot everything else ---
    df = encode_categoricals(df)

    # --- lose stray index col if present ---
    df = df.drop(columns=["index"], errors="ignore")

    # —— start reorder block ——
    # 1) your “core” columns in the exact order you want:
    cols = [
        "season",
        "race_round",
        "driver_id",
        "lap_number",
        "position",
        "grid_position",
        "finish_position",
        "lap_time",
        "Stint",
        "pit_stop_lap",
        "PitDuration",
        "FreshTyre",
        "TyreLife",
        "IsPersonalBest",
        "SpeedFL",
        "SpeedI1",
        "SpeedI2",
        "SpeedST",
        "Time",
        "LapStartDayOfYear",
        "LapStart_sin",
        "LapStart_cos",
        "sector1_time",
        "sector2_time",
        "sector3_time",
        "Sector1SessionTime",
        "Sector2SessionTime",
        "Sector3SessionTime",
        "humidity",
        "Deleted",
        "FastF1Generated",
    ]
    # 2) everything else, sorted alphabetically
    rest = sorted(c for c in df.columns if c not in cols)
    # 3) build and apply the final ordering
    ordered = [c for c in cols if c in df.columns] + rest
    df = df[ordered]
    # —— end reorder block ——

    return df


def load_data(return_groups: bool = False):
    """
    Returns X, y, and optionally groups.
    groups identifies each row’s race (season + grand_prix).
    """
    df = load_all_races()
    df = df[df["season"] <= 2024]

    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df = df.dropna(subset=["lap_time"])

    df = preprocess_times(df)
    df = df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"], errors="ignore"
    )

    groups = df["season"].astype(str) + "_" + df["grand_prix"].astype(str)

    df = encode_categoricals(df)

    y = df["lap_time"]
    X = df.drop(columns=["lap_time", "index"], errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool_"]).fillna(0)

    if return_groups:
        return X, y, groups
    return X, y
