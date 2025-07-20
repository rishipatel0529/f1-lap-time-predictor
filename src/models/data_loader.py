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
        "sector1_time",
        "sector2_time",
        "sector3_time",
        "Sector1SessionTime",
        "Sector2SessionTime",
        "Sector3SessionTime",
    ]
    for c in td_cols:
        if c in df:
            df[c] = pd.to_timedelta(df[c]).dt.total_seconds().fillna(0.0)

    # LapStartDate → day of year
    if "LapStartDate" in df:
        dt = pd.to_datetime(df["LapStartDate"], errors="coerce")
        df["LapStartDayOfYear"] = dt.dt.dayofyear.fillna(0).astype(int)
        df = df.drop(columns=["LapStartDate"])

    # LapStartTime → seconds since midnight, then cyclical encode
    if "LapStartTime" in df:
        t = pd.to_timedelta(df["LapStartTime"].fillna("0 days 00:00:00"))
        secs = t.dt.total_seconds()
        # seconds in one day
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
    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df = df.dropna(subset=["lap_time"])
    df = preprocess_times(df)
    df = df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"], errors="ignore"
    )
    df = encode_categoricals(df)
    # Drop index if present
    df = df.drop(columns=["index"], errors="ignore")
    return df


def load_data(return_groups: bool = False):
    """
    Returns X, y, and optionally groups.
    groups identifies each rows race (season + grand_prix).
    """
    # Load everything
    df = load_all_races()  # originally 2019–2025
    # **NEW** only keep 2019–2024 for tuning
    df = df[df["season"] <= 2024]

    # 1) Clean target
    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df = df.dropna(subset=["lap_time"])

    # 2) Time → seconds & drop unwanted
    df = preprocess_times(df)
    df = df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"], errors="ignore"
    )

    # 3) Build groups BEFORE you one‑hot away grand_prix
    groups = df["season"].astype(str) + "_" + df["grand_prix"].astype(str)

    # 4) One‑hot encode the categorical features
    df = encode_categoricals(df)

    # 5) Separate X & y
    y = df["lap_time"]
    X = df.drop(columns=["lap_time", "index"], errors="ignore")

    # 6) Keep only numeric & bool
    X = X.select_dtypes(include=[np.number, "bool_"]).fillna(0)

    if return_groups:
        return X, y, groups
    else:
        return X, y
