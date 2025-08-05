from pathlib import Path

import numpy as np
import pandas as pd


def load_all_races(root: str = "data/raw/three_merged_files") -> pd.DataFrame:
    "Reads, aligns, and merges all per-season CSVs under `root`."
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
    # Convert normal time fields to seconds
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

    for c in ["sector1_time", "sector2_time", "sector3_time"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Cyclic encode lap-start date & time
    if "LapStartDate" in df:
        dt = pd.to_datetime(df["LapStartDate"], errors="coerce")
        df["LapStartDayOfYear"] = dt.dt.dayofyear.fillna(0).astype(int)
        df.drop(columns=["LapStartDate"], inplace=True)

    if "LapStartTime" in df:
        t = pd.to_timedelta(df["LapStartTime"].fillna("0 days"))
        secs = t.dt.total_seconds()
        ang = 2 * np.pi * (secs / (24 * 3600))
        df["LapStart_sin"] = np.sin(ang)
        df["LapStart_cos"] = np.cos(ang)
        df.drop(columns=["LapStartTime"], inplace=True)

    # Align each pit-out with the *next* lap's pit-in, compute PitDuration
    needed = {"season", "grand_prix", "driver_id", "PitInTime", "PitOutTime"}
    if needed.issubset(df.columns):
        df["PitOutAligned"] = df.groupby(["season", "grand_prix", "driver_id"])[
            "PitOutTime"
        ].shift(-1)
        mask = (df["PitInTime"] > 0) & (df["PitOutAligned"] > 0)
        df["PitDuration"] = 0.0
        df.loc[mask, "PitDuration"] = (
            df.loc[mask, "PitOutAligned"] - df.loc[mask, "PitInTime"]
        )
        df.drop(
            columns=["PitInTime", "PitOutTime", "pit_stop_duration", "PitOutAligned"],
            inplace=True,
            errors="ignore",
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    "One-hot encode all of: grand_prix, Team, Driver, Compound."
    cats = ["grand_prix", "Team", "Driver", "Compound"]
    to_encode = [c for c in cats if c in df]
    return pd.get_dummies(df, columns=to_encode, drop_first=False)


def load_data(data_path: str = None, return_groups: bool = False):
    """
    Loads data into (X, y) or (X, y, groups).
    If data_path points to a CSV, reads that single file; otherwise
    falls back to load_all_races(data_path or default).
    """
    if data_path and Path(data_path).suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = load_all_races(data_path or "data/raw/three_merged_files")

    df = df[df["season"] <= 2024].copy()

    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df.dropna(subset=["lap_time"], inplace=True)

    df = preprocess_times(df)

    df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"],
        inplace=True,
        errors="ignore",
    )

    groups = df["season"]

    df = encode_categoricals(df)
    y = df["lap_time"]
    X = df.drop(columns=["lap_time", "index"], errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool_"]).fillna(0)

    if return_groups:
        return X, y, groups
    return X, y
