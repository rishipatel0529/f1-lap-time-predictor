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
    # 1) Convert HH:MM:SS‑style fields to seconds
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

    # 2) sector*_time are already numeric seconds
    for c in ["sector1_time", "sector2_time", "sector3_time"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 3) Cyclic encode lap‑start date & time
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

    # 4) Align each pit‑out with the *next* lap's pit‑in, compute PitDuration
    needed = {"season", "grand_prix", "driver_id", "PitInTime", "PitOutTime"}
    if needed.issubset(df.columns):
        # groupby.shift(-1) on the original row order
        df["PitOutAligned"] = df.groupby(["season", "grand_prix", "driver_id"])[
            "PitOutTime"
        ].shift(-1)

        # only compute when both PitInTime & aligned PitOut are > 0
        mask = (df["PitInTime"] > 0) & (df["PitOutAligned"] > 0)
        df["PitDuration"] = 0.0
        df.loc[mask, "PitDuration"] = (
            df.loc[mask, "PitOutAligned"] - df.loc[mask, "PitInTime"]
        )

        # drop the old columns
        df.drop(
            columns=["PitInTime", "PitOutTime", "pit_stop_duration", "PitOutAligned"],
            inplace=True,
            errors="ignore",
        )

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One‑hot encode all of: grand_prix, Team, Driver, Compound."""
    cats = ["grand_prix", "Team", "Driver", "Compound"]
    to_encode = [c for c in cats if c in df]
    return pd.get_dummies(df, columns=to_encode, drop_first=False)


def load_raw_df() -> pd.DataFrame:
    """
    Returns the full preprocessed DataFrame (with all features & lap_time target),
    but *before* train/test split.
    """
    df = load_all_races()

    # — target & drop null laps —
    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df.dropna(subset=["lap_time"], inplace=True)

    # — times → seconds, pit logic, cyclic encoding —
    df = preprocess_times(df)

    # — drop unused metadata —
    df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"],
        inplace=True,
        errors="ignore",
    )

    # — one‑hot categoricals —
    df = encode_categoricals(df)

    # — drop stray index col if present —
    df.drop(columns=["index"], inplace=True, errors="ignore")

    # — reorder columns to your “core” + the rest alphabetically —
    core = [
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
    rest = sorted(c for c in df.columns if c not in core)
    df = df[[c for c in core if c in df] + rest]

    return df


def load_data(return_groups: bool = False):
    """
    Returns (X, y) or (X, y, groups), where groups = "season_grand_prix".
    Only seasons ≤2024 are kept for tuning.
    """
    df = load_all_races()
    df = df[df["season"] <= 2024]

    df["lap_time"] = pd.to_numeric(df["lap_time"], errors="coerce")
    df.dropna(subset=["lap_time"], inplace=True)

    df = preprocess_times(df)
    df.drop(
        columns=["TrackStatus", "DeletedReason", "IsAccurate"],
        inplace=True,
        errors="ignore",
    )

    groups = df["season"].astype(str) + "_" + df["grand_prix"].astype(str)
    df = encode_categoricals(df)

    y = df["lap_time"]
    X = df.drop(columns=["lap_time", "index"], errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool_"]).fillna(0)

    if return_groups:
        return X, y, groups
    return X, y
