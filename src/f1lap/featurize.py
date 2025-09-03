"""
src/f1lap/featurize.py

Build a model-ready feature matrix from the consolidated lap dataset.
This module applies consistent cleaning, derives normalized tire/teams, adds per-driver
lag features, performs one-hot encoding for categoricals (including any prefixed one-hots),
filters out leakage-prone columns, and returns (X, y, groups, feature_names).
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, List
from .utils import coerce_booleans, add_group_key, safe_one_hot

def _normalize_compound(df: pd.DataFrame) -> pd.Series:
    # If a single 'compound' exists, keep it. Else derive from one-hot flags like Compound_SOFT, etc.
    if "compound" in df.columns:
        return df["compound"].astype(str)
    comp_cols = [c for c in df.columns if c.startswith("Compound_")]
    if comp_cols:
        # take the column name (sans prefix) with max value
        comp = df[comp_cols].values
        labels = np.array([c.replace("Compound_", "") for c in comp_cols])
        idx = np.argmax(comp, axis=1)
        return pd.Series(labels[idx], index=df.index)
    return pd.Series(pd.NA, index=df.index)

def _normalize_team(df: pd.DataFrame) -> pd.Series:
    # If a text team exists, keep it. Else derive from Team_* one‑hots
    for key in ["team", "Team", "constructor"]:
        if key in df.columns:
            return df[key].astype(str)
    tcols = [c for c in df.columns if c.startswith("Team_")]
    if tcols:
        vals = df[tcols].values
        labels = np.array([c.replace("Team_", "") for c in tcols])
        idx = np.argmax(vals, axis=1)
        return pd.Series(labels[idx], index=df.index)
    return pd.Series(pd.NA, index=df.index)

def build_dataset(csv_path: str, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = coerce_booleans(df)
    for col in cfg["data"]["required_columns"]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data.")
    df = df.dropna(subset=cfg["data"]["required_columns"])
    df = df[(df["lap_time"] >= cfg["filters"]["min_lap_time_sec"]) & (df["lap_time"] <= cfg["filters"]["max_lap_time_sec"])]
    if cfg["filters"]["drop_pit_in_laps"] and "PitDuration" in df.columns:
        df = df[(df["PitDuration"].fillna(0) <= 0.01)]

    df["season_race"] = df["season"].astype(str) + "_" + df["race_round"].astype(str)
    df["compound"] = _normalize_compound(df)
    df["TeamText"] = _normalize_team(df)

    num_cols = [c for c in cfg["candidate_numeric"] if c in df.columns]
    cat_cols = [c for c in cfg["candidate_categorical"] if c in ["race_direction","grand_prix","compound","driver_id","Team","TeamText"] and c in df.columns or c in ["compound","driver_id","TeamText","race_direction","grand_prix"]]

    auto_onehots = []
    for pref in cfg["onehot_prefixes"]:
        auto_onehots.extend([c for c in df.columns if c.startswith(pref)])

    excl = set(cfg["exclude_exact"])
    for pref in cfg["exclude_prefixes"]:
        excl.update([c for c in df.columns if c.startswith(pref)])

    if "lags" in cfg:
        group_cols = ["season","race_round","driver_id"]
        if all(c in df.columns for c in group_cols):
            df = df.sort_values(group_cols + ["lap_number"])
            for base, steps in cfg["lags"].items():
                if base in df.columns:
                    for s in steps:
                        df[f"{base}_lag{s}"] = df.groupby(group_cols)[base].shift(s)
                        num_cols.append(f"{base}_lag{s}")

    keep_cols = list(dict.fromkeys(num_cols))  # dedupe
    X_num = df[keep_cols].copy() if keep_cols else pd.DataFrame(index=df.index)
    X_cat = pd.get_dummies(df[[c for c in cat_cols if c in df.columns]].astype("category"), drop_first=False) if cat_cols else pd.DataFrame(index=df.index)
    X_auto = df[[c for c in auto_onehots if c not in excl]].copy() if auto_onehots else pd.DataFrame(index=df.index)
    X_auto = X_auto[[c for c in X_auto.columns if c not in excl]]
    X = pd.concat([X_num, X_cat, X_auto], axis=1)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(0)
    y = df[cfg["target"]].astype(float)
    groups = df["season_race"].astype(str)
    feature_names = list(X.columns)
    return X, y, groups, feature_names
