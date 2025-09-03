"""
src/f1lap/utils.py

Utility helpers used across feature building and training.
Focus areas: (1) normalizing boolean-like string columns into 0/1,
(2) composing stable group keys from multiple columns, and
(3) safe one-hot encoding that tolerates missing columns.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict

BOOL_TRUE = {"true", "True", "TRUE", True, 1, "1"}

def coerce_booleans(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object and df[c].dropna().astype(str).str.lower().isin(["true","false"]).any():
            df[c] = df[c].astype(str).str.lower().map({"true":1, "false":0})
    return df

def add_group_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    return df[cols].astype(str).agg("_".join, axis=1)

def safe_one_hot(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.DataFrame(index=df.index)
    return pd.get_dummies(df[existing].astype("category"), drop_first=False)
