from __future__ import annotations
import argparse, yaml, pandas as pd, numpy as np
from joblib import load
from pathlib import Path
from src.f1lap.featurize import build_dataset

def _key(df, keys):
    k = df[keys[0]].astype(str)
    for c in keys[1:]:
        k = k + "|" + df[c].astype(str)
    return k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", default="models/model.joblib")
    ap.add_argument("--out", default="artifacts/preds.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    bundle = load(args.model)

    # Call build_dataset WITHOUT is_inference for compatibility
    out = build_dataset(args.data, cfg)
    if len(out) == 5:
        X, y, groups, feats, df = out
    else:
        X, y, groups, feats = out
        df = None  # we won't trust raw CSV lengths

    pred_res = bundle["model"].predict(X)

    # Undo residualization if enabled
    if bundle.get("target_mode") == "residualized":
        keys = bundle.get("residual_key", ["season","race_round"])
        bmap = bundle.get("baseline_map", {})
        fallback = np.median(list(bmap.values()) or [0.0])
        if df is None:
            # If we don't have the filtered df, read then build the same key columns
            df = pd.read_csv(args.data)
        base = _key(df.loc[X.index] if df is not None and not df.index.equals(X.index) else df, keys)
        base = base.map(bmap).fillna(fallback).to_numpy()
        pred = pred_res + base
    else:
        pred = pred_res

    qs = bundle.get("residual_quantiles", {})
    p10 = pred_res + qs.get("P10", -1.0)
    p90 = pred_res + qs.get("P90",  1.0)
    if bundle.get("target_mode") == "residualized":
        p10 = p10 + (base if isinstance(base, np.ndarray) else 0)
        p90 = p90 + (base if isinstance(base, np.ndarray) else 0)

    out_df = pd.DataFrame({"pred_lap_time": pred, "pred_low": p10, "pred_high": p90})

    # Safely attach "actual" if we have a vector aligned with X
    actual_series = None
    if y is not None and len(y) == len(X):
        actual_series = np.asarray(y, dtype=float)
    elif df is not None and "lap_time" in df.columns:
        try:
            # try aligning by index if df shares index with X
            s = df.loc[X.index, "lap_time"]
            if len(s) == len(X):
                actual_series = s.to_numpy(dtype=float)
        except Exception:
            pass

    if actual_series is not None:
        out_df["actual"] = actual_series
        out_df["error"] = out_df["pred_lap_time"] - out_df["actual"]

    Path("artifacts").mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(args.out)

if __name__ == "__main__":
    main()
