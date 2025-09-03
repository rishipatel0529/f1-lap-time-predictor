"""
src/f1lap/train.py

Train script for the solo F1 lap-time predictor.
Reads YAML config, builds features from a single CSV, runs GroupKFold CV to avoid race leakage,
reports per-fold and overall RMSE/MAE, then fits a final model on all data and saves it.
Also derives quick feature importances (impurity or a small permutation fallback) for diagnostics.
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from joblib import dump
from .featurize import build_dataset

def rmse(y_true, y_pred):
    # Convenience wrapper for RMSE to keep printing/JSON concise
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def parse_args():
    # CLI: data path, config, output dirs for metrics and trained model bundle
    ap = argparse.ArgumentParser(description="Train F1 Lap-Time Predictor (solo)")
    ap.add_argument("--data", required=True, help="Path to CSV with all telemetry/metadata")
    ap.add_argument("--config", default="config/default.yaml", help="YAML config path")
    ap.add_argument("--outdir", default="artifacts", help="Where to write metrics/plots")
    ap.add_argument("--modeldir", default="models", help="Where to save trained model")
    return ap.parse_args()

def get_model(cfg):
    # Choose estimator based on config; keep seeds/params stable for reproducibility
    mtype = cfg["model"]["type"]
    if mtype == "hgb":
        p = cfg["model"]["params"]
        return HistGradientBoostingRegressor(
            max_depth=p.get("max_depth", None),
            max_iter=p.get("max_iter", 400),
            learning_rate=p.get("learning_rate", 0.08),
            min_samples_leaf=p.get("min_samples_leaf", 20),
            l2_regularization=p.get("l2_regularization", 0.0),
            random_state=42
        )
    elif mtype == "rf":
        return RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Build feature matrix X, target y, and grouping labels for CV from the CSV + config
    X, y, groups, feat_names = build_dataset(args.data, cfg)
    print(f"[data] rows={len(X):,} cols={X.shape[1]}")

    # GroupKFold ensures laps from the same (season,race_round) are not split across folds
    gkf = GroupKFold(n_splits=cfg["cv"].get("n_splits", 5))
    fold_metrics = []
    oof = np.zeros(len(y), dtype=float) # out-of-fold predictions for unbiased overall metrics

    for i, (tr, va) in enumerate(gkf.split(X, y, groups)):
        model = get_model(cfg)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        r = {
            "fold": i,
            "rmse": rmse(y.iloc[va], pred),
            "mae": float(mean_absolute_error(y.iloc[va], pred)),
        }
        fold_metrics.append(r)
        print(f"[fold {i}] RMSE={r['rmse']:.3f}s  MAE={r['mae']:.3f}s")

    # Aggregate overall quality using OOF predictions; write metrics.json for CI/reporting
    overall = {
        "rmse": rmse(y, oof),
        "mae": float(mean_absolute_error(y, oof)),
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "cv": fold_metrics,
    }
    print(f"[overall] RMSE={overall['rmse']:.3f}s  MAE={overall['mae']:.3f}s")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"metrics.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Fit the final model on all data to maximize training signal prior to persistence
    final_model = get_model(cfg).fit(X, y)

    # Try to provide feature importance: prefer impurity if available, else cheap permutation on a sample
    imp = None
    if hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
    elif hasattr(final_model, "feature_names_in_") and hasattr(final_model, "feature_importances_"):
        imp = final_model.feature_importances_
    else:
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(final_model, X.sample(min(1000, len(X)), random_state=42),
                                       y.sample(min(1000, len(y)), random_state=42), n_repeats=5, random_state=42, n_jobs=-1)
            imp = r.importances_mean
        except Exception:
            imp = None

    if imp is not None:
        fi = pd.DataFrame({"feature": feat_names[:len(imp)], "importance": imp}).sort_values("importance", ascending=False)
        fi.to_csv(outdir/"feature_importance.csv", index=False)

    # Persist model bundle (estimator + feature names) so predict_cli can reproduce the same columns
    modeldir = Path(args.modeldir); modeldir.mkdir(parents=True, exist_ok=True)
    dump({"model": final_model, "feature_names": feat_names}, modeldir/"model.joblib")
    print(f"[save] model → {modeldir/'model.joblib'}")
    print(f"[artifacts] metrics → {outdir/'metrics.json'}")

if __name__ == "__main__":
    main()
