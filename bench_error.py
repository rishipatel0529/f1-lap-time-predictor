#!/usr/bin/env python3
# bench_error.py

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    # 1) load and clean
    df = pd.read_parquet("data/train_dataset.parquet").dropna()

    # 2) re‐split to get the same validation set
    train_idx, val_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    val = df.loc[val_idx].copy()

    # 3) separate features/target
    X_val = val.drop(["lap_time", "lap_number"], axis=1)
    y_val = val["lap_time"]

    # 4) load models
    baseline = joblib.load("models/lgb_baseline_2022.pkl")
    final = joblib.load("models/lgb_final_2022.pkl")

    # 5) predict
    # drop car_id before feeding into the model
    feats = X_val.drop("car_id", axis=1)
    val["pred_baseline"] = baseline.predict(feats)
    val["pred_final"] = final.predict(feats)

    # helper to get RMSE
    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5

    # 6) overall RMSE
    rmse_base = rmse(y_val, val["pred_baseline"])
    rmse_new = rmse(y_val, val["pred_final"])
    print("Overall Validation RMSE:")
    print(f"  • Baseline: {rmse_base:.3f} s")
    print(f"  • Final   : {rmse_new:.3f} s\n")

    # 7) per-car RMSE
    rows = []
    for car, grp in val.groupby("car_id"):
        r_base = rmse(grp["lap_time"], grp["pred_baseline"])
        r_new = rmse(grp["lap_time"], grp["pred_final"])
        rows.append(
            {
                "car_id": car,
                "rmse_baseline": r_base,
                "rmse_final": r_new,
                "n_laps": len(grp),
            }
        )

    per_car = pd.DataFrame(rows).sort_values("rmse_final")
    print("Per-car RMSE (sorted by new model error):")
    print(per_car.to_string(index=False))


if __name__ == "__main__":
    main()
