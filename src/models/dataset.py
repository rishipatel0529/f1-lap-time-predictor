# src/models/dataset.py

from pathlib import Path

import pandas as pd


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # ensure car_id is int
    df["car_id"] = pd.to_numeric(df["car_id"], errors="raise").astype(int)

    feat = (
        df.groupby("car_id")
        .agg(
            {"speed": "mean", "rpm": "mean", "tire_temp": "mean", "fuel_level": "mean"}
        )
        .reset_index()
    )
    return feat


def load_labels(root: str) -> pd.DataFrame:
    files = list(Path(root).rglob("*.parquet"))
    dfs = []
    for f in files:
        tmp = pd.read_parquet(f)
        # rename and keep only the columns we need
        tmp = tmp.rename(columns={"car_number": "car_id"})
        tmp = tmp[["car_id", "lap_number", "lap_time"]]
        # cast car_id and lap_number
        tmp["car_id"] = tmp["car_id"].astype(int)
        tmp["lap_number"] = pd.to_numeric(tmp["lap_number"], errors="raise").astype(int)
        dfs.append(tmp)
    return pd.concat(dfs, ignore_index=True)


def build_dataset(feat: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    # both car_id columns are now int, so merge will work
    ds = lab.merge(feat, on="car_id", how="inner")
    return ds


if __name__ == "__main__":
    feat = load_features("data/historical_telemetry.csv")
    lab = load_labels("data/historical")
    ds = build_dataset(feat, lab)
    ds.to_parquet("data/train_dataset.parquet", index=False)
    print("Train dataset:", ds.shape)
    print(ds.head())
