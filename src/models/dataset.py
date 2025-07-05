# src/models/dataset.py

from pathlib import Path

import pandas as pd


def load_features(features_path: str) -> pd.DataFrame:
    """
    Load aggregated features per car from the Parquet feature dump.
    """
    df = pd.read_parquet(features_path)
    # ensure car_id is int
    df["car_id"] = df["car_id"].astype(int)
    # aggregate per-car statistics (mean of telemetry signals)
    feat = (
        df.groupby("car_id")
        .agg(
            {
                "speed_kmh": "mean",
                "engine_rpm": "mean",
                "gear": "mean",
                "throttle": "mean",
                "brake": "mean",
                "drs": "mean",
                "track_distance": "mean",
                "dist_to_car_ahead": "mean",
            }
        )
        .reset_index()
    )
    return feat


def load_labels(labels_root: str) -> pd.DataFrame:
    """
    Load lap-time labels for each car and lap from historical Parquet files.
    """
    paths = list(Path(labels_root).rglob("*.parquet"))
    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        df = df.rename(columns={"car_number": "car_id"})
        df = df[["car_id", "lap_number", "lap_time"]].copy()
        df["car_id"] = df["car_id"].astype(int)
        df["lap_number"] = df["lap_number"].astype(int)
        dfs.append(df)
    labels = pd.concat(dfs, ignore_index=True)
    return labels


def build_dataset(features_path: str, labels_root: str) -> pd.DataFrame:
    """
    Merge features and labels on car_id.
    (lap_number is preserved in the labels but not used for merging.)
    """
    feat = load_features(features_path)
    lab = load_labels(labels_root)
    ds = lab.merge(feat, on="car_id", how="inner")
    return ds


if __name__ == "__main__":
    FEATURES_FILE = "data/features/features_2022.parquet"
    LABELS_DIR = "data/historical"
    OUTPUT_FILE = "data/train_dataset.parquet"

    print("Loading features from", FEATURES_FILE)
    print("Loading labels from", LABELS_DIR)
    dataset = build_dataset(FEATURES_FILE, LABELS_DIR)
    print(f"Writing merged dataset ({dataset.shape[0]} rows) to {OUTPUT_FILE}")
    dataset.to_parquet(OUTPUT_FILE, index=False)
