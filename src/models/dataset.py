import glob
from pathlib import Path

import pandas as pd


def load_gp_features(features_dir: str) -> pd.DataFrame:
    """
    Read all per-GP Parquet slices under features_dir,
    concatenate them, and return the raw telemetry DataFrame.
    """
    pattern = f"{features_dir}/*_*.parquet"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No per-GP files found with pattern: {pattern}")

    # load & concat
    df = pd.concat((pd.read_parquet(fp) for fp in files), ignore_index=True)

    # rename the driver column (if present) to car_id
    if "driver" in df.columns:
        df = df.rename(columns={"driver": "car_id"})

    # ensure car_id is int
    df["car_id"] = df["car_id"].astype(int)

    return df


def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the full per-GP telemetry DataFrame, aggregate per-car means.
    """
    return (
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


def load_labels(labels_root: str) -> pd.DataFrame:
    """
    Load and concatenate historical lap-time label Parquet files under labels_root,
    returning a DataFrame with (car_id, lap_number, lap_time).
    """
    paths = list(Path(labels_root).rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No label .parquet found in {labels_root}")

    dfs = []
    for p in paths:
        df = pd.read_parquet(p)
        # rename car_number → car_id
        if "car_number" in df.columns:
            df = df.rename(columns={"car_number": "car_id"})
        if all(col in df.columns for col in ("car_id", "lap_number", "lap_time")):
            tmp = df[["car_id", "lap_number", "lap_time"]].copy()
            tmp["car_id"] = tmp["car_id"].astype(int)
            tmp["lap_number"] = tmp["lap_number"].astype(int)
            dfs.append(tmp)

    if not dfs:
        raise ValueError(f"No valid label rows in {labels_root}")

    return pd.concat(dfs, ignore_index=True)


def build_dataset(features_dir: str, labels_root: str) -> pd.DataFrame:
    """
    Build the train dataset by:
      1) loading all per-GP telemetry in features_dir
      2) aggregating per-car means
      3) loading labels from labels_root
      4) merging on car_id
    """
    df_tel = load_gp_features(features_dir)
    feat = aggregate_features(df_tel)
    lab = load_labels(labels_root)

    ds = lab.merge(feat, on="car_id", how="inner")
    if ds.empty:
        raise RuntimeError("Merged dataset is empty—check feature vs label keys")
    return ds


if __name__ == "__main__":
    import sys

    # Usage: python src/models/dataset.py <features_dir> <labels_dir>
    if len(sys.argv) != 3:
        print("Usage: python dataset.py <features_dir> <labels_dir>")
        sys.exit(1)

    features_dir = sys.argv[1]
    labels_dir = sys.argv[2]
    out_file = "data/train_dataset.parquet"

    print(f"Building train dataset from {features_dir} + {labels_dir} …")
    dataset = build_dataset(features_dir, labels_dir)
    print(f" → {len(dataset)} rows, {dataset.shape[1]} cols")

    dataset.to_parquet(out_file, index=False)
    print(f"Saved merged dataset to {out_file}")
