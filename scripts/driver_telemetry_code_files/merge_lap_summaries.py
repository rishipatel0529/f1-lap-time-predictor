#!/usr/bin/env python3
# src/features/batch_etl.py

from pathlib import Path

import pandas as pd


def run_batch_etl() -> pd.DataFrame:
    """
    1) Read all historical lap summaries from CSV or Parquet
    2) concatenate them,
    3) return the DataFrame.
    """
    out_csv = Path("data/driver_telemetry_csv_files/historical_telemetry.csv")
    raw_csv_dir = Path("data/driver_telemetry_csv_files")
    raw_parquet_dir = Path("data/driver_telemetry_csv_files")

    pieces = []
    # legacy CSVs
    if raw_csv_dir.exists():
        for fp in sorted(raw_csv_dir.glob("**/*.csv")):
            pieces.append(pd.read_csv(fp))
    # fastf1 Parquets
    if raw_parquet_dir.exists():
        for fp in sorted(raw_parquet_dir.glob("**/*.parquet")):
            pieces.append(pd.read_parquet(fp))

    if not pieces:
        raise FileNotFoundError(
            f"No historical telemetry files found in "
            f"{raw_csv_dir} or {raw_parquet_dir}"
        )

    # merge
    df = pd.concat(pieces, ignore_index=True)

    # write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return df


def main():
    run_batch_etl()


if __name__ == "__main__":
    main()
