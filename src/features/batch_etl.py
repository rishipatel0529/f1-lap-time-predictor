# src/features/batch_etl.py

from pathlib import Path

import pandas as pd


def run_batch_etl() -> pd.DataFrame:
    """
    1) Read all historical lap summaries from CSV or Parquet
       (wherever you've dumped them),
    2) concatenate them,
    3) write out data/historical_telemetry.csv,
    4) return the DataFrame.
    """
    out_csv = Path("data/historical_telemetry.csv")

    # legacy folder (pre-raw/)
    legacy_csv_dir = Path("data/historical")
    # new CSVs you dumped
    raw_csv_dir = Path("data/raw/historical")
    # fastf1‐dumped Parquets
    raw_parquet_dir = Path("data/raw/historical_fastf1")

    pieces = []

    # 1a) legacy CSVs
    if legacy_csv_dir.exists():
        for fp in sorted(legacy_csv_dir.glob("**/*.csv")):
            pieces.append(pd.read_csv(fp))

    # 1b) new CSVs
    if raw_csv_dir.exists():
        for fp in sorted(raw_csv_dir.glob("**/*.csv")):
            pieces.append(pd.read_csv(fp))

    # 1c) fastf1 Parquets
    if raw_parquet_dir.exists():
        for fp in sorted(raw_parquet_dir.glob("**/*.parquet")):
            pieces.append(pd.read_parquet(fp))

    if not pieces:
        raise FileNotFoundError(
            f"No historical telemetry files found in "
            f"{legacy_csv_dir}, {raw_csv_dir}, or {raw_parquet_dir}"
        )

    # 2) concatenate
    df = pd.concat(pieces, ignore_index=True)

    # 3) write out the canonical CSV for downstream consumers/tests
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # 4) return it
    return df
