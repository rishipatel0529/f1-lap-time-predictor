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
    legacy_csv_dir = Path("data/historical")
    raw_csv_dir = Path("data/raw/historical")
    raw_parquet_dir = Path("data/raw/historical_fastf1")

    pieces = []
    # legacy CSVs
    if legacy_csv_dir.exists():
        for fp in sorted(legacy_csv_dir.glob("**/*.csv")):
            pieces.append(pd.read_csv(fp))
    # new CSVs
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
            f"{legacy_csv_dir}, {raw_csv_dir}, or {raw_parquet_dir}"
        )

    # concatenate
    df = pd.concat(pieces, ignore_index=True)

    # write canonical CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return df


# expose main for the test:
def main() -> pd.DataFrame:
    return run_batch_etl()


if __name__ == "__main__":
    # if you want to run it as a script:
    run_batch_etl()
