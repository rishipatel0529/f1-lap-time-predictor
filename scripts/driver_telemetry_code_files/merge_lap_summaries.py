"""
scripts/driver_telemetry_code_files/merge_lap_summaries.py

Merges historical lap summaries from both CSV and Parquet sources into a single flat file.
Recursively scans the driver telemetry directory, concatenates all pieces, and writes
the combined dataset to data/historical_telemetry.csv for downstream ingestion.
Designed as a lightweight batch ETL step you can run repeatedly.
"""

from pathlib import Path
import pandas as pd

def run_batch_etl() -> pd.DataFrame:
    """
    1) Read all historical lap summaries from CSV or Parquet
    2) concatenate them,
    3) return the DataFrame.
    """

    out_csv = Path("data/historical_telemetry.csv")
    raw_csv_dir = Path("data/driver_telemetry_csv_files")
    raw_parquet_dir = Path("data/driver_telemetry_csv_files")

    pieces = []
    # load legacy CSV exports (if any) from the telemetry folder tree
    if raw_csv_dir.exists():
        for fp in sorted(raw_csv_dir.glob("**/*.csv")):
            pieces.append(pd.read_csv(fp))
    # load FastF1-derived Parquet files (more efficient column types)
    if raw_parquet_dir.exists():
        for fp in sorted(raw_parquet_dir.glob("**/*.parquet")):
            pieces.append(pd.read_parquet(fp))

    # Fail fast if nothing was found to help the operator fix paths or generate inputs
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
