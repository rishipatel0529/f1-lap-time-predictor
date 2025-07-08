#!/usr/bin/env python3
"""
Split the main features_2022.parquet into one file per Grand Prix.
"""
import re
from pathlib import Path

import pandas as pd


def sanitize(name: str) -> str:
    # lowercase, replace spaces & punctuation with underscore
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def main():
    # Path to the full-season features
    full_file = Path("data/features/features_2022.parquet")
    if not full_file.exists():
        raise FileNotFoundError(f"Could not find {full_file}")

    # Read once into memory
    print(f"Reading full features: {full_file}")
    df = pd.read_parquet(full_file)

    # Output directory for per-race files
    out_dir = Path("data/features_by_gp/2022")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group and write
    for gp, gp_df in df.groupby("grand_prix"):
        safe = sanitize(gp)
        out_path = out_dir / f"{safe}_2022.parquet"
        gp_df.to_parquet(out_path, index=False, compression="snappy")
        print(f"Wrote {len(gp_df)} rows to {out_path}")

    print("All done! Split into per-GP files in data/features_by_gp/2022.")


if __name__ == "__main__":
    main()
