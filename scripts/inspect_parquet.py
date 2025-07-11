#!/usr/bin/env python3
import sys

import pandas as pd


def inspect(path: str, n: int = 5, to_csv: bool = True):
    print(f"Loading {path!r}…")
    df = pd.read_parquet(path)
    print("\nColumns:")
    print(df.columns.tolist())
    print(f"\nShowing first {n} rows:")
    print(df.head(n))
    print(f"\nDataFrame shape: {df.shape}\n")
    if to_csv:
        out = path.replace(".parquet", ".csv")
        print(f"Writing first {n} rows to {out!r}…")
        df.head(n).to_csv(out, index=False)
        print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <path> [n_rows]")
        sys.exit(1)
    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    inspect(path, n_rows := n)
