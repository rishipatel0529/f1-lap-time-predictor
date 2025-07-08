from pathlib import Path

import pandas as pd


def main():
    # Read historical telemetry
    df = pd.read_csv("data/historical_telemetry.csv")

    # Compute features
    agg = (
        df.groupby("car_id")
        .agg(tire_temp_avg=("tire_temp", "mean"), tire_temp_std=("tire_temp", "std"))
        .reset_index()
    )

    # Add current timestamp for event_time
    agg["event_time"] = pd.Timestamp.now()

    # Ensure the output directory exists
    out_dir = Path("data") / "historical"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write features to Parquet in the test-expected location
    out_path = out_dir / "telemetry.parquet"
    agg.to_parquet(out_path, index=False)

    print(f"Parquet file written to {out_path}")
    print(agg)


# Allow running as a script
if __name__ == "__main__":
    main()
