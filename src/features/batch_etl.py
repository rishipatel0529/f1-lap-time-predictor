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

    agg["event_time"] = pd.Timestamp.now()

    agg.to_parquet("feature_repo/data/telemetry.parquet", index=False)

    print("Parquet file created at feature_repo/data/telemetry.parquet")
    print(agg)


# Allow running as a script
if __name__ == "__main__":
    main()
