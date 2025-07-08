import pandas as pd

from src.features.batch_etl import main as run_batch_etl


def test_batch_etl_output():
    # Run the ETL job
    run_batch_etl()

    # Load the generated Parquet
    df = pd.read_parquet("data/historical/telemetry.parquet")

    # Assert required columns exist
    assert "car_id" in df.columns
    assert "tire_temp_avg" in df.columns
    assert "tire_temp_std" in df.columns
    assert "event_time" in df.columns

    # Check for at least one row of data
    assert len(df) > 0

    print("test_batch_etl_output passed.")
