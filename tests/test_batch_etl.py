import pandas as pd

from src.features import batch_etl


def test_batch_etl_output():
    # Run your ETL script
    batch_etl.main()

    # Load the generated parquet
    df = pd.read_parquet("feature_repo/data/telemetry.parquet")

    # Assert required columns exist
    assert "car_id" in df.columns
    assert "tire_temp_avg" in df.columns
    assert "tire_temp_std" in df.columns
    assert "event_time" in df.columns

    # Optionally check for at least one row of data
    assert len(df) > 0

    print("test_batch_etl_output passed.")
