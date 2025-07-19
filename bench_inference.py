# bench_inference.py

import time

import joblib
import pandas as pd


def load_test_batch(path: str, batch_size: int = 1000):
    """
    Load your full dataset and return the first `batch_size` rows of features.
    """
    df = pd.read_parquet(path)
    # drop any columns you don’t feed into the model
    X = df.drop(["car_id", "lap_number", "lap_time"], axis=1)
    return X.iloc[:batch_size]


def time_it(model, X, n_runs=50):
    # a quick "warm up"
    _ = model.predict(X)
    start = time.time()
    for _ in range(n_runs):
        _ = model.predict(X)
    total = time.time() - start
    return total / n_runs  # average per run


if __name__ == "__main__":
    # 1) load a fixed test batch
    TEST_BATCH = load_test_batch("data/train_dataset.parquet", batch_size=1000)
    print(
        f"Loaded test batch: {TEST_BATCH.shape[0]} rows × {TEST_BATCH.shape[1]} feature"
    )

    # 2) load both models
    old = joblib.load("models/lgb_baseline_2022.pkl")
    new = joblib.load("models/lgb_final_2022.pkl")

    # 3) benchmark
    N = 50
    t_old = time_it(old, TEST_BATCH, n_runs=N)
    t_new = time_it(new, TEST_BATCH, n_runs=N)

    print(f"\nAverage inference time over {N} runs:")
    print(f"old baseline: {t_old*1000:.2f} ms per batch")
    print(f"new model   : {t_new*1000:.2f} ms per batch")
    print(f"new is {t_new/t_old*100:.1f}% of old speed")
