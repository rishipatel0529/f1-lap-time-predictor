# scripts/check_parquet_latency.py
import time

import numpy as np
import pandas as pd

path = "data/features_by_gp/2022/bahrain_grand_prix_2022.parquet"
times = []
for _ in range(50):
    t0 = time.time()
    # read just one column to measure overhead
    _ = pd.read_parquet(path, columns=["car_id"])
    times.append(time.time() - t0)

p95_ms = np.percentile(times, 95) * 1000
print(f"P95 parquet read latency: {p95_ms:.1f} ms")
