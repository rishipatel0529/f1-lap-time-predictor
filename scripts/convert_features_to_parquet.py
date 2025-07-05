# scripts/convert_features_to_parquet.py

import pandas as pd

# 1) read the big CSV you just generated
df = pd.read_csv("data/features/features_2022.csv", parse_dates=["time"])

# 2) write it out in Parquet for faster, compressed loads
df.to_parquet("data/features/features_2022.parquet", index=False, compression="snappy")

print("Converted CSV → Parquet!")
