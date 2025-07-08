# src/models/train_final.py

from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = Path("data/train_dataset.parquet")
MODEL_PATH = Path("models/lgb_final_2022.pkl")

best_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.23839421902678992,
    "num_leaves": 190,
    "min_data_in_leaf": 47,
    "feature_fraction": 0.6804422605238496,
    "bagging_fraction": 0.6131169614110067,
    "bagging_freq": 6,
    "verbose": -1,
    "n_estimators": 1000,
}


def main():
    df = pd.read_parquet(DATA_PATH).dropna()
    X = df.drop(["car_id", "lap_number", "lap_time"], axis=1)
    y = df["lap_time"]

    model = LGBMRegressor(**best_params)
    model.fit(X, y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = mse**0.5
    print(f"Full-data RMSE: {rmse:.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
