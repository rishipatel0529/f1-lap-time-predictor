# scripts/benchmark_hyperparams.py

import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error

from src.models.data_loader import load_raw_df  # <- use the preprocessed loader


def load_data():
    df = load_raw_df()
    # train/test split on full years
    train_df = df[df["season"] <= 2024]
    test_df = df[df["season"] == 2025]
    X_train = train_df.drop(
        columns=["lap_time", "season", "grand_prix"], errors="ignore"
    )
    y_train = train_df["lap_time"]
    X_hold = test_df.drop(columns=["lap_time", "season", "grand_prix"], errors="ignore")
    y_hold = test_df["lap_time"]
    # ensure numeric
    X_train = X_train.select_dtypes(include=[np.number, "bool_"]).fillna(0)
    X_hold = X_hold.select_dtypes(include=[np.number, "bool_"]).fillna(0)
    return X_train, X_hold, y_train, y_hold


def train_eval(params, run_name):
    X_train, X_hold, y_train, y_hold = load_data()
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=500)
        preds = model.predict(X_hold)
        rmse = np.sqrt(mean_squared_error(y_hold, preds))
        mlflow.log_metric("rmse", rmse)
        return rmse


if __name__ == "__main__":
    # 1) Baseline
    baseline = {"objective": "regression", "metric": "rmse"}
    base_rmse = train_eval(baseline, "baseline")

    # 2) Tuned: plug in your best from the Optuna study
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 83,
        "max_depth": 8,
        "learning_rate": 0.005,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 3,
    }
    tuned_rmse = train_eval(best_params, "tuned")

    print(f"Baseline RMSE: {base_rmse:.3f}")
    print(f"Tuned    RMSE: {tuned_rmse:.3f}")
