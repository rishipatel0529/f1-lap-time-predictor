#!/usr/bin/env python3
"""
Optuna-based hyperparameter search for our LightGBM lap-time regressor.
Each trial is logged as a nested MLflow run under the main experiment.
"""

import math
import os
from pathlib import Path

import mlflow
import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Disable W&B since we’re not using it here
os.environ.setdefault("WANDB_MODE", "disabled")

DATA_PATH = Path("data/train_dataset.parquet")
EXPERIMENT_NAME = "f1_lap_time_tuning"
N_TRIALS = 50


def objective(trial):
    # 1) Sample hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }

    # 2) Load & split data
    df = pd.read_parquet(DATA_PATH).dropna()
    X = df.drop(["car_id", "lap_number", "lap_time"], axis=1)
    y = df["lap_time"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Start a nested MLflow run for this trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)

        # 4) Train
        model = LGBMRegressor(**params, n_estimators=1000)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                early_stopping(stopping_rounds=20),
                log_evaluation(period=50),
            ],
        )

        # 5) Evaluate
        preds = model.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, preds))
        mlflow.log_metric("rmse", rmse)

        return rmse


def main():
    # 1) set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 2) start one outer run to group all trials
    with mlflow.start_run(run_name="optuna_lgb_search"):
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS)

        print("Best trial:", study.best_trial.number)
        print("Params:", study.best_trial.params)
        print("Best RMSE:", study.best_trial.value)


if __name__ == "__main__":
    main()
