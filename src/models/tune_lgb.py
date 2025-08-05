#!/usr/bin/env python3
import argparse
import os

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from src.models.data_loader import load_data


def objective(trial, data_path):
    # load X, y, and season-only groups from the specified CSV
    X, y, groups = load_data(data_path=data_path, return_groups=True)

    # mask out 2024 for CV
    mask = groups <= 2023
    X_tr, y_tr, grp_tr = X[mask], y[mask], groups[mask]

    # one fold per season
    cv = GroupKFold(n_splits=grp_tr.nunique())

    # sample hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "random_state": 42,
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }

    rmses = []
    for train_idx, val_idx in cv.split(X_tr, y_tr, grp_tr):
        X_train, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_train, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # train with early stopping
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=1_000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        preds = booster.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))

    mean_rmse = float(np.mean(rmses))

    # log this trial’s params + CV RMSE under a nested MLflow run
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", mean_rmse)
        mlflow.lightgbm.log_model(booster, artifact_path="model_fold")

    return mean_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tune LightGBM on F1 lap data")
    parser.add_argument(
        "--data-path",
        type=str,
        default="config/all_laps_with_track_and_corner_features.csv",
        help="Path to your laps+track+corner features CSV (or root dir of season CSVs)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=40, help="Number of Optuna trials to run"
    )
    args = parser.parse_args()

    # run hyperparameter search
    mlflow.set_experiment("f1_strategy_week8")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, args.data_path), n_trials=args.n_trials
    )

    print("Best CV RMSE:", study.best_value)
    print("Best params:  ", study.best_params)

    X, y, groups = load_data(data_path=args.data_path, return_groups=True)
    mask = groups <= 2023
    X_full, y_full = X[mask], y[mask]

    final = lgb.LGBMRegressor(**study.best_params, random_state=42)
    final.fit(X_full, y_full)

    os.makedirs("models/final", exist_ok=True)
    final_path = "models/final/lgb_final.txt"
    final.booster_.save_model(final_path)
    print(f"Final model saved to {final_path}")

    # log the final model on full data
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("cv_rmse", study.best_value)
        mlflow.lightgbm.log_model(final, artifact_path="model_full")
