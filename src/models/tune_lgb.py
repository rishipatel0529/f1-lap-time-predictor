# src/models/tune_lgb.py

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from src.models.data_loader import load_data


def objective(trial):
    # 1) Load data WITH groups
    X, y, groups = load_data(return_groups=True)

    # 2) Define your CV splitter
    cv = GroupKFold(n_splits=5)

    # 3) Sample hyperparameters
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "device": "gpu",
        "num_leaves": trial.suggest_int("num_leaves", 31, 128),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
    }

    rmses = []
    # 4) Loop over each fold
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 5) Train LightGBM on this fold
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=200,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(100)],
        )

        # 6) Predict & evaluate
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)

    # 7) Average RMSE across folds
    mean_rmse = float(np.mean(rmses))

    # 8) Log the average RMSE once
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", mean_rmse)
        # Optionally log the final fold’s model
        mlflow.lightgbm.log_model(model, artifact_path="model")

    return mean_rmse


if __name__ == "__main__":
    mlflow.set_experiment("lgbm_groupkf_tuning")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, timeout=3600)
    print("Best parameters:", study.best_trial.params)
