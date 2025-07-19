#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import joblib
import mlflow
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.models.dataset import build_dataset

# Ensure project root is on PYTHONPATH so `from src.models.dataset` works:
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WANDB_MODE", "disabled")


def main(season: int = 2022):
    # Define paths
    features_dir = ROOT / "data" / "features_by_gp" / str(season)
    labels_dir = ROOT / "data" / "historical"
    train_path = ROOT / "data" / "train_dataset.parquet"
    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"lgb_baseline_{season}.pkl"

    # 1) Build the train dataset (overwrites each run)
    print(f"Building train dataset for season {season}…")
    df = build_dataset(str(features_dir), str(labels_dir))
    df.to_parquet(train_path, index=False)
    print(f"Wrote {len(df)} rows to {train_path}")

    # 2) Train/validation split
    df = df.dropna()
    X = df.drop(["car_id", "lap_number", "lap_time"], axis=1)
    y = df["lap_time"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 3) Train under MLflow
    mlflow.set_experiment("f1_baseline")
    with mlflow.start_run(run_name=f"baseline_lgb_{season}"):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "n_estimators": 100,
            "verbose": -1,
        }
        mlflow.log_params(params)

        model = LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[early_stopping(stopping_rounds=10), log_evaluation(period=10)],
        )

        # 4) Evaluate
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        print(f"Validation RMSE: {rmse:.3f}")
        mlflow.log_metric("rmse", rmse)

        # 5) Save & log
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    main(season)
