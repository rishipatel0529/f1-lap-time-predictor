# scripts/evaluate_on_2025.py

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

from src.models.data_loader import load_raw_df


def main():
    # 1) Load the full processed DataFrame
    df = load_raw_df()

    # 2) Split into train (2019-2023) and test (2024)
    train_df = df[df["season"] <= 2023]
    test_df = df[df["season"] == 2024]

    # 3) Separate X/y
    y_train = train_df["lap_time"]
    X_train = train_df.drop(
        columns=["lap_time", "season", "grand_prix"], errors="ignore"
    )

    y_test = test_df["lap_time"]
    X_test = test_df.drop(columns=["lap_time", "season", "grand_prix"], errors="ignore")

    # Ensure numeric only
    X_train = X_train.select_dtypes(include=[np.number, "bool_"]).fillna(0)
    X_test = X_test.select_dtypes(include=[np.number, "bool_"]).fillna(0)

    # 4) Load your best hyperparams (hardcode or from MLflow)
    best_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 169,
        "max_depth": 12,
        "learning_rate": 0.005,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.7,
        "bagging_freq": 4,
    }

    # 5) Train final model on all 2019–2024
    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(best_params, dtrain, num_boost_round=500)

    # Optionally, save your model
    joblib.dump(model, "models/final_lgbm_2019_2024.pkl")
    print("Saved final model to models/final_lgbm_2019_2024.pkl")

    # 6) Predict & evaluate on 2025
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Hold‑out 2024 RMSE: {rmse:.3f} seconds")


if __name__ == "__main__":
    main()
