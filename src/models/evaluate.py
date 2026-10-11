
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.feature_config import FEATURES, BASELINE_ETA

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/eta_correction_xgb.joblib")


def report(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{name:<28} MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return mae, rmse


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run python -m src.models.train first."
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    actual = test["actual_delivery_time_minutes"].to_numpy()
    existing_eta = test[BASELINE_ETA].to_numpy()

    # Baseline 1: the original ETA
    baseline_mae, _ = report(
        "Existing ETA",
        actual,
        existing_eta,
    )

    # Baseline 2: a fixed correction learned only from training data
    mean_error = train["eta_error"].mean()
    fixed_eta = np.maximum(1, existing_eta + mean_error)

    fixed_mae, _ = report(
        "Fixed mean correction",
        actual,
        fixed_eta,
    )

    # Model: predict an individual correction for each order
    model = joblib.load(MODEL_PATH)
    predicted_error = model.predict(test[FEATURES])
    ml_eta = np.maximum(1, existing_eta + predicted_error)

    ml_mae, ml_rmse = report(
        "XGBoost correction",
        actual,
        ml_eta,
    )

    improvement = (baseline_mae - ml_mae) / baseline_mae * 100

    print(f"\nImprovement vs existing ETA: {improvement:.2f}%")
    print("Model beats existing ETA:", ml_mae < baseline_mae)
    print("Model beats fixed correction:", ml_mae < fixed_mae)


if __name__ == "__main__":
    main()
