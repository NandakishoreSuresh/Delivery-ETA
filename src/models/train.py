
"""Train and track an XGBoost ETA-correction model with MLflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.features.feature_config import (
    BASELINE_ETA,
    CATEGORICAL_FEATURES,
    FEATURES,
    TARGET,
    NUMERIC_FEATURES,
)


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/validation.csv")
MODEL_DIR = Path("models")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "delivery_eta_correction"
REGISTERED_MODEL_NAME = "delivery_eta_correction"


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""
    h = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_params() -> dict:
    """Load model parameters from params.yaml."""
    with Path("params.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["model"]


def build_pipeline(params: dict) -> Pipeline:
    """Create the preprocessing and XGBoost pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=int(params["random_state"]),
        n_jobs=1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    """Train, evaluate, log, register, and save the model."""

    # Check that the required input files exist.
    for path in (TRAIN_PATH, VAL_PATH, Path("params.yaml")):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    params = load_params()

    # Verify required columns.
    required_train_columns = set(FEATURES + [TARGET])
    required_val_columns = set(
        FEATURES + [TARGET, BASELINE_ETA, "actual_delivery_time_minutes"]
    )

    missing_train = required_train_columns - set(train.columns)
    missing_val = required_val_columns - set(val.columns)

    if missing_train:
        raise ValueError(
            f"Missing columns in training data: {sorted(missing_train)}"
        )

    if missing_val:
        raise ValueError(
            f"Missing columns in validation data: {sorted(missing_val)}"
        )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = val[FEATURES]
    y_val = val[TARGET]

    pipeline = build_pipeline(params)

    # Configure MLflow experiment tracking.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "eta_correction_xgb.joblib"

    with mlflow.start_run(run_name=params["name"]) as run:
        # Train the model.
        pipeline.fit(X_train, y_train)

        # Evaluate predicted ETA corrections.
        predicted_error = pipeline.predict(X_val)

        estimated_eta = val[BASELINE_ETA].to_numpy(dtype=float)
        actual_eta = val["actual_delivery_time_minutes"].to_numpy(
            dtype=float
        )

        predicted_eta = np.maximum(
            1.0,
            estimated_eta + predicted_error,
        )

        ml_mae = mean_absolute_error(actual_eta, predicted_eta)
        ml_rmse = np.sqrt(
            mean_squared_error(actual_eta, predicted_eta)
        )

        baseline_mae = mean_absolute_error(actual_eta, estimated_eta)
        baseline_rmse = np.sqrt(
            mean_squared_error(actual_eta, estimated_eta)
        )

        improvement = (
            (baseline_mae - ml_mae) / baseline_mae * 100
            if baseline_mae > 0
            else 0.0
        )

        # Log parameters, dataset information, and metrics.
        mlflow.log_params(params)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_param("training_rows", len(train))
        mlflow.log_param("validation_rows", len(val))
        mlflow.log_param(
            "train_dataset_sha256",
            file_sha256(TRAIN_PATH),
        )

        mlflow.log_metrics(
            {
                "validation_ml_mae": float(ml_mae),
                "validation_ml_rmse": float(ml_rmse),
                "validation_baseline_mae": float(baseline_mae),
                "validation_baseline_rmse": float(baseline_rmse),
                "validation_improvement_percent": float(improvement),
            }
        )

        # Log and register the model in MLflow.
        signature = infer_signature(X_val.head(5), predicted_error[:5])

        mlflow.sklearn.log_model(
            pipeline,
            name="eta_correction_model",
            signature=signature,
            serialization_format="cloudpickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Save a local copy for backup and other workflows.
        joblib.dump(pipeline, model_path)

        # Save model metadata.
        metadata = {
            "model_name": params["name"],
            "registered_model_name": REGISTERED_MODEL_NAME,
            "target": TARGET,
            "baseline_eta_column": BASELINE_ETA,
            "features": FEATURES,
            "validation_mae": float(ml_mae),
            "validation_rmse": float(ml_rmse),
            "baseline_mae": float(baseline_mae),
            "baseline_rmse": float(baseline_rmse),
            "improvement_percent": float(improvement),
            "mlflow_run_id": run.info.run_id,
        }

        metadata_path = MODEL_DIR / "metadata.json"

        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print("\n=== MODEL TRAINING COMPLETE ===")
        print(f"Registered model:       {REGISTERED_MODEL_NAME}")
        print(f"MLflow run ID:          {run.info.run_id}")
        print(f"Validation ML MAE:      {ml_mae:.4f}")
        print(f"Validation baseline MAE:{baseline_mae: .4f}")
        print(f"Improvement:            {improvement:.2f}%")
        print(f"Saved local model:      {model_path}")
        print(f"Saved metadata:         {metadata_path}")


if __name__ == "__main__":
    main()
