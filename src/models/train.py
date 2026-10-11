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
MLRUNS_DIR = Path("mlruns")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_params() -> dict:
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["model"]


def build_pipeline(params: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
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
        n_jobs=-1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)

    params = load_params()

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_val = val[FEATURES]
    y_val = val[TARGET]

    pipeline = build_pipeline(params)

    # Use SQLite for MLflow experiment tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("delivery_eta_correction")

    with mlflow.start_run(run_name=params["name"]):
        pipeline.fit(X_train, y_train)

        predicted_error = pipeline.predict(X_val)
        predicted_eta = val[BASELINE_ETA].to_numpy() + predicted_error
        actual_eta = val["actual_delivery_time_minutes"].to_numpy()

        ml_mae = mean_absolute_error(actual_eta, predicted_eta)
        ml_rmse = np.sqrt(mean_squared_error(actual_eta, predicted_eta))

        baseline_mae = mean_absolute_error(
            actual_eta,
            val[BASELINE_ETA],
        )
        baseline_rmse = np.sqrt(
            mean_squared_error(
                actual_eta,
                val[BASELINE_ETA],
            )
        )

        improvement = (baseline_mae - ml_mae) / baseline_mae * 100

        mlflow.log_params(params)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("feature_count", len(FEATURES))
        mlflow.log_param("training_rows", len(train))
        mlflow.log_param("validation_rows", len(val))
        mlflow.log_param("train_dataset_sha256", file_sha256(TRAIN_PATH))

        mlflow.log_metrics(
            {
                "validation_ml_mae": ml_mae,
                "validation_ml_rmse": ml_rmse,
                "validation_baseline_mae": baseline_mae,
                "validation_baseline_rmse": baseline_rmse,
                "validation_improvement_percent": improvement,
            }
        )

        signature = infer_signature(X_val.head(5), predicted_error[:5])
        mlflow.sklearn.log_model(
            pipeline,
            name="eta_correction_model",
            signature=signature,
            serialization_format="cloudpickle",
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "eta_correction_xgb.joblib"
        joblib.dump(pipeline, model_path)

        metadata = {
            "model_name": params["name"],
            "target": TARGET,
            "baseline_eta_column": BASELINE_ETA,
            "features": FEATURES,
            "validation_mae": float(ml_mae),
            "validation_rmse": float(ml_rmse),
            "baseline_mae": float(baseline_mae),
            "baseline_rmse": float(baseline_rmse),
            "improvement_percent": float(improvement),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }
        with (MODEL_DIR / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Validation ML MAE:       {ml_mae:.4f}")
        print(f"Validation baseline MAE: {baseline_mae:.4f}")
        print(f"Improvement:              {improvement:.2f}%")
        print(f"Saved model:              {model_path}")


if __name__ == "__main__":
    main()
