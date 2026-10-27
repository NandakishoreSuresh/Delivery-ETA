"""FastAPI inference service for delivery ETA correction using MLflow."""

from __future__ import annotations

from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_config import BASELINE_ETA, FEATURES
from src.monitoring.prediction_logger import log_prediction
import os
from pathlib import Path

import joblib

# Load the model version approved in the MLflow Model Registry.
MODEL_URI = "models:/delivery_eta_correction@champion"

MODEL_LOAD_MODE = os.getenv("MODEL_LOAD_MODE", "registry").lower()
LOCAL_MODEL_PATH = Path(
    os.getenv("LOCAL_MODEL_PATH", "models/eta_correction_xgb.joblib")
)

# Must match the SQLite tracking database used during training.
mlflow.set_tracking_uri("sqlite:///mlflow.db")


app = FastAPI(
    title="Delivery ETA Correction API",
    description=(
        "Predicts corrections to food delivery ETAs using a registered "
        "XGBoost model managed by MLflow."
    ),
    version="2.0.0",
)


class ETARequest(BaseModel):
    order_hour: int = Field(ge=0, le=23)
    day_of_week: str
    is_weekend: int = Field(ge=0, le=1)
    city: str
    delivery_area: str
    customer_age: int = Field(ge=13, le=100)
    customer_type: str
    restaurant_type: str
    restaurant_primary_category: str
    restaurant_rating: float = Field(ge=0, le=5)
    items_count: int = Field(gt=0)
    subtotal: float = Field(ge=0)
    discount_percent: float = Field(ge=0)
    tax_amount: float = Field(ge=0)
    service_fee: float = Field(ge=0)
    delivery_fee: float = Field(ge=0)
    order_total: float = Field(ge=0)
    payment_method: str
    distance_km: float = Field(gt=0)
    weather: str
    traffic_level: Literal["Low", "Medium", "High"]
    delivery_partner_experience_months: int = Field(ge=0)
    delivery_partner_rating: float = Field(ge=0, le=5)
    estimated_delivery_time_minutes: float = Field(gt=0)


_model = None




def get_model():
    global _model

    if _model is None:
        try:
            if MODEL_LOAD_MODE == "local":
                if not LOCAL_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Local model not found: {LOCAL_MODEL_PATH}"
                    )

                _model = joblib.load(LOCAL_MODEL_PATH)
                print(f"Loaded local model: {LOCAL_MODEL_PATH}")

            elif MODEL_LOAD_MODE == "registry":
                _model = mlflow.sklearn.load_model(MODEL_URI)
                print(f"Loaded registered model: {MODEL_URI}")

            else:
                raise ValueError(
                    f"Unsupported MODEL_LOAD_MODE: {MODEL_LOAD_MODE}"
                )

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model using mode '{MODEL_LOAD_MODE}'"
            ) from exc

    return _model


@app.get("/")
def root():
    return {
        "message": "Delivery ETA Correction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    try:
        get_model()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "status": "ok",
        "model_available": True,
        "model_uri": (
            MODEL_URI
            if MODEL_LOAD_MODE == "registry"
            else str(LOCAL_MODEL_PATH)
        ),
    }

@app.post("/predict")
def predict(request: ETARequest):
    try:
        data = request.model_dump()

        # Preserve the exact feature order used during training.
        X = pd.DataFrame([data])[FEATURES]

        model = get_model()
        predicted_error = float(model.predict(X)[0])

        estimated_eta = float(data[BASELINE_ETA])
        final_eta = max(1.0, estimated_eta + predicted_error)

        # Log the request and prediction for monitoring.
        log_prediction(
            request_features=data,
            predicted_error=predicted_error,
            estimated_eta=estimated_eta,
            final_eta=final_eta,
        )

        return {
            "estimated_eta_minutes": round(estimated_eta, 2),
            "predicted_correction_minutes": round(predicted_error, 2),
            "final_eta_minutes": round(final_eta, 2),
            "model": (
                MODEL_URI
                if MODEL_LOAD_MODE == "registry"
                else str(LOCAL_MODEL_PATH)
            ),
        }

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc