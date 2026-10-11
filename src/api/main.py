"""FastAPI inference service for delivery ETA correction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.feature_config import BASELINE_ETA, FEATURES
from src.monitoring.prediction_logger import log_prediction


MODEL_PATH = Path("models/eta_correction_xgb.joblib")

app = FastAPI(
    title="Delivery ETA Correction API",
    version="1.0.0",
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
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run `python -m src.models.train` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict(request: ETARequest):
    try:
        data = request.model_dump()
        X = pd.DataFrame([data])[FEATURES]

        predicted_error = float(get_model().predict(X)[0])
        estimated_eta = float(data[BASELINE_ETA])
        final_eta = max(1.0, estimated_eta + predicted_error)

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
            "model": "xgboost_eta_correction",
        }

    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
