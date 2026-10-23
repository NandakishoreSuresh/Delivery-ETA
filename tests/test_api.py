
"""Automated tests for the Delivery ETA Correction API."""

import numpy as np
from fastapi.testclient import TestClient

import src.api.main as api


client = TestClient(api.app)


def sample_payload():
    """Return a realistic sample order for API testing."""
    return {
        "order_hour": 15,
        "day_of_week": "Sunday",
        "is_weekend": 1,
        "city": "City_C",
        "delivery_area": "City_C_Area_3",
        "customer_age": 40,
        "customer_type": "Premium",
        "restaurant_type": "Casual Dining",
        "restaurant_primary_category": "Healthy",
        "restaurant_rating": 4.4,
        "items_count": 4,
        "subtotal": 64.82,
        "discount_percent": 0,
        "tax_amount": 9.07,
        "service_fee": 2.4,
        "delivery_fee": 0,
        "order_total": 76.29,
        "payment_method": "Credit Card",
        "distance_km": 1.1,
        "weather": "Rain",
        "traffic_level": "Medium",
        "delivery_partner_experience_months": 14,
        "delivery_partner_rating": 4.4,
        "estimated_delivery_time_minutes": 25,
    }


def test_health_endpoint(monkeypatch):
    """Health check succeeds when a registered model can be loaded."""
    monkeypatch.setattr(api, "get_model", lambda: object())

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_available"] is True
    assert data["model_uri"] == (
        "models:/delivery_eta_correction@champion"
    )


def test_predict_returns_corrected_eta_and_logs(monkeypatch):
    """Prediction returns the expected result and logs the request."""

    class FakeModel:
        def predict(self, X):
            assert X.shape[0] == 1
            return np.array([3.27])

    captured = {}

    def fake_log_prediction(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    monkeypatch.setattr(api, "log_prediction", fake_log_prediction)

    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 200

    data = response.json()
    assert data["estimated_eta_minutes"] == 25.0
    assert data["predicted_correction_minutes"] == 3.27
    assert data["final_eta_minutes"] == 28.27
    assert data["model"] == (
        "models:/delivery_eta_correction@champion"
    )

    assert captured["estimated_eta"] == 25.0
    assert captured["final_eta"] == 28.27


def test_predict_rejects_invalid_traffic_level():
    """Invalid traffic categories should fail request validation."""
    payload = sample_payload()
    payload["traffic_level"] = "Severe"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_order_hour():
    """An order hour outside 0–23 should fail validation."""
    payload = sample_payload()
    payload["order_hour"] = 25

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
