# Delivery ETA Correction & MLOps Platform

An end-to-end machine learning and MLOps project that improves food delivery time estimates using XGBoost regression. The platform includes data validation, reproducible data pipelines, MLflow model registration, FastAPI inference, prediction logging, PSI-based drift monitoring, Docker containerization, and automated CI/CD.

## Project Overview

Existing delivery ETAs can be inaccurate because of delivery distance, traffic, weather, order characteristics, and delivery-partner factors. This project predicts the error in an existing ETA and uses the predicted correction to produce an adjusted estimate.

**Correction formula**

`Corrected ETA = max(1, Existing ETA + Predicted ETA Error)`

The model predicts the difference between actual delivery time and the original estimated delivery time.

## Model Performance

Evaluated using a chronological train/validation/test split on a food delivery dataset.

| Metric | Existing ETA | XGBoost correction |
|---|---:|---:|
| Test MAE | 6.0593 min | 3.4462 min |
| Test RMSE | 8.2291 min | 4.2827 min |

The correction model achieved a **43.13% reduction in test MAE** compared with the original ETA estimates.

These results are from held-out test data. Performance on future real-world deliveries may differ.

## Tech Stack

- **Machine Learning:** Python, XGBoost, scikit-learn, pandas, NumPy
- **API:** FastAPI, Pydantic, Uvicorn
- **Experiment Tracking and Model Registry:** MLflow
- **Data and Pipeline Versioning:** DVC, DagsHub
- **Monitoring:** JSONL prediction logging, Population Stability Index (PSI)
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Testing:** pytest, FastAPI TestClient

## Architecture

1. **Data validation:** Checks the input dataset before processing.
2. **Data preparation:** Creates chronological training, validation, and test splits.
3. **Model training:** Trains an XGBoost regression pipeline to predict ETA errors.
4. **Experiment tracking:** Records parameters, metrics, and model artifacts in MLflow.
5. **Model registry:** Registers the trained model and uses the `champion` alias for the approved version.
6. **Model serving:** Exposes predictions through a FastAPI REST API.
7. **Prediction logging:** Records input features and prediction results in JSONL format.
8. **Drift monitoring:** Compares numeric feature distributions against reference training data using PSI.
9. **Deployment automation:** Builds and publishes a Docker image through GitHub Actions.

## API Usage

The API exposes the following endpoints:

- `GET /` — API information
- `GET /health` — Health check and model availability
- `POST /predict` — Predict an ETA correction

### Run the Published Docker Image

```bash
docker pull ghcr.io/nandakishoresuresh/delivery-eta-mlops:latest

docker run --rm -p 8000:8000 \
  ghcr.io/nandakishoresuresh/delivery-eta-mlops:latest
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

### Example Prediction Request

```json
{
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
  "estimated_delivery_time_minutes": 25
}
```

Example response (actual predictions may differ by model version):

```json
{
  "estimated_eta_minutes": 25,
  "predicted_correction_minutes": 3.27,
  "final_eta_minutes": 28.27,
  "model": "models/eta_correction_xgb.joblib"
}
```

## Data and Model Management

- DVC tracks datasets and pipeline artifacts.
- DagsHub stores the remote DVC cache.
- MLflow records training runs and registered model versions.
- GitHub stores source code and CI/CD workflow definitions.
- GitHub Actions runs the automated tests and builds and publishes the Docker image to GitHub Container Registry.

DVC credentials, local MLflow databases, and prediction logs should remain outside version control.

## Monitoring

The application logs predictions to `data/logs/predictions.jsonl` and includes a PSI-based feature drift check.

The live-log monitor requires at least 30 logged requests before reporting numeric feature drift. Synthetic test requests validate the monitoring implementation but should not be treated as evidence of production drift or model performance.

## Testing

Run the automated test suite from the project root:

```bash
python -m pytest -v
```

The project currently has **9 passing automated tests**, covering feature exclusions, data validation, API responses, prediction logging, and invalid input handling.

## CI/CD and Container Registry

GitHub Actions runs the test job and builds the Docker image. Successful pushes to the configured `master` branch also publish the image to GitHub Container Registry.

- **Source code:** [Delivery-ETA on GitHub](https://github.com/NandakishoreSuresh/Delivery-ETA)
- **Container image:** `ghcr.io/nandakishoresuresh/delivery-eta-mlops:latest`
- **DVC remote:** [Delivery-ETA on DagsHub](https://dagshub.com/nandakishore280704/Delivery-ETA)

## Future Improvements

- Deploy the API to a public cloud hosting service.
- Add latency, error-rate, and resource-utilization monitoring.
- Evaluate model performance using newly observed actual delivery times.
- Add automated model promotion based on evaluation criteria.
- Improve drift detection with larger live samples and categorical-feature monitoring.
