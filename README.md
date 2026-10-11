# Delivery ETA Correction & MLOps Platform

A production-oriented machine-learning system that learns to correct an existing
delivery ETA instead of blindly predicting delivery time from scratch.

## Problem

The dataset contains 50,000 food-delivery orders. We use completed orders with
a known `actual_delivery_time_minutes`.

The production baseline is:

    estimated_delivery_time_minutes

Our ML target is:

    eta_error = actual_delivery_time_minutes - estimated_delivery_time_minutes

The final prediction is:

    final_eta = estimated_delivery_time_minutes + predicted_eta_error

## Current benchmark

On the time-ordered test split created during project development:

- Existing ETA: MAE about 6.06 minutes
- Mean-error correction: MAE about 4.62 minutes

The ML model must be evaluated against these baselines.

## Project structure

```text
delivery-eta-mlops/
├── data/
│   ├── raw/
│   │   └── food_delivery_orders_dataset.csv
│   ├── processed/
│   └── logs/
├── notebooks/
│   └── 01_eda_and_baseline.ipynb
├── src/
│   ├── validation/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── api/
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md
```

## Setup in VS Code

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Validate the raw dataset

```bash
python -m src.validation.validate_data data/raw/food_delivery_orders_dataset.csv
```

### 4. Create the time-based train/validation/test datasets

```bash
python -m src.data.prepare
```

### 5. Train the XGBoost ETA-correction model

```bash
python -m src.models.train
```

### 6. Evaluate once on the untouched test split

```bash
python -m src.models.evaluate
```

### 7. Open MLflow

```bash
mlflow ui --port 5000
```

Then open:

```text
http://127.0.0.1:5000
```

### 8. Start the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
```

## MLOps roadmap

This repository starts with the core reproducible pipeline:

1. Data validation
2. Time-aware data preparation
3. Feature preprocessing
4. Model training
5. MLflow experiment tracking
6. Hold-out evaluation
7. FastAPI inference
8. Prediction logging
9. CI tests

Later phases can add DVC remote storage, Feast, a production MLflow registry,
Airflow orchestration, Evidently drift reports, Prometheus/Grafana,
automatic retraining, canary deployment and rollback.
