"""Validate raw delivery data before it enters the ML pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

from src.features.feature_config import FEATURES, POST_OUTCOME_COLUMNS, IDENTIFIER_COLUMNS


REQUIRED_COLUMNS = [
    "order_id",
    "customer_id",
    "restaurant_id",
    "driver_id",
    "order_timestamp",
    "order_date",
    "order_hour",
    "day_of_week",
    "is_weekend",
    "city",
    "delivery_area",
    "customer_age",
    "customer_type",
    "restaurant_type",
    "restaurant_primary_category",
    "restaurant_rating",
    "items_count",
    "subtotal",
    "discount_percent",
    "tax_amount",
    "service_fee",
    "delivery_fee",
    "order_total",
    "payment_method",
    "distance_km",
    "weather",
    "traffic_level",
    "delivery_partner_experience_months",
    "delivery_partner_rating",
    "restaurant_preparation_time_minutes",
    "estimated_delivery_time_minutes",
    "actual_delivery_time_minutes",
    "order_status",
]


def validate_dataframe(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    if df["order_id"].isna().any():
        errors.append("order_id contains null values")

    duplicates = int(df["order_id"].duplicated().sum())
    if duplicates:
        errors.append(f"Duplicate order_id values: {duplicates}")

    numeric_nonnegative = [
        "distance_km",
        "items_count",
        "subtotal",
        "tax_amount",
        "service_fee",
        "delivery_fee",
        "order_total",
        "estimated_delivery_time_minutes",
        "restaurant_preparation_time_minutes",
    ]

    for col in numeric_nonnegative:
        invalid = int((df[col] < 0).sum())
        if invalid:
            errors.append(f"{col} has {invalid} negative values")

    for col in ["restaurant_rating", "delivery_partner_rating"]:
        invalid = int(((df[col] < 0) | (df[col] > 5)).sum())
        if invalid:
            errors.append(f"{col} has {invalid} values outside [0, 5]")

    invalid_age = int(((df["customer_age"] < 13) | (df["customer_age"] > 100)).sum())
    if invalid_age:
        errors.append(f"customer_age has {invalid_age} values outside [13, 100]")

    invalid_weekend = set(df["is_weekend"].dropna().unique()) - {0, 1}
    if invalid_weekend:
        errors.append(f"is_weekend contains invalid values: {sorted(invalid_weekend)}")

    valid_traffic = {"Low", "Medium", "High"}
    invalid_traffic = set(df["traffic_level"].dropna().unique()) - valid_traffic
    if invalid_traffic:
        errors.append(f"traffic_level contains invalid values: {sorted(invalid_traffic)}")

    valid_status = {"Completed", "Cancelled"}
    invalid_status = set(df["order_status"].dropna().unique()) - valid_status
    if invalid_status:
        errors.append(f"order_status contains invalid values: {sorted(invalid_status)}")

    completed = df[df["actual_delivery_time_minutes"].notna()]
    if (completed["actual_delivery_time_minutes"] <= 0).any():
        errors.append("Completed orders contain actual_delivery_time_minutes <= 0")

    if (completed["estimated_delivery_time_minutes"] <= 0).any():
        errors.append("Completed orders contain estimated_delivery_time_minutes <= 0")

    return errors


def validate_file(filepath: str | Path) -> bool:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    errors = validate_dataframe(df)

    if errors:
        print("DATA VALIDATION FAILED")
        for error in errors:
            print(f" - {error}")
        return False

    print("DATA VALIDATION PASSED")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/food_delivery_orders_dataset.csv"
    ok = validate_file(path)
    raise SystemExit(0 if ok else 1)
