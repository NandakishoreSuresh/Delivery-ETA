"""Feature contract for ETA correction."""

TARGET = "eta_error"
BASELINE_ETA = "estimated_delivery_time_minutes"

# Columns that are known at prediction time in this project.
FEATURES = [
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
    "estimated_delivery_time_minutes",
]

CATEGORICAL_FEATURES = [
    "day_of_week",
    "city",
    "delivery_area",
    "customer_type",
    "restaurant_type",
    "restaurant_primary_category",
    "payment_method",
    "weather",
    "traffic_level",
]

NUMERIC_FEATURES = [
    c for c in FEATURES
    if c not in CATEGORICAL_FEATURES
]

POST_OUTCOME_COLUMNS = [
    "actual_delivery_time_minutes",
    "late_delivery",
    "tip_amount",
    "customer_rating",
    "cancellation_reason",
]

IDENTIFIER_COLUMNS = [
    "order_id",
    "customer_id",
    "restaurant_id",
    "driver_id",
]

TEMPORARILY_EXCLUDED_COLUMNS = [
    "restaurant_preparation_time_minutes",
]
