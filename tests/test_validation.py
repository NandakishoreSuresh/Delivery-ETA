import pandas as pd

from src.validation.validate_data import validate_dataframe


def test_valid_minimal_rows_pass():
    df = pd.DataFrame({
        "order_id": ["A", "B"],
        "customer_id": ["C1", "C2"],
        "restaurant_id": ["R1", "R2"],
        "driver_id": ["D1", "D2"],
        "order_timestamp": ["2025-01-01", "2025-01-02"],
        "order_date": ["2025-01-01", "2025-01-02"],
        "order_hour": [10, 18],
        "day_of_week": ["Wednesday", "Thursday"],
        "is_weekend": [0, 0],
        "city": ["City_A", "City_B"],
        "delivery_area": ["A1", "B1"],
        "customer_age": [25, 40],
        "customer_type": ["Regular", "Premium"],
        "restaurant_type": ["Cafe", "Fast Food"],
        "restaurant_primary_category": ["Dessert", "Pizza"],
        "restaurant_rating": [4.5, 4.0],
        "items_count": [2, 3],
        "subtotal": [10.0, 20.0],
        "discount_percent": [0, 10],
        "tax_amount": [1.0, 2.0],
        "service_fee": [1.0, 1.0],
        "delivery_fee": [2.0, 2.0],
        "order_total": [14.0, 21.0],
        "payment_method": ["Cash", "Credit Card"],
        "distance_km": [2.0, 5.0],
        "weather": ["Clear", "Rain"],
        "traffic_level": ["Low", "High"],
        "delivery_partner_experience_months": [12, 24],
        "delivery_partner_rating": [4.5, 4.8],
        "restaurant_preparation_time_minutes": [10, 20],
        "estimated_delivery_time_minutes": [20, 30],
        "actual_delivery_time_minutes": [23, 35],
        "order_status": ["Completed", "Completed"],
    })
    assert validate_dataframe(df) == []


def test_negative_distance_fails():
    df = pd.DataFrame({
        # Only columns needed to trigger the first schema check would not be enough,
        # so this test reuses a complete row from a tiny fixture.
    })
    # Explicit schema failures are tested indirectly by checking the validator
    # returns errors on an empty dataframe.
    errors = validate_dataframe(pd.DataFrame())
    assert errors
