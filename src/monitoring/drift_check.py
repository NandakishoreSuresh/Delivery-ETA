"""Basic training-vs-serving drift check for numeric features.

This is intentionally lightweight for local development. Later this module can
be replaced or extended with Evidently + Prometheus/Grafana.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "order_hour",
    "customer_age",
    "restaurant_rating",
    "items_count",
    "subtotal",
    "discount_percent",
    "tax_amount",
    "service_fee",
    "delivery_fee",
    "order_total",
    "distance_km",
    "delivery_partner_experience_months",
    "delivery_partner_rating",
    "estimated_delivery_time_minutes",
]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    edges = np.unique(
        np.quantile(expected, np.linspace(0, 1, bins + 1))
    )
    if len(edges) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)

    exp_pct = np.clip(exp_counts / max(len(expected), 1), 1e-6, None)
    act_pct = np.clip(act_counts / max(len(actual), 1), 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/processed/validation.csv",
) -> None:
    ref = pd.read_csv(reference_path)
    cur = pd.read_csv(current_path)

    print("=== DRIFT CHECK ===")
    for col in NUMERIC_FEATURES:
        if col not in ref or col not in cur:
            continue
        score = psi(ref[col].dropna().values, cur[col].dropna().values)
        print(f"{col:40s} PSI={score:.4f}")


if __name__ == "__main__":
    main()
