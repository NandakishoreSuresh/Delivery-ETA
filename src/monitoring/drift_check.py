"""Monitor feature drift using live API prediction logs."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_PATH = Path("data/processed/train.csv")
LOG_PATH = Path("data/logs/predictions.jsonl")
REPORT_PATH = Path("data/logs/drift_report.json")

MIN_LIVE_SAMPLES = 30
PSI_WARNING = 0.10
PSI_HIGH = 0.25

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


def calculate_psi(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0 or len(current) == 0:
        return np.nan

    if np.ptp(reference) == 0:
        return (
            0.0
            if np.allclose(current, reference[0])
            else float("inf")
        )

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        return 0.0

    edges = np.concatenate(
        ([-np.inf], edges[1:-1], [np.inf])
    )

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-6, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {REFERENCE_PATH}"
        )

    if not LOG_PATH.exists():
        print("No API prediction logs found.")
        return

    records = []

    with LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                features = record.get("features", {})
                if isinstance(features, dict):
                    records.append(features)

    print("=== API FEATURE DRIFT CHECK ===")
    print(f"Logged requests: {len(records)}")

    if len(records) < MIN_LIVE_SAMPLES:
        print(
            f"Insufficient samples: need at least "
            f"{MIN_LIVE_SAMPLES}; currently have {len(records)}."
        )
        print("No drift conclusion will be reported yet.")
        return

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.DataFrame(records)
    results = {}

    for col in NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue

        ref_values = pd.to_numeric(
            reference[col], errors="coerce"
        ).dropna()

        cur_values = pd.to_numeric(
            current[col], errors="coerce"
        ).dropna()

        score = calculate_psi(ref_values, cur_values)

        if not np.isfinite(score):
            status = "HIGH"
            score_output = str(score)
        elif score < PSI_WARNING:
            status = "LOW"
            score_output = round(score, 6)
        elif score < PSI_HIGH:
            status = "MODERATE"
            score_output = round(score, 6)
        else:
            status = "HIGH"
            score_output = round(score, 6)

        results[col] = {
            "psi": score_output,
            "status": status,
        }

        print(f"{col:40s} PSI={score_output} [{status}]")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference": str(REFERENCE_PATH),
        "logged_request_count": len(records),
        "features": results,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)

    print(f"\nDrift report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()