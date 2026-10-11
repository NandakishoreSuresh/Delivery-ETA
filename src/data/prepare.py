"""Create reproducible, time-ordered train/validation/test datasets."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.validation.validate_data import validate_dataframe


RAW_PATH = Path("data/raw/food_delivery_orders_dataset.csv")
OUT_DIR = Path("data/processed")


def main() -> None:
    df = pd.read_csv(RAW_PATH)

    errors = validate_dataframe(df)
    if errors:
        raise ValueError(
            "Raw data failed validation:\n" + "\n".join(f"- {e}" for e in errors)
        )

    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"], errors="coerce")
    if df["order_timestamp"].isna().any():
        raise ValueError("Invalid order_timestamp values found.")

    # ETA prediction is defined only for completed orders.
    completed = df[df["actual_delivery_time_minutes"].notna()].copy()

    # Learn from the systematic error in the existing ETA.
    completed["eta_error"] = (
        completed["actual_delivery_time_minutes"]
        - completed["estimated_delivery_time_minutes"]
    )

    # Chronological split prevents future orders entering training.
    completed = completed.sort_values("order_timestamp").reset_index(drop=True)

    n = len(completed)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = completed.iloc[:train_end].copy()
    val = completed.iloc[train_end:val_end].copy()
    test = completed.iloc[val_end:].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "validation.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print(f"Train:      {train.shape}")
    print(f"Validation: {val.shape}")
    print(f"Test:       {test.shape}")
    print(f"Train period: {train['order_timestamp'].min()} -> {train['order_timestamp'].max()}")
    print(f"Test period:  {test['order_timestamp'].min()} -> {test['order_timestamp'].max()}")


if __name__ == "__main__":
    main()
