"""Simple JSONL prediction logger for local development."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG_PATH = Path("data/logs/predictions.jsonl")


def log_prediction(
    request_features: dict[str, Any],
    predicted_error: float,
    estimated_eta: float,
    final_eta: float,
) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "estimated_eta": float(estimated_eta),
        "predicted_error": float(predicted_error),
        "final_eta": float(final_eta),
        "features": request_features,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
