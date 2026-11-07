"""Inference for Model 2 acceptance probability."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_DIR
from .feature_engineering import FEATURE_COLUMNS
from .topic_skill import irt_probability


def _statistical_fallback(features: dict[str, float]) -> float:
    """Documented offline Model 2 fallback when no trained artifact is present."""
    rating_probability = irt_probability(features["current_rating"], features["problem_rating"])
    topic_match = max(0.0, min(1.0, features["same_topic_skill"]))
    recent = max(0.0, min(1.0, features["prior_recent_success_rate"]))
    history = max(0.0, min(1.0, features["same_rating_success"]))
    probability = 0.50 * rating_probability + 0.25 * topic_match + 0.15 * recent + 0.10 * history
    return max(0.02, min(0.98, probability))


def predict_success(
    features: dict[str, float], model_dir: Path | str = MODEL_DIR
) -> dict[str, Any]:
    missing = [column for column in FEATURE_COLUMNS if column not in features]
    if missing:
        raise ValueError(f"Missing Model 2 success features: {missing}")
    directory = Path(model_dir)
    model_path = directory / "success_probability_model.pkl"
    metadata_path = directory / "metadata.json"
    topic_confidence = max(0.0, min(1.0, features["same_topic_confidence"]))
    history_confidence = 1.0 - math.exp(-features["prior_attempted"] / 20.0)
    prediction_confidence = 0.55 * topic_confidence + 0.45 * history_confidence
    if model_path.exists():
        model = joblib.load(model_path)
        frame = pd.DataFrame([{column: float(features[column]) for column in FEATURE_COLUMNS}])
        probability = float(model.predict_proba(frame)[:, 1][0])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return {
            "probability": max(0.0, min(1.0, probability)),
            "confidence": prediction_confidence,
            "source": "trained_model",
            "model_version": metadata.get("model_version", "user-success-v1"),
            "selected_model": metadata.get("selected_model", type(model).__name__),
            "calibration": metadata.get("test_metrics", {}),
        }
    probability = _statistical_fallback(features)
    return {
        "probability": probability,
        "confidence": prediction_confidence * 0.75,
        "source": "documented_statistical_fallback",
        "model_version": "user-success-fallback-v1",
        "selected_model": "IRT/topic/history weighted baseline",
        "calibration": {},
    }


if __name__ == "__main__":
    raise SystemExit("Use main.py or import predict_success(features).")
