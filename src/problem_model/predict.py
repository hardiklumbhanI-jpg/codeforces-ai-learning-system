"""Presentation wrapper around the unchanged legacy Model 1 predictor.

The original ``src/predict_rating.py`` and root ``models/*.pkl`` files are
preserved.  This wrapper adds empirical uncertainty metadata from the existing
held-out ``results/prediction_samples.csv``; it does not train or call Model 2.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_SRC = PROJECT_ROOT / "src"
if str(LEGACY_SRC) not in sys.path:
    sys.path.insert(0, str(LEGACY_SRC))

from predict_rating import predict_rating  # noqa: E402


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 300.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def empirical_uncertainty() -> dict[str, float]:
    path = PROJECT_ROOT / "results" / "prediction_samples.csv"
    residuals = []
    actuals = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                actual = float(row["actual_rating"])
                predicted = float(row["predicted_rating"])
                actuals.append(actual)
                residuals.append(abs(actual - predicted))
    if not residuals:
        return {"mae": 300.0, "interval_half_width": 500.0, "confidence": 0.0, "samples": 0}
    mae = statistics.mean(residuals)
    rating_span = max(actuals) - min(actuals) if len(actuals) > 1 else 2700.0
    normalized_reliability = max(0.0, min(1.0, 1.0 - mae / max(rating_span, 1.0)))
    return {
        "mae": mae,
        "interval_half_width": _quantile(residuals, 0.80),
        "confidence": normalized_reliability,
        "samples": len(residuals),
    }


def difficulty_level(rating: int) -> str:
    if rating < 1200:
        return "BEGINNER"
    if rating < 1600:
        return "INTERMEDIATE"
    if rating < 2100:
        return "ADVANCED"
    if rating < 2600:
        return "EXPERT"
    return "ELITE"


def predict_problem_difficulty(statement: str, tags: str = "") -> dict[str, Any]:
    predicted = predict_rating(statement, tags=tags)
    uncertainty = empirical_uncertainty()
    half_width = uncertainty["interval_half_width"]
    lower = int(round((predicted - half_width) / 25.0) * 25)
    upper = int(round((predicted + half_width) / 25.0) * 25)
    return {
        "predicted_rating": predicted,
        "difficulty": difficulty_level(predicted),
        "confidence": uncertainty["confidence"],
        "prediction_range": [max(800, lower), min(3500, upper)],
        "empirical_mae": uncertainty["mae"],
        "validation_samples": int(uncertainty["samples"]),
        "uncertainty_method": (
            "normalized held-out MAE reliability; range uses the empirical 80th percentile absolute residual"
        ),
        "model": "Model 1: preserved TF-IDF + structural + tag GradientBoostingRegressor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preserved Model 1 independently.")
    parser.add_argument("--statement", required=True)
    parser.add_argument("--tags", default="")
    args = parser.parse_args()
    print(predict_problem_difficulty(args.statement, args.tags))


if __name__ == "__main__":
    main()
