"""Probability-equivalent personalized difficulty and readiness."""

from __future__ import annotations

import math
from typing import Any

from .skill_gap import weighted_topic_match
from .topic_skill import irt_probability


def probability_equivalent_rating(user_rating: float, probability: float) -> float:
    """Invert the Codeforces/Elo logistic curve at the user's predicted probability."""
    probability = max(0.02, min(0.98, probability))
    return user_rating + 400.0 * math.log10((1.0 - probability) / probability)


def _difficulty_label(personalized_rating: float, user_rating: float) -> str:
    gap = personalized_rating - user_rating
    if gap <= -200:
        return "EASY FOR YOU"
    if gap <= 100:
        return "APPROPRIATE"
    if gap <= 300:
        return "CHALLENGING"
    if gap <= 500:
        return "HARD"
    return "VERY HARD"


def personalized_difficulty(
    *,
    official_rating: int,
    user_rating: int,
    acceptance_probability: float,
    prediction_confidence: float,
    gaps: list[dict[str, Any]],
    recent_success_rate: float,
) -> dict[str, Any]:
    equivalent = probability_equivalent_rating(user_rating, acceptance_probability)
    confidence = max(0.0, min(1.0, prediction_confidence))
    # Low-evidence estimates shrink toward official metadata; high-evidence
    # estimates rely more heavily on the user's probability-equivalent rating.
    blended = (1.0 - confidence) * official_rating + confidence * equivalent
    rounded = int(round(blended / 25.0) * 25)
    topic_match = weighted_topic_match(gaps)
    rating_match = irt_probability(user_rating, official_rating)
    recent = max(0.0, min(1.0, recent_success_rate))
    readiness = 100.0 * (
        0.45 * acceptance_probability
        + 0.20 * topic_match
        + 0.15 * rating_match
        + 0.10 * recent
        + 0.10 * confidence
    )
    if readiness < 40:
        readiness_label = "LOW"
    elif readiness < 70:
        readiness_label = "MEDIUM"
    else:
        readiness_label = "HIGH"
    largest_gap = max(gaps, key=lambda item: item["gap"], default=None)
    reasons = [
        f"Probability-equivalent difficulty is {equivalent:.0f} from Model 2's solve probability.",
        f"The estimate is blended {confidence:.0%} toward user-specific evidence and the rest toward the official rating.",
        f"Topic match is {topic_match:.0%}; recent success is {recent:.0%}.",
    ]
    if largest_gap and largest_gap["gap"] > 0:
        reasons.append(
            f"Largest topic gap: {largest_gap['topic']} ({largest_gap['gap']:.0%})."
        )
    return {
        "official_rating": int(official_rating),
        "user_rating": int(user_rating),
        "probability_equivalent_rating": equivalent,
        "personalized_rating": rounded,
        "difficulty_for_user": _difficulty_label(rounded, user_rating),
        "acceptance_probability": acceptance_probability,
        "prediction_confidence": confidence,
        "topic_match": topic_match,
        "rating_match": rating_match,
        "readiness_score": readiness,
        "readiness_label": readiness_label,
        "reasons": reasons,
        "readiness_formula": (
            "100 * (0.45*success_probability + 0.20*topic_match + "
            "0.15*rating_match + 0.10*recent_performance + 0.10*confidence)"
        ),
    }
