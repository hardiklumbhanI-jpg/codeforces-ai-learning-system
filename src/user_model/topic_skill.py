"""Difficulty-, recency-, and confidence-aware topic skill estimation.

The estimator is a one-dimensional regularized item-response model per topic.
For an observation with topic ability ``a`` and official problem rating ``d``:

    P(solve) = 1 / (1 + 10 ** ((d - a) / 400))

Observations are weighted by topic relevance and a recency mixture.  Skill is
the fitted ability mapped onto the documented 800-2800 scale, then subject to
controlled inactivity decay.  Confidence grows with effective evidence mass,
not with the observed success rate.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .config import (
    CONFIDENCE_EVIDENCE_SCALE,
    DECAY_FLOOR,
    DECAY_GRACE_DAYS,
    DECAY_HALF_LIFE_DAYS,
    RECENCY_HALF_LIFE_DAYS,
    RECENCY_LONG_TERM_FLOOR,
    SKILL_MAX_RATING,
    SKILL_MIN_RATING,
    SKILL_PRIOR_STRENGTH,
)


SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class SkillParameters:
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS
    long_term_floor: float = RECENCY_LONG_TERM_FLOOR
    decay_grace_days: float = DECAY_GRACE_DAYS
    decay_half_life_days: float = DECAY_HALF_LIFE_DAYS
    decay_floor: float = DECAY_FLOOR
    min_rating: float = SKILL_MIN_RATING
    max_rating: float = SKILL_MAX_RATING
    prior_strength: float = SKILL_PRIOR_STRENGTH
    confidence_evidence_scale: float = CONFIDENCE_EVIDENCE_SCALE


def recency_weight(age_days: float, parameters: SkillParameters) -> float:
    decay = 2.0 ** (-max(0.0, age_days) / parameters.recency_half_life_days)
    return parameters.long_term_floor + (1.0 - parameters.long_term_floor) * decay


def inactivity_decay(days_since_practice: float, parameters: SkillParameters) -> float:
    if days_since_practice <= parameters.decay_grace_days:
        return 1.0
    overdue = days_since_practice - parameters.decay_grace_days
    retained = 2.0 ** (-overdue / parameters.decay_half_life_days)
    return parameters.decay_floor + (1.0 - parameters.decay_floor) * retained


def irt_probability(ability_rating: float, problem_rating: float) -> float:
    exponent = max(-8.0, min(8.0, (problem_rating - ability_rating) / 400.0))
    return 1.0 / (1.0 + 10.0 ** exponent)


class DifficultyWeightedSkillModel:
    """Serializable statistical Model 2 skill estimator."""

    def __init__(self, parameters: SkillParameters | None = None) -> None:
        self.parameters = parameters or SkillParameters()

    def get_params(self) -> dict[str, Any]:
        return asdict(self.parameters)

    def estimate(
        self,
        observations: Iterable[dict[str, Any]],
        *,
        prior_rating: float,
        now: int | None = None,
    ) -> dict[str, Any] | None:
        rows = [dict(row) for row in observations if float(row.get("relevance", 0.0)) > 0]
        if not rows:
            return None
        parameters = self.parameters
        now = int(now or time.time())
        prior = max(parameters.min_rating, min(parameters.max_rating, float(prior_rating)))
        k = math.log(10.0) / 400.0
        weighted_rows = []
        for row in rows:
            age_days = max(0.0, (now - int(row["timestamp"])) / SECONDS_PER_DAY)
            relevance = max(0.05, min(1.0, float(row["relevance"])))
            weight = relevance * recency_weight(age_days, parameters)
            weighted_rows.append((row, age_days, weight))

        ability = prior
        prior_precision = parameters.prior_strength * (k * k * 0.25)
        for _ in range(40):
            gradient = -prior_precision * (ability - prior)
            hessian = -prior_precision
            for row, _, weight in weighted_rows:
                probability = irt_probability(ability, float(row["rating"]))
                outcome = float(row["accepted"])
                gradient += weight * k * (outcome - probability)
                hessian -= weight * k * k * probability * (1.0 - probability)
            if abs(hessian) < 1e-12:
                break
            step = gradient / hessian
            ability = max(parameters.min_rating - 300, min(parameters.max_rating + 300, ability - step))
            if abs(step) < 0.05:
                break

        evidence_mass = sum(weight for _, _, weight in weighted_rows)
        confidence = 1.0 - math.exp(-evidence_mass / parameters.confidence_evidence_scale)
        last_timestamp = max(int(row["timestamp"]) for row in rows)
        idle_days = max(0.0, (now - last_timestamp) / SECONDS_PER_DAY)
        decay_multiplier = inactivity_decay(idle_days, parameters)
        raw_skill = (ability - parameters.min_rating) / (parameters.max_rating - parameters.min_rating)
        skill_score = max(0.0, min(1.0, raw_skill)) * decay_multiplier

        total_relevance = sum(float(row["relevance"]) for row in rows)
        solved_relevance = sum(float(row["relevance"]) * float(row["accepted"]) for row in rows)
        long_term = (solved_relevance + 1.0) / (total_relevance + 2.0)
        recent_rows = sorted(rows, key=lambda row: int(row["timestamp"]), reverse=True)[:10]
        recent_total = sum(float(row["relevance"]) for row in recent_rows)
        recent_solved = sum(float(row["relevance"]) * float(row["accepted"]) for row in recent_rows)
        recent = (recent_solved + 1.0) / (recent_total + 2.0)
        mean_outcome = solved_relevance / total_relevance if total_relevance else 0.0
        variance = sum(
            float(row["relevance"]) * (float(row["accepted"]) - mean_outcome) ** 2 for row in rows
        ) / max(total_relevance, 1e-9)
        consistency = max(0.0, 1.0 - 2.0 * math.sqrt(variance))

        solved_ratings = [int(row["rating"]) for row in rows if int(row["accepted"]) == 1]
        average_rating = sum(float(row["rating"]) * float(row["relevance"]) for row in rows) / total_relevance
        attempts = sum(float(row.get("attempts", 1)) * float(row["relevance"]) for row in rows)
        wrong_answers = sum(float(row.get("wrong_answers", 0)) * float(row["relevance"]) for row in rows)
        tle_count = sum(float(row.get("tle_count", 0)) * float(row["relevance"]) for row in rows)

        return {
            "skill_score": skill_score,
            "confidence": confidence,
            "ability_rating": ability,
            "attempted": total_relevance,
            "solved": solved_relevance,
            "recent_performance": recent,
            "long_term_performance": long_term,
            "avg_problem_rating": average_rating,
            "max_solved_rating": max(solved_ratings) if solved_ratings else None,
            "wrong_answer_rate": wrong_answers / max(attempts, 1.0),
            "tle_rate": tle_count / max(attempts, 1.0),
            "average_attempts": attempts / max(total_relevance, 1e-9),
            "consistency": consistency,
            "last_practiced_at": last_timestamp,
            "evidence": {
                "raw_observations": len(rows),
                "effective_evidence": evidence_mass,
                "recency_half_life_days": parameters.recency_half_life_days,
                "inactivity_days": idle_days,
                "decay_multiplier": decay_multiplier,
                "prior_rating": prior,
                "formula": "regularized 1PL/Codeforces-Elo item-response model",
            },
        }


def smoothed_difficulty_curve(
    observations: Iterable[dict[str, Any]], bucket_size: int = 200
) -> list[dict[str, Any]]:
    buckets: dict[int, dict[str, float]] = {}
    for row in observations:
        rating = int(row["rating"])
        bucket = max(800, (rating // bucket_size) * bucket_size)
        current = buckets.setdefault(bucket, {"attempted": 0.0, "solved": 0.0})
        relevance = float(row.get("relevance", 1.0))
        current["attempted"] += relevance
        current["solved"] += relevance * float(row["accepted"])
    return [
        {
            "difficulty_bucket": bucket,
            "attempted": values["attempted"],
            "solved": values["solved"],
            "acceptance_rate": (values["solved"] + 1.0) / (values["attempted"] + 2.0),
        }
        for bucket, values in sorted(buckets.items())
    ]
