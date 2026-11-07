"""Leakage-safe feature engineering for Model 2 success prediction."""

from __future__ import annotations

import math
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from .config import DB_PATH
from .database import connect, initialize_database
from .problem_topics import required_topic_scores
from .profile_manager import user_summary
from .skill_estimator import get_skill_profiles


FEATURE_COLUMNS = [
    "current_rating",
    "max_rating",
    "rating_trend",
    "recent_rating_change",
    "number_of_contests",
    "prior_attempted",
    "prior_solved",
    "prior_recent_solved",
    "prior_recent_success_rate",
    "prior_overall_success_rate",
    "problem_rating",
    "rating_gap",
    "number_of_tags",
    "same_topic_skill",
    "same_topic_confidence",
    "related_topic_skill",
    "same_rating_success",
    "consistency",
    "recent_activity",
    "average_attempts",
    "wrong_answer_rate",
    "tle_rate",
    "topic_recency_days",
]


def _smoothed_rate(solved: float, attempted: float) -> float:
    return (solved + 1.0) / (attempted + 2.0)


def _weighted_topic_features(requirements, topic_stats, timestamp: int) -> tuple[float, float, float]:
    if not requirements:
        return 0.5, 0.0, 3650.0
    total_weight = sum(item["relevance"] for item in requirements)
    skill = 0.0
    confidence = 0.0
    recencies = []
    for requirement in requirements:
        relevance = requirement["relevance"]
        stats = topic_stats[requirement["topic_id"]]
        skill += relevance * _smoothed_rate(stats["solved"], stats["attempted"])
        confidence += relevance * (1.0 - math.exp(-stats["attempted"] / 8.0))
        if stats["last_time"]:
            recencies.append((relevance, max(0.0, (timestamp - stats["last_time"]) / 86_400.0)))
    topic_recency = (
        sum(weight * days for weight, days in recencies) / sum(weight for weight, _ in recencies)
        if recencies else 3650.0
    )
    return skill / total_weight, confidence / total_weight, topic_recency


def time_safe_feature_rows(interactions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create rows using only interactions strictly earlier than each target."""
    ordered = sorted((dict(row) for row in interactions), key=lambda row: (row["handle"].lower(), row["timestamp"]))
    states: dict[str, dict[str, Any]] = {}
    output = []
    for row in ordered:
        handle = row["handle"].lower()
        state = states.setdefault(handle, {
            "attempted": 0, "solved": 0, "attempts": 0, "wa": 0, "tle": 0,
            "outcomes": [], "events": deque(), "topics": defaultdict(
                lambda: {"attempted": 0.0, "solved": 0.0, "last_time": None}
            ),
            "ratings": defaultdict(lambda: {"attempted": 0.0, "solved": 0.0}),
        })
        timestamp = int(row["timestamp"])
        recent_cutoff = timestamp - 30 * 86_400
        while state["events"] and state["events"][0][0] < recent_cutoff:
            state["events"].popleft()
        recent_solved = sum(item[1] for item in state["events"])
        requirements = row.get("required_topics") or required_topic_scores(
            row.get("statement", ""), row.get("tags", [])
        )
        topic_skill, topic_confidence, topic_recency = _weighted_topic_features(
            requirements, state["topics"], timestamp
        )
        rating_bucket = int(row["problem_rating"] // 200 * 200)
        rating_stats = state["ratings"][rating_bucket]
        attempted = state["attempted"]
        solved = state["solved"]
        mean_outcome = solved / attempted if attempted else 0.0
        variance = (
            sum((value - mean_outcome) ** 2 for value in state["outcomes"]) / attempted
            if attempted else 0.0
        )
        feature_row = {
            "handle": row["handle"],
            "problem_id": row["problem_id"],
            "timestamp": timestamp,
            "accepted": int(row["accepted"]),
            "current_rating": float(row.get("current_rating") or 0),
            "max_rating": float(row.get("max_rating") or row.get("current_rating") or 0),
            "rating_trend": float(row.get("rating_trend") or 0),
            "recent_rating_change": float(row.get("recent_rating_change") or 0),
            "number_of_contests": float(row.get("number_of_contests") or 0),
            "prior_attempted": float(attempted),
            "prior_solved": float(solved),
            "prior_recent_solved": float(recent_solved),
            "prior_recent_success_rate": recent_solved / len(state["events"]) if state["events"] else 0.0,
            "prior_overall_success_rate": solved / attempted if attempted else 0.0,
            "problem_rating": float(row["problem_rating"]),
            "rating_gap": float(row["problem_rating"] - (row.get("current_rating") or 0)),
            "number_of_tags": float(len(row.get("tags", []))),
            "same_topic_skill": topic_skill,
            "same_topic_confidence": topic_confidence,
            "related_topic_skill": topic_skill,
            "same_rating_success": _smoothed_rate(rating_stats["solved"], rating_stats["attempted"]),
            "consistency": max(0.0, 1.0 - 2.0 * math.sqrt(variance)),
            "recent_activity": min(1.0, len(state["events"]) / 20.0),
            "average_attempts": state["attempts"] / attempted if attempted else 0.0,
            "wrong_answer_rate": state["wa"] / state["attempts"] if state["attempts"] else 0.0,
            "tle_rate": state["tle"] / state["attempts"] if state["attempts"] else 0.0,
            "topic_recency_days": min(3650.0, topic_recency),
        }
        output.append(feature_row)

        outcome = int(row["accepted"])
        state["attempted"] += 1
        state["solved"] += outcome
        state["attempts"] += int(row.get("attempts", 1))
        state["wa"] += int(row.get("wrong_answers", 0))
        state["tle"] += int(row.get("tle_count", 0))
        state["outcomes"].append(outcome)
        state["events"].append((timestamp, outcome))
        rating_stats["attempted"] += 1.0
        rating_stats["solved"] += outcome
        for requirement in requirements:
            stats = state["topics"][requirement["topic_id"]]
            stats["attempted"] += requirement["relevance"]
            stats["solved"] += requirement["relevance"] * outcome
            stats["last_time"] = timestamp
    return output


def build_prediction_features(
    handle: str,
    problem: dict[str, Any],
    *,
    db_path: Path | str = DB_PATH,
    now: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    now = int(now or time.time())
    summary = user_summary(handle, db_path=db_path, now=now)
    skills = get_skill_profiles(handle, db_path=db_path)
    requirements = required_topic_scores(problem.get("statement", ""), problem.get("tags", []))
    total_relevance = sum(item["relevance"] for item in requirements) or 1.0
    topic_skill = sum(
        item["relevance"] * skills.get(item["topic_id"], {}).get("skill_score", 0.5)
        for item in requirements
    ) / total_relevance
    topic_confidence = sum(
        item["relevance"] * skills.get(item["topic_id"], {}).get("confidence", 0.0)
        for item in requirements
    ) / total_relevance
    practiced = [
        (item["relevance"], skills.get(item["topic_id"], {}).get("last_practiced_at"))
        for item in requirements
        if skills.get(item["topic_id"], {}).get("last_practiced_at")
    ]
    topic_recency = (
        sum(weight * max(0, now - timestamp) / 86_400 for weight, timestamp in practiced)
        / sum(weight for weight, _ in practiced)
        if practiced else 3650.0
    )
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        lower = int(problem["official_rating"] // 200 * 200)
        rating_history = connection.execute(
            """
            SELECT COUNT(*) attempted, COALESCE(SUM(h.accepted), 0) solved
            FROM user_problem_history h JOIN problems p ON p.problem_id=h.problem_id
            WHERE h.handle=? AND p.official_rating>=? AND p.official_rating<?
            """,
            (handle, lower, lower + 200),
        ).fetchone()
    finally:
        connection.close()
    features = {
        "current_rating": float(summary.get("current_rating") or 0),
        "max_rating": float(summary.get("max_rating") or summary.get("current_rating") or 0),
        "rating_trend": float(summary["rating_trend"]),
        "recent_rating_change": float(summary["recent_rating_change"]),
        "number_of_contests": float(summary["contests"]),
        "prior_attempted": float(summary["attempted_count"]),
        "prior_solved": float(summary["solved_count"]),
        "prior_recent_solved": float(summary["recent_solved_count"]),
        "prior_recent_success_rate": float(summary["recent_success_rate"]),
        "prior_overall_success_rate": float(summary["overall_success_rate"]),
        "problem_rating": float(problem["official_rating"]),
        "rating_gap": float(problem["official_rating"] - (summary.get("current_rating") or 0)),
        "number_of_tags": float(len(problem.get("tags", []))),
        "same_topic_skill": topic_skill,
        "same_topic_confidence": topic_confidence,
        "related_topic_skill": topic_skill,
        "same_rating_success": _smoothed_rate(rating_history["solved"], rating_history["attempted"]),
        "consistency": float(summary["consistency"]),
        "recent_activity": float(summary["recent_activity"]),
        "average_attempts": float(summary["average_attempts"]),
        "wrong_answer_rate": float(summary["wrong_answer_rate"]),
        "tle_rate": float(summary["tle_rate"]),
        "topic_recency_days": min(3650.0, topic_recency),
    }
    return features, requirements
