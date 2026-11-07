"""Explainable required-topic versus inferred-user-skill comparison."""

from __future__ import annotations

from typing import Any


def analyze_skill_gaps(
    required_topics: list[dict[str, Any]],
    skill_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    report = []
    for requirement in required_topics:
        profile = skill_profiles.get(requirement["topic_id"], {})
        user_skill = float(profile.get("skill_score", 0.0))
        confidence = float(profile.get("confidence", 0.0))
        gap = max(0.0, float(requirement["relevance"]) - user_skill)
        if gap <= 0.08:
            status = "STRONG MATCH"
        elif gap <= 0.25:
            status = "MODERATE GAP"
        else:
            status = "MAJOR GAP"
        report.append({
            "topic_id": requirement["topic_id"],
            "topic": requirement["topic"],
            "required_relevance": float(requirement["relevance"]),
            "user_skill": user_skill,
            "gap": gap,
            "status": status,
            "skill_confidence": confidence,
            "evidence": profile.get("evidence", {}),
            "interpretation": (
                "Required is topic relevance/importance, not a claim about a fixed knowledge percentage."
            ),
        })
    report.sort(key=lambda item: (-item["gap"], -item["required_relevance"]))
    return report


def weighted_topic_match(gaps: list[dict[str, Any]]) -> float:
    total = sum(item["required_relevance"] for item in gaps)
    if not total:
        return 0.5
    weighted_gap = sum(
        item["required_relevance"] * item["gap"] for item in gaps
    ) / total
    return max(0.0, min(1.0, 1.0 - weighted_gap))
