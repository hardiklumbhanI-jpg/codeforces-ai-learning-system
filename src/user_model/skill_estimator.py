"""Build, persist, and read Model 2 user skill profiles."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import DB_PATH
from .database import connect, initialize_database
from .knowledge_graph import load_knowledge_graph, node_map
from .problem_topics import required_topic_scores
from .topic_skill import DifficultyWeightedSkillModel, smoothed_difficulty_curve


def _interaction_rows(connection: sqlite3.Connection, handle: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT h.*, p.official_rating, p.statement,
               GROUP_CONCAT(t.tag, ';') AS tags
        FROM user_problem_history h
        JOIN problems p ON p.problem_id = h.problem_id
        LEFT JOIN problem_tags t ON t.problem_id = p.problem_id
        WHERE h.handle = ? AND p.official_rating IS NOT NULL
        GROUP BY h.handle, h.problem_id
        ORDER BY h.first_attempt_time
        """,
        (handle,),
    ).fetchall()
    interactions = []
    graph = load_knowledge_graph()
    for row in rows:
        item = dict(row)
        requirements = required_topic_scores(
            item.get("statement") or "",
            (item.get("tags") or "").split(";"),
            graph=graph,
            limit=20,
        )
        item["required_topics"] = requirements
        interactions.append(item)
    return interactions


def estimate_all_skills(
    handle: str,
    *,
    db_path: Path | str = DB_PATH,
    now: int | None = None,
    model: DifficultyWeightedSkillModel | None = None,
) -> dict[str, dict[str, Any]]:
    initialize_database(db_path)
    connection = connect(db_path)
    now = int(now or time.time())
    model = model or DifficultyWeightedSkillModel()
    try:
        user = connection.execute(
            "SELECT * FROM users WHERE handle = ?", (handle,)
        ).fetchone()
        if user is None:
            raise LookupError(f"Unknown local user: {handle}")
        interactions = _interaction_rows(connection, user["handle"])
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for interaction in interactions:
            for requirement in interaction["required_topics"]:
                by_topic[requirement["topic_id"]].append({
                    "timestamp": interaction["last_attempt_time"],
                    "rating": interaction["official_rating"],
                    "accepted": interaction["accepted"],
                    "attempts": interaction["attempts"],
                    "wrong_answers": interaction["wrong_answers"],
                    "tle_count": interaction["tle_count"],
                    "relevance": requirement["relevance"],
                    "problem_id": interaction["problem_id"],
                })

        graph_nodes = node_map(load_knowledge_graph())
        profiles: dict[str, dict[str, Any]] = {}
        with connection:
            connection.execute("DELETE FROM topic_performance WHERE handle = ?", (user["handle"],))
            connection.execute("DELETE FROM skill_profiles WHERE handle = ?", (user["handle"],))
            for topic_id, observations in by_topic.items():
                estimate = model.estimate(
                    observations,
                    prior_rating=user["current_rating"] or user["max_rating"] or 1200,
                    now=now,
                )
                if estimate is None:
                    continue
                profile = {"topic_id": topic_id, "topic": graph_nodes[topic_id]["name"], **estimate}
                profiles[topic_id] = profile
                connection.execute(
                    """
                    INSERT INTO skill_profiles (
                        handle, topic_id, skill_score, confidence, ability_rating,
                        attempted, solved, recent_performance, long_term_performance,
                        avg_problem_rating, max_solved_rating, wrong_answer_rate,
                        tle_rate, average_attempts, consistency, last_practiced_at,
                        evidence_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user["handle"], topic_id, estimate["skill_score"], estimate["confidence"],
                        estimate["ability_rating"], estimate["attempted"], estimate["solved"],
                        estimate["recent_performance"], estimate["long_term_performance"],
                        estimate["avg_problem_rating"], estimate["max_solved_rating"],
                        estimate["wrong_answer_rate"], estimate["tle_rate"],
                        estimate["average_attempts"], estimate["consistency"],
                        estimate["last_practiced_at"], json.dumps(estimate["evidence"]), now,
                    ),
                )
                for curve in smoothed_difficulty_curve(observations):
                    connection.execute(
                        """
                        INSERT INTO topic_performance (
                            handle, topic_id, difficulty_bucket, attempted,
                            solved, acceptance_rate, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user["handle"], topic_id, curve["difficulty_bucket"],
                            curve["attempted"], curve["solved"], curve["acceptance_rate"], now,
                        ),
                    )
        return profiles
    finally:
        connection.close()


def get_skill_profiles(
    handle: str, db_path: Path | str = DB_PATH, refresh: bool = False
) -> dict[str, dict[str, Any]]:
    initialize_database(db_path)
    if refresh:
        return estimate_all_skills(handle, db_path=db_path)
    connection = connect(db_path)
    try:
        rows = connection.execute(
            "SELECT * FROM skill_profiles WHERE handle = ? ORDER BY skill_score DESC", (handle,)
        ).fetchall()
        if not rows:
            return estimate_all_skills(handle, db_path=db_path)
        nodes = node_map(load_knowledge_graph())
        return {
            row["topic_id"]: {
                **dict(row),
                "topic": nodes.get(row["topic_id"], {}).get("name", row["topic_id"]),
                "evidence": json.loads(row["evidence_json"] or "{}"),
            }
            for row in rows
        }
    finally:
        connection.close()


def difficulty_curve(handle: str, topic_id: str, db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    connection = connect(db_path)
    try:
        return [
            dict(row) for row in connection.execute(
                """SELECT difficulty_bucket, attempted, solved, acceptance_rate
                   FROM topic_performance WHERE handle=? AND topic_id=?
                   ORDER BY difficulty_bucket""",
                (handle, topic_id),
            )
        ]
    finally:
        connection.close()
