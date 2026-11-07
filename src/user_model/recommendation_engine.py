"""Model 2 similar-problem and personalized next-problem recommendations."""

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DB_PATH
from .database import connect, initialize_database
from .knowledge_graph import tag_topic_ids
from .problem_topics import required_topic_scores


MODES = {
    "skill_gap",
    "similarity",
    "difficulty_progression",
    "topic_practice",
    "revision",
    "mixed_topic",
}


def _problem_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT p.*, GROUP_CONCAT(t.tag, ';') AS tags_text
        FROM problems p LEFT JOIN problem_tags t ON t.problem_id=p.problem_id
        WHERE p.official_rating IS NOT NULL
        GROUP BY p.problem_id
        """
    ).fetchall()
    return [
        {**dict(row), "tags": [tag for tag in (row["tags_text"] or "").split(";") if tag]}
        for row in rows
    ]


def _difficulty_label(problem_rating: int, user_rating: int) -> str:
    gap = problem_rating - user_rating
    if gap <= -200:
        return "EASY"
    if gap <= 100:
        return "APPROPRIATE"
    if gap <= 300:
        return "CHALLENGING"
    return "HARD"


def similar_problems(
    target_problem: dict[str, Any], *, db_path: Path | str = DB_PATH, count: int = 5
) -> list[dict[str, Any]]:
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        problems = _problem_rows(connection)
    finally:
        connection.close()
    corpus = [
        f"{item['name']} {' '.join(item['tags'])} {item.get('statement') or ''}" for item in problems
    ]
    query = f"{target_problem['name']} {' '.join(target_problem.get('tags', []))} {target_problem.get('statement') or ''}"
    vectorizer = TfidfVectorizer(max_features=12000, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([query, *corpus])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    candidates = []
    for problem, score in zip(problems, scores):
        if problem["problem_id"] == target_problem["problem_id"]:
            continue
        candidates.append({
            **problem,
            "similarity": float(score),
            "why_recommended": "Similar statement vocabulary, official tags, and metadata.",
        })
    candidates.sort(key=lambda item: (-item["similarity"], item["official_rating"], item["name"]))
    return candidates[:count]


def _topic_ids(problem: dict[str, Any]) -> set[str]:
    requirements = required_topic_scores(problem.get("statement") or "", problem.get("tags", []), limit=20)
    return {item["topic_id"] for item in requirements if item["relevance"] >= 0.4}


def recommend_next_problems(
    handle: str,
    *,
    user_rating: int,
    target_problem: dict[str, Any],
    gaps: list[dict[str, Any]],
    mode: str = "skill_gap",
    topic: str | None = None,
    count: int = 5,
    db_path: Path | str = DB_PATH,
) -> list[dict[str, Any]]:
    if mode not in MODES:
        raise ValueError(f"Unknown recommendation mode '{mode}'. Choose from {sorted(MODES)}")
    if mode == "similarity":
        return similar_problems(target_problem, db_path=db_path, count=count)
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        solved = {
            row["problem_id"] for row in connection.execute(
                "SELECT problem_id FROM user_problem_history WHERE handle=? AND accepted=1", (handle,)
            )
        }
        candidates = [
            item for item in _problem_rows(connection)
            if item["problem_id"] not in solved and item["problem_id"] != target_problem["problem_id"]
        ]
        weak_topics = [item["topic_id"] for item in gaps if item["gap"] > 0.08]
        if topic:
            weak_topics = [topic]
        target_topics = _topic_ids(target_problem)
        ranked = []
        for problem in candidates:
            rating = int(problem["official_rating"])
            topics = _topic_ids(problem)
            gap_overlap = len(topics & set(weak_topics))
            target_overlap = len(topics & target_topics)
            if mode in {"skill_gap", "topic_practice"} and weak_topics and not gap_overlap:
                continue
            if mode == "revision" and not any(
                item["topic_id"] in topics and item.get("skill_confidence", 0) >= 0.4
                for item in gaps
            ):
                continue
            if mode == "mixed_topic" and len(topics) < 2:
                continue
            # Progression favours solvable steps from current rating toward the target.
            ceiling = min(int(target_problem["official_rating"]), user_rating + 350)
            floor = max(800, user_rating - (250 if mode == "revision" else 100))
            if not floor <= rating <= max(floor, ceiling):
                continue
            desired = min(int(target_problem["official_rating"]), user_rating + 150)
            rating_score = math.exp(-abs(rating - desired) / 250.0)
            topic_score = 0.55 * min(1.0, gap_overlap / 2.0) + 0.25 * min(1.0, target_overlap / 2.0)
            diversity = 0.1 * min(1.0, len(topics) / 3.0)
            score = 0.1 + 0.45 * rating_score + topic_score + diversity
            reason_parts = []
            if gap_overlap:
                reason_parts.append("practises a measured skill gap")
            if target_overlap:
                reason_parts.append("shares skills with the target problem")
            reason_parts.append(f"rating {rating} is a controlled step from {user_rating}")
            ranked.append({
                **problem,
                "score": score,
                "expected_difficulty": _difficulty_label(rating, user_rating),
                "why_recommended": "; ".join(reason_parts).capitalize() + ".",
            })
        ranked.sort(key=lambda item: (-item["score"], item["official_rating"], item["name"]))
        selected = ranked[:count]
        with connection:
            for item in selected:
                connection.execute(
                    """
                    INSERT INTO recommendations(handle, problem_id, mode, score, reason, expected_difficulty, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(handle, problem_id, mode) DO UPDATE SET
                        score=excluded.score, reason=excluded.reason,
                        expected_difficulty=excluded.expected_difficulty, created_at=excluded.created_at
                    """,
                    (
                        handle, item["problem_id"], mode, item["score"], item["why_recommended"],
                        item["expected_difficulty"], int(time.time()),
                    ),
                )
        return selected
    finally:
        connection.close()
