"""User-level aggregates and controlled post-attempt profile updates."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from .config import DB_PATH
from .database import connect, initialize_database, rebuild_user_problem_history, upsert_submission
from .skill_estimator import estimate_all_skills, get_skill_profiles


VERDICT_ALIASES = {
    "ACCEPTED": "OK",
    "AC": "OK",
    "OK": "OK",
    "WRONG_ANSWER": "WRONG_ANSWER",
    "WA": "WRONG_ANSWER",
    "TLE": "TIME_LIMIT_EXCEEDED",
    "TIME_LIMIT_EXCEEDED": "TIME_LIMIT_EXCEEDED",
    "RUNTIME_ERROR": "RUNTIME_ERROR",
    "RE": "RUNTIME_ERROR",
    "COMPILATION_ERROR": "COMPILATION_ERROR",
    "MEMORY_LIMIT_EXCEEDED": "MEMORY_LIMIT_EXCEEDED",
    "MLE": "MEMORY_LIMIT_EXCEEDED",
}


def normalize_verdict(verdict: str) -> str:
    key = str(verdict).strip().upper().replace(" ", "_")
    if key not in VERDICT_ALIASES:
        raise ValueError(
            "Unsupported result. Use ACCEPTED, WRONG_ANSWER, TLE, RUNTIME_ERROR, "
            "COMPILATION_ERROR, or MEMORY_LIMIT_EXCEEDED."
        )
    return VERDICT_ALIASES[key]


def user_summary(handle: str, db_path: Path | str = DB_PATH, now: int | None = None) -> dict[str, Any]:
    initialize_database(db_path)
    now = int(now or time.time())
    connection = connect(db_path)
    try:
        user = connection.execute("SELECT * FROM users WHERE handle=?", (handle,)).fetchone()
        if user is None:
            raise LookupError(f"Unknown local user: {handle}")
        history = connection.execute(
            "SELECT * FROM user_problem_history WHERE handle=? ORDER BY last_attempt_time", (handle,)
        ).fetchall()
        ratings = connection.execute(
            "SELECT * FROM rating_history WHERE handle=? ORDER BY rating_update_time", (handle,)
        ).fetchall()
        attempted = len(history)
        solved = sum(row["accepted"] for row in history)
        recent_cutoff = now - 30 * 86_400
        recent = [row for row in history if (row["last_attempt_time"] or 0) >= recent_cutoff]
        recent_solved = sum(row["accepted"] for row in recent)
        attempts_total = sum(row["attempts"] for row in history)
        wa_total = sum(row["wrong_answers"] for row in history)
        tle_total = sum(row["tle_count"] for row in history)
        outcomes = [float(row["accepted"]) for row in history]
        consistency = 1.0
        if len(outcomes) > 1:
            consistency = max(0.0, 1.0 - 2.0 * statistics.pstdev(outcomes))
        rating_trend = 0.0
        recent_rating_change = 0.0
        if ratings:
            recent_rating_change = float(ratings[-1]["new_rating"] - ratings[-1]["old_rating"])
        if len(ratings) >= 2:
            window = ratings[-5:]
            rating_trend = (window[-1]["new_rating"] - window[0]["old_rating"]) / max(1, len(window))
        return {
            **dict(user),
            "contests": len(ratings),
            "attempted_count": attempted,
            "solved_count": solved,
            "overall_success_rate": solved / attempted if attempted else 0.0,
            "recent_attempted_count": len(recent),
            "recent_solved_count": recent_solved,
            "recent_success_rate": recent_solved / len(recent) if recent else 0.0,
            "recent_activity": min(1.0, len(recent) / 20.0),
            "average_attempts": attempts_total / attempted if attempted else 0.0,
            "wrong_answer_rate": wa_total / attempts_total if attempts_total else 0.0,
            "tle_rate": tle_total / attempts_total if attempts_total else 0.0,
            "consistency": consistency,
            "rating_trend": rating_trend,
            "recent_rating_change": recent_rating_change,
        }
    finally:
        connection.close()


def record_attempt(
    handle: str,
    problem: dict[str, Any],
    verdict: str,
    *,
    solution_metadata: dict[str, Any] | None = None,
    db_path: Path | str = DB_PATH,
    timestamp: int | None = None,
) -> dict[str, Any]:
    """Persist one observed result and recompute skills from all evidence.

    The update is controlled by the IRT estimator: hard accepted problems carry
    more information, repeated evidence moves a high-confidence profile less,
    and failures can lower estimates.  No fixed percentage increment is used.
    """
    normalized = normalize_verdict(verdict)
    timestamp = int(timestamp or time.time())
    before = get_skill_profiles(handle, db_path=db_path, refresh=True)
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        prior_attempts = connection.execute(
            "SELECT COUNT(*) FROM submissions WHERE handle=? AND problem_id=?",
            (handle, problem["problem_id"]),
        ).fetchone()[0]
        # Negative ids are reserved for local terminal observations.
        submission_id = -int(timestamp * 1000 + prior_attempts)
        with connection:
            connection.execute(
                """INSERT INTO attempt_history(handle, problem_id, event_time, verdict, source, solution_metadata_json)
                   VALUES (?, ?, ?, ?, 'terminal', ?)""",
                (handle, problem["problem_id"], timestamp, normalized, json.dumps(solution_metadata or {})),
            )
            upsert_submission(
                connection,
                handle,
                {
                    "id": submission_id,
                    "contestId": problem.get("contest_id"),
                    "creationTimeSeconds": timestamp,
                    "problem": {
                        "contestId": problem.get("contest_id"),
                        "index": problem.get("problem_index"),
                        "name": problem.get("name"),
                        "rating": problem.get("official_rating"),
                        "tags": problem.get("tags", []),
                        "statement": problem.get("statement", ""),
                    },
                    "verdict": normalized,
                    "programmingLanguage": (solution_metadata or {}).get("language", "LOCAL"),
                    "attempt_number": prior_attempts + 1,
                },
                source="terminal",
            )
            rebuild_user_problem_history(connection, handle)
    finally:
        connection.close()
    after = estimate_all_skills(handle, db_path=db_path, now=timestamp)
    deltas = {
        topic_id: after[topic_id]["skill_score"] - before.get(topic_id, {}).get("skill_score", after[topic_id]["skill_score"])
        for topic_id in after
    }
    return {"verdict": normalized, "before": before, "after": after, "skill_deltas": deltas}
