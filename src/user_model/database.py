"""SQLite persistence for Model 2.

The schema is intentionally independent from Model 1's CSVs and artifacts.
All writes use stable primary keys or UPSERTs, so collection is resumable and
repeated API calls do not duplicate submissions, ratings, problems, or tags.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .config import DB_PATH, ensure_directories


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS users (
    handle TEXT PRIMARY KEY COLLATE NOCASE,
    current_rating INTEGER,
    max_rating INTEGER,
    rank TEXT,
    max_rank TEXT,
    registration_time INTEGER,
    contribution INTEGER,
    friend_count INTEGER,
    last_online_time INTEGER,
    avatar TEXT,
    title_photo TEXT,
    source TEXT NOT NULL DEFAULT 'api',
    last_synced_at INTEGER,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS problems (
    problem_id TEXT PRIMARY KEY,
    contest_id INTEGER,
    problem_index TEXT,
    display_index INTEGER UNIQUE,
    name TEXT NOT NULL,
    statement TEXT,
    official_rating INTEGER,
    problem_type TEXT,
    points REAL,
    solved_count INTEGER,
    contest_name TEXT,
    url TEXT,
    source TEXT NOT NULL DEFAULT 'api',
    raw_json TEXT,
    UNIQUE(contest_id, problem_index)
);

CREATE TABLE IF NOT EXISTS problem_tags (
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY(problem_id, tag)
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id INTEGER PRIMARY KEY,
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    contest_id INTEGER,
    problem_index TEXT,
    creation_time INTEGER NOT NULL,
    relative_time INTEGER,
    verdict TEXT,
    language TEXT,
    passed_test_count INTEGER,
    time_consumed_ms INTEGER,
    memory_consumed_bytes INTEGER,
    attempt_number INTEGER,
    accepted INTEGER NOT NULL DEFAULT 0 CHECK(accepted IN (0, 1)),
    wrong_answer INTEGER NOT NULL DEFAULT 0 CHECK(wrong_answer IN (0, 1)),
    tle INTEGER NOT NULL DEFAULT 0 CHECK(tle IN (0, 1)),
    runtime_error INTEGER NOT NULL DEFAULT 0 CHECK(runtime_error IN (0, 1)),
    source TEXT NOT NULL DEFAULT 'api',
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS rating_history (
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    contest_id INTEGER NOT NULL,
    contest_name TEXT,
    rank INTEGER,
    rating_update_time INTEGER,
    old_rating INTEGER,
    new_rating INTEGER,
    PRIMARY KEY(handle, contest_id)
);

CREATE TABLE IF NOT EXISTS user_problem_history (
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    first_attempt_time INTEGER,
    last_attempt_time INTEGER,
    attempts INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0 CHECK(accepted IN (0, 1)),
    accepted_time INTEGER,
    wrong_answers INTEGER NOT NULL DEFAULT 0,
    tle_count INTEGER NOT NULL DEFAULT 0,
    runtime_error_count INTEGER NOT NULL DEFAULT 0,
    other_failure_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(handle, problem_id)
);

CREATE TABLE IF NOT EXISTS attempt_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    event_time INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'terminal',
    solution_metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS topic_performance (
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    difficulty_bucket INTEGER NOT NULL,
    attempted REAL NOT NULL,
    solved REAL NOT NULL,
    acceptance_rate REAL NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(handle, topic_id, difficulty_bucket)
);

CREATE TABLE IF NOT EXISTS skill_profiles (
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    topic_id TEXT NOT NULL,
    skill_score REAL NOT NULL,
    confidence REAL NOT NULL,
    ability_rating REAL NOT NULL,
    attempted REAL NOT NULL,
    solved REAL NOT NULL,
    recent_performance REAL,
    long_term_performance REAL,
    avg_problem_rating REAL,
    max_solved_rating INTEGER,
    wrong_answer_rate REAL,
    tle_rate REAL,
    average_attempts REAL,
    consistency REAL,
    last_practiced_at INTEGER,
    evidence_json TEXT,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY(handle, topic_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    score REAL NOT NULL,
    reason TEXT NOT NULL,
    expected_difficulty TEXT,
    created_at INTEGER NOT NULL,
    UNIQUE(handle, problem_id, mode)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    handle TEXT NOT NULL COLLATE NOCASE REFERENCES users(handle) ON DELETE CASCADE,
    problem_id TEXT NOT NULL REFERENCES problems(problem_id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    acceptance_probability REAL NOT NULL,
    prediction_confidence REAL,
    personalized_rating INTEGER,
    readiness_score REAL,
    features_json TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_submissions_handle_time
    ON submissions(handle, creation_time DESC);
CREATE INDEX IF NOT EXISTS idx_submissions_problem
    ON submissions(problem_id, creation_time);
CREATE INDEX IF NOT EXISTS idx_submissions_handle_verdict
    ON submissions(handle, verdict);
CREATE INDEX IF NOT EXISTS idx_rating_handle_time
    ON rating_history(handle, rating_update_time);
CREATE INDEX IF NOT EXISTS idx_history_handle_time
    ON user_problem_history(handle, last_attempt_time DESC);
CREATE INDEX IF NOT EXISTS idx_problem_rating
    ON problems(official_rating);
CREATE INDEX IF NOT EXISTS idx_problem_tag_tag
    ON problem_tags(tag, problem_id);
CREATE INDEX IF NOT EXISTS idx_skill_handle_score
    ON skill_profiles(handle, skill_score DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_handle_time
    ON model_predictions(handle, created_at DESC);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def initialize_database(db_path: Path | str = DB_PATH) -> Path:
    ensure_directories()
    path = Path(db_path)
    with connect(path) as connection:
        connection.executescript(SCHEMA)
    return path


@contextmanager
def transaction(db_path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def upsert_user(connection: sqlite3.Connection, user: Mapping[str, Any]) -> None:
    connection.execute(
        """
        INSERT INTO users (
            handle, current_rating, max_rating, rank, max_rank,
            registration_time, contribution, friend_count, last_online_time,
            avatar, title_photo, source, last_synced_at, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(handle) DO UPDATE SET
            current_rating=COALESCE(excluded.current_rating, users.current_rating),
            max_rating=COALESCE(excluded.max_rating, users.max_rating),
            rank=COALESCE(excluded.rank, users.rank),
            max_rank=COALESCE(excluded.max_rank, users.max_rank),
            registration_time=COALESCE(excluded.registration_time, users.registration_time),
            contribution=COALESCE(excluded.contribution, users.contribution),
            friend_count=COALESCE(excluded.friend_count, users.friend_count),
            last_online_time=COALESCE(excluded.last_online_time, users.last_online_time),
            avatar=COALESCE(excluded.avatar, users.avatar),
            title_photo=COALESCE(excluded.title_photo, users.title_photo),
            source=excluded.source,
            last_synced_at=excluded.last_synced_at,
            raw_json=excluded.raw_json
        """,
        (
            user.get("handle"), user.get("rating"), user.get("maxRating"),
            user.get("rank"), user.get("maxRank"), user.get("registrationTimeSeconds"),
            user.get("contribution"), user.get("friendOfCount"),
            user.get("lastOnlineTimeSeconds"), user.get("avatar"),
            user.get("titlePhoto"), user.get("source", "api"),
            user.get("last_synced_at"), _json(user.get("raw", user)),
        ),
    )


def stable_problem_id(contest_id: Any, problem_index: Any, fallback: Any = None) -> str:
    if contest_id not in (None, "") and problem_index not in (None, ""):
        return f"{int(contest_id)}-{str(problem_index).strip()}"
    return str(fallback or problem_index or "unknown").strip()


def upsert_problem(
    connection: sqlite3.Connection,
    problem: Mapping[str, Any],
    display_index: int | None = None,
    source: str = "api",
) -> str:
    contest_id = problem.get("contestId", problem.get("contest_id"))
    index = problem.get("index", problem.get("problem_index"))
    problem_id = problem.get("problem_id") or stable_problem_id(contest_id, index, problem.get("name"))
    url = problem.get("url")
    if not url and contest_id is not None and index:
        url = f"https://codeforces.com/problemset/problem/{contest_id}/{index}"
    connection.execute(
        """
        INSERT INTO problems (
            problem_id, contest_id, problem_index, display_index, name,
            statement, official_rating, problem_type, points, solved_count,
            contest_name, url, source, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(problem_id) DO UPDATE SET
            name=COALESCE(NULLIF(excluded.name, ''), problems.name),
            statement=COALESCE(NULLIF(excluded.statement, ''), problems.statement),
            official_rating=COALESCE(excluded.official_rating, problems.official_rating),
            problem_type=COALESCE(excluded.problem_type, problems.problem_type),
            points=COALESCE(excluded.points, problems.points),
            solved_count=COALESCE(excluded.solved_count, problems.solved_count),
            contest_name=COALESCE(excluded.contest_name, problems.contest_name),
            url=COALESCE(excluded.url, problems.url),
            display_index=COALESCE(problems.display_index, excluded.display_index),
            raw_json=excluded.raw_json
        """,
        (
            problem_id, contest_id, index, display_index, problem.get("name", problem_id),
            problem.get("statement"), problem.get("rating", problem.get("official_rating")),
            problem.get("type", problem.get("problem_type")), problem.get("points"),
            problem.get("solvedCount", problem.get("solved_count")),
            problem.get("contest_name"), url, source, _json(problem.get("raw", problem)),
        ),
    )
    tags = problem.get("tags") or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.replace(",", ";").split(";") if tag.strip()]
    connection.executemany(
        "INSERT OR IGNORE INTO problem_tags(problem_id, tag) VALUES (?, ?)",
        [(problem_id, str(tag).strip().lower()) for tag in tags if str(tag).strip()],
    )
    return problem_id


def verdict_flags(verdict: str | None) -> tuple[int, int, int, int]:
    normalized = (verdict or "UNKNOWN").upper()
    return (
        int(normalized == "OK" or normalized == "ACCEPTED"),
        int(normalized == "WRONG_ANSWER"),
        int(normalized == "TIME_LIMIT_EXCEEDED" or normalized == "TLE"),
        int(normalized == "RUNTIME_ERROR"),
    )


def upsert_submission(
    connection: sqlite3.Connection,
    handle: str,
    submission: Mapping[str, Any],
    source: str = "api",
) -> int:
    problem = submission.get("problem", {})
    problem_id = upsert_problem(connection, problem, source=source)
    verdict = str(submission.get("verdict") or "UNKNOWN").upper()
    accepted, wrong_answer, tle, runtime_error = verdict_flags(verdict)
    submission_id = int(submission.get("id", submission.get("submission_id")))
    connection.execute(
        """
        INSERT INTO submissions (
            submission_id, handle, problem_id, contest_id, problem_index,
            creation_time, relative_time, verdict, language, passed_test_count,
            time_consumed_ms, memory_consumed_bytes, attempt_number,
            accepted, wrong_answer, tle, runtime_error, source, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(submission_id) DO UPDATE SET
            verdict=excluded.verdict,
            passed_test_count=excluded.passed_test_count,
            accepted=excluded.accepted,
            wrong_answer=excluded.wrong_answer,
            tle=excluded.tle,
            runtime_error=excluded.runtime_error,
            raw_json=excluded.raw_json
        """,
        (
            submission_id, handle, problem_id,
            submission.get("contestId", problem.get("contestId")), problem.get("index"),
            int(submission.get("creationTimeSeconds", submission.get("creation_time", 0))),
            submission.get("relativeTimeSeconds", submission.get("relative_time")),
            verdict, submission.get("programmingLanguage", submission.get("language")),
            submission.get("passedTestCount", submission.get("passed_test_count")),
            submission.get("timeConsumedMillis", submission.get("time_consumed_ms")),
            submission.get("memoryConsumedBytes", submission.get("memory_consumed_bytes")),
            submission.get("attempt_number"), accepted, wrong_answer, tle, runtime_error,
            source, _json(submission.get("raw", submission)),
        ),
    )
    return submission_id


def upsert_rating_history(
    connection: sqlite3.Connection,
    handle: str,
    changes: Iterable[Mapping[str, Any]],
) -> None:
    connection.executemany(
        """
        INSERT INTO rating_history (
            handle, contest_id, contest_name, rank, rating_update_time,
            old_rating, new_rating
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(handle, contest_id) DO UPDATE SET
            contest_name=excluded.contest_name, rank=excluded.rank,
            rating_update_time=excluded.rating_update_time,
            old_rating=excluded.old_rating, new_rating=excluded.new_rating
        """,
        [
            (
                handle, item.get("contestId"), item.get("contestName"), item.get("rank"),
                item.get("ratingUpdateTimeSeconds"), item.get("oldRating"), item.get("newRating"),
            )
            for item in changes
            if item.get("contestId") is not None
        ],
    )


def rebuild_user_problem_history(connection: sqlite3.Connection, handle: str) -> None:
    connection.execute("DELETE FROM user_problem_history WHERE handle = ?", (handle,))
    connection.execute(
        """
        INSERT INTO user_problem_history (
            handle, problem_id, first_attempt_time, last_attempt_time, attempts,
            accepted, accepted_time, wrong_answers, tle_count,
            runtime_error_count, other_failure_count
        )
        SELECT
            handle, problem_id, MIN(creation_time), MAX(creation_time), COUNT(*),
            MAX(accepted), MIN(CASE WHEN accepted = 1 THEN creation_time END),
            SUM(wrong_answer), SUM(tle), SUM(runtime_error),
            SUM(CASE WHEN accepted=0 AND wrong_answer=0 AND tle=0 AND runtime_error=0 THEN 1 ELSE 0 END)
        FROM submissions
        WHERE handle = ?
        GROUP BY handle, problem_id
        """,
        (handle,),
    )


def fetch_problem_with_tags(
    connection: sqlite3.Connection,
    *,
    display_index: int | None = None,
    problem_id: str | None = None,
) -> dict[str, Any] | None:
    if display_index is not None:
        row = connection.execute(
            "SELECT * FROM problems WHERE display_index = ?", (display_index,)
        ).fetchone()
    else:
        row = connection.execute("SELECT * FROM problems WHERE problem_id = ?", (problem_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["tags"] = [
        item["tag"] for item in connection.execute(
            "SELECT tag FROM problem_tags WHERE problem_id = ? ORDER BY tag", (result["problem_id"],)
        )
    ]
    return result
