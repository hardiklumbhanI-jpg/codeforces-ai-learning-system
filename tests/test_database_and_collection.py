import sqlite3

import pytest

from user_model.collect_user_data import CodeforcesAPIError, collect_handle, validate_handle
from user_model.database import connect, fetch_problem_with_tags, initialize_database


def test_schema_contains_required_tables_and_indexes(tmp_path):
    db = initialize_database(tmp_path / "test.db")
    connection = sqlite3.connect(db)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        connection.close()
    assert {
        "users", "problems", "problem_tags", "submissions", "rating_history",
        "user_problem_history", "topic_performance", "skill_profiles",
        "recommendations", "model_predictions", "attempt_history",
    } <= tables
    assert "idx_submissions_handle_time" in indexes
    assert "idx_problem_rating" in indexes


def test_repeated_sample_collection_is_deduplicated(sample_db):
    connection = connect(sample_db)
    before = connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
    connection.close()
    collect_handle("sample_user", db_path=sample_db, offline=True)
    connection = connect(sample_db)
    try:
        after = connection.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
        unique = connection.execute("SELECT COUNT(DISTINCT submission_id) FROM submissions").fetchone()[0]
    finally:
        connection.close()
    assert before == after == unique


def test_offline_sample_loaded_and_problem_lookup_works(sample_db):
    connection = connect(sample_db)
    try:
        user = connection.execute("SELECT * FROM users WHERE handle='sample_user'").fetchone()
        problem = fetch_problem_with_tags(connection, display_index=45)
    finally:
        connection.close()
    assert user["source"] == "offline_sample"
    assert problem is not None
    assert problem["official_rating"] is not None
    assert isinstance(problem["tags"], list)


def test_invalid_and_unknown_handles_are_friendly(tmp_path):
    with pytest.raises(ValueError, match="Invalid Codeforces handle"):
        validate_handle("!!")
    with pytest.raises(LookupError, match="No offline sample data"):
        collect_handle("unknown_user", db_path=tmp_path / "unknown.db", offline=True)


def test_api_failure_uses_sample_but_not_for_another_handle(tmp_path):
    class FailedClient:
        def request(self, *_args, **_kwargs):
            raise CodeforcesAPIError("network unavailable")

    result = collect_handle(
        "sample_user", db_path=tmp_path / "sample.db", client=FailedClient()
    )
    assert result["offline"] is True
    assert result["source"] == "offline_sample"
    with pytest.raises(LookupError, match="no cached local profile"):
        collect_handle("unknown_user", db_path=tmp_path / "other.db", client=FailedClient())
