"""Build the time-safe Model 2 interaction dataset from SQLite."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .config import DB_PATH, USER_DATA_DIR
from .database import connect, initialize_database
from .feature_engineering import FEATURE_COLUMNS, time_safe_feature_rows
from .problem_topics import required_topic_scores


def _rating_as_of(changes: list[dict[str, Any]], timestamp: int, fallback: int) -> tuple[int, int, float, float, int]:
    previous = [item for item in changes if (item.get("rating_update_time") or 0) < timestamp]
    if not previous:
        return fallback, fallback, 0.0, 0.0, 0
    current = previous[-1]["new_rating"]
    maximum = max(item["new_rating"] for item in previous)
    recent_change = previous[-1]["new_rating"] - previous[-1]["old_rating"]
    window = previous[-5:]
    trend = (window[-1]["new_rating"] - window[0]["old_rating"]) / max(1, len(window))
    return current, maximum, trend, recent_change, len(previous)


def load_interactions(db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT h.*, p.official_rating, p.statement, u.current_rating AS final_rating,
                   u.max_rating AS final_max_rating,
                   GROUP_CONCAT(t.tag, ';') AS tags
            FROM user_problem_history h
            JOIN problems p ON p.problem_id=h.problem_id
            JOIN users u ON u.handle=h.handle
            LEFT JOIN problem_tags t ON t.problem_id=p.problem_id
            WHERE p.official_rating IS NOT NULL
            GROUP BY h.handle, h.problem_id
            ORDER BY h.handle, h.first_attempt_time
            """
        ).fetchall()
        rating_by_user = {}
        for user in connection.execute("SELECT handle FROM users"):
            rating_by_user[user["handle"].lower()] = [
                dict(item) for item in connection.execute(
                    "SELECT * FROM rating_history WHERE handle=? ORDER BY rating_update_time", (user["handle"],)
                )
            ]
        interactions = []
        for row in rows:
            item = dict(row)
            tags = [tag for tag in (item.get("tags") or "").split(";") if tag]
            current, maximum, trend, recent_change, contests = _rating_as_of(
                rating_by_user.get(item["handle"].lower(), []),
                item["first_attempt_time"],
                item.get("final_rating") or 1200,
            )
            interactions.append({
                "handle": item["handle"], "problem_id": item["problem_id"],
                "timestamp": item["first_attempt_time"], "accepted": item["accepted"],
                "attempts": item["attempts"], "wrong_answers": item["wrong_answers"],
                "tle_count": item["tle_count"], "problem_rating": item["official_rating"],
                "statement": item.get("statement") or "", "tags": tags,
                "required_topics": required_topic_scores(item.get("statement") or "", tags),
                "current_rating": current, "max_rating": maximum,
                "rating_trend": trend, "recent_rating_change": recent_change,
                "number_of_contests": contests,
            })
        return interactions
    finally:
        connection.close()


def build_dataset(db_path: Path | str = DB_PATH) -> list[dict[str, Any]]:
    return time_safe_feature_rows(load_interactions(db_path))


def write_dataset(rows: list[dict[str, Any]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["handle", "problem_id", "timestamp", *FEATURE_COLUMNS, "accepted"]
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe Model 2 interactions.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=USER_DATA_DIR / "user_interactions.csv")
    args = parser.parse_args()
    rows = build_dataset(args.db)
    write_dataset(rows, args.output)
    print(f"Wrote {len(rows)} time-ordered interactions to {args.output}")


if __name__ == "__main__":
    main()
