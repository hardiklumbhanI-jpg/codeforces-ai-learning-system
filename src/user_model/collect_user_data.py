"""Collect and cache public Codeforces data for the independent Model 2.

Public profile, rating, submission, and problem metadata endpoints do not need
credentials.  The client waits at least 2.1 seconds between calls, uses a
finite timeout, persists raw JSON, and falls back to cached/offline sample data
without assigning sample history to an unrelated handle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import (
    API_BASE_URL,
    API_CACHE_DIR,
    API_MIN_INTERVAL_SECONDS,
    API_TIMEOUT_SECONDS,
    DB_PATH,
    MAX_SUBMISSIONS,
    PROJECT_ROOT,
    SAMPLE_DATA_DIR,
    ensure_directories,
)
from .database import (
    connect,
    initialize_database,
    rebuild_user_problem_history,
    upsert_problem,
    upsert_rating_history,
    upsert_submission,
    upsert_user,
)


LOGGER = logging.getLogger(__name__)
HANDLE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,24}$")


class CodeforcesAPIError(RuntimeError):
    """A recoverable Codeforces transport or API-status failure."""


def validate_handle(handle: str) -> str:
    normalized = str(handle).strip()
    if not HANDLE_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Invalid Codeforces handle format. Use 3-24 letters, digits, '.', '_' or '-'."
        )
    return normalized


class CodeforcesClient:
    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: float = API_TIMEOUT_SECONDS,
        min_interval: float = API_MIN_INTERVAL_SECONDS,
        cache_dir: Path | str = API_CACHE_DIR,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_at = 0.0

    def _cache_path(self, method: str, params: dict[str, Any]) -> Path:
        query = urllib.parse.urlencode(sorted((key, str(value)) for key, value in params.items()))
        digest = hashlib.sha256(f"{method}?{query}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{method.replace('.', '_')}_{digest}.json"

    def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        allow_cache_on_failure: bool = True,
    ) -> Any:
        cache_path = self._cache_path(method, params)
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        query = urllib.parse.urlencode(params)
        url = f"{self.base_url}/{method}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "codeforces-learning-assistant/2"})
        try:
            self._last_request_at = time.monotonic()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") != "OK":
                raise CodeforcesAPIError(payload.get("comment", "Codeforces returned FAILED."))
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload.get("result")
        except (OSError, ValueError, urllib.error.URLError, CodeforcesAPIError) as exc:
            if allow_cache_on_failure and cache_path.exists():
                LOGGER.warning("Codeforces API unavailable; using cached %s data: %s", method, exc)
                return json.loads(cache_path.read_text(encoding="utf-8")).get("result")
            raise CodeforcesAPIError(f"{method} failed: {exc}") from exc


def import_local_problem_bank(connection, csv_path: Path | None = None) -> int:
    """Load Model 1's local problem CSV as metadata only; no Model 1 output is used."""
    path = csv_path or PROJECT_ROOT / "data" / "cleaned" / "codeforces_cleaned.csv"
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for display_index, row in enumerate(csv.DictReader(file), start=1):
            row["tags"] = row.get("tags", "")
            row["source"] = "local_problem_bank"
            upsert_problem(connection, row, display_index=display_index, source="local_problem_bank")
            count += 1
    return count


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_offline_sample(connection, handle: str = "sample_user") -> dict[str, Any]:
    users = _csv_rows(SAMPLE_DATA_DIR / "sample_users.csv")
    selected = next((row for row in users if row.get("handle", "").lower() == handle.lower()), None)
    if selected is None:
        raise LookupError(f"No offline sample data exists for handle '{handle}'.")

    now = int(time.time())
    user = {
        "handle": selected["handle"],
        "rating": int(selected["current_rating"]),
        "maxRating": int(selected["max_rating"]),
        "rank": selected.get("rank") or None,
        "maxRank": selected.get("max_rank") or None,
        "registrationTimeSeconds": int(selected.get("registration_time") or 0) or None,
        "contribution": int(selected.get("contribution") or 0),
        "friendOfCount": int(selected.get("friend_count") or 0),
        "source": "offline_sample",
        "last_synced_at": now,
    }
    upsert_user(connection, user)

    for row in _csv_rows(SAMPLE_DATA_DIR / "sample_problems.csv"):
        upsert_problem(
            connection,
            {
                "contestId": int(row["contest_id"]),
                "index": row["problem_index"],
                "name": row["name"],
                "rating": int(row["rating"]),
                "tags": row["tags"],
                "statement": row.get("statement", ""),
                "url": row.get("url", ""),
            },
            source="offline_sample",
        )

    submissions = 0
    for row in _csv_rows(SAMPLE_DATA_DIR / "sample_submissions.csv"):
        if row.get("handle", "").lower() != selected["handle"].lower():
            continue
        problem = {
            "contestId": int(row["contest_id"]),
            "index": row["problem_index"],
            "name": row["problem_name"],
            "rating": int(row["problem_rating"]),
            "tags": row["tags"].split(";"),
        }
        upsert_submission(
            connection,
            selected["handle"],
            {
                "id": int(row["submission_id"]),
                "contestId": int(row["contest_id"]),
                "creationTimeSeconds": int(row["creation_time"]),
                "relativeTimeSeconds": int(row.get("relative_time") or 0),
                "problem": problem,
                "verdict": row["verdict"],
                "programmingLanguage": row.get("language", "GNU C++17"),
                "passedTestCount": int(row.get("passed_test_count") or 0),
                "attempt_number": int(row.get("attempt_number") or 1),
            },
            source="offline_sample",
        )
        submissions += 1

    changes = []
    for row in _csv_rows(SAMPLE_DATA_DIR / "sample_rating_history.csv"):
        if row.get("handle", "").lower() != selected["handle"].lower():
            continue
        changes.append({
            "contestId": int(row["contest_id"]),
            "contestName": row["contest_name"],
            "rank": int(row["rank"]),
            "ratingUpdateTimeSeconds": int(row["rating_update_time"]),
            "oldRating": int(row["old_rating"]),
            "newRating": int(row["new_rating"]),
        })
    upsert_rating_history(connection, selected["handle"], changes)
    rebuild_user_problem_history(connection, selected["handle"])
    return {"handle": selected["handle"], "source": "offline_sample", "submissions": submissions}


def sync_problemset(client: CodeforcesClient, connection) -> int:
    payload = client.request("problemset.problems", {})
    statistics = {
        (item.get("contestId"), item.get("index")): item.get("solvedCount")
        for item in payload.get("problemStatistics", [])
    }
    count = 0
    for problem in payload.get("problems", []):
        problem = dict(problem)
        problem["solvedCount"] = statistics.get((problem.get("contestId"), problem.get("index")))
        upsert_problem(connection, problem, source="api")
        count += 1
    return count


def collect_handle(
    handle: str,
    *,
    db_path: Path | str = DB_PATH,
    offline: bool = False,
    refresh_problemset: bool = False,
    client: CodeforcesClient | None = None,
) -> dict[str, Any]:
    handle = validate_handle(handle)
    initialize_database(db_path)
    connection = connect(db_path)
    try:
        with connection:
            local_problem_count = import_local_problem_bank(connection)
            # This exact handle is reserved for the bundled deterministic fixture,
            # even though an unrelated public Codeforces account may share the name.
            if handle.lower() == "sample_user":
                sample = load_offline_sample(connection, "sample_user")
                sample.update({"offline": True, "local_problems": local_problem_count})
                return sample
            if offline:
                existing = connection.execute(
                    "SELECT handle, source FROM users WHERE handle = ?", (handle,)
                ).fetchone()
                if existing:
                    rebuild_user_problem_history(connection, existing["handle"])
                    return {
                        "handle": existing["handle"], "source": existing["source"],
                        "offline": True, "local_problems": local_problem_count,
                    }
                sample = load_offline_sample(connection, handle)
                sample.update({"offline": True, "local_problems": local_problem_count})
                return sample

            client = client or CodeforcesClient()
            try:
                profile = client.request("user.info", {"handles": handle})
                if not profile:
                    raise CodeforcesAPIError(f"Codeforces returned no profile for '{handle}'.")
                user = dict(profile[0])
                canonical_handle = user.get("handle", handle)
                user.update({"source": "api", "last_synced_at": int(time.time())})
                upsert_user(connection, user)

                rating_changes = client.request("user.rating", {"handle": canonical_handle})
                upsert_rating_history(connection, canonical_handle, rating_changes)
                submissions = client.request(
                    "user.status", {"handle": canonical_handle, "from": 1, "count": MAX_SUBMISSIONS}
                )
                for submission in submissions:
                    upsert_submission(connection, canonical_handle, submission, source="api")
                if refresh_problemset:
                    sync_problemset(client, connection)
                rebuild_user_problem_history(connection, canonical_handle)
                return {
                    "handle": canonical_handle, "source": "api", "offline": False,
                    "submissions": len(submissions), "rating_changes": len(rating_changes),
                    "local_problems": local_problem_count,
                }
            except CodeforcesAPIError as exc:
                existing = connection.execute(
                    "SELECT handle, source FROM users WHERE handle = ?", (handle,)
                ).fetchone()
                if existing:
                    rebuild_user_problem_history(connection, existing["handle"])
                    return {
                        "handle": existing["handle"], "source": existing["source"],
                        "offline": True, "warning": str(exc), "local_problems": local_problem_count,
                    }
                if handle.lower() == "sample_user":
                    sample = load_offline_sample(connection, handle)
                    sample.update({"offline": True, "warning": str(exc), "local_problems": local_problem_count})
                    return sample
                raise LookupError(
                    f"Could not load '{handle}' from Codeforces and no cached local profile exists. "
                    "Check the handle/internet connection, or use 'sample_user' for offline mode."
                ) from exc
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public Codeforces data for Model 2.")
    parser.add_argument("handle")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh-problemset", action="store_true")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        result = collect_handle(
            args.handle, db_path=args.db, offline=args.offline,
            refresh_problemset=args.refresh_problemset,
        )
    except (ValueError, LookupError) as exc:
        raise SystemExit(f"Data collection stopped: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
