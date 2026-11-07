"""Terminal-first matching/presentation layer for independent Models 1 and 2."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from problem_model.predict import predict_problem_difficulty  # noqa: E402
from user_model.collect_user_data import collect_handle  # noqa: E402
from user_model.config import DB_PATH  # noqa: E402
from user_model.database import connect, fetch_problem_with_tags, initialize_database  # noqa: E402
from user_model.feature_engineering import build_prediction_features  # noqa: E402
from user_model.llm_explanation import generate_explanation  # noqa: E402
from user_model.personalized_difficulty import personalized_difficulty  # noqa: E402
from user_model.predict_success import predict_success  # noqa: E402
from user_model.profile_manager import record_attempt, user_summary  # noqa: E402
from user_model.recommendation_engine import recommend_next_problems, similar_problems  # noqa: E402
from user_model.skill_estimator import get_skill_profiles  # noqa: E402
from user_model.skill_gap import analyze_skill_gaps  # noqa: E402


LOGGER = logging.getLogger("codeforces_learning_assistant")


def section(title: str) -> None:
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)


def percent(value: float) -> str:
    return f"{100.0 * max(0.0, min(1.0, value)):.0f}%"


def analyze_solution(path_text: str) -> dict[str, Any]:
    if not path_text.strip():
        return {}
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists() or not path.is_file():
        raise ValueError(f"Solution file not found: {path}")
    suffix_languages = {
        ".cpp": "C++", ".cc": "C++", ".c": "C", ".py": "Python",
        ".java": "Java", ".kt": "Kotlin", ".rs": "Rust", ".cs": "C#",
    }
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "language": suffix_languages.get(path.suffix.lower(), path.suffix.lstrip(".").upper() or "Unknown"),
        "bytes": path.stat().st_size,
        "lines": len(text.splitlines()),
        "non_empty_lines": sum(bool(line.strip()) for line in text.splitlines()),
    }


def _display_problem_before_attempt(problem: dict[str, Any]) -> None:
    section(f"LOADING PROBLEM #{problem['display_index']}")
    print(f"Name: {problem['name']}")
    if problem.get("contest_id") and problem.get("problem_index"):
        print(f"Codeforces ID: {problem['contest_id']}{problem['problem_index']}")
    if problem.get("url"):
        print(f"URL: {problem['url']}")
    print("\nProblem statement:\n")
    print(problem.get("statement") or "Statement unavailable in local/API metadata.")
    print("\nOfficial rating and topic analysis are hidden until the attempt is complete.")


def _save_prediction(
    handle: str,
    problem: dict[str, Any],
    success: dict[str, Any],
    personalized: dict[str, Any],
    features: dict[str, float],
    db_path: Path,
) -> None:
    connection = connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO model_predictions(
                    handle, problem_id, model_version, acceptance_probability,
                    prediction_confidence, personalized_rating, readiness_score,
                    features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    handle, problem["problem_id"], success["model_version"], success["probability"],
                    success["confidence"], personalized["personalized_rating"],
                    personalized["readiness_score"], json.dumps(features, sort_keys=True), int(time.time()),
                ),
            )
    finally:
        connection.close()


def run_flow(args: argparse.Namespace) -> int:
    section("CODEFORCES AI LEARNING SYSTEM")
    handle = args.handle or input("Enter Codeforces Handle:\n> ").strip()
    try:
        sync = collect_handle(handle, db_path=args.db, offline=args.offline)
    except (ValueError, LookupError) as exc:
        print(f"\nCould not load user: {exc}")
        return 2
    canonical_handle = sync["handle"]
    if sync.get("warning"):
        print(f"Offline/cached mode: {sync['warning']}")
    elif sync.get("offline"):
        print(f"Using local data for {canonical_handle}.")

    if args.problem_index is None:
        raw_problem_index = input("\nEnter Problem Index:\n> ").strip().lstrip("#")
        try:
            problem_index = int(raw_problem_index)
        except ValueError:
            print("Friendly error: Problem Index must be a positive integer from the local problem bank.")
            return 2
    else:
        problem_index = args.problem_index
    if problem_index <= 0:
        print("Friendly error: Problem Index must be positive.")
        return 2

    initialize_database(args.db)
    connection = connect(args.db)
    try:
        problem = fetch_problem_with_tags(connection, display_index=problem_index)
    finally:
        connection.close()
    if problem is None:
        print(
            f"Friendly error: Unknown problem index #{problem_index}. "
            "Choose an index present in data/cleaned/codeforces_cleaned.csv."
        )
        return 2
    if problem.get("official_rating") is None:
        print(f"Friendly error: Problem #{problem_index} has no official rating, so Model 2 cannot assess it.")
        return 2

    _display_problem_before_attempt(problem)
    if not args.non_interactive:
        input("\nPress ENTER after solving (or after making your best attempt).")
    solution_path = args.solution or (
        "" if args.non_interactive else input("Solution file path (optional):\n> ").strip()
    )
    try:
        solution_metadata = analyze_solution(solution_path)
    except ValueError as exc:
        print(f"Solution metadata warning: {exc}")
        solution_metadata = {}
    result = args.result or input(
        "Enter result (ACCEPTED, WRONG_ANSWER, TLE, RUNTIME_ERROR, etc.):\n> "
    ).strip()

    # Independent inference calls. Model 2 receives only official metadata.
    model1 = predict_problem_difficulty(problem.get("statement") or "", ";".join(problem["tags"]))
    get_skill_profiles(canonical_handle, db_path=args.db, refresh=True)
    skills = get_skill_profiles(canonical_handle, db_path=args.db)
    features, required_topics = build_prediction_features(
        canonical_handle, problem, db_path=args.db
    )
    success = predict_success(features)
    gaps = analyze_skill_gaps(required_topics, skills)
    summary = user_summary(canonical_handle, db_path=args.db)
    personalized = personalized_difficulty(
        official_rating=int(problem["official_rating"]),
        user_rating=int(summary.get("current_rating") or 0),
        acceptance_probability=success["probability"],
        prediction_confidence=success["confidence"],
        gaps=gaps,
        recent_success_rate=summary["recent_success_rate"],
    )
    _save_prediction(canonical_handle, problem, success, personalized, features, args.db)

    similar = similar_problems(problem, db_path=args.db, count=args.recommendation_count)
    recommendations = recommend_next_problems(
        canonical_handle,
        user_rating=int(summary.get("current_rating") or 0),
        target_problem=problem,
        gaps=gaps,
        mode=args.recommendation_mode,
        count=args.recommendation_count,
        db_path=args.db,
    )
    explanation_payload = {
        "official_rating": int(problem["official_rating"]),
        "predicted_rating": model1["predicted_rating"],
        "model1_confidence": model1["confidence"],
        "required_topics": {item["topic"]: item["relevance"] for item in required_topics},
        "user_skills": {item["topic"]: item["user_skill"] for item in gaps},
        "skill_gaps": gaps,
        "acceptance_probability": success["probability"],
        "personalized_rating": personalized["personalized_rating"],
        "readiness_score": personalized["readiness_score"],
        "recommendations": [
            {"problem_id": item["problem_id"], "rating": item["official_rating"],
             "why": item["why_recommended"]} for item in recommendations
        ],
    }
    explanation = generate_explanation(explanation_payload)

    section("PROBLEM ANALYSIS")
    print(f"Problem Index: #{problem['display_index']}")
    print(f"Official Rating: {problem['official_rating']}")
    print(f"AI Predicted Rating (Model 1): {model1['predicted_rating']}")
    print(f"Model 1 Reliability: {percent(model1['confidence'])}")
    print(f"Prediction Range: {model1['prediction_range'][0]}-{model1['prediction_range'][1]}")
    print(f"Difficulty: {model1['difficulty']}")
    print(f"Model 1 held-out MAE: {model1['empirical_mae']:.1f} ({model1['validation_samples']} samples)")

    section("REQUIRED TOPICS")
    print("Scores mean estimated relevance/importance, not required knowledge percentage.")
    for item in required_topics:
        print(f"{item['topic']:<32} {percent(item['relevance']):>5}")

    section("YOUR SKILLS")
    for item in gaps:
        print(
            f"{item['topic']:<32} {percent(item['user_skill']):>5}  "
            f"confidence {percent(item['skill_confidence'])}"
        )

    section("SKILL GAP")
    for item in gaps:
        print(
            f"{item['topic']}\n"
            f"  Required relevance: {percent(item['required_relevance'])}\n"
            f"  Your skill:         {percent(item['user_skill'])}\n"
            f"  Gap:                {percent(item['gap'])}\n"
            f"  Status:             {item['status']}"
        )

    section("PERSONALIZED ANALYSIS (MODEL 2)")
    print(f"Official Rating: {personalized['official_rating']}")
    print(f"Your Current Rating: {personalized['user_rating']}")
    print(f"Your Estimated Difficulty: ~{personalized['personalized_rating']}")
    print(f"Difficulty For You: {personalized['difficulty_for_user']}")
    print(f"Acceptance Probability: {percent(success['probability'])}")
    print(f"Probability Confidence: {percent(success['confidence'])}")
    print(f"Probability Source: {success['source']} / {success['selected_model']}")
    print(f"Readiness: {personalized['readiness_score']:.0f}/100 ({personalized['readiness_label']})")
    for reason in personalized["reasons"]:
        print(f"- {reason}")

    section("SOLUTION RESULT")
    print(f"Result: {result.strip().upper().replace(' ', '_')}")
    if solution_metadata:
        print(
            f"Solution metadata: {solution_metadata['language']}, {solution_metadata['lines']} lines, "
            f"{solution_metadata['bytes']} bytes"
        )
    try:
        update = record_attempt(
            canonical_handle, problem, result, solution_metadata=solution_metadata, db_path=args.db
        )
        changed = [
            (topic_id, delta) for topic_id, delta in update["skill_deltas"].items()
            if abs(delta) >= 0.0005 and topic_id in {item["topic_id"] for item in required_topics}
        ]
        if changed:
            print("Controlled profile update (recomputed from all evidence):")
            for topic_id, delta in sorted(changed, key=lambda item: -abs(item[1]))[:5]:
                name = update["after"][topic_id]["topic"]
                print(f"- {name}: {delta * 100:+.2f} percentage points")
        else:
            print("Profile recomputed; changes were below 0.05 percentage points.")
    except ValueError as exc:
        print(f"Result was not stored: {exc}")

    section("LLM / DETERMINISTIC EXPLANATION")
    print(f"Explanation source: {explanation['source']}")
    if explanation.get("warning"):
        print(explanation["warning"])
    print(explanation["text"])

    section("SIMILAR PROBLEMS")
    for item in similar:
        print(
            f"#{item.get('display_index') or '-':<4} Rating: {item['official_rating']:<4} "
            f"Similarity: {percent(item['similarity']):>4}  {item['name']}"
        )

    section("RECOMMENDED NEXT PROBLEMS")
    if not recommendations:
        print("No suitable unsolved problems were found in the safe progression window.")
    for item in recommendations:
        print(
            f"#{item.get('display_index') or '-':<4} {item['name']}\n"
            f"  Rating: {item['official_rating']} | Topics: {', '.join(item['tags']) or 'unlabelled'}\n"
            f"  Expected for you: {item['expected_difficulty']}\n"
            f"  Why this problem? {item['why_recommended']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codeforces AI Learning System (terminal first).")
    parser.add_argument("--handle", help="Codeforces handle; use sample_user offline.")
    parser.add_argument("--problem-index", type=int, help="1-based row in the local problem bank.")
    parser.add_argument("--result", help="Observed result, e.g. ACCEPTED or WRONG_ANSWER.")
    parser.add_argument("--solution", help="Optional local solution file path.")
    parser.add_argument("--offline", action="store_true", help="Never call the Codeforces API.")
    parser.add_argument("--non-interactive", action="store_true", help="Skip the solve wait; requires other inputs.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path.")
    parser.add_argument(
        "--recommendation-mode", default="skill_gap",
        choices=["skill_gap", "difficulty_progression", "topic_practice", "revision", "mixed_topic"],
    )
    parser.add_argument("--recommendation-count", type=int, default=5)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args()
    if args.non_interactive and (not args.handle or args.problem_index is None or not args.result):
        parser.error("--non-interactive requires --handle, --problem-index, and --result")
    return run_flow(args)


if __name__ == "__main__":
    raise SystemExit(main())
