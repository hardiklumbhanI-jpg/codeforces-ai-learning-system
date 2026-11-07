from user_model.database import connect, fetch_problem_with_tags
from user_model.feature_engineering import build_prediction_features
from user_model.personalized_difficulty import personalized_difficulty
from user_model.predict_success import predict_success
from user_model.profile_manager import record_attempt, user_summary
from user_model.recommendation_engine import recommend_next_problems, similar_problems
from user_model.skill_estimator import get_skill_profiles
from user_model.skill_gap import analyze_skill_gaps


def _analysis(sample_db):
    connection = connect(sample_db)
    try:
        problem = fetch_problem_with_tags(connection, display_index=45)
    finally:
        connection.close()
    skills = get_skill_profiles("sample_user", db_path=sample_db, refresh=True)
    features, requirements = build_prediction_features("sample_user", problem, db_path=sample_db)
    gaps = analyze_skill_gaps(requirements, skills)
    return problem, features, gaps


def test_success_prediction_fallback_is_bounded(sample_db, tmp_path):
    _, features, _ = _analysis(sample_db)
    result = predict_success(features, model_dir=tmp_path / "missing_models")
    assert 0.02 <= result["probability"] <= 0.98
    assert result["source"] == "documented_statistical_fallback"


def test_skill_gap_and_personalized_difficulty_are_explainable(sample_db):
    problem, features, gaps = _analysis(sample_db)
    success = predict_success(features, model_dir=sample_db.parent / "missing")
    result = personalized_difficulty(
        official_rating=problem["official_rating"],
        user_rating=int(features["current_rating"]),
        acceptance_probability=success["probability"],
        prediction_confidence=success["confidence"],
        gaps=gaps,
        recent_success_rate=features["prior_recent_success_rate"],
    )
    assert 0 <= result["readiness_score"] <= 100
    assert result["personalized_rating"] % 25 == 0
    assert result["reasons"]
    assert all(item["status"] in {"STRONG MATCH", "MODERATE GAP", "MAJOR GAP"} for item in gaps)


def test_similar_and_personalized_recommendations_exclude_target(sample_db):
    problem, features, gaps = _analysis(sample_db)
    similar = similar_problems(problem, db_path=sample_db, count=3)
    recs = recommend_next_problems(
        "sample_user", user_rating=int(features["current_rating"]), target_problem=problem,
        gaps=gaps, db_path=sample_db, count=3,
    )
    assert len(similar) == 3
    assert all(item["problem_id"] != problem["problem_id"] for item in similar)
    assert all(item["problem_id"] != problem["problem_id"] for item in recs)
    assert all(item["why_recommended"] for item in recs)


def test_record_attempt_updates_from_evidence_not_fixed_increment(sample_db):
    problem, _, _ = _analysis(sample_db)
    update = record_attempt(
        "sample_user", problem, "ACCEPTED", db_path=sample_db, timestamp=1787769000
    )
    assert update["verdict"] == "OK"
    assert update["after"]
    assert all(abs(delta) < 1 for delta in update["skill_deltas"].values())
