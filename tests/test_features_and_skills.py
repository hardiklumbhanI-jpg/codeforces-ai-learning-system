import math

from user_model.database import connect, initialize_database, upsert_user
from user_model.feature_engineering import FEATURE_COLUMNS, time_safe_feature_rows
from user_model.skill_estimator import estimate_all_skills, get_skill_profiles
from user_model.topic_skill import DifficultyWeightedSkillModel


def test_time_safe_features_do_not_use_current_or_future_outcome():
    interactions = [
        {
            "handle": "u", "problem_id": "1-A", "timestamp": 100,
            "accepted": 1, "attempts": 1, "wrong_answers": 0, "tle_count": 0,
            "problem_rating": 1000, "current_rating": 1000, "max_rating": 1000,
            "tags": ["implementation"], "statement": "simulate an array",
            "rating_trend": 0, "recent_rating_change": 0, "number_of_contests": 0,
        },
        {
            "handle": "u", "problem_id": "2-A", "timestamp": 200,
            "accepted": 0, "attempts": 1, "wrong_answers": 1, "tle_count": 0,
            "problem_rating": 1200, "current_rating": 1000, "max_rating": 1000,
            "tags": ["implementation"], "statement": "simulate an array",
            "rating_trend": 0, "recent_rating_change": 0, "number_of_contests": 0,
        },
    ]
    rows = time_safe_feature_rows(interactions)
    assert rows[0]["prior_attempted"] == 0
    assert rows[0]["prior_solved"] == 0
    assert rows[1]["prior_attempted"] == 1
    assert rows[1]["prior_solved"] == 1
    assert all(column in rows[0] for column in FEATURE_COLUMNS)
    assert "predicted_rating" not in FEATURE_COLUMNS


def test_skill_confidence_grows_with_evidence():
    model = DifficultyWeightedSkillModel()
    two = [
        {"timestamp": 1_000_000 + i, "rating": 1400, "accepted": i % 2,
         "relevance": 1.0, "attempts": 1, "wrong_answers": 0, "tle_count": 0}
        for i in range(2)
    ]
    twenty = two * 10
    low = model.estimate(two, prior_rating=1400, now=1_000_100)
    high = model.estimate(twenty, prior_rating=1400, now=1_000_100)
    assert 0 <= low["skill_score"] <= 1
    assert high["confidence"] > low["confidence"]


def test_difficulty_weighting_rewards_harder_success():
    model = DifficultyWeightedSkillModel()
    easy = [{"timestamp": 1000, "rating": 1000, "accepted": 1, "relevance": 1,
             "attempts": 1, "wrong_answers": 0, "tle_count": 0}]
    hard = [{"timestamp": 1000, "rating": 2000, "accepted": 1, "relevance": 1,
             "attempts": 1, "wrong_answers": 0, "tle_count": 0}]
    easy_result = model.estimate(easy, prior_rating=1400, now=1000)
    hard_result = model.estimate(hard, prior_rating=1400, now=1000)
    assert hard_result["ability_rating"] > easy_result["ability_rating"]


def test_skill_decay_is_controlled_and_documented():
    model = DifficultyWeightedSkillModel()
    observations = [{"timestamp": 1000, "rating": 1600, "accepted": 1, "relevance": 1,
                     "attempts": 1, "wrong_answers": 0, "tle_count": 0}]
    fresh = model.estimate(observations, prior_rating=1400, now=1000)
    old = model.estimate(observations, prior_rating=1400, now=1000 + 730 * 86400)
    assert old["skill_score"] < fresh["skill_score"]
    assert old["evidence"]["decay_multiplier"] >= model.parameters.decay_floor


def test_sample_skill_profile_and_empty_history(sample_db, tmp_path):
    profiles = estimate_all_skills("sample_user", db_path=sample_db, now=1787769000)
    assert profiles
    assert all(0 <= item["skill_score"] <= 1 for item in profiles.values())
    empty_db = initialize_database(tmp_path / "empty.db")
    connection = connect(empty_db)
    with connection:
        upsert_user(connection, {"handle": "empty_user", "rating": 1200, "maxRating": 1200})
    connection.close()
    assert estimate_all_skills("empty_user", db_path=empty_db) == {}
