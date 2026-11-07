"""Persist the configured difficulty-weighted statistical skill model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from .config import DB_PATH, MODEL_DIR, ensure_directories
from .database import connect, initialize_database
from .skill_estimator import estimate_all_skills
from .topic_skill import DifficultyWeightedSkillModel


def train(db_path: Path | str = DB_PATH) -> Path:
    ensure_directories()
    initialize_database(db_path)
    model = DifficultyWeightedSkillModel()
    connection = connect(db_path)
    try:
        handles = [row["handle"] for row in connection.execute("SELECT handle FROM users")]
    finally:
        connection.close()
    profile_counts = {}
    for handle in handles:
        profile_counts[handle] = len(estimate_all_skills(handle, db_path=db_path, model=model))
    path = MODEL_DIR / "skill_model.pkl"
    joblib.dump(model, path)
    (MODEL_DIR / "skill_model_metadata.json").write_text(
        json.dumps({
            "model": "regularized 1PL item-response skill estimator",
            "parameters": model.get_params(),
            "profiles_built": profile_counts,
        }, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Model 2 skill profiles and artifact.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()
    print(f"Saved skill model to {train(args.db)}")


if __name__ == "__main__":
    main()
