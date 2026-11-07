import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model1_runs_independently():
    result = subprocess.run(
        [
            sys.executable, "-m", "src.problem_model.predict",
            "--statement", "Given an array, sort it and find the maximum value.",
            "--tags", "implementation;sortings",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "predicted_rating" in result.stdout


def test_terminal_offline_acceptance_flow(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "main.py", "--offline", "--non-interactive",
            "--handle", "sample_user", "--problem-index", "45",
            "--result", "WRONG_ANSWER", "--db", str(tmp_path / "flow.db"),
            "--recommendation-count", "2",
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for heading in (
        "PROBLEM ANALYSIS", "REQUIRED TOPICS", "YOUR SKILLS", "SKILL GAP",
        "PERSONALIZED ANALYSIS", "Acceptance Probability", "Readiness",
        "SIMILAR PROBLEMS", "RECOMMENDED NEXT PROBLEMS",
    ):
        assert heading in result.stdout


def test_terminal_invalid_problem_is_friendly(tmp_path):
    result = subprocess.run(
        [
            sys.executable, "main.py", "--offline", "--non-interactive",
            "--handle", "sample_user", "--problem-index", "999999",
            "--result", "ACCEPTED", "--db", str(tmp_path / "invalid.db"),
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 2
    assert "Friendly error: Unknown problem index" in result.stdout
