"""Paths and documented Model 2 parameters."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
USER_DATA_DIR = PROJECT_ROOT / "data" / "user_model"
SAMPLE_DATA_DIR = USER_DATA_DIR / "sample"
API_CACHE_DIR = USER_DATA_DIR / "cache"
DB_PATH = USER_DATA_DIR / "codeforces_users.db"
KNOWLEDGE_GRAPH_PATH = USER_DATA_DIR / "knowledge_graph.json"
MODEL_DIR = PROJECT_ROOT / "models" / "user_model"
RESULTS_DIR = PROJECT_ROOT / "results" / "user_model"

# Codeforces documents a maximum of one API request per two seconds.
API_BASE_URL = "https://codeforces.com/api"
API_MIN_INTERVAL_SECONDS = 2.1
API_TIMEOUT_SECONDS = 12
MAX_SUBMISSIONS = 10_000

# Statistical skill estimator configuration.  See README.md for formulas.
RECENCY_HALF_LIFE_DAYS = 90.0
RECENCY_LONG_TERM_FLOOR = 0.35
DECAY_GRACE_DAYS = 90.0
DECAY_HALF_LIFE_DAYS = 365.0
DECAY_FLOOR = 0.65
SKILL_MIN_RATING = 800.0
SKILL_MAX_RATING = 2800.0
SKILL_PRIOR_STRENGTH = 2.5
CONFIDENCE_EVIDENCE_SCALE = 8.0

RATING_BUCKETS = tuple(range(800, 3600, 200))


def ensure_directories() -> None:
    """Create only Model 2-owned directories."""
    for path in (USER_DATA_DIR, SAMPLE_DATA_DIR, API_CACHE_DIR, MODEL_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
