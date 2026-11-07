"""Independent Codeforces user-skill and personalization pipeline (Model 2).

Nothing in this package imports Model 1.  The two models are combined only by
``main.py`` after each pipeline has produced its own result.
"""

from .config import DB_PATH, KNOWLEDGE_GRAPH_PATH, PROJECT_ROOT

__all__ = ["DB_PATH", "KNOWLEDGE_GRAPH_PATH", "PROJECT_ROOT"]
