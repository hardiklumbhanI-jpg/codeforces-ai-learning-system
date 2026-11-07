from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture()
def sample_db(tmp_path):
    from user_model.collect_user_data import collect_handle

    db_path = tmp_path / "codeforces_users.db"
    collect_handle("sample_user", db_path=db_path, offline=True)
    return db_path
