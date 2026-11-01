"""Project-wide constants and paths."""
import os
from pathlib import Path

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
CLASS_TO_IDX = {c: i for i, c in enumerate(DAMAGE_CLASSES)}
SEVERE = {CLASS_TO_IDX["major-damage"], CLASS_TO_IDX["destroyed"]}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("AFTERSHOCK_DATA", ROOT / "data"))
RUNS_DIR = Path(os.getenv("AFTERSHOCK_RUNS", ROOT / "runs"))

# PostgreSQL + PostGIS in production, e.g. postgresql://user:pass@localhost:5432/aftershock
# SQLite works for development and tests (no spatial index; distance math done in Python).
DB_URL = os.getenv("AFTERSHOCK_DB", f"sqlite:///{(DATA_DIR / 'aftershock.db').as_posix()}")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
