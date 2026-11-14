"""Download the AfterShock data bundle (database, checkpoints, test tiles) from a Hugging Face dataset repo.

The dataset repo id comes from the Space variable AFTERSHOCK_DATASET, e.g. "your-name/aftershock-data".
Files land in ~/bundle (the AFTERSHOCK_* paths in hf/Dockerfile point there). Already-present files are skipped.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

repo = os.environ.get("AFTERSHOCK_DATASET")
if not repo:
    sys.exit("Set the Space variable AFTERSHOCK_DATASET to your dataset repo id, e.g. your-name/aftershock-data")
target = Path.home() / "bundle"
print(f"fetching {repo} -> {target} ...", flush=True)
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=target, token=os.environ.get("HF_TOKEN"))
db = target / "data" / "aftershock.db"
print(f"database: {db.stat().st_size / 1e9:.2f} GB" if db.exists() else "database missing!", flush=True)
