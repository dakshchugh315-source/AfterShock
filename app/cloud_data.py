"""Hosted mode: fetch the public data bundle (database, checkpoints, test tiles) once, then point the app at it.

Used only when the app starts on a server that has no local data/aftershock.db and no AFTERSHOCK_DB set
(e.g. Streamlit Community Cloud). On the development laptop nothing changes. The bundle is the public
Hugging Face dataset AFTERSHOCK_DATASET (default daksh-ai/aftershock-data); no login is needed to read it.
"""
import os
from pathlib import Path

DEFAULT_DATASET = "daksh-ai/aftershock-data"


def needed(root: Path) -> bool:
    return not os.getenv("AFTERSHOCK_DB") and not (root / "data" / "aftershock.db").exists()


def ensure() -> Path:
    """Download (first start only) and set AFTERSHOCK_DATA / AFTERSHOCK_RUNS / AFTERSHOCK_DB. Returns the bundle dir."""
    import streamlit as st
    from huggingface_hub import snapshot_download

    repo = os.getenv("AFTERSHOCK_DATASET") or DEFAULT_DATASET
    target = Path.home() / "aftershock_bundle"

    @st.cache_resource(show_spinner=False)
    def _download(repo_id: str, folder: str) -> str:
        return snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=folder, token=False)

    # runs before st.set_page_config, so no visible element here; Streamlit shows "Running..." meanwhile
    print(f"[aftershock] data bundle {repo} -> {target}", flush=True)
    _download(repo, str(target))  # first start: ~2 GB; later: files already present are skipped
    os.environ["AFTERSHOCK_DATA"] = str(target / "data")
    os.environ["AFTERSHOCK_RUNS"] = str(target / "runs")
    os.environ["AFTERSHOCK_DB"] = "sqlite:///" + (target / "data" / "aftershock.db").as_posix()
    return target
