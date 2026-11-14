"""Assemble the two folders to upload to Hugging Face (free hosting of the FULL dashboard):

    python scripts/prepare_hf.py --user <hf-username>

build/hf_space/  -> Space repo  <user>/aftershock       (code + Dockerfile + Space card; Docker SDK)
build/hf_data/   -> dataset repo <user>/aftershock-data (aftershock.db, checkpoints, results, test tiles)

Upload (after `hf auth login`):
    hf upload <user>/aftershock-data build/hf_data . --repo-type dataset
    hf upload <user>/aftershock      build/hf_space . --repo-type space
Only tiles of buildings that have predictions (the test split) are bundled; nothing is recomputed.
"""
import argparse
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_CARD = """---
title: AfterShock
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Satellite building-damage assessment with AI inside the DB
---

# AfterShock

Building-damage assessment from pre/post-disaster satellite images (xBD), with the model's probabilities,
uncertainty, triggers, views and a real-world check against CAL FIRE field inspections stored in SQL.
Course project, *Data Management and AI Integration*, Bennett University (Daksh Chugh).

This Space runs the full Streamlit dashboard (PyTorch on CPU, live Grad-CAM). Data is downloaded on start from
the dataset repo [{data_repo}](https://huggingface.co/datasets/{data_repo}). The first start after a pause takes
1-2 minutes.

Data: xBD (Gupta et al., 2019, CC BY-NC-SA 4.0); CAL FIRE Damage Inspection (DINS), public data.
"""
DATA_CARD = """---
license: cc-by-nc-sa-4.0
pretty_name: AfterShock demo bundle
---

# AfterShock demo bundle

Data used by the AfterShock dashboard Space: the SQLite database (buildings, 128 px pre/post crops, model
probabilities and uncertainty, hotspots, CAL FIRE field inspections, AI-layer tables), trained checkpoints and
result files, and the original xBD tiles of the test buildings (3 disasters: hurricane-harvey, hurricane-matthew,
santa-rosa-wildfire). Derived from xBD (Gupta et al., 2019, "Creating xBD", CVPR Workshops), licensed
CC BY-NC-SA 4.0, non-commercial use only. Field inspections: CAL FIRE DINS, public data.
"""


def copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    a = ap.parse_args()
    data_repo = f"{a.user}/aftershock-data"
    space, data = ROOT / "build" / "hf_space", ROOT / "build" / "hf_data"
    shutil.rmtree(space, ignore_errors=True)
    shutil.rmtree(data, ignore_errors=True)

    # ---- Space: code only
    for d in ("app", "src/aftershock", ".streamlit", "hf"):
        shutil.copytree(ROOT / d, space / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "Dockerfile"))
    docker = (ROOT / "hf" / "Dockerfile").read_text(encoding="utf-8")
    docker = docker.replace("EXPOSE 7860", f"ENV AFTERSHOCK_DATASET={data_repo}\nEXPOSE 7860")
    (space / "Dockerfile").write_text(docker, encoding="utf-8", newline="\n")
    (space / "hf" / "start.sh").write_text((ROOT / "hf" / "start.sh").read_text(encoding="utf-8"), newline="\n")
    (space / "README.md").write_text(SPACE_CARD.format(data_repo=data_repo), encoding="utf-8")

    # ---- dataset: database, run files, test tiles
    copy(ROOT / "data" / "aftershock.db", data / "data" / "aftershock.db")
    con = sqlite3.connect(f"file:{(ROOT / 'data' / 'aftershock.db').as_posix()}?mode=ro", uri=True)
    versions = {r[0] for r in con.execute("SELECT DISTINCT model_version FROM predictions")}
    tiles = [r[0] for r in con.execute("SELECT DISTINCT b.image_id FROM buildings b JOIN predictions p ON p.uid = b.uid")]
    for run in sorted((ROOT / "runs").iterdir()):
        if not run.is_dir() or run.name in ("smoke", "sanity"):
            continue
        for name in ("metrics.json", "mc_test.json", "candidates.json"):
            if (run / name).exists():
                copy(run / name, data / "runs" / run.name / name)
        if run.name in versions and (run / "best.pt").exists():
            copy(run / "best.pt", data / "runs" / run.name / "best.pt")
    xbd = ROOT / "data" / "xbd" / "train"
    for t in tiles:
        for k in ("pre", "post"):
            copy(xbd / "images" / f"{t}_{k}_disaster.png", data / "data" / "xbd" / "train" / "images" / f"{t}_{k}_disaster.png")
            copy(xbd / "labels" / f"{t}_{k}_disaster.json", data / "data" / "xbd" / "train" / "labels" / f"{t}_{k}_disaster.json")
    (data / "README.md").write_text(DATA_CARD, encoding="utf-8")

    size = lambda d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9
    print(f"build/hf_space: {sum(1 for f in space.rglob('*') if f.is_file())} files")
    print(f"build/hf_data:  {size(data):.2f} GB, {len(tiles)} tiles, checkpoints for: "
          f"{sorted(p.parent.name for p in (data / 'runs').rglob('best.pt'))}")
    print(f"\nnext:\n  hf upload {data_repo} build/hf_data . --repo-type dataset\n"
          f"  hf upload {a.user}/aftershock build/hf_space . --repo-type space")


if __name__ == "__main__":
    main()
