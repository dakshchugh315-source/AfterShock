"""Count buildings per damage class for every disaster in an xBD folder, to choose a training subset.

python scripts/disaster_stats.py --xbd data/xbd/train
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xbd", default="data/xbd/train")
    a = ap.parse_args(argv)
    counts, tiles = defaultdict(Counter), Counter()
    for f in Path(a.xbd, "labels").glob("*_post_disaster.json"):
        doc = json.loads(f.read_text(encoding="utf-8"))
        disaster = doc.get("metadata", {}).get("disaster") or f.name.rsplit("_", 3)[0]
        tiles[disaster] += 1
        for feat in doc.get("features", {}).get("xy", []):
            counts[disaster][feat.get("properties", {}).get("subtype", "un-classified")] += 1
    rows = sorted(counts, key=lambda d: -(counts[d]["major-damage"] + counts[d]["destroyed"]))
    print(f"{'disaster':28} {'tiles':>6} " + " ".join(f"{c:>13}" for c in CLASSES) + f" {'severe %':>9}")
    for d in rows:
        total = sum(counts[d][c] for c in CLASSES)
        severe = counts[d]["major-damage"] + counts[d]["destroyed"]
        print(f"{d:28} {tiles[d]:>6} " + " ".join(f"{counts[d][c]:>13}" for c in CLASSES) + f" {100 * severe / max(total, 1):>8.1f}%")


if __name__ == "__main__":
    main()
