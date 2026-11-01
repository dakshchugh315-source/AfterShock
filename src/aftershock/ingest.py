"""Load an xBD subset into the database: disasters, tiles, buildings and their pre/post crops.

python -m aftershock.ingest --xbd data/xbd/train --disasters hurricane-michael socal-fire
"""
import argparse
import hashlib
from collections import Counter

from PIL import Image

from .config import CLASS_TO_IDX, DB_URL
from .db import DB
from .xbd import crop_pair, iter_tiles, to_png_bytes


def assign_split(image_id: str, val_frac: float, test_frac: float) -> str:
    """Deterministic split by tile, so neighbouring buildings never leak across train/val/test."""
    r = int(hashlib.md5(image_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if r < test_frac else "val" if r < test_frac + val_frac else "train"


def ingest(xbd_root, db: DB, disasters=None, size=128, val_frac=0.15, test_frac=0.15, log=print):
    db.init_schema()
    counts, labels = Counter(), Counter()
    for tile in iter_tiles(xbd_root, disasters):
        split = assign_split(tile.image_id, val_frac, test_frac)
        db.upsert("disasters", ["name", "disaster_type"], [(tile.disaster, tile.disaster_type)])
        db.upsert("images", ["image_id", "disaster", "split", "width", "height"],
                  [(tile.image_id, tile.disaster, split, tile.width, tile.height)])
        pre, post = Image.open(tile.pre_path).convert("RGB"), Image.open(tile.post_path).convert("RGB")
        brows, crows = [], []
        for b in tile.buildings:
            pre_c, post_c = crop_pair(pre, post, b.pixel_bounds, size)
            brows.append((b.uid, tile.image_id, tile.disaster, split, b.label, CLASS_TO_IDX[b.label], b.lonlat_wkt, b.lon, b.lat))
            crows.append((b.uid, to_png_bytes(pre_c), to_png_bytes(post_c)))
            labels[b.label] += 1
        db.upsert("buildings", ["uid", "image_id", "disaster", "split", "label", "label_idx", "footprint_wkt", "lon", "lat"], brows)
        db.upsert("crops", ["uid", "pre_png", "post_png"], crows)
        counts[split] += len(brows)
        counts["tiles"] += 1
        if counts["tiles"] % 50 == 0:
            log(f"  {counts['tiles']} tiles, {sum(labels.values())} buildings")
    db.finalize_spatial()
    summary = {"tiles": counts["tiles"], "train": counts["train"], "val": counts["val"], "test": counts["test"], "labels": dict(labels)}
    log(f"Ingested: {summary}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xbd", required=True, help="folder with images/ and labels/")
    ap.add_argument("--disasters", nargs="*", help="limit to these disaster names")
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    a = ap.parse_args(argv)
    ingest(a.xbd, DB(a.db), a.disasters, a.size, a.val_frac, a.test_frac)


if __name__ == "__main__":
    main()
