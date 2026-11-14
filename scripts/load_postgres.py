"""Copy the AfterShock SQLite database into PostgreSQL (schema `aftershock`), with all keys, constraints,
views, functions and triggers from sql/postgres_schema.sql.

Run once after scripts/setup_postgres.ps1:
    python scripts/load_postgres.py                      # metadata + predictions + AI layer (fast)
    python scripts/load_postgres.py --crops 2000         # also copy 2,000 image pairs (BYTEA) as a sample

The connection string is read from AFTERSHOCK_PG in .env (never printed).
Predictions are inserted row by row through the trigger, so the review queue is built BY POSTGRES itself.
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aftershock.config import RUNS_DIR  # noqa: E402


def pg_url():
    url = os.environ.get("AFTERSHOCK_PG")
    if not url and (ROOT / ".env").exists():
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("AFTERSHOCK_PG="):
                url = line.split("=", 1)[1].strip()
    if not url:
        sys.exit("AFTERSHOCK_PG not set - run scripts/setup_postgres.ps1 first")
    return url


def copy_rows(cur, table, cols, rows):
    with cur.copy(f"COPY {table} ({', '.join(cols)}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", default=str(ROOT / "data" / "aftershock.db"))
    ap.add_argument("--crops", type=int, default=0, help="number of image pairs to copy (0 = none, -1 = all)")
    a = ap.parse_args()
    src = sqlite3.connect(f"file:{Path(a.sqlite).as_posix()}?mode=ro", uri=True)
    t0 = time.time()
    with psycopg.connect(pg_url()) as pg, pg.cursor() as cur:
        cur.execute((ROOT / "sql" / "postgres_schema.sql").read_text(encoding="utf-8"))
        cur.execute("SET search_path = aftershock, public")
        has_postgis = cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'postgis'").fetchone() is not None

        def q(sql):
            return src.execute(sql).fetchall()

        disasters = q("SELECT name, disaster_type FROM disasters")
        names = {d[0] for d in disasters}
        for (d,) in q("SELECT DISTINCT disaster FROM buildings"):
            if d not in names:
                disasters.append((d, None))
        copy_rows(cur, "disasters", ["name", "disaster_type"], disasters)
        copy_rows(cur, "images", ["image_id", "disaster", "split", "width", "height"],
                  q("SELECT image_id, disaster, split, width, height FROM images"))
        copy_rows(cur, "buildings", ["uid", "image_id", "disaster", "split", "label", "label_idx", "footprint_wkt", "lon", "lat"],
                  q("SELECT uid, image_id, disaster, split, label, label_idx, footprint_wkt, lon, lat FROM buildings"))
        print(f"core tables copied ({time.time() - t0:.0f}s)", flush=True)

        if a.crops:
            lim = "" if a.crops < 0 else f" ORDER BY uid LIMIT {int(a.crops)}"
            copy_rows(cur, "crops", ["uid", "pre_png", "post_png"], q("SELECT uid, pre_png, post_png FROM crops" + lim))
            print(f"crops copied ({time.time() - t0:.0f}s)", flush=True)

        # predictions go through INSERT so trg_queue_uncertain fires for every row
        preds = q("SELECT uid, model_version, pred_idx, p_no, p_minor, p_major, p_destroyed, confidence, uncertainty "
                  "FROM predictions")
        cur.executemany("INSERT INTO predictions (uid, model_version, pred_idx, p_no, p_minor, p_major, p_destroyed, "
                        "confidence, uncertainty) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", preds)
        print(f"{len(preds):,} predictions inserted, trigger filled the review queue ({time.time() - t0:.0f}s)", flush=True)

        copy_rows(cur, "hotspots", ["model_version", "cluster_id", "disaster", "n_buildings", "lon", "lat", "mean_confidence"],
                  q("SELECT model_version, cluster_id, disaster, n_buildings, lon, lat, mean_confidence FROM hotspots"))
        runs = []
        for m in sorted(Path(RUNS_DIR).glob("*/metrics.json")):
            j = json.loads(m.read_text())
            if "test" in j and j.get("model_version") not in ("smoke", "sanity"):
                runs.append((j["model_version"], j["arch"], j.get("loss"), j.get("img_size"), j.get("best_val_macro_f1"),
                             j["test"]["macro_f1"], j["test"]["accuracy"], f"runs/{m.parent.name}/best.pt"))
        copy_rows(cur, "model_runs", ["model_version", "arch", "loss", "img_size", "val_macro_f1", "test_macro_f1",
                                      "test_accuracy", "ckpt_path"], runs)
        try:
            copy_rows(cur, "field_inspections", ["id", "source", "incident", "disaster", "damage_raw", "damage_idx",
                                                 "structure", "lon", "lat"],
                      q("SELECT id, source, incident, disaster, damage_raw, damage_idx, structure, lon, lat FROM field_inspections"))
            copy_rows(cur, "field_matches", ["uid", "field_id", "dist_m", "radius_m"],
                      q("SELECT uid, field_id, dist_m, radius_m FROM field_matches"))
        except sqlite3.OperationalError:
            print("no field inspections in the SQLite database (skipped)")

        # spatial column + index
        if has_postgis:
            cur.execute("ALTER TABLE buildings ADD COLUMN geom geometry(Point, 4326) "
                        "GENERATED ALWAYS AS (ST_SetSRID(ST_MakePoint(lon, lat), 4326)) STORED")
            cur.execute("CREATE INDEX buildings_geom_gix ON buildings USING GIST (geom)")
            spatial = "PostGIS geometry + GiST index"
        else:
            cur.execute("CREATE INDEX buildings_earth_gix ON buildings USING GIST (ll_to_earth(lat, lon))")
            cur.execute("CREATE INDEX field_earth_gix ON field_inspections USING GIST (ll_to_earth(lat, lon))")
            spatial = "earthdistance GiST index on ll_to_earth(lat, lon)"
        cur.execute("ANALYZE")
        pg.commit()

        print(f"\nloaded in {time.time() - t0:.0f}s · spatial: {spatial}")
        for (t,) in cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'aftershock' "
                                "AND table_type = 'BASE TABLE' ORDER BY 1").fetchall():
            print(f"  {t:<18} {cur.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>10,}")
        pending = cur.execute("SELECT COUNT(*) FROM review_queue WHERE status = 'pending'").fetchone()[0]
        print(f"  review_queue pending rows, built by the trigger: {pending:,}")


if __name__ == "__main__":
    main()
