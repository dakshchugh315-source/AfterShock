"""Real-world check: compare the model (and the xBD labels) with CAL FIRE field inspections (DINS).

CAL FIRE inspectors visit every structure in a wildfire area and record its damage on the ground.
That is independent, real-world evidence the model never saw. This module:

  1. loads the inspections into a table `field_inspections` (with a CHECK on the damage class),
  2. matches every building to its nearest inspection point within R metres with ONE SQL statement
     (bounding-box join on an index + distance + ROW_NUMBER() OVER (PARTITION BY building)),
     stored in `field_matches`,
  3. exposes the comparison as views, and writes a markdown report.

DINS damage -> our 4 classes: No Damage -> no-damage, Affected (1-10%) and Minor (10-25%) -> minor,
Major (25-50%) -> major, Destroyed (>50%) -> destroyed.
Before 2018 DINS recorded only DAMAGED structures, so for the 2017 Tubbs fire a building with no
inspection point nearby is treated as "not recorded as damaged" (shown separately, never mixed in silently).

python -m aftershock.fieldcheck --load data/field/dins_tubbs.json --disaster santa-rosa-wildfire --radius 12 \
    --model-version siamese_fm_focal --out runs/field_check_tubbs.md
"""
import argparse
import json
import math
from pathlib import Path

from .config import DAMAGE_CLASSES, DB_URL
from .db import DB

DINS_TO_IDX = {"No Damage": 0, "Affected (1-9%)": 1, "Affected (>0-10%)": 1, "Minor (10-25%)": 1,
               "Major (26-50%)": 2, "Major (25-50%)": 2, "Destroyed (>50%)": 3}

SCHEMA = """
CREATE TABLE IF NOT EXISTS field_inspections (
    id            INTEGER PRIMARY KEY,               -- CAL FIRE OBJECTID
    source        TEXT NOT NULL DEFAULT 'CAL FIRE DINS',
    incident      TEXT NOT NULL,
    disaster      TEXT NOT NULL,                     -- our disaster name (xBD)
    damage_raw    TEXT NOT NULL,
    damage_idx    INTEGER NOT NULL CHECK (damage_idx BETWEEN 0 AND 3),
    structure     TEXT,
    lon           REAL NOT NULL,
    lat           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS field_inspections_geo_idx ON field_inspections (disaster, lat, lon);
CREATE TABLE IF NOT EXISTS field_matches (
    uid           TEXT NOT NULL REFERENCES buildings(uid),
    field_id      INTEGER NOT NULL REFERENCES field_inspections(id),
    dist_m        REAL NOT NULL CHECK (dist_m >= 0),
    radius_m      REAL NOT NULL,
    PRIMARY KEY (uid)
);
DROP VIEW IF EXISTS v_field_compare;
CREATE VIEW v_field_compare AS
SELECT b.uid, b.disaster, b.split, b.lon, b.lat,
       b.label_idx                    AS xbd_idx,
       f.damage_idx                   AS field_idx,
       f.damage_raw                   AS field_raw,
       m.dist_m
FROM buildings b
LEFT JOIN field_matches m     ON m.uid = b.uid
LEFT JOIN field_inspections f ON f.id = m.field_id
WHERE b.disaster IN (SELECT DISTINCT disaster FROM field_inspections);
DROP VIEW IF EXISTS v_field_vs_model;
CREATE VIEW v_field_vs_model AS
SELECT c.*, p.model_version, p.pred_idx, p.confidence, p.uncertainty, p.p_destroyed
FROM v_field_compare c JOIN predictions p ON p.uid = c.uid;
"""

# One statement does the spatial join: candidate pairs inside a small lat/lon box (uses the index),
# equirectangular distance in metres, then keep the nearest inspection per building.
MATCH_SQL = """
INSERT OR REPLACE INTO field_matches (uid, field_id, dist_m, radius_m)
SELECT uid, field_id, dist_m, :radius FROM (
    SELECT b.uid, f.id AS field_id,
           sqrt(((f.lon - b.lon) * :m_per_deg_lon) * ((f.lon - b.lon) * :m_per_deg_lon) +
                ((f.lat - b.lat) * :m_per_deg_lat) * ((f.lat - b.lat) * :m_per_deg_lat)) AS dist_m,
           ROW_NUMBER() OVER (PARTITION BY b.uid ORDER BY
                ((f.lon - b.lon) * :m_per_deg_lon) * ((f.lon - b.lon) * :m_per_deg_lon) +
                ((f.lat - b.lat) * :m_per_deg_lat) * ((f.lat - b.lat) * :m_per_deg_lat)) AS rn
    FROM buildings b
    JOIN field_inspections f
      ON f.disaster = b.disaster
     AND f.lat BETWEEN b.lat - :dlat AND b.lat + :dlat
     AND f.lon BETWEEN b.lon - :dlon AND b.lon + :dlon
    WHERE b.disaster = :disaster
) WHERE rn = 1 AND dist_m <= :radius
"""


def install(db: DB):
    for stmt in [s for s in SCHEMA.split(";\n") if s.strip()]:
        db.execute(stmt)
    db.commit()


def load(db: DB, path: str, disaster: str) -> int:
    d = json.loads(Path(path).read_text())
    rows = []
    for r in d["records"]:
        idx = DINS_TO_IDX.get(r.get("DAMAGE"))
        if idx is None or r.get("LATITUDE") is None or r.get("LONGITUDE") is None:
            continue  # inaccessible / unknown damage: skipped, counted in the report
        rows.append((int(r["OBJECTID"]), "CAL FIRE DINS", r["INCIDENTNAME"], disaster, r["DAMAGE"], idx,
                     r.get("STRUCTURETYPE"), float(r["LONGITUDE"]), float(r["LATITUDE"])))
    db.executemany("INSERT OR REPLACE INTO field_inspections (id, source, incident, disaster, damage_raw, damage_idx, "
                   "structure, lon, lat) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.commit()
    return len(rows)


def match(db: DB, disaster: str, radius: float) -> int:
    lat0 = db.execute("SELECT AVG(lat) FROM buildings WHERE disaster = ?", (disaster,)).fetchone()[0]
    m_lat, m_lon = 110_574.0, 111_320.0 * math.cos(math.radians(lat0))
    try:  # older SQLite builds have no math functions: register sqrt on the connection
        db.execute("SELECT sqrt(4)").fetchone()
    except Exception:
        db.conn.create_function("sqrt", 1, math.sqrt, deterministic=True)
    db.execute("DELETE FROM field_matches WHERE uid IN (SELECT uid FROM buildings WHERE disaster = ?)", (disaster,))
    db.execute(MATCH_SQL, {"radius": radius, "m_per_deg_lon": m_lon, "m_per_deg_lat": m_lat,
                           "dlat": radius / m_lat, "dlon": radius / m_lon, "disaster": disaster})
    db.commit()
    return db.execute("SELECT COUNT(*) FROM field_matches m JOIN buildings b ON b.uid = m.uid WHERE b.disaster = ?",
                      (disaster,)).fetchone()[0]


def _confusion(pairs):
    cm = [[0] * 4 for _ in range(4)]
    for t, p in pairs:
        cm[t][p] += 1
    return cm


def _binary(pairs):
    """destroyed vs not: precision, recall, F1, accuracy."""
    tp = sum(1 for t, p in pairs if t == 3 and p == 3)
    fp = sum(1 for t, p in pairs if t != 3 and p == 3)
    fn = sum(1 for t, p in pairs if t == 3 and p != 3)
    acc = sum(1 for t, p in pairs if (t == 3) == (p == 3)) / max(len(pairs), 1)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"n": len(pairs), "accuracy": acc, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def compare(db: DB, disaster: str, model_version: str | None):
    """Returns a dict of comparisons. 'matched' = buildings with an inspection within the radius."""
    res = {}
    xb = db.execute("SELECT xbd_idx, field_idx FROM v_field_compare WHERE disaster = ? AND field_idx IS NOT NULL",
                    (disaster,)).fetchall()
    res["xbd_vs_field"] = {"exact": sum(t == p for p, t in xb) / max(len(xb), 1), "n": len(xb),
                           "confusion_field_rows": _confusion([(t, p) for p, t in xb]),
                           "destroyed": _binary([(t, p) for p, t in xb])}
    if model_version:
        mv = db.execute("SELECT pred_idx, field_idx, xbd_idx FROM v_field_vs_model WHERE disaster = ? AND model_version = ? "
                        "AND field_idx IS NOT NULL", (disaster, model_version)).fetchall()
        res["model_vs_field"] = {"exact": sum(p == t for p, t, _ in mv) / max(len(mv), 1), "n": len(mv),
                                 "confusion_field_rows": _confusion([(t, p) for p, t, _ in mv]),
                                 "destroyed": _binary([(t, p) for p, t, _ in mv])}
        # xBD labels vs inspectors on EXACTLY the same buildings the model is scored on (fair side-by-side)
        xs = db.execute("SELECT xbd_idx, field_idx FROM v_field_vs_model WHERE disaster = ? AND model_version = ? "
                        "AND field_idx IS NOT NULL", (disaster, model_version)).fetchall()
        res["xbd_vs_field_same"] = {"exact": sum(t == p for p, t in xs) / max(len(xs), 1), "n": len(xs),
                                    "confusion_field_rows": _confusion([(t, p) for p, t in xs]),
                                    "destroyed": _binary([(t, p) for p, t in xs])}
        mx = db.execute("SELECT pred_idx, xbd_idx FROM v_field_vs_model WHERE disaster = ? AND model_version = ? "
                        "AND field_idx IS NOT NULL", (disaster, model_version)).fetchall()
        res["model_vs_xbd_same_buildings"] = {"exact": sum(p == t for p, t in mx) / max(len(mx), 1), "n": len(mx)}
        un = db.execute("SELECT pred_idx FROM v_field_vs_model WHERE disaster = ? AND model_version = ? "
                        "AND field_idx IS NULL", (disaster, model_version)).fetchall()
        res["model_on_unrecorded"] = {"n": len(un), "pred_destroyed": sum(1 for (p,) in un if p == 3),
                                      "pred_counts": [sum(1 for (p,) in un if p == i) for i in range(4)]}
    res["field_counts"] = dict(db.execute("SELECT damage_raw, COUNT(*) FROM field_inspections WHERE disaster = ? "
                                          "GROUP BY damage_raw ORDER BY 2 DESC", (disaster,)).fetchall())
    res["buildings"] = db.execute("SELECT COUNT(*) FROM buildings WHERE disaster = ?", (disaster,)).fetchone()[0]
    res["dist"] = db.execute("SELECT AVG(dist_m), MAX(dist_m) FROM field_matches m JOIN buildings b ON b.uid = m.uid "
                             "WHERE b.disaster = ?", (disaster,)).fetchone()
    return res


def report(res, disaster, model_version, radius) -> str:
    L = [f"# Real-world check: {disaster} vs CAL FIRE field inspections", "",
         f"Source: CAL FIRE Damage Inspection (DINS), public ArcGIS service. Match radius {radius:.0f} m "
         f"(nearest inspection point per building, SQL spatial join). Buildings in DB: {res['buildings']:,}; "
         f"matched: {res['xbd_vs_field']['n']:,} (mean distance {res['dist'][0] or 0:.1f} m).", "",
         "Field records: " + ", ".join(f"{k} {v:,}" for k, v in res["field_counts"].items()), ""]

    def block(title, r):
        b = r["destroyed"]
        L.extend([f"## {title}", "",
                  f"- buildings compared: {r['n']:,}", f"- same damage class: {r['exact']:.1%}",
                  f"- destroyed vs not: accuracy {b['accuracy']:.1%}, precision {b['precision']:.1%}, "
                  f"recall {b['recall']:.1%}, F1 {b['f1']:.3f}", "",
                  "Confusion (rows = field inspectors, columns = " + title.split(' vs ')[0] + "):", "",
                  "| field \\ pred | " + " | ".join(DAMAGE_CLASSES) + " |", "|---" * 5 + "|"]
                 + [f"| {DAMAGE_CLASSES[i]} | " + " | ".join(f"{v:,}" for v in row) + " |"
                    for i, row in enumerate(r["confusion_field_rows"])] + [""])

    block("xBD labels vs field inspectors", res["xbd_vs_field"])
    if model_version and "model_vs_field" in res:
        block(f"Model `{model_version}` vs field inspectors", res["model_vs_field"])
        block("xBD labels vs field inspectors (same buildings as the model)", res["xbd_vs_field_same"])
        L.append(f"Same buildings, model vs xBD labels: {res['model_vs_xbd_same_buildings']['exact']:.1%} "
                 f"({res['model_vs_xbd_same_buildings']['n']:,} buildings).")
        u = res["model_on_unrecorded"]
        L.append(f"\nBuildings with no inspection point within {radius:.0f} m (pre-2018 DINS lists only damaged "
                 f"structures, so these are 'not recorded as damaged'): {u['n']:,}; the model calls "
                 f"{u['pred_destroyed']:,} of them destroyed.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--load", help="JSON written by scripts/fetch_dins.py")
    ap.add_argument("--disaster", required=True)
    ap.add_argument("--radius", type=float, default=12.0)  # 12 m: see the radius sweep in docs/CHANGES.md (item 45)
    ap.add_argument("--model-version")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    db = DB(a.db)
    install(db)
    if a.load:
        print(f"loaded {load(db, a.load, a.disaster):,} inspections")
    print(f"matched {match(db, a.disaster, a.radius):,} buildings within {a.radius:.0f} m")
    res = compare(db, a.disaster, a.model_version)
    md = report(res, a.disaster, a.model_version, a.radius)
    print(md)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        (Path(a.out).with_suffix(".json")).write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
