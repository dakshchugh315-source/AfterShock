"""AI inside the database (SQLite).

Adds a layer on top of the base schema so the model and its probabilistic output can be used from plain SQL:
  * SQL functions that run the trained network on the image bytes stored in `crops`
    (SELECT ai_predict(pre_png, post_png) FROM crops ...), plus entropy / expected-severity helpers
  * a model registry (`model_runs`), a human-review queue filled by a TRIGGER, and a human feedback table
  * views for probabilistic aggregation (expected counts), calibration, selective accuracy (window functions)

python -m aftershock.sqlai --install                       # tables, views, triggers; registers runs/*/metrics.json
python -m aftershock.sqlai --demo --model-version siamese_fm_focal --ckpt runs/siamese_fm_focal/best.pt --out runs/sql_demo.md
"""
import argparse
import functools
import io
import json
import math
import sqlite3
import time
from pathlib import Path

from .config import DAMAGE_CLASSES, DB_URL, RUNS_DIR
from .db import DB

AI_LAYER = """
CREATE TABLE IF NOT EXISTS model_runs (
    model_version TEXT PRIMARY KEY,
    arch          TEXT NOT NULL,
    loss          TEXT,
    img_size      INT  CHECK (img_size > 0),
    val_macro_f1  REAL CHECK (val_macro_f1 BETWEEN 0 AND 1),
    test_macro_f1 REAL CHECK (test_macro_f1 BETWEEN 0 AND 1),
    test_accuracy REAL,
    ckpt_path     TEXT,
    registered_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS review_queue (
    uid           TEXT NOT NULL,
    model_version TEXT NOT NULL,
    reason        TEXT NOT NULL,
    uncertainty   REAL,
    status        TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done')),
    queued_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (uid, model_version));

CREATE TABLE IF NOT EXISTS human_reviews (
    review_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    uid           TEXT NOT NULL REFERENCES buildings (uid),
    model_version TEXT NOT NULL,
    reviewer      TEXT NOT NULL,
    label_idx     INT  NOT NULL CHECK (label_idx BETWEEN 0 AND 3),
    reviewed_at   TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE INDEX IF NOT EXISTS predictions_unc_idx ON predictions (model_version, uncertainty);

-- Every new prediction the model is unsure about goes to the human-review queue automatically.
CREATE TRIGGER IF NOT EXISTS trg_queue_uncertain AFTER INSERT ON predictions
WHEN NEW.uncertainty >= 0.5 OR NEW.confidence < 0.6
BEGIN
    INSERT OR REPLACE INTO review_queue (uid, model_version, reason, uncertainty)
    VALUES (NEW.uid, NEW.model_version,
            CASE WHEN NEW.uncertainty >= 0.5 THEN 'high uncertainty' ELSE 'low confidence' END,
            NEW.uncertainty);
END;

-- A human answer closes the queue item.
CREATE TRIGGER IF NOT EXISTS trg_review_done AFTER INSERT ON human_reviews
BEGIN
    UPDATE review_queue SET status = 'done' WHERE uid = NEW.uid AND model_version = NEW.model_version;
END;

CREATE VIEW IF NOT EXISTS v_predictions AS
SELECT b.uid, b.disaster, b.split, b.label_idx AS true_idx, p.model_version, p.pred_idx,
       p.p_no, p.p_minor, p.p_major, p.p_destroyed, p.confidence, p.uncertainty,
       (p.pred_idx = b.label_idx) AS correct
FROM buildings b JOIN predictions p ON p.uid = b.uid;

-- Probabilistic aggregation: expected counts = sum of probabilities, next to the hard counts.
CREATE VIEW IF NOT EXISTS v_disaster_damage AS
SELECT model_version, disaster, COUNT(*) AS n_buildings,
       SUM(pred_idx = 3)                   AS hard_destroyed,
       ROUND(SUM(p_destroyed), 1)          AS expected_destroyed,
       SUM(true_idx = 3)                   AS true_destroyed,
       ROUND(SUM(p_major + p_destroyed), 1) AS expected_severe,
       ROUND(AVG(uncertainty), 3)          AS mean_uncertainty,
       ROUND(AVG(correct), 3)              AS accuracy
FROM v_predictions WHERE split = 'test'
GROUP BY model_version, disaster;

CREATE VIEW IF NOT EXISTS v_calibration AS
SELECT model_version, MIN(CAST(confidence * 10 AS INT), 9) / 10.0 AS conf_bin, COUNT(*) AS n,
       ROUND(AVG(confidence), 3) AS mean_confidence, ROUND(AVG(correct), 3) AS accuracy
FROM v_predictions WHERE split = 'test'
GROUP BY model_version, conf_bin;

-- Window functions: rank buildings by uncertainty to answer "accuracy if we only answer the surest X%".
CREATE VIEW IF NOT EXISTS v_uncertainty_rank AS
SELECT model_version, uid, uncertainty, correct,
       ROW_NUMBER() OVER (PARTITION BY model_version ORDER BY uncertainty) AS rnk,
       COUNT(*)     OVER (PARTITION BY model_version)                      AS total
FROM v_predictions WHERE split = 'test';

CREATE VIEW IF NOT EXISTS v_review_queue AS
SELECT q.uid, q.model_version, q.reason, ROUND(q.uncertainty, 3) AS uncertainty, b.disaster, b.lon, b.lat
FROM review_queue q JOIN buildings b ON b.uid = q.uid
WHERE q.status = 'pending'
ORDER BY q.uncertainty DESC;

CREATE VIEW IF NOT EXISTS v_model_leaderboard AS
SELECT model_version, arch, loss, img_size, ROUND(val_macro_f1, 3) AS val_macro_f1,
       ROUND(test_macro_f1, 3) AS test_macro_f1,
       RANK() OVER (ORDER BY val_macro_f1 DESC) AS rank_by_val
FROM model_runs;

CREATE VIEW IF NOT EXISTS v_model_vs_human AS
SELECT h.model_version, COUNT(*) AS reviewed, ROUND(AVG(p.pred_idx = h.label_idx), 3) AS model_agrees_with_human
FROM human_reviews h JOIN predictions p ON p.uid = h.uid AND p.model_version = h.model_version
GROUP BY h.model_version;
"""

BACKFILL = """
INSERT OR IGNORE INTO review_queue (uid, model_version, reason, uncertainty)
SELECT uid, model_version, CASE WHEN uncertainty >= 0.5 THEN 'high uncertainty' ELSE 'low confidence' END, uncertainty
FROM predictions WHERE uncertainty >= 0.5 OR confidence < 0.6
"""


# ---------------------------------------------------------------- SQL functions
def entropy4(a, b, c, d):
    ps = [x for x in (a, b, c, d) if x and x > 0]
    return -sum(p * math.log(p) for p in ps) / math.log(4)


def expected_severity(a, b, c, d):
    """0 = no damage ... 3 = destroyed, weighted by probability."""
    return 0 * a + 1 * b + 2 * c + 3 * d


def damage_label(idx):
    return DAMAGE_CLASSES[int(idx)] if idx is not None else None


def register_functions(conn: sqlite3.Connection, ckpt_path=None):
    """Make the model callable from SQL on this connection."""
    conn.create_function("entropy4", 4, entropy4, deterministic=True)
    conn.create_function("expected_severity", 4, expected_severity, deterministic=True)
    conn.create_function("damage_label", 1, damage_label, deterministic=True)
    if ckpt_path is None:
        return
    import torch
    from PIL import Image  # noqa: F401  (png decoding inside png_to_tensor)
    from .data import png_to_tensor
    from .model import load_checkpoint
    model, ckpt = load_checkpoint(ckpt_path, torch.device("cpu"))
    model.eval()
    size = ckpt["img_size"]

    @functools.lru_cache(maxsize=512)
    def probs(pre: bytes, post: bytes):
        with torch.no_grad():
            x1, x2 = png_to_tensor(bytes(pre), size)[None], png_to_tensor(bytes(post), size)[None]
            return tuple(torch.softmax(model(x1, x2), 1)[0].tolist())

    conn.create_function("ai_predict", 2, lambda a, b: DAMAGE_CLASSES[max(range(4), key=probs(a, b).__getitem__)],
                         deterministic=True)
    conn.create_function("ai_confidence", 2, lambda a, b: round(max(probs(a, b)), 4), deterministic=True)
    conn.create_function("ai_probs", 2, lambda a, b: json.dumps(dict(zip(DAMAGE_CLASSES, map(lambda v: round(v, 4), probs(a, b))))),
                         deterministic=True)


# ---------------------------------------------------------------- install / registry
def install(db: DB):
    if db.pg:
        raise SystemExit("The AI layer is written for SQLite (the engine used for the experiments).")
    db.conn.executescript(AI_LAYER)
    db.conn.execute(BACKFILL)
    db.commit()


def register_runs(db: DB, runs_dir: Path = RUNS_DIR):
    rows = []
    for m in sorted(Path(runs_dir).glob("*/metrics.json")):
        j = json.loads(m.read_text())
        if "test" not in j or j.get("model_version") in ("smoke", "sanity"):
            continue
        rows.append((j["model_version"], j["arch"], j.get("loss"), j.get("img_size"), j.get("best_val_macro_f1"),
                     j["test"]["macro_f1"], j["test"]["accuracy"], str(m.parent / "best.pt")))
    db.executemany("INSERT OR REPLACE INTO model_runs (model_version, arch, loss, img_size, val_macro_f1, test_macro_f1, "
                   "test_accuracy, ckpt_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    db.commit()
    return len(rows)


# ---------------------------------------------------------------- demo report
def _table(cur, max_rows=12):
    cols = [d[0] for d in cur.description]
    rows = cur.fetchmany(max_rows)
    fmt = lambda v: f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out) if rows else "_(no rows)_"


def demo(db: DB, model_version: str, ckpt_path=None, out=None):
    register_functions(db.conn, ckpt_path)
    mv = model_version.replace("'", "''")
    queries = [
        ("Model registry, ranked by validation macro-F1 (window function RANK)", "SELECT * FROM v_model_leaderboard"),
        ("Table sizes", "SELECT 'buildings' AS tbl, COUNT(*) AS n FROM buildings UNION ALL SELECT 'crops', COUNT(*) FROM crops "
                        "UNION ALL SELECT 'predictions', COUNT(*) FROM predictions UNION ALL SELECT 'review_queue', COUNT(*) FROM review_queue"),
        ("Expected vs hard destroyed count per disaster (probabilistic aggregation)",
         f"SELECT disaster, n_buildings, hard_destroyed, expected_destroyed, true_destroyed, mean_uncertainty, accuracy "
         f"FROM v_disaster_damage WHERE model_version = '{mv}' ORDER BY n_buildings DESC"),
        ("Threshold query: destroyed with probability >= 0.8",
         f"SELECT COUNT(*) AS n, SUM(true_idx = 3) AS really_destroyed FROM v_predictions "
         f"WHERE model_version = '{mv}' AND split = 'test' AND p_destroyed >= 0.8"),
        ("Accuracy if the system answers only its surest X% (window functions)",
         f"SELECT ROUND(100.0 * cut, 0) AS coverage_pct, ROUND(AVG(correct), 4) AS accuracy, MAX(total) - SUM(1) AS to_human "
         f"FROM v_uncertainty_rank, (SELECT 0.5 AS cut UNION ALL SELECT 0.8 UNION ALL SELECT 1.0) "
         f"WHERE model_version = '{mv}' AND rnk <= cut * total GROUP BY cut ORDER BY cut"),
        ("Calibration: does 80% confidence mean 80% correct?",
         f"SELECT conf_bin, n, mean_confidence, accuracy FROM v_calibration WHERE model_version = '{mv}' ORDER BY conf_bin"),
        ("Human review queue filled by the trigger (top 8)",
         f"SELECT uid, reason, uncertainty, disaster FROM v_review_queue WHERE model_version = '{mv}' LIMIT 8"),
        ("SQL functions on stored probabilities",
         f"SELECT uid, damage_label(pred_idx) AS predicted, ROUND(expected_severity(p_no, p_minor, p_major, p_destroyed), 2) AS exp_severity, "
         f"ROUND(entropy4(p_no, p_minor, p_major, p_destroyed), 3) AS entropy FROM predictions WHERE model_version = '{mv}' "
         f"ORDER BY p_destroyed DESC LIMIT 5"),
    ]
    if ckpt_path:
        queries.append(("In-database inference: the network runs inside SELECT on the stored image bytes",
                        "SELECT b.uid, damage_label(b.label_idx) AS truth, ai_predict(c.pre_png, c.post_png) AS ai, "
                        "ai_confidence(c.pre_png, c.post_png) AS conf FROM buildings b JOIN crops c ON c.uid = b.uid "
                        "WHERE b.split = 'test' AND b.label_idx = 3 LIMIT 6"))

    md = [f"# AfterShock SQL demo ({model_version})", "", f"Database: `{db.url}`", ""]
    for title, sql in queries:
        t = time.perf_counter()
        cur = db.execute(sql)
        tbl = _table(cur)
        md += [f"## {title}", "", "```sql", sql, "```", "", tbl, "", f"_{(time.perf_counter() - t) * 1000:.1f} ms_", ""]

    # Index vs full scan on the training query
    q = "SELECT COUNT(*) FROM buildings{} WHERE split = 'test'"
    md += ["## Index vs full table scan", ""]
    for label, hint in (("with buildings_split_idx", ""), ("forced full scan (NOT INDEXED)", " NOT INDEXED")):
        plan = " / ".join(r[-1] for r in db.execute("EXPLAIN QUERY PLAN " + q.format(hint)).fetchall())
        t = time.perf_counter()
        for _ in range(20):
            db.execute(q.format(hint)).fetchone()
        md += [f"- **{label}**: plan `{plan}`, {(time.perf_counter() - t) / 20 * 1000:.2f} ms per query"]
    md.append("")
    text = "\n".join(md)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--model-version")
    ap.add_argument("--ckpt", help="checkpoint for the ai_predict() SQL function")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    db = DB(a.db)
    if a.install or a.demo:
        install(db)
        print(f"AI layer installed; {register_runs(db)} model runs registered")
    if a.demo:
        print(demo(db, a.model_version, a.ckpt, a.out))


if __name__ == "__main__":
    main()
