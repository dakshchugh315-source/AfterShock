"""Thin database layer: PostgreSQL + PostGIS in production, SQLite for development and tests."""
import math
import sqlite3
from pathlib import Path

from .config import DB_URL, ROOT

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS disasters (name TEXT PRIMARY KEY, disaster_type TEXT);
CREATE TABLE IF NOT EXISTS images (image_id TEXT PRIMARY KEY, disaster TEXT, split TEXT, width INT, height INT);
CREATE TABLE IF NOT EXISTS buildings (uid TEXT PRIMARY KEY, image_id TEXT, disaster TEXT, split TEXT, label TEXT,
    label_idx INT, footprint_wkt TEXT, lon REAL, lat REAL);
CREATE INDEX IF NOT EXISTS buildings_split_idx ON buildings (split);
CREATE TABLE IF NOT EXISTS crops (uid TEXT PRIMARY KEY, pre_png BLOB, post_png BLOB);
CREATE TABLE IF NOT EXISTS predictions (uid TEXT, model_version TEXT, pred_idx INT, p_no REAL, p_minor REAL,
    p_major REAL, p_destroyed REAL, confidence REAL, uncertainty REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (uid, model_version));
CREATE TABLE IF NOT EXISTS facilities (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, ftype TEXT, lon REAL, lat REAL);
CREATE TABLE IF NOT EXISTS hotspots (model_version TEXT, cluster_id INT, disaster TEXT, n_buildings INT, lon REAL,
    lat REAL, mean_confidence REAL, PRIMARY KEY (model_version, cluster_id));
"""


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


class DB:
    def __init__(self, url: str = DB_URL):
        self.url = url
        self.pg = url.startswith("postgres")
        if self.pg:
            import psycopg
            self.conn = psycopg.connect(url, autocommit=True)
        else:
            path = Path(url.removeprefix("sqlite:///"))
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path)

    def execute(self, sql: str, params=()):
        if self.pg:
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows):
        if self.pg:
            with self.conn.cursor() as cur:
                cur.executemany(sql.replace("?", "%s"), rows)
        else:
            self.conn.executemany(sql, rows)

    def commit(self):
        if not self.pg:
            self.conn.commit()

    def init_schema(self):
        if self.pg:
            self.execute((ROOT / "sql" / "schema_postgis.sql").read_text(encoding="utf-8"))
        else:
            self.conn.executescript(SQLITE_SCHEMA)
        self.commit()

    def upsert(self, table: str, cols: list, rows: list):
        """Insert rows, replacing existing primary keys (works on both engines)."""
        ph = ", ".join("?" * len(cols))
        if self.pg:
            pk = {"disasters": "name", "images": "image_id", "buildings": "uid", "crops": "uid",
                  "predictions": "uid, model_version", "hotspots": "model_version, cluster_id"}[table]
            sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in pk.split(", "))
            sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph}) ON CONFLICT ({pk}) DO UPDATE SET {sets}"
        else:
            sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({ph})"
        self.executemany(sql, rows)
        self.commit()

    def finalize_spatial(self):
        """PostGIS: build polygon geometries from the stored WKT."""
        if self.pg:
            self.execute("UPDATE buildings SET footprint = ST_GeomFromText(footprint_wkt, 4326) WHERE footprint IS NULL")

    # ---------- reads used by training, prediction and the dashboard ----------
    def split_rows(self, split: str, limit: int | None = None):
        sql = "SELECT uid, label_idx FROM buildings WHERE split = ? ORDER BY uid"
        rows = self.execute(sql + (f" LIMIT {int(limit)}" if limit else ""), (split,)).fetchall()
        return [(r[0], r[1]) for r in rows]

    def crop_bytes(self, uid: str):
        row = self.execute("SELECT pre_png, post_png FROM crops WHERE uid = ?", (uid,)).fetchone()
        if row is None:
            raise KeyError(uid)
        return bytes(row[0]), bytes(row[1])

    def buildings_near(self, lon, lat, meters, model_version, pred_idx=None, min_conf=0.0):
        """Predicted buildings within `meters` of a point, filtered by class and confidence (probabilistic query)."""
        cond = "p.model_version = ? AND p.confidence >= ?" + (" AND p.pred_idx = ?" if pred_idx is not None else "")
        params = [model_version, min_conf] + ([pred_idx] if pred_idx is not None else [])
        if self.pg:
            sql = (f"SELECT b.uid, b.lon, b.lat, p.pred_idx, p.confidence FROM buildings b JOIN predictions p USING (uid) "
                   f"WHERE {cond} AND ST_DWithin(b.geom::geography, ST_SetSRID(ST_MakePoint(?, ?), 4326)::geography, ?)")
            return self.execute(sql, params + [lon, lat, meters]).fetchall()
        sql = f"SELECT b.uid, b.lon, b.lat, p.pred_idx, p.confidence FROM buildings b JOIN predictions p ON p.uid = b.uid WHERE {cond}"
        return [r for r in self.execute(sql, params).fetchall() if haversine_m(lon, lat, r[1], r[2]) <= meters]
