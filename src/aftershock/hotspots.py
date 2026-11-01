"""Knowledge discovery: DBSCAN over predicted major/destroyed buildings -> ranked damage hotspots.

python -m aftershock.hotspots --model-version siamese_focal --eps-m 150 --min-samples 5 --min-conf 0.6
"""
import argparse

import numpy as np
from sklearn.cluster import DBSCAN

from .config import DB_URL, SEVERE
from .db import DB

EARTH_RADIUS_M = 6371000.0


def find_hotspots(model_version, db_url=DB_URL, eps_m=150.0, min_samples=5, min_conf=0.6, log=print) -> list:
    db = DB(db_url)
    rows = db.execute(
        "SELECT b.uid, b.disaster, b.lon, b.lat, p.confidence FROM buildings b JOIN predictions p ON p.uid = b.uid "
        f"WHERE p.model_version = ? AND p.confidence >= ? AND p.pred_idx IN ({','.join(map(str, sorted(SEVERE)))}) "
        "AND b.lon IS NOT NULL", (model_version, min_conf)).fetchall()
    if len(rows) < min_samples:
        log(f"Only {len(rows)} severe buildings above confidence {min_conf}; no hotspots.")
        return []
    coords = np.radians([[r[3], r[2]] for r in rows])  # haversine expects [lat, lon] in radians
    labels = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine", algorithm="ball_tree").fit_predict(coords)
    hotspots = []
    for k in sorted(set(labels) - {-1}):
        members = [rows[i] for i in np.where(labels == k)[0]]
        hotspots.append({"cluster_id": int(k), "disaster": members[0][1], "n_buildings": len(members),
                         "lon": float(np.mean([m[2] for m in members])), "lat": float(np.mean([m[3] for m in members])),
                         "mean_confidence": float(np.mean([m[4] for m in members]))})
    hotspots.sort(key=lambda h: -h["n_buildings"])                      # rescue priority: biggest cluster first
    db.execute("DELETE FROM hotspots WHERE model_version = ?", (model_version,))
    db.upsert("hotspots", ["model_version", "cluster_id", "disaster", "n_buildings", "lon", "lat", "mean_confidence"],
              [(model_version, h["cluster_id"], h["disaster"], h["n_buildings"], h["lon"], h["lat"], h["mean_confidence"]) for h in hotspots])
    noise = int((labels == -1).sum())
    log(f"{len(hotspots)} hotspots from {len(rows)} severe buildings ({noise} isolated)")
    for rank, h in enumerate(hotspots, 1):
        log(f"  #{rank}: {h['n_buildings']} buildings near ({h['lat']:.5f}, {h['lon']:.5f}), mean confidence {h['mean_confidence']:.0%}")
    return hotspots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--eps-m", type=float, default=150.0)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--min-conf", type=float, default=0.6)
    a = ap.parse_args(argv)
    find_hotspots(a.model_version, a.db, a.eps_m, a.min_samples, a.min_conf)


if __name__ == "__main__":
    main()
