"""Download CAL FIRE DINS field inspections for one fire (public ArcGIS service) into a local JSON file.

Only damage level, structure type and location are kept - no addresses, parcel numbers or values.

python scripts/fetch_dins.py --incident Tubbs --bbox -122.79,38.40,-122.63,38.57 --out data/field/dins_tubbs.json
python scripts/fetch_dins.py --incident Woolsey --disaster socal-fire --out data/field/dins_woolsey.json
  (with --disaster the box is the extent of that disaster's buildings in $AFTERSHOCK_DB, plus ~200 m)
"""
import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

SERVICE = ("https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/ArcGIS/rest/services/"
           "POSTFIRE_MASTER_DATA_SHARE/FeatureServer/0/query")
FIELDS = "OBJECTID,INCIDENTNAME,DAMAGE,STRUCTURETYPE,STRUCTURECATEGORY,LATITUDE,LONGITUDE"


def fetch(incident: str, bbox: str, page: int = 2000):
    out, offset = [], 0
    while True:
        q = {"where": f"INCIDENTNAME = '{incident}'", "geometry": bbox, "geometryType": "esriGeometryEnvelope",
             "inSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": FIELDS, "returnGeometry": "false",
             "resultOffset": offset, "resultRecordCount": page, "orderByFields": "OBJECTID", "f": "json"}
        with urllib.request.urlopen(SERVICE + "?" + urllib.parse.urlencode(q), timeout=60) as r:
            d = json.load(r)
        if "error" in d:
            raise RuntimeError(d["error"])
        feats = [f["attributes"] for f in d.get("features", [])]
        out += feats
        if len(feats) < page:
            return out
        offset += page


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incident", required=True)
    ap.add_argument("--bbox", help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--disaster", help="take the box from this disaster's buildings in the database instead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.bbox:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from aftershock.config import DB_URL
        from aftershock.db import DB
        x0, y0, x1, y1 = DB(DB_URL).execute("SELECT MIN(lon), MIN(lat), MAX(lon), MAX(lat) FROM buildings "
                                            "WHERE disaster = ?", (a.disaster,)).fetchone()
        a.bbox = f"{x0 - 0.0025},{y0 - 0.002},{x1 + 0.0025},{y1 + 0.002}"
        print(f"bbox from {a.disaster}: {a.bbox}")
    rows = fetch(a.incident, a.bbox)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"source": SERVICE, "incident": a.incident, "bbox": a.bbox, "records": rows}))
    print(f"{len(rows)} DINS records for {a.incident} -> {a.out}")


if __name__ == "__main__":
    main()
