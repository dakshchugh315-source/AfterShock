"""Read xBD / xView2 labels and cut pre/post crops around each building.

xBD layout:  <root>/images/<disaster>_<id>_{pre,post}_disaster.png
             <root>/labels/<disaster>_<id>_{pre,post}_disaster.json
Post-disaster label JSON: features.xy[*] = pixel polygons with properties.subtype (damage) and uid,
features.lng_lat[*] = the same buildings in lon/lat; metadata has disaster, disaster_type, width, height.
"""
import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from shapely import wkt

from .config import CLASS_TO_IDX


@dataclass
class Building:
    uid: str
    label: str
    pixel_bounds: tuple      # (minx, miny, maxx, maxy) in image pixels
    lonlat_wkt: str
    lon: float
    lat: float


@dataclass
class Tile:
    image_id: str            # e.g. hurricane-michael_00000123
    disaster: str
    disaster_type: str
    width: int
    height: int
    pre_path: Path
    post_path: Path
    buildings: list


def iter_tiles(root: Path, disasters=None):
    """Yield every post-disaster tile that has a matching pre image."""
    root = Path(root)
    for label_path in sorted((root / "labels").glob("*_post_disaster.json")):
        tile = parse_tile(label_path, root)
        if tile and (not disasters or tile.disaster in disasters):
            yield tile


def parse_tile(label_path: Path, root: Path):
    doc = json.loads(Path(label_path).read_text(encoding="utf-8"))
    meta = doc.get("metadata", {})
    image_id = Path(label_path).name.replace("_post_disaster.json", "")
    post = root / "images" / f"{image_id}_post_disaster.png"
    pre = root / "images" / f"{image_id}_pre_disaster.png"
    if not (pre.exists() and post.exists()):
        return None
    lonlat = {f["properties"].get("uid"): f["wkt"] for f in doc.get("features", {}).get("lng_lat", [])}
    buildings = []
    for f in doc.get("features", {}).get("xy", []):
        props = f.get("properties", {})
        label = props.get("subtype")
        if props.get("feature_type", "building") != "building" or label not in CLASS_TO_IDX:
            continue                                    # skips "un-classified" and non-buildings
        uid = props.get("uid")
        poly = wkt.loads(f["wkt"])
        ll_wkt = lonlat.get(uid)
        c = wkt.loads(ll_wkt).centroid if ll_wkt else None
        buildings.append(Building(uid, label, poly.bounds, ll_wkt, c.x if c else None, c.y if c else None))
    return Tile(image_id, meta.get("disaster", image_id.rsplit("_", 1)[0]), meta.get("disaster_type", ""),
                int(meta.get("width", 1024)), int(meta.get("height", 1024)), pre, post, buildings)


def crop_pair(pre: Image.Image, post: Image.Image, bounds, size=128, pad=0.25):
    """Square crop around a building with some context, from both images, resized to `size`."""
    minx, miny, maxx, maxy = bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny, 8) * (1 + pad) / 2
    box = (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, post.width)), int(min(cy + half, post.height)))
    return [img.crop(box).resize((size, size), Image.BILINEAR) for img in (pre, post)]


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
