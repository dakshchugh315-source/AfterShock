"""Labelled review figures for one building (DISPLAY ONLY - nothing here feeds the model).

evidence_sheet(): Before | After | What changed, cut from the original xBD tiles, each panel with a title bar
                  (date, satellite), the building outline, a scale bar in metres and a colour legend.
gradcam_sheet():  the network's own input (before, after) and its Grad-CAM heatmap, with an importance legend.

The metres-per-pixel scale is measured, not assumed: the xBD label file stores every building polygon twice,
in pixel coordinates (xy) and in longitude/latitude, so the same edge measured both ways gives metres per pixel.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.ndimage import gaussian_filter
from shapely import wkt as shapely_wkt

BG, BAR, INK, MUTED = (7, 11, 24), (15, 23, 43), (240, 243, 252), (154, 166, 200)
OUTLINE = (255, 70, 70)
CHANGE_STOPS = np.array([[11, 18, 36], [126, 34, 206], [244, 63, 94], [251, 146, 60], [250, 204, 21]], np.float32)
JET_STOPS = np.array([[0, 0, 140], [0, 120, 255], [0, 230, 200], [250, 230, 0], [255, 60, 0], [160, 0, 0]], np.float32)


def _font(size, bold=False):
    # Windows fonts first, then DejaVu (installed in the Hugging Face Docker image)
    names = (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
             if bold else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _ramp(x, stops):
    """Map values in 0..1 to RGB along colour stops."""
    pos = np.clip(x, 0, 1) * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    f = (pos - lo)[..., None]
    return stops[lo] * (1 - f) + stops[lo + 1] * f


def _haversine(lon1, lat1, lon2, lat2):
    r = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def metres_per_pixel(poly_xy, poly_ll):
    """Same building edge measured in pixels and in metres -> metres per pixel of the tile."""
    xy, ll = list(poly_xy.exterior.coords), list(poly_ll.exterior.coords)
    best = None
    for i in range(min(len(xy), len(ll))):
        for j in range(i + 1, min(len(xy), len(ll))):
            d_px = math.dist(xy[i], xy[j])
            if best is None or d_px > best[0]:
                best = (d_px, i, j)
    d_px, i, j = best
    return _haversine(*ll[i], *ll[j]) / d_px if d_px > 0 else None


def _header(img, title, subtitle, accent):
    w = img.width
    out = Image.new("RGB", (w, img.height + 58), BAR)
    out.paste(img, (0, 58))
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, 5, 57], fill=accent)
    d.text((16, 8), title, font=_font(20, True), fill=INK)
    d.text((16, 34), subtitle, font=_font(14), fill=MUTED)
    return out


def _scale_bar(img, m_per_px):
    """Bar of a round length (2/5/10/20/50/100 m) close to a quarter of the width, bottom-left."""
    if not m_per_px:
        return img
    target = img.width * 0.25 * m_per_px
    length = min((2, 5, 10, 20, 50, 100, 200), key=lambda v: abs(v - target))
    px = length / m_per_px
    d = ImageDraw.Draw(img)
    x0, y0 = 14, img.height - 22
    d.rectangle([x0 - 6, y0 - 22, x0 + px + 6, y0 + 10], fill=(0, 0, 0))
    d.rectangle([x0, y0, x0 + px, y0 + 5], fill=(255, 255, 255))
    d.text((x0, y0 - 20), f"{length} m", font=_font(14, True), fill=(255, 255, 255))
    return img


def _legend(width, stops, left, right, title):
    h = 52
    img = Image.new("RGB", (width, h), BAR)
    bar = _ramp(np.tile(np.linspace(0, 1, width - 32)[None, :], (12, 1)), stops).astype(np.uint8)
    img.paste(Image.fromarray(bar), (16, 26))
    d = ImageDraw.Draw(img)
    d.text((16, 5), title, font=_font(13, True), fill=INK)
    d.text((16, 39), left, font=_font(11), fill=MUTED)
    rw = d.textlength(right, font=_font(11))
    d.text((width - 16 - rw, 39), right, font=_font(11), fill=MUTED)
    return img


def _stack_row(panels, gap=14):
    h = max(p.height for p in panels)
    out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), h), BG)
    x = 0
    for p in panels:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def _meta(label_path):
    try:
        m = json.loads(Path(label_path).read_text(encoding="utf-8")).get("metadata", {})
        return (m.get("capture_date", "")[:10] or "date ?"), m.get("sensor", "satellite ?")
    except Exception:
        return "date ?", "satellite ?"


def evidence_sheet(data_dir, image_id, uid, pad=3.0, size=420):
    """Returns (figure, info) or None. info: m_per_px, building_m, change_in, change_out, dates."""
    root = Path(data_dir) / "xbd" / "train"
    post_lab, pre_lab = root / "labels" / f"{image_id}_post_disaster.json", root / "labels" / f"{image_id}_pre_disaster.json"
    pre_path, post_path = root / "images" / f"{image_id}_pre_disaster.png", root / "images" / f"{image_id}_post_disaster.png"
    if not (post_lab.exists() and pre_path.exists() and post_path.exists()):
        return None
    feats = json.loads(post_lab.read_text(encoding="utf-8")).get("features", {})
    fxy = next((f for f in feats.get("xy", []) if f.get("properties", {}).get("uid") == uid), None)
    fll = next((f for f in feats.get("lng_lat", []) if f.get("properties", {}).get("uid") == uid), None)
    if fxy is None:
        return None
    poly = shapely_wkt.loads(fxy["wkt"])
    mpp = metres_per_pixel(poly, shapely_wkt.loads(fll["wkt"])) if fll else None
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny, 8) * (1 + pad) / 2
    pre_img, post_img = Image.open(pre_path).convert("RGB"), Image.open(post_path).convert("RGB")
    box = (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, post_img.width)), int(min(cy + half, post_img.height)))
    a_raw, b_raw = pre_img.crop(box), post_img.crop(box)
    scale = size / max(a_raw.size)
    disp = (round(a_raw.width * scale), round(a_raw.height * scale))
    pts = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in poly.exterior.coords]

    def photo(img):
        img = ImageOps.autocontrast(img, cutoff=1).resize(disp, Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
        ImageDraw.Draw(img).line(pts + [pts[0]], fill=OUTLINE, width=3)
        return _scale_bar(img, mpp / scale if mpp else None)

    # change map: brightness-normalised difference, smoothed, on a fixed 0..1 scale
    a = np.asarray(a_raw, np.float32); b = np.asarray(b_raw, np.float32)
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6); b = (b - b.mean((0, 1))) / (b.std((0, 1)) + 1e-6)
    diff = np.clip(gaussian_filter(np.sqrt(((a - b) ** 2).mean(2)), 2) / 3.0, 0, 1)
    mask_img = Image.new("L", a_raw.size, 0)
    ImageDraw.Draw(mask_img).polygon([(x - box[0], y - box[1]) for x, y in poly.exterior.coords], fill=1)
    mask = np.asarray(mask_img, bool)
    c_in = float(diff[mask].mean()) if mask.any() else 0.0
    c_out = float(diff[~mask].mean()) if (~mask).any() else 0.0
    base = np.asarray(ImageOps.autocontrast(b_raw, cutoff=1), np.float32) * 0.8
    alpha = np.clip((diff - 0.15) * 1.6, 0, 0.85)[..., None]
    chg = Image.fromarray((base * (1 - alpha) + _ramp(diff, CHANGE_STOPS) * alpha).clip(0, 255).astype(np.uint8))
    chg = chg.resize(disp, Image.LANCZOS)
    ImageDraw.Draw(chg).line(pts + [pts[0]], fill=(255, 255, 255), width=3)
    chg = _scale_bar(chg, mpp / scale if mpp else None)

    pre_date, pre_sensor = _meta(pre_lab)
    post_date, post_sensor = _meta(post_lab)
    p1 = _header(photo(a_raw), "BEFORE", f"{pre_date} · {pre_sensor}", (96, 165, 250))
    p2 = _header(photo(b_raw), "AFTER", f"{post_date} · {post_sensor}", (244, 63, 94))
    p3 = _header(chg, "WHAT CHANGED", f"inside {c_in:.2f} vs around {c_out:.2f}", (250, 204, 21))
    p3 = _stack_col(p3, _legend(p3.width, CHANGE_STOPS, "no change", "strong change", "Change (before vs after)"))
    p1 = _stack_col(p1, _note(p1.width, "Red outline = this building"))
    p2 = _stack_col(p2, _note(p2.width, "Same place, after the disaster"))
    info = {"m_per_px": mpp, "building_m": (max(maxx - minx, maxy - miny) * mpp) if mpp else None,
            "building_px": max(maxx - minx, maxy - miny), "change_in": c_in, "change_out": c_out,
            "pre_date": pre_date, "post_date": post_date, "pre_sensor": pre_sensor, "post_sensor": post_sensor}
    return _stack_row([p1, p2, p3]), info


def _stack_col(top, bottom):
    out = Image.new("RGB", (top.width, top.height + bottom.height), BAR)
    out.paste(top, (0, 0)); out.paste(bottom, (0, top.height))
    return out


def _note(width, text):
    img = Image.new("RGB", (width, 52), BAR)
    ImageDraw.Draw(img).text((16, 17), text, font=_font(13), fill=MUTED)
    return img


def gradcam_sheet(pre_rgb, post_rgb, cam, network, pred, conf, size=420):
    """pre_rgb/post_rgb: HxWx3 uint8 (the network's own input); cam: HxW in 0..1."""
    def up(arr):
        return Image.fromarray(arr).resize((size, size), Image.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    px = pre_rgb.shape[0]
    heat = _ramp(np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((px, px), Image.BILINEAR)) / 255.0,
                 JET_STOPS)
    overlay = (post_rgb.astype(np.float32) * 0.55 + heat * 0.45).clip(0, 255).astype(np.uint8)
    p1 = _stack_col(_header(up(pre_rgb), "MODEL INPUT · BEFORE", f"{px}×{px} px, what {network} sees", (96, 165, 250)),
                    _note(size, f"Upscaled from {px} px for display, nothing added"))
    p2 = _stack_col(_header(up(post_rgb), "MODEL INPUT · AFTER", f"{px}×{px} px", (244, 63, 94)),
                    _note(size, "Same crop after the disaster"))
    p3 = _stack_col(_header(up(overlay), "GRAD-CAM", f"{network} predicts {pred} ({conf:.0%})", (167, 139, 250)),
                    _legend(size, JET_STOPS, "ignored", "drove the decision", "Where the network looked"))
    return _stack_row([p1, p2, p3])
