"""AfterShock dashboard: damage map, human-review queue, hotspots, explanations, probabilistic queries.

streamlit run app/dashboard.py
"""
import io
import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from shapely import wkt as shapely_wkt
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from city3d import render_city3d, render_compare3d  # noqa: E402
from aftershock.config import DAMAGE_CLASSES, DATA_DIR, RUNS_DIR  # noqa: E402
from aftershock.data import png_to_tensor  # noqa: E402
from aftershock.db import DB  # noqa: E402
from aftershock.explain import explanation_panel, gradcam  # noqa: E402
from aftershock.model import load_checkpoint  # noqa: E402
from aftershock.sqlai import register_functions  # noqa: E402
from aftershock import fieldcheck  # noqa: E402

SHORT_NAMES = {"post_focal": "Post-only", "siamese_focal": "Siamese", "siamese_fm_focal": "Siamese-FM"}


def pretty(run: str) -> str:
    base = run.removesuffix("_192")
    return SHORT_NAMES.get(base, base) + (" 192px" if run.endswith("_192") else "")

# Validated categorical palette (dataviz skill, references/palette.md) - fixed hue order, never cycled.
BLUE, ORANGE, AQUA, VIOLET = "#60a5fa", "#f59e0b", "#2dd4bf", "#a78bfa"  # brightened for the dark theme
# UI accents are cool (cyan/indigo) so warm colours always mean damage.
ACCENT, ACCENT2 = "#22d3ee", "#818cf8"
# Status palette (severity, not identity): good -> critical.
GOOD, WARNING, SERIOUS, CRITICAL = "#34d399", "#facc15", "#fb923c", "#f43f5e"
COLORS = {0: GOOD, 1: WARNING, 2: SERIOUS, 3: CRITICAL}
INK_MUTED, GRID = "#94a0c2", "#1f2a44"  # tuned for the dark theme in .streamlit/config.toml
# Known-good disasters to default to (busy + has hotspots) - picked over an alphabetical default
# (e.g. guatemala-volcano has only 168 buildings / 7 damaged, a flat first impression).
PREFERRED_DEFAULT_DISASTERS = ["hurricane-harvey", "hurricane-michael", "socal-fire"]


def crisp(img: Image.Image, size: int = 480) -> Image.Image:
    """Upscale a small crop/Grad-CAM panel with Lanczos + a touch of sharpening so it reads clean at display size."""
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))


@st.cache_data(show_spinner=False)
def context_crop_with_outline(image_id: str, uid: str, pad: float = 3.0, size: int = 560):
    """DISPLAY ONLY - a wider crop straight from the original xBD tile (not the 128px DB crop the
    model trains/predicts on), with the building's own polygon outlined in red, so a reviewer sees
    real surrounding context instead of a tiny image stretched to look bigger than it is.
    Does not touch model input, the stored DB crops, or any training/predict code path."""
    labels_path = DATA_DIR / "xbd" / "train" / "labels" / f"{image_id}_post_disaster.json"
    images_dir = DATA_DIR / "xbd" / "train" / "images"
    if not labels_path.exists():
        return None
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    feat = next((f for f in doc.get("features", {}).get("xy", []) if f.get("properties", {}).get("uid") == uid), None)
    if feat is None:
        return None
    poly = shapely_wkt.loads(feat["wkt"])
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny, 8) * (1 + pad) / 2
    pre_path, post_path = images_dir / f"{image_id}_pre_disaster.png", images_dir / f"{image_id}_post_disaster.png"
    if not (pre_path.exists() and post_path.exists()):
        return None
    panels = []
    for path in (pre_path, post_path):
        img = Image.open(path).convert("RGB")
        box = (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, img.width)), int(min(cy + half, img.height)))
        crop = img.crop(box)
        scale = size / max(crop.size)
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)
        # display enhancement only: stretch contrast (1% clip) and sharpen what upscaling softened
        crop = ImageOps.autocontrast(crop, cutoff=1).filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))
        draw = ImageDraw.Draw(crop)
        pts = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in poly.exterior.coords]
        draw.line(pts + [pts[0]], fill=(255, 64, 64), width=4)
        panels.append(crop)
    return panels[0], panels[1]


@st.cache_data(show_spinner=False)
def change_view(image_id: str, uid: str, pad: float = 3.0, size: int = 560):
    """DISPLAY ONLY - what changed between the pre and post tile around one building.
    Both crops are brightness-normalised (mean/std), the per-pixel difference is smoothed and drawn as a
    heatmap over the 'after' image. Returns (image, change inside the building, change in its surroundings),
    both on the same 0-1 scale. Not used by the model."""
    labels_path = DATA_DIR / "xbd" / "train" / "labels" / f"{image_id}_post_disaster.json"
    images_dir = DATA_DIR / "xbd" / "train" / "images"
    if not labels_path.exists():
        return None
    doc = json.loads(labels_path.read_text(encoding="utf-8"))
    feat = next((f for f in doc.get("features", {}).get("xy", []) if f.get("properties", {}).get("uid") == uid), None)
    pre_path, post_path = images_dir / f"{image_id}_pre_disaster.png", images_dir / f"{image_id}_post_disaster.png"
    if feat is None or not (pre_path.exists() and post_path.exists()):
        return None
    poly = shapely_wkt.loads(feat["wkt"])
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny, 8) * (1 + pad) / 2
    pre_img, post_img = Image.open(pre_path).convert("RGB"), Image.open(post_path).convert("RGB")
    box = (int(max(cx - half, 0)), int(max(cy - half, 0)), int(min(cx + half, post_img.width)),
           int(min(cy + half, post_img.height)))
    a = np.asarray(pre_img.crop(box), dtype=np.float32)
    b = np.asarray(post_img.crop(box), dtype=np.float32)
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6)   # removes overall brightness / colour shifts between dates
    b = (b - b.mean((0, 1))) / (b.std((0, 1)) + 1e-6)
    diff = np.sqrt(((a - b) ** 2).mean(2))
    diff = gaussian_filter(diff, sigma=2)               # smooth pixel noise
    diff = np.clip(diff / 3.0, 0, 1)                    # fixed scale, so buildings are comparable to each other
    mask_img = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
    ImageDraw.Draw(mask_img).polygon([(x - box[0], y - box[1]) for x, y in poly.exterior.coords], fill=1)
    mask = np.asarray(mask_img, dtype=bool)
    inside = float(diff[mask].mean()) if mask.any() else 0.0
    outside = float(diff[~mask].mean()) if (~mask).any() else 0.0
    # heatmap: dark blue (no change) -> magenta -> orange -> yellow (strong change), blended over the after image
    stops = np.array([[11, 18, 36], [126, 34, 206], [244, 63, 94], [251, 146, 60], [250, 204, 21]], dtype=np.float32)
    pos = diff * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    heat = stops[lo] * (1 - frac) + stops[lo + 1] * frac
    base = np.asarray(ImageOps.autocontrast(post_img.crop(box), cutoff=1), dtype=np.float32) * 0.8
    alpha = np.clip((diff - 0.15) * 1.6, 0, 0.85)[..., None]   # little change -> the photo shows through
    out = Image.fromarray((base * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8))
    scale = size / max(out.size)
    out = out.resize((round(out.width * scale), round(out.height * scale)), Image.LANCZOS)
    pts = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in poly.exterior.coords]
    ImageDraw.Draw(out).line(pts + [pts[0]], fill=(255, 255, 255), width=3)
    return out, inside, outside


def plotly_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="system-ui, -apple-system, 'Segoe UI', sans-serif")),
        font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK_MUTED, size=12),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=76, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font=dict(family="system-ui, sans-serif")),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def hero_tile(value: str, label: str, accent: str) -> str:
    # Value text uses the accent colour itself (not a fixed ink colour) so it stays legible
    # in both Streamlit's light and dark themes - a fixed near-black value broke in dark mode.
    # Wraps instead of truncating, and scales with the viewport so values fit at ~1000px wide.
    return f"""<div class="as-tile" style="--acc:{accent}; flex:1; min-width:0; background:linear-gradient(135deg, {accent}26, {accent}08);
        border:1px solid {accent}55; border-radius:16px; padding:14px 16px;">
        <div style="font-size:clamp(1.15rem, 2.1vw, 1.8rem); line-height:1.15; font-weight:800; color:{accent};
            overflow-wrap:anywhere;">{value}</div>
        <div style="font-size:0.8rem; line-height:1.25; color:{INK_MUTED}; margin-top:4px; overflow-wrap:anywhere;">{label}</div>
    </div>"""


@st.cache_data(show_spinner=False)
def bootstrap_macro_f1_ci(confusion_matrix, n_boot: int = 500, seed: int = 0):
    """95% CI for macro-F1 by resampling the model's OWN recorded test confusion matrix.

    This is not invented data - the confusion matrix is the real per-class test outcome
    already stored in metrics.json. Bootstrapping it answers "how much would macro-F1 move
    around if we re-ran evaluation on a different same-sized sample from the same error
    distribution" - a lightweight stand-in for the "train 3x and report variance" ideal,
    which a single training run per architecture can't give us directly.
    """
    cm = np.asarray(confusion_matrix, dtype=np.int64)
    k = cm.shape[0]
    probs = cm.flatten() / cm.sum()
    n = int(cm.sum())
    rng = np.random.default_rng(seed)
    cell_idx = np.arange(k * k)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        draws = rng.choice(cell_idx, size=n, p=probs)
        boot_cm = np.bincount(draws, minlength=k * k).reshape(k, k)
        f1s = []
        for c in range(k):
            tp, fp, fn = boot_cm[c, c], boot_cm[:, c].sum() - boot_cm[c, c], boot_cm[c, :].sum() - boot_cm[c, c]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        scores[i] = np.mean(f1s)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


@st.cache_resource(show_spinner=False)
def get_model(ckpt_path: str):
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    return model, ckpt


def live_gradcam(ckpt_path: str, db: DB, uid: str):
    """Grad-CAM computed on the fly for a building that wasn't in the precomputed batch,
    instead of dead-ending the demo on a 'run this command yourself' message."""
    model, ckpt = get_model(ckpt_path)
    pre_png, post_png = db.crop_bytes(uid)
    pre_t, post_t = png_to_tensor(pre_png, ckpt["img_size"]), png_to_tensor(post_png, ckpt["img_size"])
    cam, cls, probs = gradcam(model, pre_t, post_t)
    return explanation_panel(pre_t, post_t, cam), DAMAGE_CLASSES[cls], float(probs[cls])


st.set_page_config(page_title="AfterShock", page_icon="🛰️", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; max-width: 1500px;}}
h1 {{font-weight: 800 !important; letter-spacing: -0.02em;}}
h2, h3, h4 {{font-weight: 700 !important;}}
/* ---------- Mission-control theme (colours in .streamlit/config.toml) ---------- */
.stApp {{
    background:
        radial-gradient(1200px 600px at 85% -10%, #312e8155, transparent 60%),
        radial-gradient(900px 500px at -10% 30%, #08334477, transparent 60%),
        linear-gradient(#ffffff05 1px, transparent 1px) 0 0 / 100% 42px,
        linear-gradient(90deg, #ffffff05 1px, transparent 1px) 0 0 / 42px 100%,
        #070b18;
}}
header[data-testid="stHeader"] {{background: transparent;}}
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0c1530, #070b18) !important;
    border-right: 1px solid #1f2a44;
}}
.as-brand {{display:flex; align-items:center; gap:10px; padding:4px 0 14px; margin-bottom:10px; border-bottom:1px solid #1f2a44;}}
.as-brand-logo {{width:38px; height:38px; border-radius:11px; display:grid; place-items:center; font-size:20px;
    background:linear-gradient(135deg, {ACCENT}, {VIOLET}); box-shadow:0 8px 20px -8px {ACCENT};}}
.as-brand-name {{font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:1.1rem; color:#fff; line-height:1.1;}}
.as-brand-sub {{font-size:0.72rem; color:#8f9bbd; letter-spacing:.08em; text-transform:uppercase;}}
.as-side-label {{font-size:0.7rem; letter-spacing:.12em; text-transform:uppercase; color:#8f9bbd; margin:14px 0 2px;}}

h1, h2, h3, h4 {{letter-spacing:-0.015em;}}
h3 {{color:#ffffff !important;}}

/* Tabs as a segmented pill bar */
[role="tablist"] {{gap:2px; justify-content:space-between; background:#0f172bcc; border:1px solid #1f2a44; padding:6px; border-radius:14px;
    backdrop-filter:blur(8px);}}
[role="tablist"] {{border-bottom:0 !important;}}
[data-testid="stTab"], button[data-baseweb="tab"] {{border-radius:10px !important; padding:7px 9px !important; margin:0 !important; height:auto !important; border-bottom:0 !important;}}
[data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"] {{background:linear-gradient(135deg, {ACCENT}33, {VIOLET}33) !important;
    box-shadow:inset 0 0 0 1px {ACCENT}66;}}
[data-testid="stTab"] p, button[data-baseweb="tab"] p {{font-weight:600 !important; font-size:0.86rem !important; white-space:nowrap;}}
[data-testid="stTab"][aria-selected="true"] p {{color:#ffffff !important;}}
[data-baseweb="tab-border"] {{display:none;}}

/* Glass panels for every container type */
[data-testid="stMetric"], div[data-testid="stExpander"], [data-testid="stAlert"],
[data-testid="stPlotlyChart"], [data-testid="stCode"] {{
    background: linear-gradient(160deg, #131d34cc, #0b1224cc) !important;
    border: 1px solid #1f2a44 !important; border-radius: 14px !important;
    box-shadow: 0 10px 30px -18px #000;
}}
[data-testid="stMetric"] {{padding:14px 16px 10px;}}
[data-testid="stPlotlyChart"] {{padding:6px;}}
[data-testid="stMetricValue"] {{background:linear-gradient(90deg, #ffffff, #b9c4ff); -webkit-background-clip:text;
    background-clip:text; color:transparent !important;}}
[data-testid="stAlert"] {{border-left:3px solid {ACCENT} !important;}}
.stButton > button {{background:linear-gradient(135deg, {ACCENT}, {ACCENT2}) !important; color:#06111f !important;
    border:0 !important; font-weight:600; box-shadow:0 8px 20px -10px {ACCENT}; transition:transform .2s, box-shadow .2s;}}
.stButton > button:hover {{transform:translateY(-2px); box-shadow:0 12px 26px -10px {ACCENT};}}
::-webkit-scrollbar {{width:10px; height:10px;}}
::-webkit-scrollbar-thumb {{background:#24304d; border-radius:10px;}}
::-webkit-scrollbar-track {{background:transparent;}}
[data-testid="stMetricValue"] {{font-weight: 800 !important; font-size: clamp(1.2rem, 2.2vw, 2rem) !important;}}
/* Let metric labels/values wrap instead of being cut off with an ellipsis at narrower widths. */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
img {{border-radius: 10px;}}
hr {{margin: 0.6rem 0 1.2rem;}}
.legend-chip {{display:inline-flex; align-items:center; gap:6px; margin-right:16px; font-size:0.85rem; color:{INK_MUTED};}}
.legend-dot {{width:10px; height:10px; border-radius:50%; display:inline-block;}}
.uid-mono {{font-family: ui-monospace, Consolas, monospace; font-size:0.8rem;}}
/* Let the page scroll normally; the map only zooms on click, not on every page-scroll pass-over. */
iframe {{pointer-events:auto;}}

/* ---------- UI polish: pure CSS (no JavaScript), so nothing extra runs in the browser ---------- */
@keyframes as-fade-up {{from {{opacity:0; transform:translateY(14px);}} to {{opacity:1; transform:none;}}}}
@keyframes as-spin {{to {{transform:rotateY(360deg);}}}}
@keyframes as-orbit {{from {{transform:rotateX(70deg) rotateZ(0deg);}} to {{transform:rotateX(70deg) rotateZ(360deg);}}}}
@keyframes as-shift {{0%,100% {{background-position:0% 50%;}} 50% {{background-position:100% 50%;}}}}
@keyframes as-pulse {{0%,100% {{box-shadow:0 0 0 0 {CRITICAL}66;}} 50% {{box-shadow:0 0 0 10px {CRITICAL}00;}}}}
@keyframes as-scan {{from {{transform:translateX(-100%);}} to {{transform:translateX(400%);}}}}

.as-hero {{position:relative; overflow:hidden; border-radius:22px; padding:26px 30px; margin-bottom:6px;
    background:linear-gradient(120deg, #0b1736, #1e1b4b, #083344, #0b1736); background-size:300% 300%;
    animation:as-shift 18s ease infinite; color:#eef1ff; display:flex; align-items:center; gap:28px;
    border:1px solid #ffffff22; box-shadow:0 20px 50px -20px #0b1230cc;}}
.as-hero::after {{content:""; position:absolute; top:0; left:0; width:25%; height:100%; pointer-events:none;
    background:linear-gradient(90deg, transparent, #ffffff14, transparent); animation:as-scan 6s linear infinite;}}
.as-hero h1 {{color:#ffffff !important; margin:0 !important; padding:0 !important; font-size:clamp(1.5rem, 3vw, 2.4rem) !important;}}
.as-hero .as-sub {{color:#c9cff5; font-size:0.95rem; margin-top:6px;}}
.as-pills {{margin-top:12px; display:flex; flex-wrap:wrap; gap:8px;}}
.as-pill {{font-size:0.78rem; padding:4px 10px; border-radius:999px; background:#ffffff14; border:1px solid #ffffff2a; color:#e6e9ff;}}
.as-live {{display:inline-block; width:9px; height:9px; border-radius:50%; background:{CRITICAL}; margin-right:6px;
    animation:as-pulse 2s infinite; vertical-align:middle;}}

/* 3D globe: a shaded sphere, a spinning meridian and a satellite on a tilted orbit */
.as-globe-wrap {{flex:0 0 auto; width:120px; height:120px; perspective:600px; position:relative;}}
.as-globe {{position:absolute; inset:10px; border-radius:50%;
    background:radial-gradient(circle at 32% 30%, #7fd1ff, #2a78d6 45%, #0b2a5c 75%, #040c22);
    box-shadow:inset -14px -10px 30px #00000088, 0 0 35px #2a78d666;}}
.as-meridian {{position:absolute; inset:10px; border-radius:50%; border:2px solid #ffffff40; transform-style:preserve-3d;
    animation:as-spin 6s linear infinite;}}
.as-orbit {{position:absolute; inset:-4px; border-radius:50%; border:1.5px dashed #ffffff55; transform-style:preserve-3d;
    animation:as-orbit 7s linear infinite;}}
.as-sat {{position:absolute; top:-6px; left:50%; width:12px; height:12px; margin-left:-6px; border-radius:3px;
    background:{ACCENT}; box-shadow:0 0 12px {ACCENT};}}

/* Glass tiles that tilt in 3D on hover */
.as-tile {{backdrop-filter:blur(6px); transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
    animation:as-fade-up .6s ease both; transform-style:preserve-3d;}}
.as-tile:hover {{transform:perspective(700px) rotateX(7deg) rotateY(-7deg) translateY(-4px);
    box-shadow:0 18px 35px -14px var(--acc);}}
[data-testid="stMetric"] {{transition:transform .3s, box-shadow .3s; animation:as-fade .6s ease;}}
[data-testid="stMetric"]:hover {{transform:perspective(700px) rotateX(6deg) translateY(-3px); box-shadow:0 14px 28px -14px {VIOLET};}}

/* Tabs: content fades in, active tab gets a gradient underline */
/* opacity only: a transform here would trap fullscreen tables/charts inside the tab */
@keyframes as-fade {{from {{opacity:0;}} to {{opacity:1;}}}}
[role="tabpanel"], [data-baseweb="tab-panel"] {{animation:as-fade .4s ease;}}  /* no fill-mode: a lingering animation makes a stacking context */
[data-testid="stTab"], button[data-baseweb="tab"] {{transition:color .2s, transform .2s, background .2s;}}
[data-testid="stTab"]:hover, button[data-baseweb="tab"]:hover {{transform:translateY(-2px);}}
[data-baseweb="tab-highlight"] {{background:linear-gradient(90deg, {BLUE}, {VIOLET}, {ACCENT}) !important; height:3px !important;}}
[data-testid="stPlotlyChart"] {{animation:as-fade .6s ease;}}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{animation:none !important; transition:none !important;}} }}
.as-eyebrow {{font-size:0.72rem; letter-spacing:.14em; text-transform:uppercase; color:#67e8f9; margin-bottom:4px;}}
.as-hero .as-sub {{max-width:820px; line-height:1.5;}}
.as-flow {{display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-top:14px;}}
.as-step {{font-size:0.8rem; padding:5px 11px; border-radius:9px; background:#ffffff12; border:1px solid #ffffff26; color:#f1f3ff;}}
.as-arrow {{color:{ACCENT}; font-weight:700;}}
.as-stats {{display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; margin-top:16px;}}
.as-stat {{background:#0007; border:1px solid #ffffff1f; border-radius:12px; padding:10px 14px;}}
.as-stat-v {{font-family:'Space Grotesk', sans-serif; font-size:1.55rem; font-weight:700; color:#fff; line-height:1.1;}}
.as-stat-l {{font-size:0.75rem; color:#b8c0e0; margin-top:2px;}}
.as-foot {{margin-top:14px; font-size:0.78rem; color:#aab3d6; display:flex; flex-wrap:wrap; align-items:center; gap:8px;}}
.as-foot .as-pill {{font-size:0.7rem; padding:2px 8px;}}
.as-verdict {{border-radius:16px; padding:16px 18px; margin-bottom:12px; border:1px solid var(--c);
    background:linear-gradient(135deg, color-mix(in srgb, var(--c) 22%, transparent), #0b1224cc);}}
.as-verdict-k {{font-size:0.7rem; letter-spacing:.12em; text-transform:uppercase; color:#aab3d6;}}
.as-verdict-v {{font-family:'Space Grotesk', sans-serif; font-size:1.6rem; font-weight:700; color:var(--c);}}
.as-verdict-row {{display:flex; flex-wrap:wrap; gap:22px; margin-top:6px;}}
.as-badge {{display:inline-block; font-size:0.75rem; padding:3px 10px; border-radius:999px; font-weight:600;}}
@media (max-width: 700px) {{ .as-globe-wrap {{display:none;}} .as-hero {{padding:20px;}} }}
</style>
""", unsafe_allow_html=True)
def fieldcheck_radius(db_, disaster_):
    r = db_.execute("SELECT MAX(radius_m) FROM field_matches m JOIN buildings b ON b.uid = m.uid WHERE b.disaster = ?",
                    (disaster_,)).fetchone()[0]
    return r or 0.0


def render_hero(stats=None, model=None):
    """Header banner. `stats` = live COUNT(*)s from the database (None if no DB is configured)."""
    steps = [("🛰️", "Pre + post imagery"), ("🗄️", "SQL database"), ("🧠", "Siamese CNN"),
             ("🎲", "Probabilities + uncertainty"), ("🧑‍⚖️", "Human review")]
    flow = '<span class="as-arrow">→</span>'.join(f'<span class="as-step">{i} {t}</span>' for i, t in steps)
    stat_html = ""
    model_txt = f" · numbers for the final model <code>{model}</code> (best validation macro-F1)" if model else ""
    if stats:
        stat_html = '<div class="as-stats">' + "".join(
            f'<div class="as-stat"><div class="as-stat-v">{v}</div><div class="as-stat-l">{l}</div></div>'
            for v, l in stats) + "</div>"
    st.markdown(f"""
<div class="as-hero">
  <div class="as-globe-wrap"><div class="as-globe"></div><div class="as-meridian"></div>
    <div class="as-orbit"><div class="as-sat"></div></div></div>
  <div style="flex:1; min-width:0;">
    <div class="as-eyebrow"><span class="as-live"></span>Satellite damage intelligence · xBD benchmark</div>
    <h1>AfterShock</h1>
    <div class="as-sub">Which buildings collapsed, how sure is the model, and why? AfterShock compares before and after
      satellite images building by building, stores every answer as probabilities in SQL, and sends the unsure ones to a person.</div>
    <div class="as-flow">{flow}</div>
    {stat_html}
    <div class="as-foot">Data Management and AI Integration · Bennett University · Daksh Chugh{model_txt}</div>
  </div>
</div>""", unsafe_allow_html=True)


# The database comes only from the environment (scripts/start_demo.ps1 sets it) - never from the UI,
# and its local path is never displayed.
db_url = os.environ.get("AFTERSHOCK_DB")
if not db_url:
    render_hero()
    st.error("Database not configured. Start the dashboard with `scripts\\start_demo.ps1`.")
    st.stop()
db = DB(db_url)


def _count(sql):
    try:
        return db.execute(sql).fetchone()[0]
    except Exception:  # table not installed on this database (e.g. review_queue before sqlai --install)
        return None


hero_slot = st.container()  # filled below, once the final model is known

# ---------------------------------------------------------------- architecture comparison
# Only the controlled trio (same loss/epochs/data, architecture is the one thing that differs) -
# other runs in runs/ (e.g. an older post_baseline with weighted-CE loss) are a different experiment
# and would make "best architecture" a misleading, apples-to-oranges claim.
RUN_GROUPS = {
    "Pilot: 64px, 5 epochs, 3 disasters": ["small_post", "small_siamese", "small_siamese_fm"],
    "128px, 15 epochs": ["post_focal", "siamese_focal", "siamese_fm_focal"],
    "192px, 12 epochs": ["post_focal_192", "siamese_focal_192", "siamese_fm_focal_192"],
}
rows = []
for resolution, names in RUN_GROUPS.items():
    for name in names:
        mf = Path(RUNS_DIR) / name / "metrics.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        t = m.get("test", {})
        if not t:
            continue
        lo, hi = bootstrap_macro_f1_ci(t["confusion_matrix"])
        rows.append({"model": name, "resolution": resolution, "architecture": m["arch"],
                     "val macro F1": m["best_val_macro_f1"], "test macro F1": t["macro_f1"],
                     "macro F1 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                     "accuracy": t["accuracy"], "recall (destroyed)": t["per_class"]["destroyed"]["recall"],
                     "precision (destroyed)": t["per_class"]["destroyed"]["precision"]})
cmp_df = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
# Lead with a group only when its full trio exists - a partial group would have the headline
# comparing a model against itself.
active_group = next((g for g in reversed(RUN_GROUPS)
                     if not cmp_df.empty and (cmp_df["resolution"] == g).sum() == len(RUN_GROUPS[g])), "128px, 15 epochs")
group_df = cmp_df[cmp_df["resolution"] == active_group] if not cmp_df.empty else cmp_df


def lift(gdf):
    """Best-by-validation vs worst-by-validation: (best, worst, point gap, relative %) on test macro-F1."""
    best, worst = gdf["val macro F1"].idxmax(), gdf["val macro F1"].idxmin()
    b, w = gdf.loc[best, "test macro F1"], gdf.loc[worst, "test macro F1"]
    return best, worst, (b - w) * 100, (b / w - 1) * 100

tab_overview, tab_map, tab_explain, tab_prob, tab_sql, tab_real, tab_quality = st.tabs(
    ["🏠 Overview", "🗺️ Map & hotspots", "🧠 Grad-CAM", "🔎 Probabilistic DB", "🗄️ Database & SQL",
     "🌍 Real-world check", "📈 Quality"])

# ============================================================== TAB: Overview & honesty
with tab_overview:
    if not group_df.empty:
        # Correctness fix: the model that goes forward must be chosen on VALIDATION macro-F1,
        # never on the test score - picking by test score is the model leaking into its own
        # selection criterion. (It happens to be the same winner here, but the method matters.)
        best_row, worst_row, point_diff, rel_lift = lift(group_df)
        n_disasters = len({r[0] for r in db.execute("SELECT DISTINCT disaster FROM buildings").fetchall()})
        total_buildings = db.execute("SELECT COUNT(*) FROM buildings").fetchone()[0]
        hc = st.columns(4)
        for col, html in zip(hc, [
            hero_tile(f"{total_buildings:,}", "buildings", BLUE),
            hero_tile(str(n_disasters), "disasters", ORANGE),
            hero_tile(pretty(best_row), f"best architecture in this experiment, by validation ({best_row})", AQUA),
            hero_tile(f"+{point_diff:.1f} pts", f"test macro-F1 vs {pretty(worst_row)} ({rel_lift:+.1f}% rel.)", VIOLET),
        ]):
            col.markdown(html, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    groups_present = [g for g in RUN_GROUPS if not cmp_df.empty and (cmp_df["resolution"] == g).any()] or [active_group]
    st.radio("Experiment", groups_present,
             index=groups_present.index(active_group) if active_group in groups_present else 0,
             key="active_group_radio", horizontal=True,
             help="Each group trains the three architectures with the same settings; only the architecture differs.")
    group_df = cmp_df[cmp_df["resolution"] == st.session_state.active_group_radio] if not cmp_df.empty else cmp_df

    # Same architecture, only the crop resolution changed - the direct answer to "does more pixels help?"
    if not cmp_df.empty and {"siamese_fm_focal", "siamese_fm_focal_192"} <= set(cmp_df.index):
        lo_r, hi_r = cmp_df.loc["siamese_fm_focal"], cmp_df.loc["siamese_fm_focal_192"]
        gap = (hi_r["test macro F1"] - lo_r["test macro F1"]) * 100
        st.info(f"🔍 **Resolution effect (same architecture, siamese_fm):** 128px → 192px changes test macro-F1 "
                f"{lo_r['test macro F1']:.3f} → {hi_r['test macro F1']:.3f} (**{gap:+.1f} pts**), "
                f"destroyed recall {lo_r['recall (destroyed)']:.3f} → {hi_r['recall (destroyed)']:.3f}. "
                f"Note the 192px run used 12 epochs vs 15 at 128px.")

    with st.expander("📊 Architecture comparison — post-only vs. Siamese vs. Siamese feature-map fusion", expanded=True):
        if len(group_df) == 1:
            only = group_df.index[0]
            st.info(f"Only `{only}` has finished at this resolution so far - the architecture comparison needs "
                    f"all three. Its test macro-F1 is {group_df.loc[only, 'test macro F1']:.3f} "
                    f"(95% CI {group_df.loc[only, 'macro F1 95% CI']}).")
        elif not group_df.empty:
            best_row, worst_row, point_diff, rel_lift = lift(group_df)
            bc1, bc2 = st.columns([1, 1])
            with bc1:
                fig = go.Figure()
                for metric, color in [("val macro F1", BLUE), ("test macro F1", ORANGE), ("recall (destroyed)", AQUA)]:
                    fig.add_bar(x=group_df.index, y=group_df[metric], name=metric, marker_color=color,
                                marker_line_width=0, hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>")
                fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_range=[0, 1], height=360)
                st.plotly_chart(plotly_theme(fig, "Validation & test metrics by architecture"), width="stretch",
                                 config={"displayModeBar": False})
            with bc2:
                show_cols = ["val macro F1", "test macro F1", "macro F1 95% CI", "accuracy"]
                st.dataframe(group_df[show_cols].style.format(
                    {"val macro F1": "{:.3f}", "test macro F1": "{:.3f}", "accuracy": "{:.3f}"}
                ).highlight_max(subset=["val macro F1"], color=f"{AQUA}30"), width="stretch", height=38 + 35 * len(group_df))
            st.success(f"🏆 Chosen by **validation** macro-F1 (not test - that would be the model peeking at its own "
                       f"exam answers): **{best_row}** ({group_df.loc[best_row, 'architecture']}), "
                       f"**+{point_diff:.1f} percentage points** test macro-F1 over the weakest architecture tried "
                       f"({rel_lift:+.1f}% relative). 95% CI from bootstrapping each model's own test confusion matrix "
                       f"(a single training run per architecture, so treat small gaps with caution).")

            # The trade-off the headline number hides: better macro-F1 is not strictly better on every axis
            # that matters for rescue triage.
            recall_best = group_df["recall (destroyed)"].idxmax()
            if recall_best != best_row:
                r_best, r_top = group_df.loc[recall_best, "recall (destroyed)"], group_df.loc[best_row, "recall (destroyed)"]
                st.warning(
                    f"⚠️ **Hidden trade-off for rescue use:** `{recall_best}` finds more of the actually-destroyed "
                    f"buildings (recall **{r_best:.1%}**) than the overall-best `{best_row}` (recall **{r_top:.1%}**). "
                    f"For search-and-rescue prioritisation, missing a destroyed building is worse than a false alarm, "
                    f"so recall-on-destroyed is arguably the number that matters most here — and the overall-best "
                    f"model is **not** the best on it.")
            precisions = group_df["precision (destroyed)"]
            plist = ", ".join(f"{m} {p:.0%}" for m, p in precisions.items())
            false_share = 1 - precisions.min()
            st.info(f"📎 **Destroyed-class precision:** {plist}. For the weakest model, {false_share:.0%} of its "
                    f"'destroyed' calls are false alarms. The MC-Dropout uncertainty routing (Probabilistic DB tab) "
                    f"exists for this: unsure calls go to a person instead of being trusted as they are.")
        else:
            st.info("Train post / siamese / siamese_fm first to populate this comparison.")

versions = [r[0] for r in db.execute("SELECT DISTINCT model_version FROM predictions").fetchall()]
if not versions:
    with tab_overview:
        st.info("No predictions yet. Run `python -m aftershock.predict --ckpt runs/<name>/best.pt` first for the map, "
                "queries and review queue in the other tabs.")
    st.stop()

# Default to the final model = best by VALIDATION macro-F1 among models that have stored predictions.
val_scores = {v: json.loads((Path(RUNS_DIR) / v / "metrics.json").read_text()).get("best_val_macro_f1", -1)
              for v in versions if (Path(RUNS_DIR) / v / "metrics.json").exists()}
final_version = max(val_scores, key=val_scores.get) if val_scores else versions[0]
# An ensemble / calibrated result has no single network file. Grad-CAM and the SQL functions then use
# the best single network (by validation macro-F1) that has a checkpoint.
_nets = [v for v in sorted(val_scores, key=val_scores.get, reverse=True) if (Path(RUNS_DIR) / v / "best.pt").exists()]
net_version = _nets[0] if _nets else final_version


def prob_source(v):
    """How the stored probabilities of model v were made, from runs/<v>/metrics.json or mc_test.json."""
    run = Path(RUNS_DIR) / v
    try:
        m = json.loads((run / "metrics.json").read_text())
        if "chosen" in m:  # result of scripts/improve_model.py
            return f"{m['chosen']}, {m.get('passes', '?')} MC-Dropout passes per model"
        mc = run / "mc_test.json"
        if mc.exists():
            return f"mean of {json.loads(mc.read_text()).get('passes', 20)} MC-Dropout passes"
    except Exception:
        pass
    return "MC-Dropout"


def _count_p(sql, params):
    try:
        return db.execute(sql, params).fetchone()[0]
    except Exception:  # table not installed on this database
        return None


# Header numbers are for the FINAL model only (adding rows of several models would double-count buildings).
hero_stats = [(f"{n:,}", label) for n, label in [
    (_count("SELECT COUNT(*) FROM buildings"), "buildings in the DB"),
    (_count_p("SELECT COUNT(*) FROM predictions WHERE model_version = ?", (final_version,)),
     "test buildings scored with probabilities"),
    (_count_p("SELECT COUNT(*) FROM review_queue WHERE status = 'pending' AND model_version = ?", (final_version,)),
     "queued for human review (DB trigger)"),
    (_count_p("SELECT COUNT(*) FROM hotspots WHERE model_version = ?", (final_version,)), "damage hotspots (DBSCAN)"),
] if n is not None]
with hero_slot:
    render_hero(hero_stats, final_version)
st.sidebar.markdown("""<div class="as-brand"><div class="as-brand-logo">🛰️</div>
<div><div class="as-brand-name">AfterShock</div><div class="as-brand-sub">Damage control panel</div></div></div>
<div class="as-side-label">Model &amp; area</div>""", unsafe_allow_html=True)
version = st.sidebar.selectbox("Model", versions, help="Used by every tab. '(final)' = best validation macro-F1.",
                               index=versions.index(final_version), format_func=lambda v: f"{v} (final)" if v == final_version else v)
disasters = [r[0] for r in db.execute("SELECT DISTINCT disaster FROM buildings").fetchall()]
default_disaster = next((d for d in PREFERRED_DEFAULT_DISASTERS if d in disasters), disasters[0] if disasters else None)
disaster = st.sidebar.selectbox("Disaster", disasters, index=disasters.index(default_disaster) if default_disaster in disasters else 0)
st.sidebar.markdown('<div class="as-side-label">Filters</div>', unsafe_allow_html=True)
min_conf = st.sidebar.slider("Minimum confidence on map", 0.0, 1.0, 0.0, 0.05)
review_threshold = st.sidebar.slider("Review if uncertainty ≥", 0.0, 1.0, 0.6, 0.05,
                                     help="Buildings at or above this MC-Dropout uncertainty go to a person.")

df = pd.DataFrame(db.execute(
    "SELECT b.uid, b.lon, b.lat, b.label, b.split, b.image_id, p.pred_idx, p.confidence, p.uncertainty FROM buildings b "
    "JOIN predictions p ON p.uid = b.uid WHERE p.model_version = ? AND b.disaster = ?", (version, disaster)).fetchall(),
    columns=["uid", "lon", "lat", "true_label", "split", "image_id", "pred_idx", "confidence", "uncertainty"])
df["predicted"] = df["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])
df_test = df[df["split"] == "test"]  # the only split that's honest to score agreement on

with tab_overview:
    st.markdown(f"#### {disaster.replace('-', ' ').title()} · model `{version}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test buildings scored", f"{len(df):,}")
    c2.metric("Predicted major / destroyed", f"{int(df['pred_idx'].isin([2, 3]).sum()):,}")
    c3.metric("To human review", f"{int((df['uncertainty'] >= review_threshold).sum()):,}",
              help=f"uncertainty ≥ {review_threshold:.2f} (sidebar)")
    c4.metric("Test accuracy", f"{(df_test['predicted'] == df_test['true_label']).mean():.0%}" if len(df_test) else "–",
              help="Share of test buildings whose predicted class equals the xBD label. Test split only: the model "
                   "never trained on these buildings.")

if df.empty:
    n_total = db.execute("SELECT COUNT(*) FROM buildings WHERE disaster = ?", (disaster,)).fetchone()[0]
    n_test = db.execute("SELECT COUNT(*) FROM buildings WHERE disaster = ? AND split = 'test'", (disaster,)).fetchone()[0]
    tab_overview.warning(
        f"**No predictions to show for {disaster.replace('-', ' ').title()}.** This disaster has "
        f"{n_total} buildings total, but the deterministic train/val/test split (hashed per tile, so "
        f"neighbouring buildings never leak across splits) happened to put **{n_test} of them in the test "
        f"split** — and `predict` only scores the test split. This is a real quirk of how few tiles this "
        f"disaster has, not a bug; pick a different disaster from the sidebar to see the map, hotspots and "
        f"queries below.")

# ============================================================== TAB: Map & hotspots
with tab_map:
    # Map full width (the 3D city needs room); hotspot + per-disaster tables below it.
    left, right = st.container(), st.container()
    with left:
        st.subheader("🗺️ Damage map")
        shown = df[(df["confidence"] >= min_conf) & df["lon"].notna()]
        hotspot_rows = db.execute("SELECT cluster_id, n_buildings, lon, lat, mean_confidence FROM hotspots "
                                   "WHERE model_version = ? AND disaster = ?", (version, disaster)).fetchall()
        view_mode = st.radio("View", ["🏙️ 3D city", "🛰️ 2D satellite"], horizontal=True,
                             label_visibility="collapsed", key="map_view_mode")
        # A disaster spans tens of km, so the plain mean can land on an empty field. Both views open on
        # the biggest hotspot, else on the ~1 km grid cell with the most major/destroyed predictions.
        if len(shown):
            if hotspot_rows:
                top = max(hotspot_rows, key=lambda r: r[1])
                c_lon, c_lat = float(top[2]), float(top[3])
            else:
                sev = shown[shown["pred_idx"] >= 2] if (shown["pred_idx"] >= 2).any() else shown
                cell = sev.assign(gx=(sev["lon"] / 0.01).round(), gy=(sev["lat"] / 0.01).round()).groupby(["gx", "gy"])
                members = cell.get_group(cell.size().idxmax())
                c_lon, c_lat = float(members["lon"].mean()), float(members["lat"].mean())
        if len(shown):
            # Real footprint polygons (lon/lat) for every shown building - used by both views.
            fp = dict(db.execute(
                "SELECT b.uid, b.footprint_wkt FROM buildings b JOIN predictions p ON p.uid = b.uid "
                "WHERE p.model_version = ? AND b.disaster = ?", (version, disaster)).fetchall())
        if len(shown) and view_mode.startswith("🏙️"):
            # Each footprint is extruded on satellite imagery: height + colour = predicted damage (app/city3d.py).
            render_city3d(shown, fp, hotspot_rows, (c_lon, c_lat), disaster.replace("-", " ").title(), height=640)
            st.caption("Buildings rise from their real footprints; destroyed ones pulse red, DBSCAN hotspots are violet "
                       "rings. The camera orbits until you touch the map. Hover a building for its confidence.")
        elif len(shown):
            fmap = folium.Map(location=[c_lat, c_lon], zoom_start=17, tiles=None,
                               scrollWheelZoom=False)  # don't hijack the page's scroll - click the map to zoom instead
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri, Maxar, Earthstar Geographics", name="Satellite", control=False).add_to(fmap)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="Labels", control=False, overlay=True).add_to(fmap)
            # Outline every building's real footprint polygon (stored in buildings.footprint_wkt, lon/lat)
            # in its predicted-damage colour, so the marks sit exactly on the roofs in the satellite image.
            features = []
            for r in shown.itertuples():
                if fp.get(r.uid):
                    features.append({"type": "Feature", "geometry": shapely_wkt.loads(fp[r.uid]).__geo_interface__,
                                     "properties": {"damage": r.predicted, "sure": f"{r.confidence:.0%}",
                                                    "uncertainty": f"{r.uncertainty:.2f}", "color": COLORS[r.pred_idx],
                                                    "severe": int(r.pred_idx >= 2)}})
                else:
                    folium.CircleMarker([r.lat, r.lon], radius=5, color="white", weight=1, fill=True,
                                        fill_color=COLORS[r.pred_idx], fill_opacity=0.9).add_to(fmap)
            if features:
                folium.GeoJson(
                    {"type": "FeatureCollection", "features": features}, name="Buildings",
                    style_function=lambda f: {"color": f["properties"]["color"], "weight": 3 if f["properties"]["severe"] else 2,
                                              "fillColor": f["properties"]["color"],
                                              "fillOpacity": 0.45 if f["properties"]["severe"] else 0.2},
                    highlight_function=lambda f: {"weight": 5, "fillOpacity": 0.7},
                    tooltip=folium.GeoJsonTooltip(fields=["damage", "sure", "uncertainty"],
                                                  aliases=["Predicted", "Confidence", "Uncertainty"]),
                ).add_to(fmap)
            for cid, n_b, lon, lat, mean_conf in hotspot_rows:
                folium.Circle([lat, lon], radius=120, color=VIOLET, weight=3, fill=True, fill_color=VIOLET, fill_opacity=0.12,
                              tooltip=f"Hotspot {cid}: {n_b} severe buildings, {mean_conf:.0%} mean confidence").add_to(fmap)
            st_folium(fmap, height=520, use_container_width=True)
            st.caption("Click the map before scrolling/zooming it - the page keeps scrolling normally otherwise.")
        else:
            st.info("No buildings pass the confidence filter for this disaster.")
        st.markdown(f"""
        <span class="legend-chip"><span class="legend-dot" style="background:{GOOD}"></span>no damage</span>
        <span class="legend-chip"><span class="legend-dot" style="background:{WARNING}"></span>minor</span>
        <span class="legend-chip"><span class="legend-dot" style="background:{SERIOUS}"></span>major</span>
        <span class="legend-chip"><span class="legend-dot" style="background:{CRITICAL}"></span>destroyed</span>
        <span class="legend-chip"><span class="legend-dot" style="background:{VIOLET}"></span>hotspot (DBSCAN)</span>
        """, unsafe_allow_html=True)

    with right:
        st.subheader(f"📍 Hotspots in {disaster.replace('-', ' ').title()}")
        if hotspot_rows:
            hs_df = pd.DataFrame(hotspot_rows, columns=["cluster", "buildings", "lon", "lat", "mean confidence"])
            st.dataframe(hs_df.style.format({"mean confidence": "{:.0%}", "lon": "{:.4f}", "lat": "{:.4f}"}),
                         hide_index=True, width="stretch", height=min(38 + 35 * len(hs_df), 320))
        else:
            st.info(f"No DBSCAN hotspots stored for {disaster} with model `{version}`. Hotspots are mined only "
                    f"for the final model; pick another disaster or model from the sidebar.")
        st.subheader("🗂️ Per-disaster breakdown")
        pd_rows = db.execute(
            "SELECT b.disaster, b.label, p.pred_idx FROM buildings b JOIN predictions p ON p.uid = b.uid "
            "WHERE p.model_version = ? AND b.split = 'test'", (version,)).fetchall()
        if pd_rows:
            pdf = pd.DataFrame(pd_rows, columns=["disaster", "true_label", "pred_idx"])
            pdf["predicted"] = pdf["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])
            per_disaster = pdf.groupby("disaster").apply(
                lambda g: pd.Series({"test buildings": len(g), "accuracy": (g["predicted"] == g["true_label"]).mean()}),
                include_groups=False)
            per_disaster = per_disaster.sort_values("test buildings", ascending=False)
            per_disaster["accuracy"] = per_disaster["accuracy"] * 100
            st.dataframe(per_disaster, width="stretch", height=38 + 35 * len(per_disaster),  # no empty filler rows
                         column_config={"test buildings": st.column_config.NumberColumn(format="localized"),
                                        "accuracy": st.column_config.ProgressColumn("test accuracy", min_value=0,
                                                                                    max_value=100, format="%.0f%%")})
        else:
            st.caption("No test-split predictions stored for this model yet.")

# ============================================================== TAB: Explainability
with tab_explain:
    left, right = st.columns([5, 6])
    with left:
        st.subheader("🧑‍⚖️ Human review queue")
        queue = df[df["uncertainty"] >= review_threshold].sort_values("uncertainty", ascending=False).copy()
        queue["uid_short"] = queue["uid"].str.slice(0, 8) + "…"
        # confidence is stored as a probability (0-1); show it as a percentage bar.
        queue["conf_pct"] = queue["confidence"] * 100
        qa, qb = st.columns(2)
        qa.metric("Queued for review", f"{len(queue):,}", help=f"Buildings with MC-Dropout uncertainty ≥ {review_threshold:.2f} "
                  "(sidebar). The database trigger uses its own fixed rule: uncertainty ≥ 0.5 or confidence < 0.6.")
        qb.metric("Share of this disaster", f"{len(queue) / max(len(df), 1):.0%}")
        st.dataframe(
            queue[["uid_short", "predicted", "conf_pct", "uncertainty"]].rename(columns={"uid_short": "uid"}),
            hide_index=True, width="stretch", height=280,
            column_config={
                "uid": st.column_config.TextColumn("building", width="small"),
                "predicted": st.column_config.TextColumn("model says"),
                "conf_pct": st.column_config.ProgressColumn("confidence", min_value=0, max_value=100, format="%.0f%%"),
                "uncertainty": st.column_config.NumberColumn("uncert.", format="%.2f", width="small"),
            })
        st.caption("Most uncertain first. Confidence = the model's top probability; uncertainty = normalised entropy of the "
                   f"stored probabilities ({prob_source(version)}); 0 = sure, 1 = no idea. Pick a building below to inspect it.")
        pick_from = queue["uid"].tolist() + df.sort_values("confidence")["uid"].tolist()[:50]
        uid = st.selectbox("Building (uncertain ones first)", pick_from, format_func=lambda u: u[:8] + "…")
    with right:
        if uid:
            # Verdict card: the stored probabilistic row for this building, next to the xBD ground truth.
            prow = db.execute("SELECT p_no, p_minor, p_major, p_destroyed, pred_idx, confidence, uncertainty "
                              "FROM predictions WHERE uid = ? AND model_version = ?", (uid, version)).fetchone()
            truth = df.loc[df["uid"] == uid, "true_label"]
            if prow:
                *probs4, p_idx, p_conf, p_unc = prow
                col = COLORS[p_idx]
                # If the model is unsure, don't headline the arg-max class as if it were an answer:
                # say "Unsure" and show the top-2 classes, which is what sends it to a person.
                order = sorted(range(4), key=lambda i: probs4[i], reverse=True)
                unsure = p_unc >= review_threshold
                if unsure:
                    col = INK_MUTED
                    headline = (f'Unsure <span style="font-size:0.95rem; font-weight:500; color:#c9d0ea">· leaning '
                                f'<b style="color:{COLORS[order[0]]}">{DAMAGE_CLASSES[order[0]]}</b> '
                                f'{probs4[order[0]]:.0%} vs <b style="color:{COLORS[order[1]]}">{DAMAGE_CLASSES[order[1]]}</b> '
                                f'{probs4[order[1]]:.0%}</span>')
                else:
                    headline = DAMAGE_CLASSES[p_idx]
                truth_txt = truth.iloc[0] if len(truth) else "–"
                match = len(truth) and truth_txt == DAMAGE_CLASSES[p_idx]
                badge = (f'<span class="as-badge" style="background:{GOOD}33; color:{GOOD}">matches xBD label</span>' if match
                         else f'<span class="as-badge" style="background:{CRITICAL}33; color:#ff8a8a">xBD label: {truth_txt}</span>')
                route = (f'<span class="as-badge" style="background:{ORANGE}33; color:#ffb08a">→ human review</span>'
                         if p_unc >= review_threshold else
                         f'<span class="as-badge" style="background:{BLUE}33; color:#8cc4ff">model decides</span>')
                st.markdown(f"""<div class="as-verdict" style="--c:{col}">
                    <div class="as-verdict-k">Model verdict · <span class="uid-mono">{uid[:13]}…</span></div>
                    <div class="as-verdict-v">{headline}</div>
                    <div class="as-verdict-row"><div><div class="as-verdict-k">confidence</div><b>{p_conf:.0%}</b></div>
                    <div><div class="as-verdict-k">uncertainty</div><b>{p_unc:.2f}</b></div>
                    <div style="display:flex; gap:6px; align-items:end;">{badge}{route}</div></div>
                    {f'<div style="font-size:0.8rem; color:#aab3d6; margin-top:8px;">Uncertainty {p_unc:.2f} is above the '
                     f'review threshold ({review_threshold:.2f}), so "{DAMAGE_CLASSES[p_idx]}" is only the top guess, not a '
                     f'decision. The system hands this building to a person.</div>' if unsure else ''}</div>""",
                            unsafe_allow_html=True)
                figp = go.Figure(go.Bar(x=list(probs4), y=DAMAGE_CLASSES, orientation="h",
                                        marker_color=[COLORS[i] for i in range(4)], text=[f"{p:.0%}" for p in probs4],
                                        textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                figp.update_layout(height=200, xaxis_range=[0, 1.12], xaxis_tickformat=".0%", showlegend=False)
                figp.update_yaxes(autorange="reversed")
                st.plotly_chart(plotly_theme(figp, f"Stored probabilities · {prob_source(version)}"),
                                width="stretch", config={"displayModeBar": False})
            image_id_row = df.loc[df["uid"] == uid, "image_id"]
            context = context_crop_with_outline(image_id_row.iloc[0], uid) if len(image_id_row) else None
            if context:
                st.markdown("**Context** (native-resolution tile crop, building outlined in red)")
                ca, cb = st.columns(2)
                ca.image(context[0], caption="Before", width="stretch")
                cb.image(context[1], caption="After", width="stretch")
                st.caption("xBD imagery is ~0.5–0.8 m/pixel, so a house is only ~25 pixels wide. This wider crop "
                           "(contrast-stretched for display) is for a human reviewer, not what the model is shown.")
                cv = change_view(image_id_row.iloc[0], uid)
                if cv:
                    cimg, c_in, c_out = cv
                    ratio = c_in / max(c_out, 1e-6)
                    verdict_txt = ("clearly more than its surroundings" if ratio >= 1.5 else
                                   "about the same as its surroundings" if ratio >= 0.8 else "less than its surroundings")
                    flood = any(k in disaster for k in ("hurricane", "flood", "tsunami"))
                    cc1, cc2 = st.columns(2)
                    cc1.image(cimg, caption="What changed (yellow = strong change)", width="stretch")
                    cc2.markdown(f"**What changed?**\n\nInside the building (white outline): **{c_in:.2f}**  \n"
                                 f"Surroundings: **{c_out:.2f}**  \n(0 = identical, 1 = completely different)\n\n"
                                 f"→ the building changed **{verdict_txt}**.")
                    cc2.caption("Display aid only, not used by the model. Both photos are brightness-normalised first; "
                                "a small misalignment between the two dates, shadows and season also show up as change."
                                + (" Flood damage is mostly inside the house, so a roof can look unchanged from above; "
                                   "such buildings are exactly the ones the model is unsure about." if flood else ""))
            pre_png, post_png = db.crop_bytes(uid)
            net_for_cam = version if (Path(RUNS_DIR) / version / "best.pt").exists() else net_version
            try:
                net_px = json.loads((Path(RUNS_DIR) / net_for_cam / "metrics.json").read_text()).get("img_size", "?")
            except Exception:
                net_px = "?"
            st.markdown(f"**What the model sees** (stored {Image.open(io.BytesIO(pre_png)).width} px crop, "
                        f"resized to {net_px} px for `{net_for_cam}`)")
            a, b = st.columns(2)
            a.image(crisp(Image.open(io.BytesIO(pre_png))), caption="Before", width="stretch")
            b.image(crisp(Image.open(io.BytesIO(post_png))), caption="After", width="stretch")
            cam_path = Path(RUNS_DIR) / version / "explanations" / f"{uid}.png"
            ckpt_path = Path(RUNS_DIR) / (version if (Path(RUNS_DIR) / version / "best.pt").exists() else net_version) / "best.pt"
            if cam_path.exists():
                st.image(crisp(Image.open(cam_path), size=960), caption="Grad-CAM: before | after | heatmap", width="stretch")
            elif ckpt_path.exists():
                with st.spinner("Computing Grad-CAM for this building (not in the precomputed batch)…"):
                    panel, pred, conf = live_gradcam(str(ckpt_path), db, uid)
                st.image(crisp(panel, size=960), caption=f"Grad-CAM (live) of network {net_for_cam}: before | after | "
                                                          f"heatmap — this network alone predicts {pred} ({conf:.0%})",
                         width="stretch")
                if net_for_cam != version:
                    st.caption(f"`{version}` combines several networks, and Grad-CAM needs a single network, so the "
                               f"heatmap is from `{net_for_cam}` (its best member by validation). Its own prediction can "
                               f"differ from the combined verdict above.")
            else:
                st.caption("No checkpoint available to compute Grad-CAM for this model.")

# ============================================================== TAB: Probabilistic DB queries
with tab_prob:
    st.subheader("📐 Calibration: is a 90%-confident prediction actually right 90% of the time?")
    calib_rows = db.execute(
        "SELECT b.label, p.pred_idx, p.confidence FROM buildings b JOIN predictions p ON p.uid = b.uid "
        "WHERE p.model_version = ? AND b.split = 'test'", (version,)).fetchall()
    if calib_rows:
        cdf = pd.DataFrame(calib_rows, columns=["true_label", "pred_idx", "confidence"])
        cdf["predicted"] = cdf["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])
        cdf["correct"] = cdf["predicted"] == cdf["true_label"]
        cdf["bucket"] = pd.cut(cdf["confidence"], bins=[0, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
        reliability = cdf.groupby("bucket", observed=True).agg(n=("correct", "size"), observed_accuracy=("correct", "mean"),
                                                                 mean_confidence=("confidence", "mean")).reset_index()
        fig3 = go.Figure()
        fig3.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=INK_MUTED, dash="dash"), name="perfectly calibrated")
        fig3.add_scatter(x=reliability["mean_confidence"], y=reliability["observed_accuracy"], mode="markers+lines",
                          marker=dict(size=10, color=BLUE), line=dict(color=BLUE), name="this model",
                          hovertemplate="confidence ~%{x:.0%}<br>observed accuracy %{y:.0%}<extra></extra>")
        fig3.update_layout(xaxis_title="mean predicted confidence", yaxis_title="observed accuracy",
                            xaxis_range=[0.4, 1], yaxis_range=[0.4, 1], height=380)
        st.plotly_chart(plotly_theme(fig3, "Reliability diagram (test split)"), width="stretch",
                         config={"displayModeBar": False})
        st.caption("Points below the dashed line mean the model is over-confident in that bucket (says e.g. 90% sure, "
                   "right less often than 90%) - exactly what MC-Dropout uncertainty is meant to catch downstream.")
    else:
        st.info("No test-split predictions stored for this model yet.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🧊 3D probability space: what the database actually stores")
    st.caption("Every point is one test building of the selected disaster, placed by its stored probabilities "
               "(p_minor, p_major, p_destroyed; p_no = 1 - the rest). Confident predictions sit near a corner, "
               "uncertain ones float in the middle; marker size = MC-Dropout uncertainty. Drag to rotate.")
    prob_rows = db.execute(
        "SELECT p.p_minor, p.p_major, p.p_destroyed, p.pred_idx, p.uncertainty FROM buildings b "
        "JOIN predictions p ON p.uid = b.uid WHERE p.model_version = ? AND b.disaster = ? AND b.split = 'test' "
        "LIMIT 4000", (version, disaster)).fetchall()
    if prob_rows:
        pdf3 = pd.DataFrame(prob_rows, columns=["p_minor", "p_major", "p_destroyed", "pred_idx", "uncertainty"])
        fig3d = go.Figure()
        for i, cls_name in enumerate(DAMAGE_CLASSES):
            part = pdf3[pdf3["pred_idx"] == i]
            if len(part):
                fig3d.add_scatter3d(x=part["p_minor"], y=part["p_major"], z=part["p_destroyed"], mode="markers",
                                    name=cls_name, marker=dict(size=2 + 6 * part["uncertainty"], color=COLORS[i],
                                                               opacity=0.75, line=dict(width=0)),
                                    hovertemplate="p_minor %{x:.2f}<br>p_major %{y:.2f}<br>p_destroyed %{z:.2f}"
                                                  f"<extra>{cls_name}</extra>")
        axis = dict(range=[0, 1], gridcolor=GRID, backgroundcolor="rgba(0,0,0,0)", showbackground=False)
        fig3d.update_layout(height=520, scene=dict(xaxis=dict(title="p_minor", **axis), yaxis=dict(title="p_major", **axis),
                                                   zaxis=dict(title="p_destroyed", **axis), aspectmode="cube",
                                                   camera=dict(eye=dict(x=1.6, y=1.4, z=0.9))))
        st.plotly_chart(plotly_theme(fig3d, f"Stored class probabilities, {len(pdf3):,} test buildings"),
                        width="stretch", config={"displayModeBar": False})
    else:
        st.info("No test-split predictions for this disaster with this model.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("🔎 Probabilistic + spatial query")
    st.caption("The same kind of query the project report shows in SQL - e.g. \"destroyed buildings within 2km of a "
               "point, predicted with at least 80% confidence\" - run live against the database, with a confidence "
               "threshold instead of a bare class label.")
    all_test_buildings = df_test[["uid", "lon", "lat"]].dropna()
    if len(all_test_buildings):
        # Default the query to the biggest hotspot's neighbourhood, not an arbitrary (possibly
        # empty-looking) building - an empty-by-default panel reads as broken, not "try the sliders".
        hotspots_here = db.execute("SELECT lon, lat FROM hotspots WHERE model_version = ? AND disaster = ? "
                                    "ORDER BY n_buildings DESC LIMIT 1", (version, disaster)).fetchone()
        # Default class: the severe class (destroyed, else major) with the most confident (>= 0.5) predictions here.
        sure = df_test[(df_test["confidence"] >= 0.5) & df_test["lon"].notna()]
        default_cls = next((c for c in (3, 2) if (sure["pred_idx"] == c).sum() >= 5), 3)
        if hotspots_here:
            cx, cy = hotspots_here
        else:  # densest ~500 m cell of confident predictions of that class
            pool = sure[sure["pred_idx"] == default_cls]
            pool = pool if len(pool) else all_test_buildings
            cells = pool.groupby([(pool["lon"] / 0.005).round(), (pool["lat"] / 0.005).round()])
            g = cells.get_group(cells.size().idxmax())
            cx, cy = g["lon"].mean(), g["lat"].mean()
        d2 = (all_test_buildings["lon"] - cx) ** 2 + (all_test_buildings["lat"] - cy) ** 2
        default_uid = all_test_buildings.loc[d2.idxmin(), "uid"]
        uid_list = all_test_buildings["uid"].tolist()
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            center_uid = st.selectbox("Center on building", uid_list, index=uid_list.index(default_uid),
                                       format_func=lambda u: u[:10] + "…")
        with qcol2:
            radius_m = st.slider("Radius (metres)", 50, 3000, 500, 50)
        with qcol3:
            class_filter = st.selectbox("Damage class", ["(any)"] + DAMAGE_CLASSES, index=1 + default_cls)
        min_q_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.5, 0.05,
                                help="Starts at 0.5 so the default query isn't empty - raise it to see precision drop off.")
        center = all_test_buildings[all_test_buildings["uid"] == center_uid].iloc[0]
        pred_idx = DAMAGE_CLASSES.index(class_filter) if class_filter != "(any)" else None
        results = db.buildings_near(center["lon"], center["lat"], radius_m, version, pred_idx=pred_idx, min_conf=min_q_conf)
        # The exact statement db.buildings_near() runs on SQLite (same string, same parameter order).
        cond = "p.model_version = ? AND p.confidence >= ?" + (" AND p.pred_idx = ?" if pred_idx is not None else "")
        st.code("SELECT b.uid, b.lon, b.lat, p.pred_idx, p.confidence\nFROM buildings b\nJOIN predictions p ON p.uid = b.uid\n"
                f"WHERE {cond};", language="sql")
        params = [version, min_q_conf] + ([pred_idx] if pred_idx is not None else [])
        st.caption(f"Parameters: {params}. SQLite has no spatial index, so the radius filter (haversine distance "
                   f"≤ {radius_m} m) is applied to these rows in Python; on PostGIS the same filter is "
                   f"`ST_DWithin(b.geom::geography, ST_MakePoint(lon, lat)::geography, {radius_m})` inside the WHERE.")
        st.success(f"**{len(results)} buildings** within {radius_m} m of `{center_uid[:10]}…` match "
                   f"{'any damage class' if pred_idx is None else class_filter} with confidence ≥ {min_q_conf:.0%}.")
        if results:
            rdf = pd.DataFrame(results, columns=["uid", "lon", "lat", "pred_idx", "confidence"])
            rdf["predicted"] = rdf["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])
            st.dataframe(rdf[["uid", "predicted", "confidence", "lon", "lat"]].style.format({"confidence": "{:.0%}"}),
                         hide_index=True, width="stretch", height=240)
    else:
        st.info("No test-split buildings with coordinates for this disaster/model.")

# ============================================================== TAB: Database & SQL
_sql_lock = threading.Lock()


def _ro_uri(url: str) -> str:
    return Path(url.removeprefix("sqlite:///")).resolve().as_uri() + "?mode=ro"


@st.cache_resource(show_spinner=False)
def ro_connection(url: str):
    """Read-only connection for this tab: SQLite itself refuses any write made through it."""
    conn = sqlite3.connect(_ro_uri(url), uri=True, check_same_thread=False)
    register_functions(conn)  # entropy4 / expected_severity / damage_label - no model needed
    return conn


@st.cache_resource(show_spinner="Loading the final model into SQLite…")
def ai_connection(url: str, ckpt: str):
    conn = sqlite3.connect(_ro_uri(url), uri=True, check_same_thread=False)
    register_functions(conn, ckpt)  # adds ai_predict / ai_confidence / ai_probs backed by the network
    return conn


def run_sql(conn, sql: str, params=()):
    with _sql_lock:
        t0 = time.perf_counter()
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        ms = (time.perf_counter() - t0) * 1000
    return pd.DataFrame(rows, columns=[d[0] for d in cur.description]), ms


def check_constraints(create_sql: str):
    """Pull every CHECK (...) clause out of a CREATE TABLE statement (balanced parentheses)."""
    out, i = [], create_sql.find("CHECK")
    while i != -1:
        start = create_sql.find("(", i)
        depth, j = 0, start
        while j < len(create_sql):
            depth += {"(": 1, ")": -1}.get(create_sql[j], 0)
            if depth == 0:
                break
            j += 1
        out.append(" ".join(create_sql[start + 1:j].split()))
        i = create_sql.find("CHECK", j)
    return out


@st.cache_data(ttl=60, show_spinner=False)
def schema_overview(url: str):
    conn = ro_connection(url)
    rows = []
    for name, create_sql in run_sql(conn, "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
                                          "AND name NOT LIKE 'sqlite_%' ORDER BY name")[0].itertuples(index=False):
        n = run_sql(conn, f'SELECT COUNT(*) FROM "{name}"')[0].iat[0, 0]
        cols = run_sql(conn, f'PRAGMA table_info("{name}")')[0]
        pk = cols[cols["pk"] > 0].sort_values("pk")["name"].tolist()
        fks = run_sql(conn, f'PRAGMA foreign_key_list("{name}")')[0]
        parts = [f"PK ({', '.join(pk)})"] if pk else []
        parts += [f"FK {r['from']} → {r['table']}.{r['to']}" for _, r in fks.iterrows()]
        parts += [f"CHECK {c}" for c in check_constraints(create_sql or "")]
        nn = cols[(cols["notnull"] == 1) & (cols["pk"] == 0)]["name"].tolist()
        if nn:
            parts.append(f"NOT NULL {', '.join(nn)}")
        rows.append({"table": name, "rows": int(n), "keys / constraints": " · ".join(parts) or "-"})
    return pd.DataFrame(rows)


with tab_sql:
    conn_ro = ro_connection(db_url)
    has_ai_layer = run_sql(conn_ro, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'model_runs'")[0].iat[0, 0] == 1
    st.caption("This tab opens the database **read-only** (`mode=ro`): every query below is a fixed, parameterised "
               "statement, and SQLite would refuse a write even if one were attempted.")
    if not has_ai_layer:
        st.warning("The SQL/AI layer isn't installed in this database yet: run `python -m aftershock.sqlai --install`.")
    else:
        st.subheader("Schema")
        schema_df = schema_overview(db_url)
        st.dataframe(schema_df, hide_index=True, width="stretch", height=38 + 35 * len(schema_df),
                     column_config={"rows": st.column_config.NumberColumn(format="localized"),
                                    "keys / constraints": st.column_config.TextColumn(width="large")})

        st.subheader("Preset queries")
        PRESETS = {
            "Model leaderboard - RANK() window function over validation macro-F1": (
                "SELECT model_version, arch, img_size, val_macro_f1, test_macro_f1, rank_by_val\n"
                "FROM v_model_leaderboard\nORDER BY rank_by_val;", (), "v_model_leaderboard"),
            "Expected vs hard 'destroyed' count per disaster (probabilistic aggregation)": (
                "SELECT disaster, n_buildings, hard_destroyed, expected_destroyed, true_destroyed,\n"
                "       mean_uncertainty, accuracy\nFROM v_disaster_damage\nWHERE model_version = ?\n"
                "ORDER BY n_buildings DESC;", (version,), "v_disaster_damage"),
            "Threshold query: P(destroyed) ≥ 0.8 - and how many really are destroyed": (
                "SELECT COUNT(*)                 AS flagged_destroyed,\n"
                "       SUM(true_idx = 3)        AS truly_destroyed,\n"
                "       ROUND(AVG(true_idx = 3), 3) AS precision_at_0_8\nFROM v_predictions\n"
                "WHERE model_version = ? AND split = 'test' AND p_destroyed >= 0.8;", (version,), "v_predictions"),
            "Human review queue - top 20 most uncertain (filled by a trigger)": (
                "SELECT uid, reason, uncertainty, disaster\nFROM v_review_queue\nWHERE model_version = ?\nLIMIT 20;",
                (version,), "v_review_queue"),
            "Calibration - does 80% confidence mean 80% correct?": (
                "SELECT conf_bin, n, mean_confidence, accuracy\nFROM v_calibration\nWHERE model_version = ?\n"
                "ORDER BY conf_bin;", (version,), "v_calibration"),
            "Selective accuracy at 50 / 80 / 100% coverage (ROW_NUMBER() window function)": (
                "SELECT ROUND(100.0 * cut, 0)      AS coverage_pct,\n"
                "       ROUND(AVG(correct), 4)     AS accuracy,\n"
                "       MAX(total) - COUNT(*)      AS sent_to_human\n"
                "FROM v_uncertainty_rank,\n     (SELECT 0.5 AS cut UNION ALL SELECT 0.8 UNION ALL SELECT 1.0)\n"
                "WHERE model_version = ? AND rnk <= cut * total\nGROUP BY cut\nORDER BY cut;", (version,),
                "v_uncertainty_rank"),
        }
        choice = st.selectbox("Query", list(PRESETS))
        sql, params, view = PRESETS[choice]
        st.code(sql, language="sql")
        if params:
            st.caption(f"Parameters: model_version = '{version}' (the model selected in the sidebar)")
        result, ms = run_sql(conn_ro, sql, params)
        st.dataframe(result, hide_index=True, width="stretch")
        st.caption(f"{len(result)} row(s) · {ms:.1f} ms")
        view_sql = run_sql(conn_ro, "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = ?", (view,))[0]
        if len(view_sql):
            with st.expander(f"View definition: {view}"):
                st.code(view_sql.iat[0, 0], language="sql")

        st.subheader("Index vs full table scan")
        if st.button("Run EXPLAIN QUERY PLAN + timing"):
            q = "SELECT COUNT(*) FROM buildings{} WHERE split = 'test';"
            out = []
            for label, hint in (("uses buildings_split_idx", ""), ("forced full scan", " NOT INDEXED")):
                stmt = q.format(hint)
                plan = " / ".join(run_sql(conn_ro, "EXPLAIN QUERY PLAN " + stmt)[0]["detail"])
                total = 0.0
                for _ in range(20):
                    total += run_sql(conn_ro, stmt)[1]
                out.append({"variant": label, "SQL": stmt, "query plan": plan, "ms per query (avg of 20)": round(total / 20, 2)})
            st.session_state["index_demo"] = pd.DataFrame(out)
        if "index_demo" in st.session_state:
            st.dataframe(st.session_state["index_demo"], hide_index=True, width="stretch")

        st.subheader("AI inside SQL")
        final_ckpt = Path(RUNS_DIR) / net_version / "best.pt"
        ai_sql = ("SELECT b.uid,\n       damage_label(b.label_idx)            AS truth,\n"
                  "       ai_predict(c.pre_png, c.post_png)    AS ai,\n"
                  "       ai_confidence(c.pre_png, c.post_png) AS conf\n"
                  "FROM buildings b JOIN crops c ON c.uid = b.uid\nWHERE b.split = 'test'\nLIMIT 5;")
        st.code(ai_sql, language="sql")
        st.caption(f"`ai_predict()` / `ai_confidence()` are SQL functions registered on the connection that run the "
                   f"network `{net_version}` (best single network by validation macro-F1) on the PNG bytes stored in `crops`.")
        if st.button("Run the network inside SELECT", disabled=not final_ckpt.exists()):
            st.session_state["ai_demo"] = run_sql(ai_connection(db_url, str(final_ckpt)), ai_sql)
        if "ai_demo" in st.session_state:
            ai_df, ai_ms = st.session_state["ai_demo"]
            st.dataframe(ai_df, hide_index=True, width="stretch")
            st.caption(f"{ai_ms:.0f} ms for {len(ai_df)} buildings, on CPU")

        st.subheader("Trigger: uncertain predictions queue themselves for review")
        trig = run_sql(conn_ro, "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = 'trg_queue_uncertain'")[0]
        if len(trig):
            st.code(trig.iat[0, 0], language="sql")
        rq, _ = run_sql(conn_ro, "SELECT model_version, status, COUNT(*) AS n FROM review_queue "
                                 "GROUP BY model_version, status ORDER BY model_version, status")
        st.dataframe(rq, hide_index=True, width="stretch")
        st.caption(f"review_queue holds {int(rq['n'].sum()):,} rows in total.")

# ============================================================== TAB: Real-world check (CAL FIRE field inspections)
with tab_real:
    st.subheader("🌍 Does the model agree with inspectors on the ground?")
    st.markdown("After a California wildfire, **CAL FIRE inspectors visit every structure** and record its damage "
                "(the public *Damage Inspection* database, DINS). That is real-world evidence collected on foot, "
                "independent of the satellite labels and never seen by the model. Each satellite building is matched "
                "to its nearest inspection point **inside the database, with one SQL spatial join**.")
    try:
        field_disasters = [r[0] for r in db.execute("SELECT DISTINCT disaster FROM field_inspections").fetchall()]
    except Exception:
        field_disasters = []
    if not field_disasters:
        st.info("No field inspections loaded in this database yet. Run `python scripts/fetch_dins.py ...` and "
                "`python -m aftershock.fieldcheck --load ...` (see the module docstring).")
    else:
        fd = st.selectbox("Disaster with field data", field_disasters,
                          index=field_disasters.index(disaster) if disaster in field_disasters else 0, key="field_disaster")
        res = fieldcheck.compare(db, fd, version)
        inc = db.execute("SELECT GROUP_CONCAT(DISTINCT incident), COUNT(*) FROM field_inspections WHERE disaster = ?",
                         (fd,)).fetchone()
        xall, mv = res["xbd_vs_field"], res.get("model_vs_field", {"n": 0})
        # Score the xBD labels on the SAME buildings as the model when possible, so the two are directly comparable.
        xv = res["xbd_vs_field_same"] if mv["n"] else xall
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Field inspections", f"{inc[1]:,}", help=f"CAL FIRE DINS, incident: {inc[0]}")
        k2.metric("Buildings compared", f"{xv['n']:,}",
                  help=f"test buildings with both a model prediction and an inspection within the radius. "
                       f"All matched buildings (any split): {xall['n']:,}; mean distance {(res['dist'][0] or 0):.1f} m")
        if mv["n"]:
            k3.metric("Model vs inspectors", f"{mv['destroyed']['accuracy']:.1%}",
                      help=f"destroyed vs not, on {mv['n']:,} test buildings with an inspection. "
                           f"Recall {mv['destroyed']['recall']:.1%}, precision {mv['destroyed']['precision']:.1%}.")
        else:
            k3.metric("Model vs inspectors", "–", help="this model has no test predictions with a matched inspection")
        k4.metric("xBD labels vs inspectors", f"{xv['destroyed']['accuracy']:.1%}",
                  help=f"destroyed vs not, on the same {xv['n']:,} buildings. On all {xall['n']:,} matched buildings: "
                       f"{xall['destroyed']['accuracy']:.1%}")

        def cm_fig(cm, title, xname):
            z = np.array(cm, dtype=float)
            pct = z / np.maximum(z.sum(1, keepdims=True), 1)
            fig = go.Figure(go.Heatmap(z=pct, x=DAMAGE_CLASSES, y=DAMAGE_CLASSES,
                                       colorscale=[[0, "#0b1224"], [1, ACCENT]], zmin=0, zmax=1, showscale=False,
                                       text=[[f"{int(v):,}" for v in row] for row in z], texttemplate="%{text}",
                                       hovertemplate="field %{y}<br>" + xname +
                                       " %{x}<br>%{z:.0%} of this field class<extra></extra>"))
            fig.update_layout(height=330, xaxis_title=xname, yaxis_title="field inspectors")
            fig.update_yaxes(autorange="reversed")
            return plotly_theme(fig, title)

        c1, c2 = st.columns(2)
        if mv["n"]:
            c1.plotly_chart(cm_fig(mv["confusion_field_rows"], f"Model ({version}) vs inspectors · {mv['n']:,} test buildings",
                                   "model"), width="stretch", config={"displayModeBar": False})
        else:
            c1.info("Pick a model with test predictions for this disaster in the sidebar.")
        c2.plotly_chart(cm_fig(xv["confusion_field_rows"], f"xBD labels vs inspectors · same {xv['n']:,} buildings",
                               "xBD label"), width="stretch", config={"displayModeBar": False})
        if mv["n"]:
            comp = pd.DataFrame([
                {"compared with inspectors": f"Model ({version})", "buildings": mv["n"],
                 "same class": mv["exact"] * 100, "destroyed: accuracy": mv["destroyed"]["accuracy"] * 100,
                 "destroyed: recall": mv["destroyed"]["recall"] * 100,
                 "destroyed: precision": mv["destroyed"]["precision"] * 100},
                {"compared with inspectors": "xBD labels (human annotators)", "buildings": xv["n"],
                 "same class": xv["exact"] * 100, "destroyed: accuracy": xv["destroyed"]["accuracy"] * 100,
                 "destroyed: recall": xv["destroyed"]["recall"] * 100,
                 "destroyed: precision": xv["destroyed"]["precision"] * 100}])
            pct = {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%")
                   for c in ["same class", "destroyed: accuracy", "destroyed: recall", "destroyed: precision"]}
            st.dataframe(comp, hide_index=True, width="stretch", height=38 + 35 * 2, column_config=pct)
            st.caption(f"Same {mv['n']:,} test buildings in both rows, so the numbers compare directly. "
                       f"The xBD labels were drawn by human annotators from the same satellite images; the model "
                       f"learned from such labels. On all {xall['n']:,} matched buildings (every split) the xBD "
                       f"labels agree {xall['destroyed']['accuracy']:.1%} on destroyed vs not.")

        # Plain-language findings, computed from the matrices above
        cmx = np.array(xall["confusion_field_rows"])
        light, light_as_none = int(cmx[1].sum()), int(cmx[1][0])
        notes = [f"**Destroyed homes are clear from space.** The satellite labels agree with the inspectors on "
                 f"*destroyed vs not* for {xall['destroyed']['accuracy']:.1%} of all {xall['n']:,} matched buildings."]
        if mv["n"]:
            notes.append(f"**The model finds {mv['destroyed']['recall']:.1%} of the buildings inspectors recorded as "
                         f"destroyed**, and {mv['destroyed']['precision']:.1%} of its 'destroyed' calls are confirmed "
                         f"on the ground.")
        if light:
            notes.append(f"**Light damage is invisible from above.** Of {light:,} buildings inspectors rated "
                         f"'affected/minor' (1-25%), {light_as_none:,} are labelled *no-damage* even in xBD: scorched "
                         f"siding or a broken window cannot be seen in a roof-top image at ~0.5 m/pixel.")
        un = res.get("model_on_unrecorded")
        if un and un["n"]:
            notes.append(f"Before 2018 DINS listed only damaged structures, so {un['n']:,} test buildings have no "
                         f"inspection nearby ('not recorded as damaged'); the model calls {un['pred_destroyed']:,} of "
                         f"them destroyed. These are kept out of the scores above, not counted as right or wrong.")
        for n_ in notes:
            st.markdown("- " + n_)

        st.markdown("### 🏙️ Model vs inspectors in 3D: same buildings, one camera")
        st.caption("Left: what the model predicted from the satellite images. Right: what CAL FIRE inspectors recorded "
                   "on the ground. Both maps share one camera, so rotating or zooming one moves the other. "
                   "Press **Highlight differences** to fade every building where they agree.")
        dis = pd.DataFrame(db.execute(
            "SELECT uid, lon, lat, pred_idx, field_idx, field_raw, confidence, dist_m FROM v_field_vs_model "
            "WHERE disaster = ? AND model_version = ? AND field_idx IS NOT NULL", (fd, version)).fetchall(),
            columns=["uid", "lon", "lat", "pred_idx", "field_idx", "field_raw", "confidence", "dist_m"])
        if len(dis):
            def verdict(r):
                if (r.pred_idx == 3) == (r.field_idx == 3):
                    return "agree"
                return "missed (field: destroyed)" if r.field_idx == 3 else "false alarm"
            vcol = {"agree": GOOD, "missed (field: destroyed)": CRITICAL, "false alarm": WARNING}
            dis["verdict"] = [verdict(r) for r in dis.itertuples()]
            fpf = dict(db.execute(
                "SELECT b.uid, b.footprint_wkt FROM buildings b JOIN v_field_vs_model v ON v.uid = b.uid "
                "WHERE v.disaster = ? AND v.model_version = ? AND v.field_idx IS NOT NULL", (fd, version)).fetchall())
            # open on the ~1 km cell with the most disagreements (else the most buildings)
            focus = dis[dis["verdict"] != "agree"] if (dis["verdict"] != "agree").any() else dis
            cells = focus.groupby([(focus["lon"] / 0.01).round(), (focus["lat"] / 0.01).round()])
            fcell = cells.get_group(cells.size().idxmax())
            vc = dis["verdict"].value_counts()
            render_compare3d(dis, fpf, (float(fcell["lon"].mean()), float(fcell["lat"].mean())),
                             f"🤖 Model · {version}", "👷 CAL FIRE inspectors (ground truth)",
                             sub_a=f"{len(dis):,} test buildings · prediction from satellite images",
                             sub_b=f"field inspection matched within {fieldcheck_radius(db, fd):.0f} m",
                             height=560)
            st.markdown(" ".join(f'<span class="legend-chip"><span class="legend-dot" style="background:{c}"></span>'
                                 f'{v} ({int(vc.get(v, 0)):,})</span>' for v, c in vcol.items()), unsafe_allow_html=True)

            st.markdown("#### 📋 Building-by-building data")
            show = st.radio("Show", ["Disagreements only", "All matched buildings"], horizontal=True,
                            key="field_table_mode", label_visibility="collapsed")
            tbl = dis if show.startswith("All") else dis[dis["verdict"] != "agree"]
            tbl = tbl.assign(building=tbl["uid"].str.slice(0, 10) + "…",
                             model=tbl["pred_idx"].map(lambda i: DAMAGE_CLASSES[i]),
                             conf=tbl["confidence"] * 100)
            st.dataframe(tbl[["building", "model", "conf", "field_raw", "verdict", "dist_m", "lat", "lon"]],
                         hide_index=True, width="stretch", height=min(38 + 35 * len(tbl), 420),
                         column_config={
                             "model": st.column_config.TextColumn("model says"),
                             "conf": st.column_config.ProgressColumn("model confidence", min_value=0, max_value=100,
                                                                     format="%.0f%%"),
                             "field_raw": st.column_config.TextColumn("inspectors recorded"),
                             "dist_m": st.column_config.NumberColumn("match distance (m)", format="%.1f"),
                             "lat": st.column_config.NumberColumn(format="%.5f"),
                             "lon": st.column_config.NumberColumn(format="%.5f")})
            st.caption(f"{len(tbl):,} rows. Use the table's download button for a CSV. 'Agree' is judged on destroyed vs "
                       f"not, the distinction a satellite can see; the 4-class confusion is in the matrices above.")

        with st.expander("🗄️ The SQL behind this (spatial join + window function)"):
            st.code(fieldcheck.MATCH_SQL.strip(), language="sql")
            st.caption("Candidate pairs come from a small latitude/longitude box (served by the index "
                       "field_inspections_geo_idx); ROW_NUMBER() OVER (PARTITION BY building ORDER BY distance) keeps "
                       "the nearest inspection. Results are stored in field_matches and read through the views "
                       "v_field_compare and v_field_vs_model.")
            st.code("SELECT field_raw, pred_idx, COUNT(*) AS buildings\nFROM v_field_vs_model\n"
                    "WHERE disaster = 'santa-rosa-wildfire' AND model_version = ?\nGROUP BY field_raw, pred_idx;",
                    language="sql")
        st.caption("Source: CAL FIRE Damage Inspection Program (DINS), public data; only damage level and location are "
                   "stored. Mapping: No Damage → no-damage; Affected (1-10%) and Minor (10-25%) → minor; "
                   "Major (25-50%) → major; Destroyed (>50%) → destroyed.")


# ============================================================== TAB: Model quality
with tab_quality:
    metrics_file = Path(RUNS_DIR) / version / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text()).get("test", {})
        if m:
            st.caption("All numbers below are computed on the held-out **test split only** (never seen in training).")
            pc = pd.DataFrame(m["per_class"]).T
            mc1, mc2 = st.columns([3, 2])
            with mc1:
                fig2 = go.Figure()
                for metric, color in [("precision", BLUE), ("recall", ORANGE), ("f1", AQUA)]:
                    if metric in pc.columns:
                        fig2.add_bar(x=pc.index, y=pc[metric], name=metric, marker_color=color, marker_line_width=0,
                                     hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>")
                fig2.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, yaxis_range=[0, 1], height=340)
                st.plotly_chart(plotly_theme(fig2, "Per-class precision / recall / F1 (test)"), width="stretch",
                                 config={"displayModeBar": False})
            with mc2:
                st.dataframe(pc, width="stretch", height=38 + 35 * len(pc))
            lo, hi = bootstrap_macro_f1_ci(m["confusion_matrix"])
            st.caption(f"Macro F1 {m['macro_f1']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) · harmonic F1 {m['harmonic_f1']:.3f} "
                       f"· accuracy {m['accuracy']:.3f}")
    else:
        st.info("No metrics.json for this model yet.")
