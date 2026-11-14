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
import pydeck as pdk
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFilter
from shapely import wkt as shapely_wkt
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aftershock.config import DAMAGE_CLASSES, DATA_DIR, RUNS_DIR  # noqa: E402
from aftershock.data import png_to_tensor  # noqa: E402
from aftershock.db import DB  # noqa: E402
from aftershock.explain import explanation_panel, gradcam  # noqa: E402
from aftershock.model import load_checkpoint  # noqa: E402
from aftershock.sqlai import register_functions  # noqa: E402

SHORT_NAMES = {"post_focal": "Post-only", "siamese_focal": "Siamese", "siamese_fm_focal": "Siamese-FM"}


def pretty(run: str) -> str:
    base = run.removesuffix("_192")
    return SHORT_NAMES.get(base, base) + (" 192px" if run.endswith("_192") else "")

# Validated categorical palette (dataviz skill, references/palette.md) - fixed hue order, never cycled.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
# Status palette (severity, not identity): good -> critical.
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
COLORS = {0: GOOD, 1: WARNING, 2: SERIOUS, 3: CRITICAL}
INK_MUTED, GRID = "#898781", "#e1e0d9"
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
def context_crop_with_outline(image_id: str, uid: str, pad: float = 3.0, size: int = 320):
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
        draw = ImageDraw.Draw(crop)
        pts = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in poly.exterior.coords]
        draw.line(pts + [pts[0]], fill=(220, 38, 38), width=3)
        panels.append(crop)
    return panels[0], panels[1]


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
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {VIOLET}14, {VIOLET}05);
    border: 1px solid {VIOLET}30;
    border-radius: 14px;
    padding: 14px 16px 10px;
}}
[data-testid="stMetricValue"] {{font-weight: 800 !important; font-size: clamp(1.2rem, 2.2vw, 2rem) !important;}}
/* Let metric labels/values wrap instead of being cut off with an ellipsis at narrower widths. */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
div[data-testid="stExpander"] {{
    border: 1px solid {VIOLET}40;
    border-radius: 14px;
    background: {VIOLET}08;
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
    background:linear-gradient(120deg, #0f1b3d, #2a1f5c, #0d3b4f, #0f1b3d); background-size:300% 300%;
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
    background:{ORANGE}; box-shadow:0 0 12px {ORANGE};}}

/* Glass tiles that tilt in 3D on hover */
.as-tile {{backdrop-filter:blur(6px); transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s;
    animation:as-fade-up .6s ease both; transform-style:preserve-3d;}}
.as-tile:hover {{transform:perspective(700px) rotateX(7deg) rotateY(-7deg) translateY(-4px);
    box-shadow:0 18px 35px -14px var(--acc);}}
[data-testid="stMetric"] {{transition:transform .3s, box-shadow .3s; animation:as-fade-up .6s ease both;}}
[data-testid="stMetric"]:hover {{transform:perspective(700px) rotateX(6deg) translateY(-3px); box-shadow:0 14px 28px -14px {VIOLET};}}

/* Tabs: content fades in, active tab gets a gradient underline */
[data-baseweb="tab-panel"] {{animation:as-fade-up .45s ease both;}}
button[data-baseweb="tab"] {{transition:color .2s, transform .2s;}}
button[data-baseweb="tab"]:hover {{transform:translateY(-2px);}}
[data-baseweb="tab-highlight"] {{background:linear-gradient(90deg, {BLUE}, {VIOLET}, {ORANGE}) !important; height:3px !important;}}
[data-testid="stPlotlyChart"], [data-testid="stDeckGlJsonChart"] {{animation:as-fade-up .6s ease both;}}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{animation:none !important; transition:none !important;}} }}
@media (max-width: 700px) {{ .as-globe-wrap {{display:none;}} .as-hero {{padding:20px;}} }}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="as-hero">
  <div class="as-globe-wrap"><div class="as-globe"></div><div class="as-meridian"></div>
    <div class="as-orbit"><div class="as-sat"></div></div></div>
  <div>
    <h1>AfterShock</h1>
    <div class="as-sub">Building damage assessment from pre/post-disaster satellite images · AI inside the database</div>
    <div class="as-pills">
      <span class="as-pill"><span class="as-live"></span>live database</span>
      <span class="as-pill">Deep learning on DB data</span><span class="as-pill">Probabilistic DB</span>
      <span class="as-pill">Data mining (DBSCAN)</span><span class="as-pill">XAI (Grad-CAM)</span>
    </div>
    <div class="as-sub" style="font-size:0.8rem; margin-top:10px;">Data Management and AI Integration · Bennett University · Daksh Chugh</div>
  </div>
</div>""", unsafe_allow_html=True)

# The database comes only from the environment (scripts/start_demo.ps1 sets it) - never from the UI,
# and its local path is never displayed.
db_url = os.environ.get("AFTERSHOCK_DB")
if not db_url:
    st.error("Database not configured. Start the dashboard with `scripts\\start_demo.ps1`.")
    st.stop()
db = DB(db_url)

# ---------------------------------------------------------------- architecture comparison
# Only the controlled trio (same loss/epochs/data, architecture is the one thing that differs) -
# other runs in runs/ (e.g. an older post_baseline with weighted-CE loss) are a different experiment
# and would make "best architecture" a misleading, apples-to-oranges claim.
RUN_GROUPS = {
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

tab_overview, tab_map, tab_explain, tab_prob, tab_sql, tab_quality = st.tabs(
    ["🏠 Overview & honesty", "🗺️ Map & hotspots", "🧠 Explainability (Grad-CAM)",
     "🔎 Probabilistic DB queries", "🗄️ Database & SQL", "📈 Model quality"])

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
            hero_tile(pretty(best_row), f"best by validation ({best_row})", AQUA),
            hero_tile(f"+{point_diff:.1f} pts", f"test macro-F1 vs {pretty(worst_row)} ({rel_lift:+.1f}% rel.)", VIOLET),
        ]):
            col.markdown(html, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    st.radio("Resolution", list(RUN_GROUPS) if not cmp_df.empty else ["128px, 15 epochs"],
             index=list(RUN_GROUPS).index(active_group) if not cmp_df.empty else 0,
             key="active_group_radio", horizontal=True,
             help="192px results appear here as each 192px run finishes.")
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
            bc1, bc2 = st.columns([3, 2])
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
                ).highlight_max(subset=["val macro F1"], color=f"{AQUA}30"), width="stretch", height=360)
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
            st.info(f"📎 **Destroyed-class precision is weak across the board** ({', '.join(f'{m} {p:.0%}' for m, p in precisions.items())}) "
                    f"— roughly half of what every model flags as 'destroyed' is a false alarm. That's exactly what the "
                    f"MC-Dropout uncertainty routing (Probabilistic DB tab) is for: send the unsure ones to a human "
                    f"instead of trusting the raw class label.")
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
version = st.sidebar.selectbox("Model (for map / queries / explanations below)", versions,
                               index=versions.index(final_version), format_func=lambda v: f"{v} (final)" if v == final_version else v)
disasters = [r[0] for r in db.execute("SELECT DISTINCT disaster FROM buildings").fetchall()]
default_disaster = next((d for d in PREFERRED_DEFAULT_DISASTERS if d in disasters), disasters[0] if disasters else None)
disaster = st.sidebar.selectbox("Disaster", disasters, index=disasters.index(default_disaster) if default_disaster in disasters else 0)
min_conf = st.sidebar.slider("Minimum confidence on map", 0.0, 1.0, 0.0, 0.05)
review_threshold = st.sidebar.slider("Send to human review if uncertainty ≥", 0.0, 1.0, 0.6, 0.05)

df = pd.DataFrame(db.execute(
    "SELECT b.uid, b.lon, b.lat, b.label, b.split, b.image_id, p.pred_idx, p.confidence, p.uncertainty FROM buildings b "
    "JOIN predictions p ON p.uid = b.uid WHERE p.model_version = ? AND b.disaster = ?", (version, disaster)).fetchall(),
    columns=["uid", "lon", "lat", "true_label", "split", "image_id", "pred_idx", "confidence", "uncertainty"])
df["predicted"] = df["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])
df_test = df[df["split"] == "test"]  # the only split that's honest to score agreement on

st.markdown(f"#### {disaster.replace('-', ' ').title()} · model `{version}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Buildings", f"{len(df):,}")
c2.metric("Major / destroyed", f"{int(df['pred_idx'].isin([2, 3]).sum()):,}")
c3.metric("To human review", f"{int((df['uncertainty'] >= review_threshold).sum()):,}")
c4.metric("Test agreement", f"{(df_test['predicted'] == df_test['true_label']).mean():.0%}" if len(df_test) else "–",
          help="Computed on test-split buildings only - the model never trained on these, so this isn't inflated "
               "by buildings it has already memorised.")

if df.empty:
    n_total = db.execute("SELECT COUNT(*) FROM buildings WHERE disaster = ?", (disaster,)).fetchone()[0]
    n_test = db.execute("SELECT COUNT(*) FROM buildings WHERE disaster = ? AND split = 'test'", (disaster,)).fetchone()[0]
    st.warning(
        f"**No predictions to show for {disaster.replace('-', ' ').title()}.** This disaster has "
        f"{n_total} buildings total, but the deterministic train/val/test split (hashed per tile, so "
        f"neighbouring buildings never leak across splits) happened to put **{n_test} of them in the test "
        f"split** — and `predict` only scores the test split. This is a real quirk of how few tiles this "
        f"disaster has, not a bug; pick a different disaster from the sidebar to see the map, hotspots and "
        f"queries below.")

# ============================================================== TAB: Map & hotspots
with tab_map:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🗺️ Damage map")
        shown = df[(df["confidence"] >= min_conf) & df["lon"].notna()]
        hotspot_rows = db.execute("SELECT cluster_id, n_buildings, lon, lat, mean_confidence FROM hotspots "
                                   "WHERE model_version = ? AND disaster = ?", (version, disaster)).fetchall()
        view_mode = st.radio("View", ["🏙️ 3D damage towers", "🛰️ 2D satellite"], horizontal=True,
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
        if len(shown) and view_mode.startswith("🏙️"):
            # One column per building: height and colour = predicted severity, so damaged blocks
            # stand out as "towers". Hotspots (DBSCAN) are translucent violet discs on the ground.
            sev_rgb = {i: [int(COLORS[i][k:k + 2], 16) for k in (1, 3, 5)] for i in COLORS}
            deck_df = shown[["uid", "lon", "lat", "pred_idx", "predicted", "confidence", "uncertainty"]].copy()
            deck_df["height"] = deck_df["pred_idx"].map({0: 4, 1: 25, 2: 60, 3: 110}) * (0.5 + deck_df["confidence"])
            deck_df["color"] = deck_df["pred_idx"].map(lambda i: sev_rgb[i] + [230])
            deck_df["conf_pct"] = (deck_df["confidence"] * 100).round(0)
            deck_df["uncertainty"] = deck_df["uncertainty"].round(2)
            layers = [pdk.Layer("ColumnLayer", data=deck_df, get_position=["lon", "lat"], get_elevation="height",
                                elevation_scale=1.5, radius=7, get_fill_color="color", pickable=True, extruded=True,
                                auto_highlight=True)]
            if hotspot_rows:
                hs_deck = pd.DataFrame(hotspot_rows, columns=["cluster_id", "n_buildings", "lon", "lat", "mean_confidence"])
                layers.append(pdk.Layer("ScatterplotLayer", data=hs_deck, get_position=["lon", "lat"], get_radius=120,
                                        get_fill_color=[98, 80, 214, 45], get_line_color=[98, 80, 214, 220],
                                        stroked=True, line_width_min_pixels=2, pickable=False))
            view =pdk.ViewState(latitude=c_lat, longitude=c_lon, zoom=15, pitch=55, bearing=-25)
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="dark",
                                     tooltip={"html": "<b>{predicted}</b> · {conf_pct}% sure<br/>uncertainty {uncertainty}",
                                              "style": {"fontSize": "12px"}}),
                            height=520)
            st.caption("Drag to pan, right-drag (or Ctrl+drag) to rotate and tilt. Column height and colour = predicted "
                       "damage; violet discs = DBSCAN hotspots.")
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
            fp = dict(db.execute(
                "SELECT b.uid, b.footprint_wkt FROM buildings b JOIN predictions p ON p.uid = b.uid "
                "WHERE p.model_version = ? AND b.disaster = ?", (version, disaster)).fetchall())
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
                         hide_index=True, width="stretch", height=240)
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
            st.dataframe(per_disaster.sort_values("test buildings", ascending=False).style.format({"accuracy": "{:.0%}", "test buildings": "{:,.0f}"}),
                         width="stretch", height=260)
        else:
            st.caption("No test-split predictions stored for this model yet.")

# ============================================================== TAB: Explainability
with tab_explain:
    left, right = st.columns([2, 3])
    with left:
        st.subheader("🧑‍⚖️ Human review queue")
        queue = df[df["uncertainty"] >= review_threshold].sort_values("uncertainty", ascending=False).copy()
        queue["uid_short"] = queue["uid"].str.slice(0, 8) + "…"
        st.dataframe(
            queue[["uid_short", "predicted", "confidence", "uncertainty"]].rename(columns={"uid_short": "uid"}),
            hide_index=True, width="stretch", height=260,
            column_config={
                "confidence": st.column_config.NumberColumn(format="%.0f%%", width="small"),
                "uncertainty": st.column_config.NumberColumn(format="%.2f", width="small"),
            })
        st.caption("Full IDs are truncated for display - select a building below to see its full id and explanation.")
        pick_from = queue["uid"].tolist() + df.sort_values("confidence")["uid"].tolist()[:50]
        uid = st.selectbox("Building (uncertain ones first)", pick_from, format_func=lambda u: u[:8] + "…")
    with right:
        if uid:
            st.markdown(f'<span class="uid-mono">{uid}</span>', unsafe_allow_html=True)
            image_id_row = df.loc[df["uid"] == uid, "image_id"]
            context = context_crop_with_outline(image_id_row.iloc[0], uid) if len(image_id_row) else None
            if context:
                st.markdown("**Context** (native-resolution tile crop, building outlined in red)")
                ca, cb = st.columns(2)
                ca.image(context[0], caption="Before", width="stretch")
                cb.image(context[1], caption="After", width="stretch")
                st.caption("xBD imagery is ~0.5–0.8 m/pixel, so a house is only ~25 pixels wide — this wider crop is "
                           "for a human reviewer's context, not what the model is shown.")
            st.markdown("**What the model sees** (128px input crop)")
            pre_png, post_png = db.crop_bytes(uid)
            a, b = st.columns(2)
            a.image(crisp(Image.open(io.BytesIO(pre_png))), caption="Before", width="stretch")
            b.image(crisp(Image.open(io.BytesIO(post_png))), caption="After", width="stretch")
            cam_path = Path(RUNS_DIR) / version / "explanations" / f"{uid}.png"
            ckpt_path = Path(RUNS_DIR) / version / "best.pt"
            if cam_path.exists():
                st.image(crisp(Image.open(cam_path), size=960), caption="Grad-CAM: before | after | heatmap", width="stretch")
            elif ckpt_path.exists():
                with st.spinner("Computing Grad-CAM for this building (not in the precomputed batch)…"):
                    panel, pred, conf = live_gradcam(str(ckpt_path), db, uid)
                st.image(crisp(panel, size=960), caption=f"Grad-CAM (live): before | after | heatmap — "
                                                          f"predicted {pred} ({conf:.0%})", width="stretch")
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
        fig3.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRID, dash="dash"), name="perfectly calibrated")
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
        if hotspots_here:
            d2 = (all_test_buildings["lon"] - hotspots_here[0]) ** 2 + (all_test_buildings["lat"] - hotspots_here[1]) ** 2
            default_uid = all_test_buildings.loc[d2.idxmin(), "uid"]
        else:
            default_uid = all_test_buildings["uid"].iloc[0]
        uid_list = all_test_buildings["uid"].tolist()
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            center_uid = st.selectbox("Center on building", uid_list, index=uid_list.index(default_uid),
                                       format_func=lambda u: u[:10] + "…")
        with qcol2:
            radius_m = st.slider("Radius (metres)", 50, 3000, 500, 50)
        with qcol3:
            class_filter = st.selectbox("Damage class", ["(any)"] + DAMAGE_CLASSES, index=4)
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
        st.dataframe(schema_overview(db_url), hide_index=True, width="stretch",
                     column_config={"rows": st.column_config.NumberColumn(format="%d"),
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
        final_ckpt = Path(RUNS_DIR) / final_version / "best.pt"
        ai_sql = ("SELECT b.uid,\n       damage_label(b.label_idx)            AS truth,\n"
                  "       ai_predict(c.pre_png, c.post_png)    AS ai,\n"
                  "       ai_confidence(c.pre_png, c.post_png) AS conf\n"
                  "FROM buildings b JOIN crops c ON c.uid = b.uid\nWHERE b.split = 'test'\nLIMIT 5;")
        st.code(ai_sql, language="sql")
        st.caption(f"`ai_predict()` / `ai_confidence()` are SQL functions registered on the connection that run the "
                   f"final model (`{final_version}`, chosen by validation macro-F1) on the PNG bytes stored in `crops`.")
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
                st.dataframe(pc, width="stretch", height=340)
            lo, hi = bootstrap_macro_f1_ci(m["confusion_matrix"])
            st.caption(f"Macro F1 {m['macro_f1']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) · harmonic F1 {m['harmonic_f1']:.3f} "
                       f"· accuracy {m['accuracy']:.3f}")
    else:
        st.info("No metrics.json for this model yet.")
