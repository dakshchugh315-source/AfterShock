"""AfterShock dashboard: damage map, human-review queue, hotspots and explanations.

streamlit run app/dashboard.py
"""
import io
import json
import sys
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aftershock.config import DAMAGE_CLASSES, DB_URL, RUNS_DIR  # noqa: E402
from aftershock.db import DB  # noqa: E402

COLORS = {0: "#16a34a", 1: "#facc15", 2: "#f97316", 3: "#dc2626"}

st.set_page_config(page_title="AfterShock", page_icon="🛰️", layout="wide")
st.title("🛰️ AfterShock: building damage after disasters")

db_url = st.sidebar.text_input("Database URL", DB_URL)
db = DB(db_url)
versions = [r[0] for r in db.execute("SELECT DISTINCT model_version FROM predictions").fetchall()]
if not versions:
    st.info("No predictions yet. Run `python -m aftershock.predict --ckpt runs/<name>/best.pt` first.")
    st.stop()
version = st.sidebar.selectbox("Model", versions)
disasters = [r[0] for r in db.execute("SELECT DISTINCT disaster FROM buildings").fetchall()]
disaster = st.sidebar.selectbox("Disaster", disasters)
min_conf = st.sidebar.slider("Minimum confidence on map", 0.0, 1.0, 0.0, 0.05)
review_threshold = st.sidebar.slider("Send to human review if uncertainty ≥", 0.0, 1.0, 0.6, 0.05)

df = pd.DataFrame(db.execute(
    "SELECT b.uid, b.lon, b.lat, b.label, p.pred_idx, p.confidence, p.uncertainty FROM buildings b "
    "JOIN predictions p ON p.uid = b.uid WHERE p.model_version = ? AND b.disaster = ?", (version, disaster)).fetchall(),
    columns=["uid", "lon", "lat", "true_label", "pred_idx", "confidence", "uncertainty"])
df["predicted"] = df["pred_idx"].map(lambda i: DAMAGE_CLASSES[i])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Buildings assessed", len(df))
c2.metric("Major / destroyed", int(df["pred_idx"].isin([2, 3]).sum()))
c3.metric("Sent to human review", int((df["uncertainty"] >= review_threshold).sum()))
c4.metric("Agreement with labels", f"{(df['predicted'] == df['true_label']).mean():.0%}" if len(df) else "–")

left, right = st.columns([3, 2])
with left:
    st.subheader("Damage map")
    shown = df[(df["confidence"] >= min_conf) & df["lon"].notna()]
    if len(shown):
        fmap = folium.Map(location=[shown["lat"].mean(), shown["lon"].mean()], zoom_start=17, tiles="CartoDB positron")
        for r in shown.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=5, color=COLORS[r.pred_idx], fill=True, fill_opacity=0.85,
                                popup=f"{r.uid}<br>{r.predicted} ({r.confidence:.0%})<br>uncertainty {r.uncertainty:.2f}").add_to(fmap)
        for h in db.execute("SELECT cluster_id, n_buildings, lon, lat FROM hotspots WHERE model_version = ? AND disaster = ?",
                            (version, disaster)).fetchall():
            folium.Circle([h[3], h[2]], radius=120, color="#7c3aed", weight=2, fill=False,
                          tooltip=f"Hotspot {h[0]}: {h[1]} severe buildings").add_to(fmap)
        st_folium(fmap, height=520, use_container_width=True)
    st.caption("🟢 no damage · 🟡 minor · 🟠 major · 🔴 destroyed · 🟣 hotspot")

with right:
    st.subheader("Human review queue")
    queue = df[df["uncertainty"] >= review_threshold].sort_values("uncertainty", ascending=False)
    st.dataframe(queue[["uid", "predicted", "confidence", "uncertainty"]], hide_index=True, height=240)
    st.subheader("Inspect a building")
    uid = st.selectbox("Building", queue["uid"].tolist() + df.sort_values("confidence")["uid"].tolist()[:50])
    if uid:
        pre_png, post_png = db.crop_bytes(uid)
        a, b = st.columns(2)
        a.image(Image.open(io.BytesIO(pre_png)), caption="Before", width="stretch")
        b.image(Image.open(io.BytesIO(post_png)), caption="After", width="stretch")
        cam = Path(RUNS_DIR) / version / "explanations" / f"{uid}.png"
        if cam.exists():
            st.image(str(cam), caption="Grad-CAM: before | after | heatmap")
        else:
            st.caption(f"No Grad-CAM yet: python -m aftershock.explain --ckpt runs/{version}/best.pt --uid {uid}")

metrics_file = Path(RUNS_DIR) / version / "metrics.json"
if metrics_file.exists():
    st.subheader("Model quality (test split)")
    m = json.loads(metrics_file.read_text()).get("test", {})
    if m:
        st.dataframe(pd.DataFrame(m["per_class"]).T, width="stretch")
        st.caption(f"Macro F1 {m['macro_f1']:.3f} · harmonic F1 {m['harmonic_f1']:.3f} · accuracy {m['accuracy']:.3f}")
