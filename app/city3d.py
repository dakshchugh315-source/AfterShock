"""3D "damage city" for the dashboard.

Every building's real footprint (buildings.footprint_wkt) is extruded on top of Esri satellite
imagery with deck.gl. Height and colour show the model's predicted damage class. The view is
a small self-contained HTML page shown in an iframe (streamlit.components.v1.html), so the
camera can orbit and the buildings can "rise" from the ground when the map loads. Only data
that is already in the database is drawn. Nothing here changes predictions.
"""
import json

import streamlit.components.v1 as components
from shapely import wkt as shapely_wkt

NAMES = ["no damage", "minor damage", "major damage", "destroyed"]
RGB = [[52, 211, 153], [250, 204, 21], [251, 146, 60], [244, 63, 94]]  # same as dashboard COLORS

_HTML = r"""
<div id="wrap">
  <div id="map"></div>
  <div class="hud top">
    <div class="title">3D damage city <span>· __TITLE__</span></div>
    <div class="legend">__LEGEND__</div>
  </div>
  <div class="hud stats">__STATS__</div>
  <div class="hud ctrl">
    <button id="bOrbit">⏸ Orbit</button>
    <button id="bRise">⬆ Replay</button>
    <button id="bTop">🛰 Top view</button>
    <button id="bTilt">🎥 Tilt</button>
  </div>
  <div class="hud hint">drag = pan · right-drag / ctrl+drag = rotate &amp; tilt · scroll = zoom</div>
</div>
<style>
  html, body {margin:0; background:transparent; font-family:Inter, system-ui, sans-serif;}
  #wrap {position:relative; width:100%; height:__H__px; border-radius:16px; overflow:hidden;
         background:#05080f; border:1px solid #1f2a44;}
  #map {position:absolute; inset:0;}
  .hud {position:absolute; z-index:2; color:#e7ebf5; backdrop-filter:blur(8px);
        background:#070b18cc; border:1px solid #ffffff1f; border-radius:12px; padding:8px 12px;}
  .top {left:12px; top:12px;}
  .title {font-weight:700; font-size:14px;} .title span {color:#9aa6c8; font-weight:500;}
  .legend {display:flex; flex-wrap:wrap; gap:10px; margin-top:6px; font-size:11.5px; color:#c9d0ea;}
  .legend i {display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px; vertical-align:-1px;}
  .stats {right:12px; top:12px; display:flex; gap:14px; font-size:11px; color:#9aa6c8;}
  .stats b {display:block; font-size:18px; color:#fff;}
  .ctrl {left:12px; bottom:12px; display:flex; gap:6px; padding:6px;}
  .ctrl button {background:#ffffff14; color:#fff; border:1px solid #ffffff2a; border-radius:8px;
                padding:6px 10px; font-size:12px; cursor:pointer;}
  .ctrl button:hover {background:#22d3ee33; border-color:#22d3ee;}
  .hint {right:12px; bottom:12px; font-size:11px; color:#9aa6c8;}
  @media (max-width:640px) {.stats, .hint {display:none;}}
</style>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script>
const DATA = __DATA__, HOT = __HOT__, RGB = __RGB__, NAMES = __NAMES__;
const HEIGHT = [4, 12, 26, 44];                       // metres per damage class (exaggerated)
const DESTROYED = DATA.filter(d => d.c === 3).map(d => ({lon: d.x, lat: d.y}));
let vs = {longitude: __LON__, latitude: __LAT__, zoom: 16.3, pitch: 60, bearing: -25, maxPitch: 85, minZoom: 11, maxZoom: 20};
let target = null, orbit = true, t0 = performance.now(), rise = 0, eased = 0;
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduce) { orbit = false; rise = 1; eased = 1; }

function layers(now) {
  const pulse = 0.5 + 0.5 * Math.sin(now / 380);
  return [
    new deck.TileLayer({
      id: 'satellite',
      data: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      minZoom: 0, maxZoom: 19, tileSize: 256,
      renderSubLayers: p => {
        const b = p.tile.bbox;
        return new deck.BitmapLayer(p, {data: null, image: p.data, bounds: [b.west, b.south, b.east, b.north]});
      }
    }),
    new deck.ScatterplotLayer({
      id: 'hotspots', data: HOT, getPosition: d => [d.lon, d.lat],
      radiusUnits: 'meters', getRadius: 120, radiusScale: 1 + 0.25 * pulse,
      stroked: true, filled: true, getFillColor: [139, 124, 246, 35], getLineColor: [170, 150, 255, 230],
      lineWidthMinPixels: 2
    }),
    new deck.ScatterplotLayer({
      id: 'beacons', data: DESTROYED, getPosition: d => [d.lon, d.lat],
      radiusUnits: 'meters', getRadius: 9, radiusScale: 1 + 1.4 * pulse,
      filled: true, getFillColor: [244, 63, 94, Math.round(90 * (1 - pulse) + 20)]
    }),
    new deck.PolygonLayer({
      id: 'buildings', data: DATA, getPolygon: d => d.p,
      extruded: true, getElevation: d => HEIGHT[d.c], elevationScale: eased,
      getFillColor: d => RGB[d.c].concat(d.c >= 2 ? 240 : 185),
      pickable: true, autoHighlight: true, highlightColor: [255, 255, 255, 110],
      material: {ambient: 0.35, diffuse: 0.65, shininess: 48, specularColor: [255, 225, 205]}
    })
  ];
}

const map = new deck.DeckGL({
  container: 'map', viewState: vs, controller: true, layers: layers(0),
  onViewStateChange: ({viewState, interactionState}) => {
    vs = viewState;
    if (interactionState && (interactionState.isDragging || interactionState.isZooming ||
        interactionState.isRotating || interactionState.isPanning)) { orbit = false; target = null; syncBtn(); }
    map.setProps({viewState: vs});
  },
  getTooltip: ({object}) => object && object.c !== undefined && {
    html: `<b style="color:rgb(${RGB[object.c]})">${NAMES[object.c]}</b><br>${Math.round(object.f * 100)}% confident · uncertainty ${object.u.toFixed(2)}`,
    style: {background: '#0b1224', color: '#e7ebf5', fontSize: '12px', borderRadius: '8px', border: '1px solid #1f2a44'}
  }
});

function frame(now) {
  if (!reduce) rise = Math.min(1, (now - t0) / 1800);
  eased = 1 - Math.pow(1 - rise, 3);
  if (orbit) vs = {...vs, bearing: vs.bearing + 0.06};
  if (target) {
    const k = 0.08; let done = true; const nv = {...vs};
    for (const key of Object.keys(target)) {
      nv[key] = vs[key] + (target[key] - vs[key]) * k;
      if (Math.abs(target[key] - nv[key]) > 0.05) done = false;
    }
    vs = nv; if (done) target = null;
  }
  map.setProps({viewState: vs, layers: layers(now)});
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

const bOrbit = document.getElementById('bOrbit');
function syncBtn() { bOrbit.textContent = orbit ? '⏸ Orbit' : '▶ Orbit'; }
bOrbit.onclick = () => { orbit = !orbit; syncBtn(); };
document.getElementById('bRise').onclick = () => { t0 = performance.now(); };
document.getElementById('bTop').onclick = () => { orbit = false; syncBtn(); target = {pitch: 0, bearing: 0, zoom: 15.6}; };
document.getElementById('bTilt').onclick = () => { target = {pitch: 62, zoom: 16.4}; };
syncBtn();
</script>
"""


def render_city3d(shown, footprints: dict, hotspot_rows, center, title: str, height: int = 600):
    """shown: DataFrame with uid, lon, lat, pred_idx, confidence, uncertainty.
    footprints: uid -> footprint WKT (lon/lat). hotspot_rows: (cluster_id, n, lon, lat, mean_conf)."""
    data = []
    for r in shown.itertuples():
        wkt = footprints.get(r.uid)
        if not wkt:
            continue
        geom = shapely_wkt.loads(wkt)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        ring = [[round(x, 7), round(y, 7)] for x, y in geom.exterior.coords]
        data.append({"p": ring, "c": int(r.pred_idx), "f": round(float(r.confidence), 3),
                     "u": round(float(r.uncertainty), 3), "x": round(float(r.lon), 7), "y": round(float(r.lat), 7)})
    hot = [{"lon": float(h[2]), "lat": float(h[3])} for h in hotspot_rows]
    counts = [sum(1 for d in data if d["c"] == i) for i in range(4)]
    legend = "".join(f'<span><i style="background:rgb({",".join(map(str, RGB[i]))})"></i>{NAMES[i]}</span>' for i in range(4))
    if hot:
        legend += '<span><i style="background:#8b7cf6"></i>hotspot</span>'
    stats = (f"<div><b>{len(data):,}</b>buildings</div><div><b>{counts[2] + counts[3]:,}</b>major / destroyed</div>"
             f"<div><b>{len(hot)}</b>hotspots</div>")

    def js(obj):  # JSON safe to embed inside <script>
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (_HTML.replace("__DATA__", js(data)).replace("__HOT__", js(hot)).replace("__RGB__", js(RGB))
            .replace("__NAMES__", js(NAMES)).replace("__LON__", str(center[0])).replace("__LAT__", str(center[1]))
            .replace("__H__", str(height)).replace("__TITLE__", title).replace("__LEGEND__", legend)
            .replace("__STATS__", stats))
    components.html(html, height=height + 4)


# ---------------------------------------------------------------- side-by-side comparison (one shared camera)
_COMPARE_HTML = r"""
<div id="wrap">
  <div class="pane"><div id="mapA" class="map"></div><div class="hud top"><div class="title">__TITLE_A__</div>
    <div class="sub">__SUB_A__</div></div></div>
  <div class="pane"><div id="mapB" class="map"></div><div class="hud top"><div class="title">__TITLE_B__</div>
    <div class="sub">__SUB_B__</div></div></div>
  <div class="hud legend">__LEGEND__</div>
  <div class="hud ctrl"><button id="bOrbit">⏸ Orbit</button><button id="bTop">🛰 Top view</button>
    <button id="bTilt">🎥 Tilt</button><button id="bDiff">◐ Highlight differences</button></div>
</div>
<style>
  html, body {margin:0; background:transparent; font-family:Inter, system-ui, sans-serif;}
  #wrap {position:relative; display:grid; grid-template-columns:1fr 1fr; gap:6px; height:__H__px;}
  .pane {position:relative; border-radius:14px; overflow:hidden; background:#05080f; border:1px solid #1f2a44;}
  .map {position:absolute; inset:0;}
  .hud {position:absolute; z-index:2; color:#e7ebf5; backdrop-filter:blur(8px); background:#070b18cc;
        border:1px solid #ffffff1f; border-radius:12px; padding:7px 11px;}
  .top {left:10px; top:10px;} .title {font-weight:700; font-size:13.5px;} .sub {font-size:11px; color:#9aa6c8; margin-top:2px;}
  .legend {right:10px; bottom:58px; display:flex; gap:10px; font-size:11px; color:#c9d0ea;}
  .legend i {display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:4px; vertical-align:-1px;}
  .ctrl {left:50%; transform:translateX(-50%); bottom:10px; display:flex; gap:6px; padding:6px;}
  .ctrl button {background:#ffffff14; color:#fff; border:1px solid #ffffff2a; border-radius:8px; padding:6px 10px;
                font-size:12px; cursor:pointer;}
  .ctrl button:hover, .ctrl button.on {background:#22d3ee33; border-color:#22d3ee;}
  @media (max-width:700px) {#wrap {grid-template-columns:1fr; height:auto;} .pane {height:360px;} .legend {display:none;}}
</style>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<script>
const A = __DATA_A__, B = __DATA_B__, RGB = __RGB__, NAMES = __NAMES__;
const HEIGHT = [4, 12, 26, 44];
let vs = {longitude: __LON__, latitude: __LAT__, zoom: 16.2, pitch: 58, bearing: -25, maxPitch: 85, minZoom: 11, maxZoom: 20};
let orbit = true, target = null, diffOnly = false, t0 = performance.now(), eased = 0;
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if (reduce) { orbit = false; eased = 1; }
const sat = id => new deck.TileLayer({id: 'sat-' + id,
  data: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  minZoom: 0, maxZoom: 19, tileSize: 256,
  renderSubLayers: p => { const b = p.tile.bbox;
    return new deck.BitmapLayer(p, {data: null, image: p.data, bounds: [b.west, b.south, b.east, b.north]}); }});
function blocks(id, data) {
  return new deck.PolygonLayer({id: 'b-' + id, data, getPolygon: d => d.p, extruded: true,
    getElevation: d => HEIGHT[d.c], elevationScale: eased,
    getFillColor: d => RGB[d.c].concat(diffOnly && !d.d ? 35 : (d.c >= 2 ? 240 : 190)),
    updateTriggers: {getFillColor: diffOnly}, pickable: true, autoHighlight: true, highlightColor: [255, 255, 255, 110],
    material: {ambient: 0.35, diffuse: 0.65, shininess: 48, specularColor: [255, 225, 205]}});
}
const tip = ({object}) => object && object.c !== undefined && {
  html: `<b>model:</b> <span style="color:rgb(${RGB[object.m]})">${NAMES[object.m]}</span> (${Math.round(object.f * 100)}%)<br>` +
        `<b>inspectors:</b> <span style="color:rgb(${RGB[object.g]})">${NAMES[object.g]}</span> · ${object.r}` +
        (object.d ? '<br><b style="color:#fde047">disagree</b>' : ''),
  style: {background: '#0b1224', color: '#e7ebf5', fontSize: '12px', borderRadius: '8px', border: '1px solid #1f2a44'}};
function onVS({viewState, interactionState}) {
  vs = viewState;
  if (interactionState && (interactionState.isDragging || interactionState.isZooming || interactionState.isRotating ||
      interactionState.isPanning)) { orbit = false; target = null; sync(); }
}
const da = new deck.DeckGL({container: 'mapA', viewState: vs, controller: true, onViewStateChange: onVS, getTooltip: tip});
const db = new deck.DeckGL({container: 'mapB', viewState: vs, controller: true, onViewStateChange: onVS, getTooltip: tip});
function frame(now) {
  if (!reduce) { const r = Math.min(1, (now - t0) / 1800); eased = 1 - Math.pow(1 - r, 3); }
  if (orbit) vs = {...vs, bearing: vs.bearing + 0.06};
  if (target) { const nv = {...vs}; let done = true;
    for (const k of Object.keys(target)) { nv[k] = vs[k] + (target[k] - vs[k]) * 0.08; if (Math.abs(target[k] - nv[k]) > 0.05) done = false; }
    vs = nv; if (done) target = null; }
  da.setProps({viewState: vs, layers: [sat('a'), blocks('a', A)]});
  db.setProps({viewState: vs, layers: [sat('b'), blocks('b', B)]});
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
const bO = document.getElementById('bOrbit'), bD = document.getElementById('bDiff');
function sync() { bO.textContent = orbit ? '⏸ Orbit' : '▶ Orbit'; bD.classList.toggle('on', diffOnly); }
bO.onclick = () => { orbit = !orbit; sync(); };
bD.onclick = () => { diffOnly = !diffOnly; sync(); };
document.getElementById('bTop').onclick = () => { orbit = false; sync(); target = {pitch: 0, bearing: 0, zoom: 15.6}; };
document.getElementById('bTilt').onclick = () => { target = {pitch: 58, zoom: 16.3}; };
sync();
</script>
"""


def render_compare3d(df, footprints: dict, center, title_a: str, title_b: str, sub_a: str = "", sub_b: str = "",
                     height: int = 560):
    """Two synced 3D maps of the SAME buildings. df columns: uid, lon, lat, pred_idx (model), field_idx (inspectors),
    field_raw, confidence. Left = model class, right = inspector class; 'Highlight differences' fades agreeing ones."""
    a, b = [], []
    for r in df.itertuples():
        wkt = footprints.get(r.uid)
        if not wkt:
            continue
        geom = shapely_wkt.loads(wkt)
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        ring = [[round(x, 7), round(y, 7)] for x, y in geom.exterior.coords]
        m, g = int(r.pred_idx), int(r.field_idx)
        common = {"p": ring, "m": m, "g": g, "f": round(float(r.confidence), 3), "r": str(r.field_raw),
                  "d": int((m == 3) != (g == 3))}  # disagreement on destroyed vs not
        a.append({**common, "c": m})
        b.append({**common, "c": g})
    legend = "".join(f'<span><i style="background:rgb({",".join(map(str, RGB[i]))})"></i>{NAMES[i]}</span>' for i in range(4))

    def js(obj):
        return json.dumps(obj, separators=(",", ":")).replace("</", "<\\/")

    html = (_COMPARE_HTML.replace("__DATA_A__", js(a)).replace("__DATA_B__", js(b)).replace("__RGB__", js(RGB))
            .replace("__NAMES__", js(NAMES)).replace("__LON__", str(center[0])).replace("__LAT__", str(center[1]))
            .replace("__H__", str(height)).replace("__TITLE_A__", title_a).replace("__TITLE_B__", title_b)
            .replace("__SUB_A__", sub_a).replace("__SUB_B__", sub_b).replace("__LEGEND__", legend))
    components.html(html, height=height + 6)
