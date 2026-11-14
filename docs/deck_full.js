const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.SKILL + "/scripts/apply_theme.js");

const SLIDES_DIR = path.join(__dirname, "..", "deck", "project", "slides");
const notes = (id, fallback = "") => {
  if (!fs.existsSync(path.join(SLIDES_DIR, id + ".html"))) return fallback;
  const h = fs.readFileSync(path.join(SLIDES_DIR, id + ".html"), "utf8");
  const m = h.match(/<aside>([\s\S]*?)<\/aside>/);
  return m ? m[1].replace(/&gt;/g, ">").replace(/&lt;/g, "<").replace(/&amp;/g, "&").trim() : "";
};

const THEME = {
  name: "AfterShock", headFontFace: "Arial", bodyFontFace: "Calibri",
  colors: { dk1: "1B222B", lt1: "FFFFFF", dk2: "11161D", lt2: "F2F4F7", accent1: "E8692E", accent2: "3E7CB1",
    accent3: "E3A63B", accent4: "A8321D", accent5: "2F6E4F", accent6: "B4501C", hlink: "3E7CB1", folHlink: "B4501C" },
};
const K = { dark: "11161D", cardD: "1C2530", light: "FFFFFF", cardL: "F2F4F7", text: "1B222B", muted: "4A5563", soft: "BFC6CF",
  or: "E8692E", orD: "B4501C", orL: "F08A4B", blue: "3E7CB1", yel: "E3A63B", red: "A8321D", green: "2F6E4F", grey: "9AA3AE", tint: "FBE3D3" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "Daksh Chugh";
pres.title = "AfterShock: Building Damage Assessment from Satellite Images";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

for (const [name, bg, fg, numc] of [["LIGHT", K.light, K.text, "6B7280"], ["DARK", K.dark, "E9E6DF", "8B95A1"]]) {
  pres.defineSlideMaster({
    title: name, background: { color: bg },
    objects: [
      { text: { text: "AfterShock · Daksh Chugh", options: { x: 0.6, y: 6.95, w: 6, h: 0.35, fontSize: 10, color: numc, fontFace: "Calibri", margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.85, w: 12.1, h: 1.15, fontSize: 30, bold: true, color: fg, fontFace: "Arial", valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.2, y: 6.95, w: 0.55, h: 0.35, fontSize: 10, color: numc, fontFace: "Calibri", align: "right" },
  });
}
pres.defineSlideMaster({ title: "COVER", background: { color: K.dark }, objects: [] });
for (const t of ["Why", "Architecture", "Database", "Models and results", "Trust", "Wrap-up"]) pres.addSection({ title: t });

// ---------- helpers ----------
const T = (s, text, o) => s.addText(text, { isTextBox: true, margin: 0, fontFace: "Calibri", valign: "top", ...o });
let N = 0;
const num = (e) => String(++N).padStart(2, "0") + " · " + e;
function slide(id, layout, section, eyebrow, title, noteText = "") {
  const s = pres.addSlide({ masterName: layout, sectionTitle: section });
  const dark = layout === "DARK";
  if (eyebrow) eyebrow = num(eyebrow);
  if (eyebrow) T(s, eyebrow.toUpperCase(), { x: 0.6, y: 0.45, w: 12, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: dark ? K.orL : K.orD });
  if (title) s.addText(title, { placeholder: "title" });
  s.addNotes(noteText || notes(id));
  return s;
}
function card(s, x, y, w, h, head, body, dark, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: o.fill || (dark ? K.cardD : K.cardL) }, line: o.line ? { color: o.line, width: 1.5 } : { type: "none" } });
  let ty = y + 0.25;
  if (o.badge) {
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: ty, w: 0.5, h: 0.5, fill: { color: K.or }, line: { type: "none" } });
    T(s, o.badge, { x: x + 0.3, y: ty, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    ty += 0.7;
  }
  if (o.kicker) { T(s, o.kicker, { x: x + 0.3, y: ty, w: w - 0.6, h: 0.3, fontSize: 11, bold: true, color: o.kickerColor || K.grey, charSpacing: 1 }); ty += 0.35; }
  T(s, head, { x: x + 0.3, y: ty, w: w - 0.6, h: 0.5, fontSize: o.headSize || 19, bold: true, fontFace: "Arial", color: o.headColor || (dark ? "E9E6DF" : K.text) });
  T(s, body, { x: x + 0.3, y: ty + 0.55, w: w - 0.6, h: h - (ty - y) - 0.7, fontSize: o.bodySize || 15, color: dark ? K.soft : K.muted, lineSpacingMultiple: 1.1 });
}
function table(s, x, y, w, header, rows, colW, o = {}) {
  const head = header.map((h, i) => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: K.text }, align: (o.right || []).includes(i) ? "right" : "left" } }));
  const body = rows.map((r, ri) => r.map((c, i) => ({ text: String(c), options: {
    fill: { color: o.hi === ri ? K.tint : (ri % 2 ? K.cardL : "FFFFFF") }, color: K.text, bold: o.hi === ri || undefined,
    align: (o.right || []).includes(i) ? "right" : "left" } })));
  s.addTable([head, ...body], { x, y, w, colW, fontSize: o.size || 14, fontFace: "Calibri", border: { type: "solid", color: "D5D9DF", pt: 0.75 }, margin: [0.06, 0.1, 0.06, 0.1], valign: "middle", rowH: o.rowH || 0.42 });
}
function bigNum(s, x, y, w, num, label, color, dark) {
  T(s, num, { x, y, w, h: 0.95, fontSize: 54, bold: true, fontFace: "Arial", color });
  T(s, label, { x, y: y + 0.95, w, h: 0.7, fontSize: 14, color: dark ? K.soft : K.muted });
}
function bar(s, x, y, label, value, maxW, frac, color) {
  T(s, label, { x, y, w: 8, h: 0.32, fontSize: 14, color: K.text });
  s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.34, w: Math.max(maxW * frac, 0.05), h: 0.3, fill: { color }, line: { type: "none" } });
}

// ---------- slides ----------
// 1 cover
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Why" });
  s.addShape(pres.shapes.OVAL, { x: 8.6, y: -1.2, w: 6.5, h: 6.5, fill: { color: "1C2A3A" }, line: { type: "none" } });
  s.addShape(pres.shapes.OVAL, { x: 10.2, y: 0.6, w: 2.6, h: 2.6, fill: { color: K.or, transparency: 20 }, line: { type: "none" } });
  T(s, "DATA MANAGEMENT AND AI INTEGRATION · COURSE PROJECT", { x: 0.8, y: 0.8, w: 9, h: 0.35, fontSize: 13, bold: true, charSpacing: 3, color: K.orL });
  T(s, "AfterShock", { x: 0.8, y: 2.3, w: 9, h: 1.4, fontSize: 72, bold: true, fontFace: "Arial", color: "E9E6DF" });
  T(s, "Which buildings are damaged after a disaster, how sure are we, and why? Answered from satellite images.", { x: 0.8, y: 3.8, w: 8.2, h: 1.1, fontSize: 22, color: "C9C4BA" });
  T(s, [{ text: "Daksh Chugh", options: { bold: true, color: "E9E6DF" } }, { text: "   ·   Bennett University", options: { color: "C9C4BA" } }], { x: 0.8, y: 6.3, w: 9, h: 0.45, fontSize: 18 });
  s.addNotes(notes("cover"));
}

// ---------- extra helpers for diagrams and SQL ----------
// Full database (10 disasters), final model siamese_fm_focal: Project_Report.md 10.6 (sqlai demo on aftershock_full.db).
const DBN = { src: "Full DB (10 disasters, 107,441 buildings), final model siamese_fm_focal · python -m aftershock.sqlai --demo",
  idxMs: "0.3 ms", scanMs: "60 ms", speedup: "≈ 190×", thrN: "104", thrTrue: "97", thrPct: "93.3%", sel50: "88.6%", selAll: "73.8%" };
function codeBox(s, x, y, w, h, head, code, size = 12) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: K.dark }, line: { type: "none" } });
  let cy = y + 0.2;
  if (head) { T(s, head, { x: x + 0.25, y: cy, w: w - 0.5, h: 0.32, fontSize: 13, bold: true, color: K.orL }); cy += 0.42; }
  T(s, code, { x: x + 0.25, y: cy, w: w - 0.5, h: h - (cy - y) - 0.15, fontSize: size, fontFace: "Courier New", color: "E9E6DF", lineSpacingMultiple: 1.0 });
}
function entity(s, x, y, w, h, name, fields, color) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "FFFFFF" }, line: { color, width: 1.5 } });
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.38, fill: { color }, line: { color, width: 1.5 } });
  T(s, name, { x: x + 0.12, y: y + 0.04, w: w - 0.2, h: 0.3, fontSize: 13, bold: true, fontFace: "Arial", color: "FFFFFF", valign: "middle" });
  T(s, fields.map((f, i) => {
    const key = f.startsWith("PK") || f.startsWith("FK");
    return { text: f, options: { bold: key, color: key ? K.orD : K.text, breakLine: i < fields.length - 1 } };
  }), { x: x + 0.12, y: y + 0.45, w: w - 0.2, h: h - 0.5, fontSize: 10.5, fontFace: "Courier New" });
}
function line(s, x1, y1, x2, y2, color = "5B6675", arrow = false, dash) {
  const o = { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1), line: { color, width: 1.5 } };
  if (x2 < x1) o.flipH = true;
  if (y2 < y1) o.flipV = true;
  if (arrow) o.line.endArrowType = "triangle";
  if (dash) o.line.dashType = dash;
  s.addShape(pres.shapes.LINE, o);
}
function box(s, x, y, w, h, head, body, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: o.fill || K.cardD }, line: { color: o.line || o.fill || K.cardD, width: 1.5 } });
  T(s, head, { x: x + 0.12, y: y + 0.1, w: w - 0.24, h: 0.32, fontSize: o.hs || 13, bold: true, fontFace: "Arial", color: o.hc || "E9E6DF", align: o.align || "left" });
  if (body) T(s, body, { x: x + 0.12, y: y + 0.45, w: w - 0.24, h: h - 0.5, fontSize: o.bs || 11, color: o.bc || K.soft, align: o.align || "left" });
}
const lbl = (s, x, y, w, t, c = K.muted, size = 10) => T(s, t, { x, y, w, h: 0.25, fontSize: size, bold: true, color: c, align: "center" });

// 2 problem
if (0) {
  const s = slide("problem", "LIGHT", "Why", "The problem", "After a disaster, rescue teams first need to know which buildings are damaged, and where");
  const items = [["1", "Slow", "Analysts still mark damage by hand on satellite images, one building at a time. A city takes days."],
    ["2", "Risky", "A model that is wrong without warning is dangerous: a missed collapse can cost lives."],
    ["3", "No priority", "A list of damaged buildings is not a plan. Teams need to know where to go first."]];
  items.forEach(([b, h, t], i) => card(s, 0.6 + i * 4.1, 2.6, 3.85, 3.4, h, t, false, { badge: b, headSize: 22, bodySize: 16 }));
}
// 3 idea
{
  const s = slide("idea", "DARK", "Why", "The problem and the idea", "After a disaster, teams need to know which buildings are damaged, how sure we are, and where to go first", "After an earthquake, flood or cyclone, rescue teams first need to know which buildings are damaged and where. Today analysts mark this by hand on satellite images, which takes days, and a model that is wrong without warning is dangerous. My idea: compare every building before and after the disaster, and give four answers instead of one: the damage level, how confident the model is, a heatmap showing why, and clusters of heavy damage ranked for rescue. All four are stored in the database as rows that can be queried with SQL.");
  const items = [["Damage", "No damage, minor, major or destroyed"], ["Confidence", "How sure it is; unsure cases go to a human"],
    ["Reason", "A heatmap of where in the image it looked"], ["Hotspot", "Clusters of heavy damage, ranked for rescue"]];
  items.forEach(([h, t], i) => card(s, 0.6 + i * 3.08, 2.6, 2.88, 2.6, h, t, true, { badge: String(i + 1), headSize: 21, bodySize: 15 }));
  T(s, [{ text: "All four live in the database ", options: { bold: true, color: K.orL } }, { text: "as rows you can query with SQL: the AI output is data, not a screenshot.", options: { color: "E9E6DF" } }], { x: 0.6, y: 5.6, w: 12, h: 0.5, fontSize: 18 });
}
// 4 topics
{
  const s = slide("topics", "LIGHT", "Why", "Course alignment", "The database is the spine; the AI reads from it and writes back to it");
  table(s, 0.6, 2.2, 12.1, ["Course topic", "Where it lives in AfterShock"], [
    ["Data management (the spine)", "10 tables, keys, constraints, indexes, 7 views, 2 triggers; ETL of 107,441 buildings; SQL functions that run the model"],
    ["4 · Deep learning on large data (core)", "Siamese CNN in PyTorch, trained on image bytes streamed from the database by SQL"],
    ["1 · Probabilistic database", "Every building stores 4 class probabilities + uncertainty; threshold and expected-count queries"],
    ["3 · Data mining / KDD", "DBSCAN finds damage hotspots from stored coordinates and writes them back as a table"],
    ["5 · Explainable AI (XAI part)", "Grad-CAM heatmaps for the most uncertain buildings, picked by an ORDER BY query"],
  ], [4.3, 7.8], { size: 14, rowH: 0.62, hi: 0 });
}
// 5 dataset
{
  const s = slide("dataset", "LIGHT", "Architecture", "The data", "xBD: real before/after satellite pairs, labelled per building");
  bigNum(s, 0.6, 2.3, 4.2, "107,441", "buildings in our database", K.orD, false);
  bigNum(s, 0.6, 4.2, 4.2, "10", "disasters: hurricanes, floods, fires, an earthquake, a tsunami, a volcano", K.orD, false);
  T(s, "Test set: 15,812 buildings, heavily imbalanced", { x: 5.4, y: 2.2, w: 7.3, h: 0.4, fontSize: 18, bold: true, fontFace: "Arial", color: K.text });
  const max = 7.0;
  bar(s, 5.4, 2.8, "No damage · 9,973 (63%)", 0, max, 0.631, K.blue);
  bar(s, 5.4, 3.65, "Minor damage · 2,198 (14%)", 0, max, 0.139, K.yel);
  bar(s, 5.4, 4.5, "Major damage · 2,811 (18%)", 0, max, 0.178, K.or);
  bar(s, 5.4, 5.35, "Destroyed · 830 (5%)", 0, max, 0.052, K.red);
  T(s, "Source: xBD / xView2 (Gupta et al., CVPR-W 2019)", { x: 5.4, y: 6.4, w: 7, h: 0.3, fontSize: 10, color: "6B7280" });
  T(s, [{ text: "No leakage: ", options: { bold: true } }, { text: "split by whole tiles (MD5 of tile id) 70 / 15 / 15; best model chosen on validation, test used once." }], { x: 0.6, y: 5.95, w: 4.5, h: 0.75, fontSize: 12, color: K.text });
}
// 6 system architecture
{
  const s = slide("sysarch", "DARK", "Architecture", "System architecture", "Everything goes through the database: the model reads from it and writes back to it",
    "This is the architecture. On the left, the raw xBD files: JSON labels with building polygons, and PNG satellite tiles. My ETL script, ingest.py, cuts one crop per building, assigns the split by hashing the tile id, and loads everything with batched upserts. In the middle is the database: it is the single source of truth. On the right, every AI module talks to the database with SQL: training reads crops with SELECT, prediction writes probabilities back with an upsert, Grad-CAM picks the most uncertain buildings with ORDER BY, and DBSCAN reads a join and writes a hotspots table. At the bottom, every panel of the dashboard is a SQL query. Nothing reads image folders directly.");
  box(s, 0.5, 2.25, 2.7, 1.75, "1 · Sources", "xBD JSON labels (polygons, damage)\nPNG tiles, pre + post\n10 disasters", { line: K.blue });
  box(s, 0.5, 4.4, 2.7, 1.8, "2 · ETL  ingest.py", "Extract polygons\nTransform: crop, resize, MD5 split\nLoad: batched UPSERT", { line: K.blue });
  line(s, 1.85, 4.0, 1.85, 4.4, K.blue, true);
  line(s, 3.2, 5.3, 3.7, 5.3, K.blue, true);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.7, y: 2.25, w: 5.1, h: 3.95, rectRadius: 0.1, fill: { color: "2A2018" }, line: { color: K.or, width: 2 } });
  T(s, "3 · DATABASE  (SQLite · PostgreSQL + PostGIS schema)", { x: 3.9, y: 2.35, w: 4.8, h: 0.3, fontSize: 12, bold: true, color: K.orL });
  const chips = ["disasters", "images", "buildings", "crops", "predictions", "hotspots", "facilities", "model_runs", "review_queue", "human_reviews"];
  chips.forEach((c, i) => {
    const cx = 3.9 + (i % 2) * 2.4, cy = 2.8 + Math.floor(i / 2) * 0.48;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y: cy, w: 2.25, h: 0.38, rectRadius: 0.06, fill: { color: i < 7 ? K.cardD : "3A2A1C" }, line: { type: "none" } });
    T(s, c, { x: cx, y: cy, w: 2.25, h: 0.38, fontSize: 11, fontFace: "Courier New", color: "E9E6DF", align: "center", valign: "middle" });
  });
  T(s, "+ 7 views · 2 triggers · indexes (B-tree, GiST) · SQL functions ai_predict(), entropy4()", { x: 3.9, y: 5.35, w: 4.7, h: 0.7, fontSize: 11.5, color: K.orL });
  const mods = [["train.py", "SELECT crops  →  PyTorch", "read"], ["predict.py", "MC Dropout  →  UPSERT predictions", "write"], ["explain.py", "ORDER BY uncertainty  →  Grad-CAM", "read"], ["hotspots.py", "JOIN  →  DBSCAN  →  INSERT hotspots", "write"]];
  mods.forEach(([h, t, d], i) => {
    const y = 2.25 + i * 1.0;
    box(s, 9.4, y, 3.45, 0.85, "4 · " + h, t, { line: d === "read" ? K.blue : K.or, hs: 12, bs: 11 });
    if (d === "read") line(s, 8.8, y + 0.42, 9.4, y + 0.42, K.blue, true); else line(s, 9.4, y + 0.42, 8.8, y + 0.42, K.or, true);
  });
  box(s, 3.7, 6.3, 9.15, 0.5, "5 · Dashboard (Streamlit + Folium): every panel is a SQL query", "", { fill: K.cardD, hs: 12 });
  T(s, "→ blue: reads (SELECT)\n← orange: writes back", { x: 0.5, y: 6.3, w: 3.1, h: 0.55, fontSize: 10, color: K.soft });
}
// 7 model architecture
{
  const s = slide("modelarch", "LIGHT", "Architecture", "Model architecture", "Model C: compare the two images feature map by feature map, then decide",
    "This is my main model, the feature-map fusion Siamese network. The before and the after crop each go through the same ResNet-18, with shared weights, up to layer 3. At 128 pixels that gives a 256-channel, 8 by 8 feature map for each image, so the location of things is still there. I stack three maps: the before map, the after map, and the absolute difference, which is a change map. That is 768 channels. A 1 by 1 convolution mixes them back to 256 channels, then layer 4 of ResNet reads the change, average pooling gives 512 numbers, and a small head with dropout outputs four probabilities. The dropout matters later, because I keep it on at prediction time for uncertainty. The late-fusion model B instead pools each image to 512 numbers first, so it loses where the change happened.");
  const L = K.blue, O = K.or;
  const lightBox = (x, y, w, h, head, body, c) => box(s, x, y, w, h, head, body, { fill: "FFFFFF", line: c, hc: K.text, bc: K.muted, hs: 13, bs: 11 });
  lightBox(0.5, 2.35, 1.7, 1.0, "Pre image", "3 × 128 × 128", L);
  lightBox(0.5, 4.75, 1.7, 1.0, "Post image", "3 × 128 × 128", L);
  lightBox(2.6, 2.35, 2.3, 1.0, "ResNet-18 → layer3", "out: 256 × 8 × 8", L);
  lightBox(2.6, 4.75, 2.3, 1.0, "ResNet-18 → layer3", "out: 256 × 8 × 8", L);
  line(s, 2.2, 2.85, 2.6, 2.85, L, true); line(s, 2.2, 5.25, 2.6, 5.25, L, true);
  line(s, 3.75, 3.35, 3.75, 4.75, K.grey, false, "dash");
  T(s, "shared weights", { x: 3.85, y: 3.85, w: 1.4, h: 0.4, fontSize: 11, italic: true, color: K.muted });
  lightBox(5.4, 3.3, 2.4, 1.5, "Concatenate", "[ pre, post, |post − pre| ]\n768 × 8 × 8", O);
  line(s, 4.9, 2.85, 5.4, 3.7, O, true); line(s, 4.9, 5.25, 5.4, 4.4, O, true);
  lightBox(8.2, 3.3, 1.55, 1.5, "1×1 conv", "+ BN + ReLU\n256 × 8 × 8", O);
  lightBox(10.15, 3.3, 1.35, 1.5, "layer4", "512 × 4 × 4", O);
  line(s, 7.8, 4.05, 8.2, 4.05, O, true); line(s, 9.75, 4.05, 10.15, 4.05, O, true);
  box(s, 11.85, 2.6, 1.0, 2.9, "Head", "avg pool\n512\n↓\nFC 256\nReLU\nDropout 0.3\n↓\nFC 4\nsoftmax", { fill: K.dark, hs: 12, bs: 10.5, align: "center" });
  line(s, 11.5, 4.05, 11.85, 4.05, O, true);
  T(s, [{ text: "Why: ", options: { bold: true } }, { text: "fusing after layer 3 keeps an 8 × 8 map, so the network sees where the roof changed. Late fusion (model B) pools each image to 512 numbers first and loses that." }],
    { x: 0.5, y: 6.0, w: 7.6, h: 0.75, fontSize: 13, color: K.text });
  T(s, [{ text: "Training: ", options: { bold: true } }, { text: "ImageNet weights, AdamW 3e-4, cosine LR, focal loss (γ = 2) + class weights, 15 epochs, best epoch by validation macro-F1." }],
    { x: 8.3, y: 6.0, w: 4.55, h: 0.75, fontSize: 11, color: K.muted });
}
// 8 ER diagram
{
  const s = slide("er", "LIGHT", "Database", "ER design", "Ten tables: the data, the model output, and the human feedback",
    "This is the entity-relationship design. One disaster has many image tiles, one tile has many buildings: both one-to-many, enforced with foreign keys in the PostgreSQL schema. Crops is one-to-one with buildings: I split the image bytes into their own table on purpose. Predictions is one-to-many from buildings, with a composite primary key of uid and model version, so the same building can hold predictions from every model I trained and I can compare them in SQL. The orange tables are the AI layer: model_runs is a registry of trained models, review_queue is filled automatically by a trigger, and human_reviews stores a person's corrected label. Hotspots, in green, are written back by the data-mining step.");
  const B = K.blue, O = K.or, G = K.green;
  entity(s, 0.5, 2.25, 2.3, 1.15, "disasters", ["PK name", "disaster_type"], B);
  entity(s, 3.35, 2.25, 2.6, 1.65, "images", ["PK image_id", "FK disaster", "split CHECK(...)", "width, height"], B);
  entity(s, 6.5, 2.25, 2.85, 2.2, "buildings", ["PK uid", "FK image_id", "disaster, split", "label, label_idx", "lon, lat", "geom POINT (GiST)"], B);
  entity(s, 9.95, 2.25, 2.9, 2.2, "predictions", ["PK,FK uid", "PK model_version", "pred_idx", "p_no … p_destroyed", "confidence", "uncertainty"], O);
  entity(s, 6.5, 5.0, 2.85, 1.15, "crops", ["PK,FK uid", "pre_png, post_png BLOB"], B);
  entity(s, 9.95, 5.0, 2.9, 1.3, "review_queue", ["PK uid, model_version", "reason, uncertainty", "status CHECK(...)"], O);
  entity(s, 3.35, 4.3, 2.6, 1.15, "model_runs", ["PK model_version", "arch, img_size, F1"], O);
  entity(s, 0.5, 4.0, 2.3, 1.45, "hotspots", ["PK model_version,", "   cluster_id", "lon, lat, n"], G);
  entity(s, 3.35, 5.7, 2.6, 0.95, "human_reviews", ["FK uid", "label_idx CHECK"], O);
  const rel = (x1, y1, x2, y2, a, b, lx1, ly1, lx2, ly2) => { line(s, x1, y1, x2, y2, "5B6675"); lbl(s, lx1, ly1, 0.4, a, K.orD, 11); lbl(s, lx2, ly2, 0.4, b, K.orD, 11); };
  rel(2.8, 2.7, 3.35, 2.7, "1", "N", 2.75, 2.4, 3.0, 2.4);
  rel(5.95, 2.7, 6.5, 2.7, "1", "N", 5.9, 2.4, 6.15, 2.4);
  rel(9.35, 2.7, 9.95, 2.7, "1", "N", 9.3, 2.4, 9.6, 2.4);
  rel(7.9, 4.45, 7.9, 5.0, "1", "1", 7.95, 4.45, 7.95, 4.73);
  rel(11.4, 4.45, 11.4, 5.0, "1", "0..1", 11.45, 4.45, 11.5, 4.73);
  T(s, "Blue: data · Orange: AI output + feedback · Green: data mining", { x: 6.5, y: 6.35, w: 6.35, h: 0.3, fontSize: 10, color: "6B7280", align: "right" });
}
// 9 constraints and design decisions
{
  const s = slide("dbdesign", "LIGHT", "Database", "Keys, constraints, design decisions", "Integrity is enforced by the database, not by hope",
    "Here are the DBMS concepts and exactly where they are used. Primary keys everywhere; a composite key on predictions so one building can have one prediction per model. Foreign keys in the PostgreSQL schema, and in SQLite the load order guarantees that parents exist before children. CHECK constraints restrict the split to train, val or test, and labels to 0 to 3. A generated column in PostGIS builds the point geometry from longitude and latitude automatically. On normalisation I want to be honest: the buildings table repeats disaster and split, which you could get through image_id, so strictly it is not in third normal form. I did that on purpose because the training query filters by split thousands of times, and the data is loaded once and never updated, so there is no update anomaly risk. And I split the 1.5 gigabytes of image bytes into the crops table, which is vertical partitioning, so the buildings table stays small and fast.");
  table(s, 0.6, 2.2, 8.0, ["Concept", "Where in AfterShock"], [
    ["Primary key", "uid, image_id, name; serial id in facilities"],
    ["Composite key", "predictions (uid, model_version); hotspots (model_version, cluster_id)"],
    ["Foreign key", "images → disasters, buildings → images, crops / predictions → buildings"],
    ["CHECK", "split IN ('train','val','test'); label_idx BETWEEN 0 AND 3; status IN (…)"],
    ["DEFAULT", "created_at DEFAULT now(); status DEFAULT 'pending'"],
    ["Generated column", "geom = ST_SetSRID(ST_MakePoint(lon, lat), 4326) STORED"],
    ["BLOB / BYTEA", "crop images stored as PNG bytes inside the database"],
    ["UPSERT", "INSERT … ON CONFLICT DO UPDATE: re-running never duplicates"],
  ], [1.9, 6.1], { size: 12, rowH: 0.47 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y: 2.2, w: 3.9, h: 4.25, rectRadius: 0.12, fill: { color: K.dark }, line: { type: "none" } });
  T(s, "Normalisation, honestly", { x: 9.2, y: 2.4, w: 3.5, h: 0.4, fontSize: 17, bold: true, fontFace: "Arial", color: K.orL });
  T(s, [
    { text: "buildings repeats disaster and split (reachable via image_id): not strict 3NF, on purpose.", options: { bullet: true, breakLine: true } },
    { text: "Why: the hot training query filters by split with no JOIN. Data is load-once, read-many, so no update anomalies.", options: { bullet: true, breakLine: true } },
    { text: "Images moved to crops (vertical partitioning): 1.5 GB of bytes kept out of the hot table.", options: { bullet: true } },
  ], { x: 9.2, y: 2.95, w: 3.45, h: 3.4, fontSize: 13, color: "E9E6DF", paraSpaceAfter: 8 });
}
// 10 ETL
if (0) {
  const s = slide("etl", "LIGHT", "Database", "ETL: loading the data", "Extract, transform, load: 107,441 buildings into the database",
    "This is how data gets in. Extract: read each xBD label file, which holds building polygons in pixel and geographic coordinates and the damage label. Transform: cut a crop around each polygon from the pre and the post tile, resize to 128 pixels, encode as PNG, and assign the split with an MD5 hash of the tile id, so a whole tile always lands in the same split. Load: batched upserts, using executemany, in foreign-key order: disasters, then images, then buildings, then crops. The upsert means I can re-run ingest after a crash without creating duplicates. And every query uses placeholders, never string concatenation, which protects against SQL injection.");
  const steps = [["E", "Extract", "Read xBD JSON: polygons (pixel + lon/lat), damage label, disaster; pre/post PNG tiles."],
    ["T", "Transform", "Crop around each polygon, resize to 128 px, encode PNG. Split = MD5(tile id): 70 / 15 / 15."],
    ["L", "Load", "Batched UPSERT with executemany, in FK order: disasters → images → buildings → crops."]];
  steps.forEach(([b, h, t], i) => {
    const y = 2.2 + i * 1.45;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y, w: 5.6, h: 1.3, rectRadius: 0.1, fill: { color: K.cardL }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: 0.85, y: y + 0.35, w: 0.6, h: 0.6, fill: { color: K.or }, line: { type: "none" } });
    T(s, b, { x: 0.85, y: y + 0.35, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, h, { x: 1.7, y: y + 0.15, w: 4.3, h: 0.35, fontSize: 16, bold: true, fontFace: "Arial", color: K.text });
    T(s, t, { x: 1.7, y: y + 0.52, w: 4.35, h: 0.75, fontSize: 12.5, color: K.muted });
  });
  codeBox(s, 6.6, 2.2, 6.25, 2.4, "UPSERT (PostgreSQL, db.py)", "INSERT INTO predictions (uid, model_version,\n    pred_idx, p_no, …, uncertainty)\nVALUES (%s, %s, %s, %s, …, %s)\nON CONFLICT (uid, model_version)\nDO UPDATE SET pred_idx = EXCLUDED.pred_idx, …;", 12);
  codeBox(s, 6.6, 4.8, 6.25, 1.65, "Parameterised read (no SQL injection)", "db.execute(\"SELECT pre_png, post_png\n            FROM crops WHERE uid = ?\", (uid,))", 12);
}
// 11 split
if (0) {
  const s = slide("split", "LIGHT", "Database", "Honest evaluation", "We split by whole tiles, so the test set is truly unseen");
  card(s, 0.6, 2.3, 5.9, 3.0, "Random split by building: leaks", "Two houses on the same street share lighting, roofs and damage. One lands in training, its neighbour in test: the model has seen the answer. Scores look better than they are.", false, { headColor: K.red, kicker: "✗  WRONG", kickerColor: K.red, bodySize: 16 });
  card(s, 6.8, 2.3, 5.9, 3.0, "Split by tile: what we did", "A hash of the tile id sends every tile to train (70%), validation (15%) or test (15%). All buildings in a tile move together, so test areas are new places.", false, { headColor: K.green, kicker: "✓  RIGHT", kickerColor: K.green, bodySize: 16 });
  T(s, [{ text: "And: ", options: { bold: true } }, { text: "the best model is chosen on " }, { text: "validation", options: { bold: true } }, { text: "; the test set is used once, for the final number." }], { x: 0.6, y: 5.65, w: 12, h: 0.45, fontSize: 17, color: K.text });
  T(s, "Code: ingest.assign_split (MD5 hash of tile id) · stored in images.split with a CHECK constraint", { x: 0.6, y: 6.5, w: 9, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 12 indexes
{
  const s = slide("indexes", "DARK", "Database", "Indexes and query plans", "One index made the training query about 190 times faster",
    "Training and evaluation constantly ask for all buildings in one split. I measured that query on the full database of 107 thousand buildings. With the B-tree index on split, SQLite's query plan says SEARCH using a covering index, and it takes about 0.3 milliseconds. When I force a full table scan with NOT INDEXED, the plan says SCAN buildings and it takes about 60 milliseconds, roughly 190 times slower. For the review queue I added a composite index on model_version and uncertainty, which matches the ORDER BY. In PostgreSQL, the geometry column has a GiST spatial index, which is what makes the ST_DWithin distance query fast.");
  bigNum(s, 0.6, 2.3, 3.3, DBN.scanMs, "full table scan  (NOT INDEXED)", K.grey, true);
  bigNum(s, 4.1, 2.3, 3.3, DBN.idxMs, "with buildings_split_idx", K.orL, true);
  bigNum(s, 8.2, 2.3, 4.5, DBN.speedup, "faster on SELECT COUNT(*) … WHERE split = 'test'", "E9E6DF", true);
  codeBox(s, 0.6, 4.2, 6.4, 2.25, "EXPLAIN QUERY PLAN", "-- with the index\nSEARCH buildings USING COVERING INDEX\n       buildings_split_idx (split=?)\n\n-- forced scan\nSCAN buildings", 12);
  T(s, [
    { text: "B-tree  buildings(split): the training / test query", options: { bullet: true, breakLine: true } },
    { text: "Composite  predictions(model_version, uncertainty): review queue ORDER BY", options: { bullet: true, breakLine: true } },
    { text: "GiST  buildings(geom) in PostGIS: ST_DWithin distance queries", options: { bullet: true } },
  ], { x: 7.3, y: 4.25, w: 5.5, h: 2.2, fontSize: 14, color: "E9E6DF", paraSpaceAfter: 8 });
  T(s, DBN.src + " · timed on the training laptop", { x: 0.6, y: 6.55, w: 11, h: 0.3, fontSize: 10, color: "8B95A1" });
}
// 13 probabilistic queries
{
  const s = slide("probsql", "LIGHT", "Database", "Probabilistic database queries", "Each building stores a distribution, so SQL can ask “how sure?”",
    "A normal database says this building is destroyed. Mine stores a probability for each of the four classes plus an uncertainty score, which is called attribute-level uncertainty. That allows two new kinds of query. A threshold query: give me buildings that are destroyed with probability at least 0.8. With the final model that returned 104 test buildings and 97 of them really were destroyed, 93.3 percent, while the plain arg-max destroyed label is right only about half the time, 51 percent. And an expected count: instead of counting hard labels, I sum the destroyed probability. If ten buildings are each fifty percent destroyed, the hard count says zero or ten, but the expected count says five, which is the honest answer. I also use window functions, ROW_NUMBER over uncertainty, to compute accuracy if the system answers only the buildings it is surest about.");
  codeBox(s, 0.6, 2.2, 6.3, 1.7, "Threshold query", "SELECT COUNT(*), SUM(true_idx = 3)\nFROM v_predictions\nWHERE split = 'test' AND p_destroyed >= 0.8;", 12);
  codeBox(s, 0.6, 4.05, 6.3, 2.4, "Expected vs hard count (probabilistic aggregation)", "SELECT disaster,\n  SUM(pred_idx = 3)          AS hard_destroyed,\n  ROUND(SUM(p_destroyed), 1) AS expected_destroyed\nFROM v_predictions WHERE split = 'test'\nGROUP BY disaster;", 12);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.3, y: 2.2, w: 5.55, h: 1.7, rectRadius: 0.12, fill: { color: K.tint }, line: { type: "none" } });
  T(s, DBN.thrPct, { x: 7.55, y: 2.35, w: 2.5, h: 0.9, fontSize: 44, bold: true, fontFace: "Arial", color: K.orD });
  T(s, `of the ${DBN.thrN} buildings with p_destroyed ≥ 0.8 really were destroyed (${DBN.thrTrue})`, { x: 10.1, y: 2.4, w: 2.6, h: 1.4, fontSize: 13, color: K.text });
  card(s, 7.3, 4.05, 5.55, 2.4, "Why expected counts", "10 buildings, each 50% likely destroyed: the hard count says 0 or 10, the expected count says 5. Window functions (ROW_NUMBER over uncertainty) give accuracy on the surest X%: " + DBN.sel50 + " on the surest half vs " + DBN.selAll + " on all.", false, { headSize: 16, bodySize: 13 });
  T(s, DBN.src, { x: 0.6, y: 6.55, w: 11, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 14 AI inside SQL
{
  const s = slide("aisql", "DARK", "Database", "AI inside the database", "The neural network is callable from SQL, on images stored in the database",
    "This is the part where AI is integrated into the database itself. I register the trained PyTorch model as SQL functions on the database connection. So in plain SQL I can write SELECT ai_predict of pre_png and post_png FROM crops, and the network runs on the image bytes that are already stored in the database; the images never leave it. There is also ai_confidence, ai_probs which returns the four probabilities as JSON, and pure SQL helpers on stored predictions: entropy, expected severity, and damage_label. This is the same idea as in-database machine learning in PostgreSQL with PL/Python, PostgresML or BigQuery ML: bring the model to the data, not the data to the model.");
  codeBox(s, 0.6, 2.2, 7.2, 2.75, "In-database inference (runs on the stored PNG bytes)", "SELECT b.uid,\n       damage_label(b.label_idx)            AS truth,\n       ai_predict(c.pre_png, c.post_png)    AS ai,\n       ai_confidence(c.pre_png, c.post_png) AS conf\nFROM buildings b JOIN crops c ON c.uid = b.uid\nWHERE b.split = 'test'\nLIMIT 6;", 12.5);
  codeBox(s, 0.6, 5.1, 7.2, 1.35, "How: register the model on the connection (sqlai.py)", "conn.create_function(\"ai_predict\", 2, predict_fn,\n                     deterministic=True)", 12);
  const fns = [["ai_predict(pre, post)", "damage class from the network"], ["ai_confidence(pre, post)", "top probability"], ["ai_probs(pre, post)", "4 probabilities as JSON"],
    ["entropy4(p1..p4)", "uncertainty of a stored row"], ["expected_severity(p1..p4)", "0 to 3, probability-weighted"]];
  fns.forEach(([f, d], i) => {
    const y = 2.2 + i * 0.85;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y, w: 4.65, h: 0.72, rectRadius: 0.08, fill: { color: K.cardD }, line: { color: i < 3 ? K.or : K.blue, width: 1.2 } });
    T(s, f, { x: 8.35, y: y + 0.07, w: 4.4, h: 0.3, fontSize: 12.5, bold: true, fontFace: "Courier New", color: i < 3 ? K.orL : "9CC3E6" });
    T(s, d, { x: 8.35, y: y + 0.38, w: 4.4, h: 0.3, fontSize: 11.5, color: K.soft });
  });
  T(s, "Bring the model to the data: same idea as PL/Python in PostgreSQL, PostgresML, BigQuery ML", { x: 8.2, y: 6.5, w: 4.7, h: 0.3, fontSize: 10, color: "8B95A1" });
}
// 15 triggers
{
  const s = slide("triggers", "LIGHT", "Database", "Triggers: human in the loop", "Uncertain predictions route themselves to a human, automatically",
    "The second integration is a feedback loop built with triggers. When predict.py inserts a prediction, the trigger trg_queue_uncertain checks it: if the uncertainty is at least 0.5 or the confidence below 0.6, it inserts the building into review_queue. No application code is needed; the database enforces the rule. A human expert then records the correct label in human_reviews, and a second trigger marks the queue item as done. A view, v_model_vs_human, then measures how often the model agrees with the experts, and those corrected labels are exactly the data you would use to retrain. So the database is not just storage: it runs the workflow between the AI and the people.");
  const flow = [["predict.py", "INSERT into predictions", K.blue], ["trg_queue_uncertain", "WHEN uncertainty ≥ 0.5\nOR confidence < 0.6", K.or], ["review_queue", "status = 'pending'", K.or], ["Human expert", "INSERT into human_reviews", K.green], ["trg_review_done", "status = 'done'", K.or]];
  flow.forEach(([h, t, c], i) => {
    const x = 0.6 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.25, w: 2.2, h: 1.3, rectRadius: 0.1, fill: { color: "FFFFFF" }, line: { color: c, width: 2 } });
    T(s, h, { x: x + 0.05, y: 2.35, w: 2.1, h: 0.35, fontSize: 11.5, bold: true, fontFace: "Courier New", color: K.text, align: "center" });
    T(s, t, { x: x + 0.1, y: 2.75, w: 2.0, h: 0.75, fontSize: 11, color: K.muted, align: "center" });
    if (i < 4) line(s, x + 2.2, 2.9, x + 2.5, 2.9, "5B6675", true);
  });
  codeBox(s, 0.6, 3.85, 7.3, 2.6, "The trigger (sqlai.py)", "CREATE TRIGGER trg_queue_uncertain\nAFTER INSERT ON predictions\nWHEN NEW.uncertainty >= 0.5 OR NEW.confidence < 0.6\nBEGIN\n  INSERT OR REPLACE INTO review_queue\n    (uid, model_version, reason, uncertainty)\n  VALUES (NEW.uid, NEW.model_version, …);\nEND;", 11.5);
  card(s, 8.2, 3.85, 4.65, 2.6, "Closing the loop", "v_model_vs_human measures how often the model agrees with the experts. Corrected labels in human_reviews are the next training data.", false, { headSize: 16, bodySize: 13.5 });
}
// 16 views
if (0) {
  const s = slide("views", "DARK", "Database", "Views and analytics", "Seven views turn raw predictions into answers",
    "On top of the tables I defined seven views, so the dashboard and the report just query a view. v_predictions joins buildings and predictions and adds a correct column. v_disaster_damage aggregates per disaster with expected and hard counts. v_calibration groups by confidence bins to check whether eighty percent confident really means eighty percent correct. v_uncertainty_rank uses window functions for selective accuracy. v_review_queue is the pending queue ordered by uncertainty. v_model_leaderboard ranks models by validation macro-F1 with RANK, and v_model_vs_human compares the model with expert labels.");
  const v = [["v_predictions", "buildings ⋈ predictions + correct flag"], ["v_disaster_damage", "GROUP BY disaster: expected vs hard counts"], ["v_calibration", "confidence bins vs real accuracy"],
    ["v_uncertainty_rank", "ROW_NUMBER() / COUNT() OVER (PARTITION BY model)"], ["v_review_queue", "pending items ORDER BY uncertainty DESC"], ["v_model_leaderboard", "RANK() OVER (ORDER BY val_macro_f1 DESC)"], ["v_model_vs_human", "agreement with expert labels"]];
  v.forEach(([n, d], i) => {
    const col = i % 2, r = Math.floor(i / 2), x = 0.6 + col * 6.2, y = 2.2 + r * 1.07;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 5.95, h: 0.9, rectRadius: 0.08, fill: { color: K.cardD }, line: { type: "none" } });
    T(s, n, { x: x + 0.25, y: y + 0.1, w: 5.5, h: 0.35, fontSize: 14, bold: true, fontFace: "Courier New", color: K.orL });
    T(s, d, { x: x + 0.25, y: y + 0.48, w: 5.5, h: 0.35, fontSize: 12.5, color: K.soft });
  });
}

// 17 models
if (0) {
  const s = slide("models", "DARK", "Models and results", "Three models, one question", "Does seeing the “before” image help, and how should we combine it?");
  card(s, 0.6, 2.3, 3.9, 3.4, "Post-only", "ResNet-18 sees only the after image. Cannot tell an old ruin from new damage.", true, { kicker: "A · BASELINE" });
  card(s, 4.7, 2.3, 3.9, 3.4, "Siamese", "One shared ResNet-18 for both images. Each is squeezed to a 512-number vector, then compared. Where the change was is lost.", true, { kicker: "B · LATE FUSION" });
  card(s, 8.8, 2.3, 3.9, 3.4, "Feature-map fusion", "Compare the two images position by position after layer 3: [pre, post, |post − pre|]. Layer 4 then reads the change map.", true, { kicker: "C · MY IMPROVEMENT", kickerColor: K.orL, fill: "2A2018", line: K.or });
  T(s, "All three: ImageNet-pretrained ResNet-18, focal loss, same data and epochs. Only the architecture changes. Every run is a row in model_runs.", { x: 0.6, y: 6.0, w: 12, h: 0.4, fontSize: 15, color: K.soft });
}
// 18 surprise
{
  const s = slide("surprise", "LIGHT", "Models and results", "A surprise in the pilot", "The first Siamese model lost to the baseline. Fusion was the reason.");
  const sc = 6.6 / 0.616;
  bar(s, 0.6, 2.4, "A · Post-only · 0.616", 0, sc, 0.616, K.blue);
  bar(s, 0.6, 3.3, "B · Siamese, late fusion · 0.550", 0, sc, 0.550, K.grey);
  bar(s, 0.6, 4.2, "C · Siamese, feature-map fusion · 0.581", 0, sc, 0.581, K.or);
  T(s, "Test macro-F1, bars start at 0. Pilot: 3 disasters, 6,000 training buildings, 64 px, 5 epochs, 7,880 test buildings.", { x: 0.6, y: 5.2, w: 6.8, h: 0.6, fontSize: 12, color: "6B7280" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 2.3, w: 4.5, h: 3.6, rectRadius: 0.12, fill: { color: K.dark }, line: { type: "none" } });
  T(s, "What I learned", { x: 8.5, y: 2.55, w: 4, h: 0.45, fontSize: 20, bold: true, fontFace: "Arial", color: K.orL });
  T(s, "Adding the before image does not help by itself. Squeezing first throws away where things changed. Comparing feature maps won back over half the gap.", { x: 8.5, y: 3.15, w: 3.9, h: 2.6, fontSize: 16, color: "E9E6DF", lineSpacingMultiple: 1.1 });
}
// 19 results
{
  const s = slide("results", "LIGHT", "Models and results", "Full-scale results", "At full scale, feature-map fusion wins on overall quality");
  table(s, 0.6, 2.3, 8.3, ["Model", "Macro-F1", "Accuracy", "Recall destroyed"], [
    ["A · Post-only", "0.600", "0.710", "0.716"], ["B · Siamese, late fusion", "0.611", "0.716", "0.628"], ["C · Siamese, feature-map fusion", "0.632", "0.739", "0.661"]],
    [3.5, 1.5, 1.5, 1.8], { size: 16, rowH: 0.6, right: [1, 2, 3], hi: 2 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.3, y: 2.3, w: 3.4, h: 2.4, rectRadius: 0.12, fill: { color: K.dark }, line: { type: "none" } });
  T(s, "+3.2", { x: 9.6, y: 2.5, w: 3, h: 1.0, fontSize: 54, bold: true, fontFace: "Arial", color: K.orL });
  T(s, "points macro-F1 over post-only (+5.3% relative)", { x: 9.6, y: 3.55, w: 2.9, h: 0.9, fontSize: 14, color: "E9E6DF" });
  T(s, "Test set: 15,812 buildings, 10 disasters, 128 px, 15 epochs; C chosen on validation macro-F1 (0.647). Honest trade-off: post-only misses fewer collapses (recall 0.716 vs 0.661), so doubtful “destroyed” calls go to a human. A 192 px retrain of C did not help (val 0.643, test 0.616), so 128 px stays final.", { x: 0.6, y: 5.0, w: 12.1, h: 1.2, fontSize: 14, color: K.muted });
  T(s, "Runs: post_focal, siamese_focal, siamese_fm_focal (runs/*/metrics.json; also in the model_runs table)", { x: 0.6, y: 6.5, w: 10, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 20 perclass
if (0) {
  const s = slide("perclass", "LIGHT", "Models and results", "Where it struggles", "Clear cases are easy; minor damage is the hard class");
  table(s, 0.6, 2.3, 6.6, ["Class", "Precision", "Recall", "F1"], [["No damage", "0.942", "0.821", "0.877"], ["Minor", "0.386", "0.673", "0.491"], ["Major", "0.658", "0.521", "0.582"], ["Destroyed", "0.514", "0.661", "0.579"]],
    [2.4, 1.4, 1.4, 1.4], { size: 16, rowH: 0.58, right: [1, 2, 3], hi: 1 });
  card(s, 7.6, 2.3, 5.1, 1.9, "Why minor is hard", "xBD is about 0.5–0.8 m per pixel, so a house is only about 25 pixels wide. A cracked roof is 1–2 pixels.", false, { bodySize: 15 });
  card(s, 7.6, 4.4, 5.1, 1.6, "Disasters differ", "Test accuracy ranges from 46% (Hurricane Matthew) to 99% (Mexico earthquake).", false, { bodySize: 15 });
  T(s, "Model C, test set · macro-F1 0.632 · harmonic F1 0.604", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 21 tradeoff
if (0) {
  const s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Models and results" });
  s.background = { color: K.orD };
  T(s, num("The honest trade-off").toUpperCase(), { x: 0.6, y: 0.45, w: 12, h: 0.3, fontSize: 12, bold: true, charSpacing: 2, color: "FFE1CC" });
  T(s, "The best overall model is not the one that misses the fewest collapses", { x: 0.6, y: 1.0, w: 11.5, h: 1.6, fontSize: 34, bold: true, fontFace: "Arial", color: "FBF3EC" });
  T(s, "0.716", { x: 0.6, y: 3.0, w: 4.5, h: 1.2, fontSize: 66, bold: true, fontFace: "Arial", color: "FBF3EC" });
  T(s, "recall on destroyed · post-only", { x: 0.6, y: 4.2, w: 4.5, h: 0.4, fontSize: 16, color: "FFE1CC" });
  T(s, "0.661", { x: 5.6, y: 3.0, w: 4.5, h: 1.2, fontSize: 66, bold: true, fontFace: "Arial", color: "FBF3EC" });
  T(s, "recall on destroyed · feature-map fusion", { x: 5.6, y: 4.2, w: 5, h: 0.4, fontSize: 16, color: "FFE1CC" });
  T(s, "For rescue, a missed collapse costs more than a false alarm. So I report both, and the uncertainty trigger sends doubtful “destroyed” calls to a human.", { x: 0.6, y: 5.1, w: 11.5, h: 1.0, fontSize: 18, color: "FBF3EC" });
  s.addNotes(notes("tradeoff"));
}
// 22 uncertainty
{
  const s = slide("uncertainty", "DARK", "Trust", "Knowing when it doesn’t know", "The model rates its own doubt, and unsure buildings go to a human");
  const steps = [["Keep dropout on at prediction time and predict each building 20 times."], ["The average gives the 4 probabilities; the spread gives the uncertainty."], ["Both are stored in the predictions table; the trigger sends unsure buildings to the review queue."]];
  steps.forEach(([t], i) => {
    const y = 2.4 + i * 1.15;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.55, h: 0.55, fill: { color: K.or }, line: { type: "none" } });
    T(s, String(i + 1), { x: 0.6, y, w: 0.55, h: 0.55, fontSize: 16, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
    T(s, t, { x: 1.4, y: y + 0.02, w: 6.2, h: 0.95, fontSize: 17, color: "E9E6DF" });
  });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.2, y: 2.3, w: 4.5, h: 3.9, rectRadius: 0.12, fill: { color: K.cardD }, line: { type: "none" } });
  bigNum(s, 8.5, 2.5, 4, "73.9%", "accuracy on all buildings", K.grey, true);
  bigNum(s, 8.5, 4.2, 4, "88.6%", "on the 50% it is most sure about; the rest go to a human", K.orL, true);
  T(s, "Monte Carlo Dropout (Gal and Ghahramani, ICML 2016)", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 10, color: "8B95A1" });
}
// 23 xai
if (0) {
  const s = slide("xai", "LIGHT", "Trust", "Explainable AI", "Grad-CAM shows where the model looked before it decided");
  const panels = [["Before", "7E9A6E", "C9C2B3"], ["After", "8A8270", "5B5248"]];
  panels.forEach(([lab, bg, house], i) => {
    const x = 0.6 + i * 2.25;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.4, w: 2.05, h: 2.05, rectRadius: 0.1, fill: { color: bg }, line: { type: "none" } });
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.55, y: 2.95, w: 0.95, h: 0.8, fill: { color: house }, line: { type: "none" } });
    T(s, lab, { x, y: 4.55, w: 2.05, h: 0.35, fontSize: 14, align: "center", color: K.text });
  });
  const hx = 0.6 + 2 * 2.25;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: hx, y: 2.4, w: 2.05, h: 2.05, rectRadius: 0.1, fill: { color: "24456B" }, line: { type: "none" } });
  [[1.6, "3E7CB1"], [1.15, "F3D36B"], [0.75, "F08A4B"], [0.4, "D7301F"]].forEach(([d, c]) => s.addShape(pres.shapes.OVAL, { x: hx + 1.025 - d / 2, y: 3.425 - d / 2, w: d, h: d, fill: { color: c }, line: { type: "none" } }));
  T(s, "Heatmap", { x: hx, y: 4.55, w: 2.05, h: 0.35, fontSize: 14, align: "center", color: K.text });
  T(s, "Illustration; real examples in the live demo", { x: 0.6, y: 5.0, w: 6.6, h: 0.3, fontSize: 11, italic: true, color: "6B7280" });
  T(s, [
    { text: "Gradients of the predicted class show which regions mattered; red = high importance.", options: { bullet: true, breakLine: true } },
    { text: "For model C the heatmap is taken on the fused change map, so it shows where the change drove the decision.", options: { bullet: true, breakLine: true } },
    { text: "Which buildings? The 30 most uncertain: SELECT uid FROM predictions ORDER BY uncertainty DESC LIMIT 30.", options: { bullet: true } },
  ], { x: 7.6, y: 2.4, w: 5.1, h: 3.5, fontSize: 17, color: K.text, paraSpaceAfter: 12 });
  T(s, "Grad-CAM (Selvaraju et al., ICCV 2017)", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 24 hotspots
if (0) {
  const s = slide("hotspots", "LIGHT", "Trust", "Knowledge discovery", "From thousands of dots to a short list of places to go first");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.3, w: 3.6, h: 3.6, rectRadius: 0.12, fill: { color: K.dark }, line: { type: "none" } });
  T(s, "37", { x: 0.9, y: 2.6, w: 3, h: 1.6, fontSize: 96, bold: true, fontFace: "Arial", color: K.orL });
  T(s, "damage hotspots found across the 10 disasters", { x: 0.9, y: 4.3, w: 3, h: 1.2, fontSize: 16, color: "E9E6DF" });
  const rows = [["Input (SQL)", "buildings JOIN predictions WHERE pred_idx IN (2, 3) AND confidence ≥ 0.6."], ["Method", "DBSCAN with real-world (haversine) distance: at least 5 such buildings within 150 m."],
    ["Why DBSCAN", "No need to guess the number of clusters; isolated buildings stay as noise."], ["Output (SQL)", "DELETE old rows, then INSERT into the hotspots table; drawn on the map."]];
  rows.forEach(([h, t], i) => T(s, [{ text: h + ": ", options: { bold: true } }, { text: t }], { x: 4.7, y: 2.35 + i * 0.9, w: 8, h: 0.8, fontSize: 16, color: K.text }));
  T(s, "Code: src/aftershock/hotspots.py · DBSCAN (Ester et al., KDD 1996)", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 23b xai + hotspots
{
  const s = slide("xaihot", "LIGHT", "Trust", "Explainable AI and knowledge discovery", "Why it decided, and where to send teams first",
    "Two more steps turn predictions into something responders can use. Grad-CAM uses the gradients of the predicted class to show which part of the image drove the decision; for my model it is computed on the fused change map, so it shows where the change mattered. I run it on the 30 most uncertain buildings, chosen with an ORDER BY uncertainty query. Then DBSCAN, a density-based clustering algorithm, reads from a join of buildings and predictions all buildings predicted major or destroyed with confidence at least 0.6, and groups them when at least 5 are within 150 metres, using real-world haversine distance. I do not need to choose the number of clusters, and isolated buildings stay as noise. It found 37 hotspots, which are written back into the hotspots table and drawn on the map.");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 2.2, w: 5.95, h: 4.25, rectRadius: 0.12, fill: { color: K.cardL }, line: { type: "none" } });
  T(s, "Grad-CAM (XAI)", { x: 0.9, y: 2.4, w: 5.4, h: 0.4, fontSize: 19, bold: true, fontFace: "Arial", color: K.orD });
  T(s, [
    { text: "Gradients of the predicted class show which regions mattered; red = high importance.", options: { bullet: true, breakLine: true } },
    { text: "For model C it is taken on the fused change map: where the change drove the decision.", options: { bullet: true, breakLine: true } },
    { text: "Run on the 30 most uncertain buildings:", options: { bullet: true } },
  ], { x: 0.9, y: 2.95, w: 5.4, h: 2.3, fontSize: 14.5, color: K.text, paraSpaceAfter: 8 });
  T(s, "SELECT uid FROM predictions\nORDER BY uncertainty DESC LIMIT 30;", { x: 0.9, y: 5.4, w: 5.4, h: 0.8, fontSize: 12, fontFace: "Courier New", color: K.text });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.9, y: 2.2, w: 5.95, h: 4.25, rectRadius: 0.12, fill: { color: K.dark }, line: { type: "none" } });
  T(s, "DBSCAN hotspots (data mining)", { x: 7.2, y: 2.4, w: 5.4, h: 0.4, fontSize: 19, bold: true, fontFace: "Arial", color: K.orL });
  T(s, "37", { x: 7.2, y: 2.9, w: 1.9, h: 1.2, fontSize: 66, bold: true, fontFace: "Arial", color: K.orL });
  T(s, "damage hotspots across the 10 disasters", { x: 9.1, y: 3.15, w: 3.5, h: 0.8, fontSize: 14, color: "E9E6DF" });
  T(s, [
    { text: "In (SQL): buildings ⋈ predictions, major or destroyed, confidence ≥ 0.6", options: { bullet: true, breakLine: true } },
    { text: "≥ 5 such buildings within 150 m (haversine distance)", options: { bullet: true, breakLine: true } },
    { text: "No need to fix the number of clusters; isolated buildings stay noise", options: { bullet: true, breakLine: true } },
    { text: "Out (SQL): INSERT into the hotspots table, drawn on the map", options: { bullet: true } },
  ], { x: 7.2, y: 4.25, w: 5.4, h: 2.1, fontSize: 13, color: "E9E6DF", paraSpaceAfter: 6 });
  T(s, "Grad-CAM (Selvaraju et al., ICCV 2017) · DBSCAN (Ester et al., KDD 1996)", { x: 0.6, y: 6.55, w: 10, h: 0.3, fontSize: 10, color: "6B7280" });
}
// 25 demo
if (0) {
  const s = slide("demo", "DARK", "Trust", "Live demo", "Everything above, on one interactive map and one SQL console",
    "Now the live demo. The dashboard shows the damage map with hotspots, the review queue of the most uncertain buildings, and for any building the before, after and Grad-CAM images. Then I run the SQL demo: python -m aftershock.sqlai --demo. It shows the model leaderboard ranked with a window function, the expected counts per disaster, the review queue filled by the trigger, the index versus full scan timing, and finally ai_predict running the neural network inside a SELECT statement.");
  const items = [["Damage map", "buildings coloured by predicted damage, plus hotspots"], ["Review queue", "filled by the trigger, most uncertain first"], ["Inspect", "before, after and the Grad-CAM heatmap for any building"],
    ["SQL demo", "python -m aftershock.sqlai --demo"], ["ai_predict()", "the network running inside a SELECT"], ["Index vs scan", "EXPLAIN QUERY PLAN, live timing"]];
  items.forEach(([h, t], i) => card(s, 0.6 + (i % 3) * 4.1, 2.3 + Math.floor(i / 3) * 2.0, 3.85, 1.8, h, t, true, { headSize: 18, bodySize: 14 }));
  T(s, "Streamlit + Folium dashboard · sqlai.py SQL demo", { x: 0.6, y: 6.5, w: 8, h: 0.3, fontSize: 10, color: "8B95A1" });
}
// 26 limits
{
  const s = slide("limits", "LIGHT", "Wrap-up", "Problems and limits", "What went wrong, what I fixed, and what is still open",
    "I want to be upfront about problems. My laptop ran out of memory during training, so I cut the data loader workers and moved the full run to an 8 GB GPU. The Siamese model lost in the pilot, which led to my fusion design. During review I found the dashboard was choosing the best model by test score, which is a methodology mistake, and that one metric included training buildings; both are fixed. Class imbalance is handled with focal loss and macro-F1. Limits: one run per model; destroyed precision is 0.51; satellites only see roofs; it is untested outside xBD. On the database side, experiments ran on SQLite; the PostgreSQL and PostGIS schema is written but the AI layer with triggers and SQL functions is SQLite-only for now. It is a tool to support responders, not to replace them.");
  const left = ["Laptop ran out of RAM mid-training → fewer loader workers, then an 8 GB GPU", "Siamese lost in the pilot → designed feature-map fusion", "Dashboard picked the best model by test score → now by validation", "“Agreement” counted training buildings → now test only", "63% of buildings undamaged → focal loss and macro-F1"];
  const right = ["One training run per model; bootstrap CIs, not repeated seeds", "Destroyed precision is 0.51: false alarms are common", "Satellites see roofs, not interiors or walls", "Experiments on SQLite; PostGIS schema written, AI layer SQLite-only so far", "A support tool for responders, not a replacement"];
  [[0.6, "Problems I hit and fixed", K.green, left], [6.8, "Limits that remain", K.red, right]].forEach(([x, h, c, list]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.25, w: 5.9, h: 4.1, rectRadius: 0.12, fill: { color: K.cardL }, line: { type: "none" } });
    T(s, h, { x: x + 0.3, y: 2.45, w: 5.3, h: 0.45, fontSize: 19, bold: true, fontFace: "Arial", color: c });
    T(s, list.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < list.length - 1 } })), { x: x + 0.3, y: 3.0, w: 5.3, h: 3.2, fontSize: 14, color: K.text, paraSpaceAfter: 6 });
  });
}
// 27 conclusion
{
  const s = slide("conclusion", "DARK", "Wrap-up", "Conclusion", "Four things to take away",
    "To conclude, four points. One: the database is the spine of the system; every module reads and writes through SQL, with keys, constraints, indexes, views and triggers. Two: AI is integrated into the database itself: the network is a SQL function, and a trigger routes uncertain predictions to people. Three: on 107 thousand real buildings, comparing before and after at the feature-map level gave the best overall result, and I only got there because a pilot showed the simple way fails. Four: storing probabilities lets the system answer only what it is sure about, 88.6 percent accuracy on half the buildings, and hand the rest to people.");
  const pts = ["The database is the spine: every module reads and writes through SQL, with keys, constraints, indexes, views and triggers.",
    "AI lives inside the database: the network is a SQL function, and a trigger routes uncertain predictions to a human.",
    "On 107,441 real buildings, feature-map fusion gave the best macro-F1 (0.632), after a pilot showed that naive fusion fails.",
    "Stored probabilities let the system answer only what it is sure of: 88.6% accuracy on half, the rest to humans."];
  pts.forEach((t, i) => {
    const y = 2.15 + i * 0.98;
    T(s, String(i + 1), { x: 0.6, y, w: 0.6, h: 0.8, fontSize: 32, bold: true, fontFace: "Arial", color: K.orL });
    T(s, t, { x: 1.4, y: y + 0.05, w: 11.2, h: 0.85, fontSize: 17, color: "E9E6DF" });
  });
  T(s, [{ text: "Next: ", options: { bold: true, color: "E9E6DF" } }, { text: "the AI layer on PostgreSQL (PL/Python), tuning the destroyed threshold, several seeds per model, testing on a new disaster.", options: { color: K.soft } }], { x: 0.6, y: 6.15, w: 12, h: 0.5, fontSize: 15 });
}
// 28 thanks
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Wrap-up" });
  s.addShape(pres.shapes.OVAL, { x: 8.6, y: 2.4, w: 6.5, h: 6.5, fill: { color: "1C2A3A" }, line: { type: "none" } });
  T(s, "AFTERSHOCK", { x: 0.8, y: 0.8, w: 6, h: 0.35, fontSize: 13, bold: true, charSpacing: 3, color: K.orL });
  T(s, "Thank you", { x: 0.8, y: 2.4, w: 9, h: 1.3, fontSize: 66, bold: true, fontFace: "Arial", color: "E9E6DF" });
  T(s, "Questions are welcome.", { x: 0.8, y: 3.75, w: 9, h: 0.6, fontSize: 24, color: "C9C4BA" });
  T(s, "References: Gupta et al. 2019 (xBD) · He et al. 2016 (ResNet) · Daudt et al. 2018 (Siamese change detection) · Lin et al. 2017 (focal loss) · Gal and Ghahramani 2016 (MC Dropout) · Selvaraju et al. 2017 (Grad-CAM) · Ester et al. 1996 (DBSCAN)", { x: 0.8, y: 5.7, w: 7.6, h: 1.0, fontSize: 12, color: "8B95A1" });
  s.addNotes(notes("thanks"));
}

(async () => {
  const out = process.argv[2];
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("written", out);
})();
