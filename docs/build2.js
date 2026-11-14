const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, Header, PageNumber,
  TableOfContents, HighlightColor, ImageRun,
} = require("docx");

const FONT = "Calibri";
const ACCENT = "B4501C";
const DARK = "1B222B";
const W = 9026; // A4 text width in DXA with 1" margins

// ---------- helpers ----------
// text with **bold** and [[placeholder]] (highlighted) markup
function runs(text, opts = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...opts }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...opts }));
    else out.push(new TextRun({ text: "[" + t.slice(2, -2) + "]", highlight: HighlightColor.YELLOW, ...opts }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...opts }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 60, line: 288 } });
const N = (t, ref = "nums") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 288 } });
const CODE = (lines) => lines.map((l, i) => new Paragraph({
  children: [new TextRun({ text: l || " ", font: "Consolas", size: 19 })],
  shading: { type: ShadingType.CLEAR, fill: "F2F0EB", color: "auto" },
  spacing: { after: 0, before: i === 0 ? 60 : 0, line: 260 },
  indent: { left: 200, right: 200 },
}));
const CAPTION = (t) => new Paragraph({ children: [new TextRun({ text: t, italics: true, size: 19, color: "555555" })], spacing: { before: 60, after: 200 }, alignment: AlignmentType.CENTER });
const BREAK = () => new Paragraph({ children: [new PageBreak()] });

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(header, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, isHead, fill) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: { type: ShadingType.CLEAR, fill: fill || (isHead ? DARK : "FFFFFF"), color: "auto" },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: (opts.right || []).includes(i) && !isHead ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(String(text), { size: 20, color: isHead ? "FFFFFF" : "222222", bold: isHead || undefined }),
    })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, ri) => new TableRow({ children: r.map((c, i) => cell(c, i, false, (opts.highlightRow === ri) ? "FBE3D3" : (ri % 2 ? "F5F3EE" : "FFFFFF"))) })),
    ],
  });
}

// ---------- content ----------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2400, after: 120 }, children: [new TextRun({ text: "DATA MANAGEMENT AND AI INTEGRATION · PROJECT REPORT", size: 20, color: ACCENT, bold: true, characterSpacing: 40 })] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "AfterShock", size: 88, bold: true, font: "Calibri Light", color: DARK })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "Satellite-based building damage assessment after disasters, with uncertainty-aware predictions stored in a spatial database, explainable AI and damage-hotspot discovery", size: 30, color: "444444" })] }),
  new Paragraph({ border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: ACCENT, space: 4 } }, children: [] }),
  new Paragraph({ spacing: { before: 400, after: 80 }, children: runs("**Student:** Daksh Chugh", { size: 24 }) }),
  new Paragraph({ spacing: { after: 80 }, children: runs("**Roll No.:** [[Roll number]]", { size: 24 }) }),
  new Paragraph({ spacing: { after: 80 }, children: runs("**Institution:** Bennett University", { size: 24 }) }),
  new Paragraph({ spacing: { after: 80 }, children: runs("**Course topics combined:** 4 (core), 1, 3 and 5 (XAI part)", { size: 24 }) }),
  new Paragraph({ spacing: { after: 80 }, children: runs("**Date:** November 2026", { size: 24 }) }),
  BREAK(),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Contents", font: "Calibri Light", size: 34, bold: true, color: DARK })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  BREAK(),
);

// Abstract
children.push(
  H1("Abstract"),
  P("After an earthquake, flood or cyclone, rescue teams first need to know which buildings are damaged and where. This is still largely done by hand on satellite images, which takes days. AfterShock is a system that compares pre-disaster and post-disaster satellite crops of every building and classifies its damage on the four-level Joint Damage Scale (no damage, minor, major, destroyed). Building footprints, image crops, labels and predictions are stored in a database; a PyTorch model is trained on crops streamed from it; Monte Carlo Dropout turns each prediction into class probabilities plus an uncertainty that are written back to the database and can be queried by confidence; Grad-CAM explains decisions; and DBSCAN finds damage hotspots for rescue prioritisation. An AI layer inside the database makes the network callable from SQL, routes uncertain predictions to human review with triggers, and answers probabilistic questions with views and window functions. Everything is presented on an interactive map dashboard."),
  P("On the public xBD benchmark (10 disasters, 107,441 buildings, 15,812 test buildings split by whole tiles), a Siamese network with **feature-map fusion**, which I designed after a pilot showed that the standard late-fusion Siamese network underperforms, achieved the best test macro-F1 of **0.632** (accuracy 0.739), against 0.611 for late fusion and 0.600 for a post-image-only baseline. The simpler baseline, however, has higher recall on destroyed buildings (0.716 vs 0.661), a trade-off reported openly. Using uncertainty to answer only the 50% most confident buildings raises accuracy to **88.6%**, with the rest routed to human review. DBSCAN found **37** damage hotspots."),
);

// 1 Introduction
children.push(
  H1("1. Introduction and Problem Statement"),
  P("Damage assessment is the first step of disaster response: it decides where search-and-rescue teams, medical aid and engineers go first. High-resolution satellite imagery is available within hours of most large disasters, but turning it into a building-by-building damage map is slow manual work."),
  P("Three problems motivate this project:"),
  B("**Speed.** Manual mapping does not scale to city-sized disasters."),
  B("**Trust.** A model that is wrong without warning is dangerous; a missed collapse can cost lives. Predictions must carry a confidence, and an explanation."),
  B("**Prioritisation.** A list of damaged buildings is not a plan; teams need ranked areas."),
  P("**Goal:** build an accurate, explainable and uncertainty-aware damage classifier on large-scale data stored in a database, and turn its output into rescue priorities."),
  H1("2. Objectives"),
  N("Store satellite image metadata, building polygons and image crops in a (spatial) database."),
  N("Train deep learning models that classify building damage from pre/post image pairs, and compare architectures fairly."),
  N("Estimate prediction uncertainty, store it per building, and route low-confidence buildings to human review."),
  N("Explain predictions visually with Grad-CAM."),
  N("Discover damage hotspots with clustering and answer spatial, confidence-filtered queries."),
  N("Present results on an interactive dashboard."),
);

// 3 Topics
children.push(
  H1("3. Alignment with the Course Topics"),
  P("The project combines four of the five prescribed topics, with Topic 4 as the core."),
  table(["Course topic", "Requirement", "How AfterShock meets it"], [
    ["**4. Deep Learning for Large-Scale Data Analysis** (core)", "PyTorch/TensorFlow on large data from a DB; image classification", "Three PyTorch CNNs classify damage for 107,441 buildings whose crops are streamed from the database"],
    ["**1. Probabilistic and Uncertain Databases**", "Store uncertain values; query with confidence", "Four class probabilities, confidence and MC-Dropout uncertainty stored per building; queries filter by confidence and distance"],
    ["**3. Data Mining and KDD**", "Classification/clustering; knowledge discovery from DB data", "DBSCAN on database coordinates finds damage hotspots; ranked hotspot table"],
    ["**5. Explainable AI** (XAI part only)", "Interpret model results on DB-stored data", "Grad-CAM heatmaps for buildings stored in the DB; LIME available as a cross-check"],
  ], [2400, 2700, 3926]),
  CAPTION("Table 1. Mapping of the project to the course topics. Time-series forecasting (the other half of Topic 5) is not used."),
);

// 4 Dataset
children.push(
  H1("4. Dataset"),
  P("**xBD (xView2 challenge)** (Gupta et al., 2019) is the standard public benchmark for building damage assessment: pairs of pre- and post-disaster high-resolution satellite tiles (1024 × 1024 pixels, roughly 0.5–0.8 m per pixel) with every building outlined as a polygon and labelled on the Joint Damage Scale."),
  table(["Property", "Value used in this project"], [
    ["Disasters", "10: guatemala-volcano, hurricane-florence, hurricane-harvey, hurricane-matthew, hurricane-michael, mexico-earthquake, midwest-flooding, palu-tsunami, santa-rosa-wildfire, socal-fire"],
    ["Buildings in the database", "107,441"],
    ["Labels", "no-damage, minor-damage, major-damage, destroyed"],
    ["Split", "By tile (hash of tile id): train 70% / validation 15% / test 15%"],
    ["Test set", "15,812 buildings"],
  ], [2600, 6426]),
  CAPTION("Table 2. Dataset used."),
  P("The data is strongly imbalanced, which shapes every later decision (loss function and metric):"),
  table(["Class", "Test buildings", "Share"], [
    ["No damage", "9,973", "63.1%"], ["Minor damage", "2,198", "13.9%"], ["Major damage", "2,811", "17.8%"], ["Destroyed", "830", "5.2%"], ["**Total**", "**15,812**", "100%"],
  ], [3600, 2700, 2726], { right: [1, 2] }),
  CAPTION("Table 3. Class distribution of the test set. A model that always predicts 'no damage' reaches 63% accuracy but a macro-F1 of only about 0.19 on this set."),
);

// 5 Architecture
children.push(
  H1("5. System Architecture"),
  P("The architecture is database-centred: raw files enter once through an ETL step, and every later module reads from the database with SQL and writes its results back to it. The pipeline has eight steps; steps 1–3 are data management, 4–5 deep learning and probabilistic storage, 6–8 trust and knowledge discovery."),
  N("**xBD tiles** – pre and post images with polygon labels.", "steps"),
  N("**Crops** – each building is cut from both images using its polygon bounding box and resized (128 × 128 px in the main run).", "steps"),
  N("**Database** – buildings, crops (PNG bytes), labels, splits and coordinates are stored (SQLite in development; PostgreSQL + PostGIS schema provided).", "steps"),
  N("**Siamese CNN** – a PyTorch DataLoader streams crop pairs from the database to train and evaluate the models.", "steps"),
  N("**MC Dropout** – 20 stochastic passes per building give probabilities and uncertainty, written to the predictions table.", "steps"),
  N("**Grad-CAM** – heatmaps explain individual predictions.", "steps"),
  N("**DBSCAN** – clusters of confidently damaged buildings become ranked hotspots.", "steps"),
  N("**Dashboard** – Streamlit + Folium map, review queue, explanations, queries and model comparison.", "steps"),
);

// 6 Database
children.push(
  H1("6. Database Design and AI Integration"),
  P("The database is the spine of AfterShock. No module reads image folders: the ETL step loads everything into the database, the training DataLoader streams image bytes from it with SQL, and every AI step writes its output back as rows. On top of the base schema, an **AI layer** (src/aftershock/sqlai.py) makes the model and its probabilistic output usable from plain SQL. Experiments ran on SQLite (one portable file, easy to move between machines); a PostgreSQL + PostGIS schema with the same tables is provided in sql/schema_postgis.sql."),
  H2("6.1 Schema (ten tables)"),
  table(["Table", "Key columns", "Purpose"], [
    ["disasters", "**PK** name; disaster_type", "One row per disaster"],
    ["images", "**PK** image_id; **FK** disaster; split CHECK (train/val/test); width, height", "One row per tile"],
    ["buildings", "**PK** uid; **FK** image_id; disaster, split, label, label_idx, footprint_wkt, lon, lat; PostGIS footprint and generated geom", "One row per building"],
    ["crops", "**PK, FK** uid; pre_png, post_png (BLOB/BYTEA)", "Image bytes, 1:1 with buildings"],
    ["predictions", "**PK** (uid, model_version); pred_idx, p_no, p_minor, p_major, p_destroyed, confidence, uncertainty, created_at", "Probabilistic output, 1:N per building (one row per model)"],
    ["hotspots", "**PK** (model_version, cluster_id); disaster, n_buildings, lon, lat, mean_confidence", "Data-mining output"],
    ["facilities", "**PK** id (serial); name, ftype, lon, lat", "Hospitals, schools for spatial queries"],
    ["model_runs", "**PK** model_version; arch, loss, img_size, val_macro_f1 CHECK (0..1), test_macro_f1, test_accuracy", "Model registry (AI layer)"],
    ["review_queue", "**PK** (uid, model_version); reason, uncertainty, status CHECK (pending/done)", "Filled by a trigger (AI layer)"],
    ["human_reviews", "**PK** review_id; **FK** uid; model_version, reviewer, label_idx CHECK (0..3)", "Expert feedback (AI layer)"],
  ], [1500, 5000, 2526]),
  CAPTION("Table 4. Database schema. Relationships: disasters 1:N images 1:N buildings; buildings 1:1 crops; buildings 1:N predictions; predictions 1:0..1 review_queue."),
  H2("6.2 Keys, constraints and design decisions"),
  B("**Composite keys.** predictions uses (uid, model_version), so every trained model keeps its own predictions and models can be compared with a single query."),
  B("**Referential integrity.** Foreign keys are declared in the PostgreSQL schema. In SQLite the loader inserts in foreign-key order (disasters → images → buildings → crops), so parents always exist before children."),
  B("**CHECK and DEFAULT constraints** restrict split, label and status values, and timestamp rows automatically."),
  B("**Generated column (PostGIS).** geom is GENERATED ALWAYS AS ST_SetSRID(ST_MakePoint(lon, lat), 4326) STORED, so geometry can never disagree with the coordinates."),
  B("**Vertical partitioning.** The ~1.5 GB of PNG bytes live in crops, not in buildings, so scans and joins on buildings stay small and fast."),
  B("**Deliberate denormalisation.** buildings repeats disaster and split, which are also reachable through image_id; strictly this violates 3NF (a transitive dependency). It was done on purpose: the training query filters by split thousands of times without a join, and the data is load-once and read-many, so update anomalies cannot occur in practice."),
  B("**UPSERT and parameterised queries.** All writes use INSERT … ON CONFLICT DO UPDATE (PostgreSQL) or INSERT OR REPLACE (SQLite), so re-running any step never creates duplicates; all values are passed as parameters (?), never concatenated into SQL, which prevents SQL injection."),
  H2("6.3 ETL"),
  P("**Extract:** read every xBD label file (polygons in pixel and geographic coordinates, damage label, disaster). **Transform:** crop each building from the pre and post tile around its polygon, resize to 128 px, encode as PNG, and assign train/val/test by an MD5 hash of the tile id (Section 7.2). **Load:** batched upserts with executemany. The result is 107,441 buildings with 107,441 image pairs in one database."),
  H2("6.4 Indexes and query plans"),
  P("Training and evaluation repeatedly select all buildings of one split. On the full database (107,441 buildings), SELECT COUNT(*) FROM buildings WHERE split = 'test' with the B-tree index buildings_split_idx uses the plan SEARCH … USING COVERING INDEX and takes about **0.3 ms**; forcing a full scan (NOT INDEXED) gives SCAN buildings and about **60 ms**, roughly 190 times slower (timed on the training laptop). On the smaller pilot database the same test gave 1.7 ms vs 107 ms. A composite index predictions(model_version, uncertainty) serves the review queue's ORDER BY, and in PostGIS a GiST index on geom serves ST_DWithin distance queries."),
  H2("6.5 Probabilistic queries"),
  P("Because each prediction is stored as a probability distribution with an uncertainty (attribute-level uncertainty), questions about confidence become ordinary SQL. A **threshold query** keeps only confident answers:"),
  ...CODE([
    "SELECT COUNT(*), SUM(true_idx = 3) AS really_destroyed",
    "FROM v_predictions",
    "WHERE model_version = 'small_siamese_fm' AND split = 'test' AND p_destroyed >= 0.8;",
  ]),
  P("For the final model this returned 104 test buildings, of which 97 (93.3%) were truly destroyed, while the model's plain arg-max 'destroyed' label has a precision of only 0.514. A confidence threshold in SQL makes the 'destroyed' list far more reliable. An **expected count** sums probabilities instead of counting hard labels: if ten buildings are each 50% likely destroyed, the hard count says 0 or 10 but the expected count says 5.", { spacing: { before: 160, after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED }),
  ...CODE([
    "SELECT disaster, SUM(pred_idx = 3) AS hard_destroyed,",
    "       ROUND(SUM(p_destroyed), 1) AS expected_destroyed",
    "FROM v_predictions WHERE split = 'test' GROUP BY disaster;",
  ]),
  P("A spatial-probabilistic query (PostGIS): destroyed buildings within 2 km of a hospital that the model is at least 80% sure about:", { spacing: { before: 160, after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED }),
  ...CODE([
    "SELECT b.uid, p.p_destroyed",
    "FROM buildings b JOIN predictions p USING (uid)",
    "JOIN facilities f ON f.ftype = 'hospital'",
    "WHERE p.pred_idx = 3 AND p.p_destroyed >= 0.8",
    "  AND ST_DWithin(b.geom::geography,",
    "      ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)::geography, 2000);",
  ]),
  H2("6.6 AI inside the database"),
  P("**SQL functions that run the model.** The trained PyTorch network is registered on the database connection (sqlite3 create_function), so it can be called inside a query on the image bytes already stored in the database; the images never leave it. The same idea is used by in-database machine learning systems (PL/Python in PostgreSQL, PostgresML, BigQuery ML): bring the model to the data."),
  ...CODE([
    "SELECT b.uid, damage_label(b.label_idx) AS truth,",
    "       ai_predict(c.pre_png, c.post_png)    AS ai,",
    "       ai_confidence(c.pre_png, c.post_png) AS conf",
    "FROM buildings b JOIN crops c ON c.uid = b.uid",
    "WHERE b.split = 'test' LIMIT 6;",
  ]),
  P("Further functions: ai_probs (the four probabilities as JSON), and pure SQL helpers on stored rows: entropy4 (normalised entropy), expected_severity (0–3, probability-weighted) and damage_label.", { spacing: { before: 160, after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED }),
  P("**Triggers for human-in-the-loop review.** When a prediction is inserted, trg_queue_uncertain adds it to review_queue if its uncertainty is at least 0.5 or its confidence below 0.6; when an expert inserts a corrected label into human_reviews, trg_review_done marks the item done. The workflow between the model and people is enforced by the database rather than by application code, and the corrected labels are ready to be used for retraining."),
  ...CODE([
    "CREATE TRIGGER trg_queue_uncertain AFTER INSERT ON predictions",
    "WHEN NEW.uncertainty >= 0.5 OR NEW.confidence < 0.6",
    "BEGIN",
    "  INSERT OR REPLACE INTO review_queue (uid, model_version, reason, uncertainty)",
    "  VALUES (NEW.uid, NEW.model_version, 'high uncertainty', NEW.uncertainty);",
    "END;",
  ]),
  P("**Views and window functions.** Seven views serve the dashboard and the report: v_predictions (join + correct flag), v_disaster_damage (expected vs hard counts per disaster), v_calibration (confidence bins vs accuracy), v_uncertainty_rank (ROW_NUMBER and COUNT OVER PARTITION BY model, for selective accuracy), v_review_queue, v_model_leaderboard (RANK OVER validation macro-F1) and v_model_vs_human. For the final model, v_uncertainty_rank reproduces the Python selective-accuracy results in SQL: 88.6% on the most certain half of the test buildings, 80.1% at 80% coverage and 73.8% on all of them (Section 8.5).", { spacing: { before: 160, after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED }),
  P("Numbers in this chapter come from python -m aftershock.sqlai --demo run on the full database (aftershock_full.db) with the final model siamese_fm_focal. The trigger has filled review_queue with 17,251 rows for the two models that have stored predictions. The dashboard tab Database & SQL runs the same queries live, on a read-only connection with fixed, parameterised presets."),
);

// 7 Methodology
children.push(
  H1("7. Methodology"),
  H2("7.1 Preprocessing"),
  P("Each building's polygon gives a bounding box in the 1024 × 1024 tile; the same box is cropped from the pre and post image and resized. Training uses identical random flips and 90° rotations on both crops, so that the change between them is preserved. Inputs are normalised with ImageNet statistics."),
  H2("7.2 Leakage-free split"),
  P("Neighbouring buildings share lighting, roof styles and damage. A random split by building would put near-duplicates in train and test and inflate scores. The split is therefore made by **tile**: an MD5 hash of the tile id assigns every tile, and all its buildings, to train, validation or test. The final model is chosen on validation macro-F1; the test set is used once."),
  H2("7.3 Models"),
  table(["Model", "Input", "How it works"], [
    ["A · Post-only baseline", "post crop", "ImageNet-pretrained ResNet-18 → 512-d vector → classifier head"],
    ["B · Siamese, late fusion", "pre + post", "Shared ResNet-18 encodes each crop to a 512-d vector; head classifies [pre, post, |post − pre|]"],
    ["C · Siamese, feature-map fusion (proposed)", "pre + post", "Shared ResNet-18 up to layer 3 on both crops; feature maps compared position by position: 1×1 conv on [pre, post, |post − pre|]; layer 4 then reads this change map; pool → head"],
  ], [2600, 1400, 5026]),
  CAPTION("Table 5. The three architectures. All share the backbone, loss, data and schedule; only the fusion differs."),
  P("The motivation for model C came from an experiment (Section 8.1): late fusion pools each image to a single vector before comparing, which discards **where** the change is. Mid-level fusion keeps the spatial layout, so a collapsed roof in one corner remains visible to the classifier."),
  H2("7.4 Class imbalance"),
  P("Focal loss (Lin et al., 2017) with inverse-frequency class weights down-weights easy, frequent examples. Results are reported with **macro-F1** (unweighted mean of per-class F1), per-class precision/recall, recall on 'destroyed' (safety-critical), and the harmonic mean of class F1s used in xView2."),
  H2("7.5 Training setup"),
  P("AdamW (lr 3e-4, weight decay 1e-4), cosine learning-rate schedule, dropout 0.3 in the head, batch size 128, 15 epochs, 128 × 128 crops, seed 42. The checkpoint with the best validation macro-F1 is kept. The full run used an NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, PyTorch 2.9.1 + CUDA 12.8); pilots used an NVIDIA MX450 (2 GB)."),
  H2("7.6 Uncertainty: Monte Carlo Dropout"),
  P("Dropout is kept active at inference and each building is predicted 20 times (Gal and Ghahramani, 2016). The mean of the softmax outputs gives the four class probabilities; their spread gives the uncertainty. Both are stored in the predictions table. Buildings whose uncertainty exceeds a threshold are placed in a human-review queue. Selective accuracy (accuracy on the most confident fraction of buildings) and a reliability diagram measure whether confidence is meaningful."),
  H2("7.7 Explainability: Grad-CAM"),
  P("Grad-CAM (Selvaraju et al., 2017) weights the last convolutional feature maps by the gradient of the predicted class and shows the result as a heatmap. For model C it is computed on the fused change map, so it highlights where the before/after difference drove the decision. It is used to check that 'destroyed' is driven by the roof rather than shadows or vegetation."),
  H2("7.8 Knowledge discovery: DBSCAN hotspots"),
  P("Buildings predicted major or destroyed with confidence ≥ 0.6 are read from the database and clustered with DBSCAN (Ester et al., 1996) using the haversine distance: a hotspot is at least 5 such buildings within 150 m. DBSCAN needs no preset number of clusters and leaves isolated buildings as noise. Hotspots are stored, ranked by size and shown on the map."),
);

// 8 Results
children.push(
  H1("8. Experiments and Results"),
  P("All numbers below are read from the saved result files (runs/*/metrics.json and runs/*/mc_test.json) and the database; none were typed by hand."),
  H2("8.1 Pilot experiments (3 disasters, laptop GPU)"),
  P("To check that the data carries a learnable signal and to compare architectures cheaply, pilots used 3 disasters (hurricane-harvey, hurricane-matthew, santa-rosa-wildfire; 48,777 buildings; 7,880 test buildings)."),
  table(["Run", "Setup", "Test macro-F1", "Accuracy", "Recall destroyed"], [
    ["Always 'no damage'", "–", "0.14", "0.403", "0.000"],
    ["Post-only, weighted CE", "33k train, 128 px, 10 epochs", "0.665", "0.681", "0.716"],
    ["A · Post-only, focal", "6k train (balanced), 64 px, 5 epochs", "0.616", "0.622", "0.735"],
    ["B · Siamese late fusion, focal", "same as above", "0.550", "0.561", "0.799"],
    ["C · Siamese feature-map fusion, focal", "same as above", "0.581", "0.585", "0.77"],
  ], [2700, 2700, 1200, 1100, 1326], { right: [2, 3, 4] }),
  CAPTION("Table 6. Pilot results on the 3-disaster test set (7,880 buildings)."),
  P("Findings: (i) every model is far above the trivial baseline, so the data is worth the effort; (ii) more data and larger crops help (0.616 → 0.665 for post-only); (iii) **surprisingly, the late-fusion Siamese network lost to the post-only model**. Diagnosing this led to model C, which recovered more than half of the gap, and to taking all three models to the full-scale run."),
  H2("8.2 Full-scale comparison (10 disasters)"),
  table(["Model", "Test macro-F1", "Accuracy", "Recall destroyed"], [
    ["A · Post-only", "0.600", "0.710", "**0.716**"],
    ["B · Siamese, late fusion", "0.611", "0.716", "0.628"],
    ["C · Siamese, feature-map fusion", "**0.632**", "**0.739**", "0.661"],
  ], [3400, 1800, 1700, 2126], { right: [1, 2, 3], highlightRow: 2 }),
  CAPTION("Table 7. Full run: 10 disasters, 128 px, 15 epochs, focal loss, test set of 15,812 buildings. Model C was selected on validation macro-F1 (0.647) before the test set was evaluated."),
  P("Model C improves macro-F1 by **3.2 points** over the post-only baseline (5.3% relative) and has the best accuracy. With a full training set, the ordering matches the hypothesis that seeing the before image helps, provided the images are fused at the feature-map level. Only one training run per model was possible; bootstrap 95% confidence intervals over the test set are shown in the dashboard, and differences of a few points should be read with that in mind."),
  H2("8.3 Per-class performance of model C"),
  table(["Class", "Precision", "Recall", "F1", "Support"], [
    ["No damage", "0.942", "0.821", "0.877", "9,973"],
    ["Minor damage", "0.386", "0.673", "0.491", "2,198"],
    ["Major damage", "0.658", "0.521", "0.582", "2,811"],
    ["Destroyed", "0.514", "0.661", "0.579", "830"],
  ], [2826, 1550, 1550, 1550, 1550], { right: [1, 2, 3, 4] }),
  CAPTION("Table 8. Per-class test metrics of model C. Macro-F1 0.632, harmonic F1 0.604, accuracy 0.739."),
  P("Minor damage is the hardest class. This is largely physical: at 0.5–0.8 m per pixel a house is only about 25 pixels wide, so a cracked roof or a few missing tiles span one or two pixels. Performance also varies strongly by disaster: test accuracy ranges from 46% (Hurricane Matthew, dominated by minor and major damage) to 99% (Mexico earthquake, mostly undamaged buildings)."),
  H2("8.4 The recall trade-off"),
  P("The best overall model is not the one that misses the fewest collapses: the post-only model has higher recall on destroyed buildings (0.716 vs 0.661). For rescue, a missed collapse is worse than a false alarm, and model C's precision on 'destroyed' is only 0.514. Both numbers are reported; tuning the decision threshold for 'destroyed' on the validation set, or combining both models, is future work. In the current system, uncertain 'destroyed' calls are routed to human review."),
  H2("8.5 Uncertainty and selective prediction"),
  P("With MC Dropout (20 passes), accuracy on all test buildings is 73.8% (80% coverage: 80.1%). When the model answers only the 50% of buildings it is most confident about, accuracy rises to **88.6%**; the other 50% go to human review. A reliability diagram in the dashboard compares stated confidence with observed accuracy."),
  H2("8.6 Hotspots"),
  P("DBSCAN found **37** hotspots containing 1,288 severe buildings across the 10 disasters. They are stored in the hotspots table, ranked by the number of damaged buildings, and drawn on the dashboard map."),
  H2("8.7 Does higher resolution help? (128 px vs 192 px)"),
  P("Because a house is only about 25 pixels wide at 128 px, model C was retrained on 192 × 192 crops with the same buildings and split (12 epochs instead of 15 because of time, batch size 64). It did **not** help: validation macro-F1 0.643 vs 0.647, test macro-F1 0.616 vs 0.632, accuracy 0.731 vs 0.739; destroyed recall was unchanged (0.663 vs 0.661) and destroyed precision fell (0.453 vs 0.514). Selection by validation therefore keeps the 128 px model as final. Part of the gap may come from the shorter training, so this comparison is not fully controlled. Its uncertainty ranking still works: accuracy rises from 0.731 at full coverage to 0.889 at 50% coverage."),
);

// 9 Problems faced
children.push(
  H1("9. Problems Faced and How They Were Solved"),
  P("Several problems came up during the project. They are listed honestly because each one changed the design or the way results are reported."),
  table(["#", "Problem", "Cause", "What I did", "Lesson"], [
    ["1", "Dataset is large and gated", "xBD needs registration; the image set used here is about 8 GB", "Started with a 3-disaster subset; moved the data to the GPU machine via Google Drive with a packed code bundle", "Plan data movement early"],
    ["2", "GPU not used at first", "The installed PyTorch was a CPU-only build", "Installed the CUDA build of PyTorch; verified with a benchmark", "Check torch.cuda.is_available() first"],
    ["3", "Training logs stayed empty", "Python buffers output when redirected to a file", "Ran with unbuffered output (python -u / PYTHONUNBUFFERED)", "Make long jobs observable"],
    ["4", "Training killed: out of RAM", "6 processes of ~700 MB each plus a 1.5 GB database on a laptop", "Fewer DataLoader workers; full run moved to an 8 GB GPU machine", "Data loading, not the GPU, was the bottleneck"],
    ["5", "Slow training on a 2 GB GPU", "~10 min per epoch on MX450", "Small balanced pilots (6k buildings, 64 px) for quick decisions", "Pilot first, then scale"],
    ["6", "Siamese model lost to the baseline", "Late fusion pools away where the change is", "Designed feature-map (mid-level) fusion; it became the best model at full scale", "A negative result can lead to the main contribution"],
    ["7", "Severe class imbalance (63% undamaged)", "Nature of disaster data", "Focal loss with class weights; report macro-F1, per-class and destroyed recall", "Accuracy alone hides failure"],
    ["8", "Risk of data leakage", "Neighbouring buildings look alike", "Split by whole tiles using a hash of the tile id", "Evaluation design matters as much as the model"],
    ["9", "Best model chosen by test score in the first dashboard", "Methodology mistake found in review", "Selection now uses validation macro-F1; test used once", "Never tune on test"],
    ["10", "Inflated 'agreement' metric", "It counted training buildings", "Restricted to test buildings and relabelled", "Every number needs its split stated"],
    ["11", "Misleading '+5%' headline", "Relative and absolute gains mixed", "Shown as +3.2 points (+5.3% relative)", "Say which kind of percentage"],
    ["12", "Blurry building images in the dashboard", "A house is ~25 px; a 128 px crop was upscaled to 480 px", "Explained in the UI; display can use the original tile with context", "Resolution limits what can be seen"],
    ["13", "Dashboard usability issues", "Unreadable KPI text, map captured page scroll, cut-off columns, local DB path visible", "Fixed contrast, disabled scroll-zoom, set column widths, hid the path", "Review the UI with fresh eyes"],
    ["14", "Public demo link", "A quick tunnel exposes the dashboard to anyone with the link", "Tunnel only during demos; no free-text SQL; path input removed", "Security applies to demos too"],
    ["15", "Review queue too large with the pilot model", "The trigger thresholds (uncertainty ≥ 0.5 or confidence < 0.6) queued 6,191 of 7,880 pilot test buildings", "Kept the thresholds as an explicit, documented rule; to be tuned on validation for the final model", "A rule in the database must be calibrated like a model"],
  ], [400, 1800, 2100, 2826, 1900]),
  CAPTION("Table 9. Problems faced during the project."),
);

// 10 Dashboard, 11 limitations, 12 conclusion
children.push(
  H1("10. Dashboard"),
  P("The Streamlit dashboard brings the pipeline together: an overview with the architecture comparison (selected by validation score, with bootstrap confidence intervals and the recall trade-off), a damage map per disaster with hotspots, a human-review queue of the most uncertain buildings, a building inspector with before/after crops and an on-the-fly Grad-CAM heatmap, a probabilistic query panel (distance, class and minimum confidence), a calibration chart, and per-class and per-disaster quality tables."),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/dash_map3d.png"), transformation: { width: 420, height: 451 } })] }),
  CAPTION("Figure 1. 3D damage map: one column per building, height and colour = predicted damage (pydeck). Shown on the pilot database (Hurricane Harvey)."),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, children: [new ImageRun({ type: "png", data: fs.readFileSync(__dirname + "/dash_prob3d.png"), transformation: { width: 560, height: 348 } })] }),
  CAPTION("Figure 2. The stored class probabilities of each test building in 3D (p_minor, p_major, p_destroyed); marker size = MC-Dropout uncertainty."),
  H1("11. Limitations and Ethics"),
  B("One training run per architecture; differences of a few points may partly be noise (bootstrap intervals are a partial substitute for repeated seeds)."),
  B("Destroyed-class precision is 0.514: false alarms are frequent, and post-only has higher destroyed recall."),
  B("Satellites see roofs, not interiors or walls; minor damage is near the limit of the image resolution."),
  B("Trained and tested on xBD disasters only; generalisation to new regions and disaster types is untested."),
  B("Experiments ran on SQLite. The PostgreSQL + PostGIS schema is written, but the AI layer (SQL functions, triggers, views) is implemented for SQLite only so far."),
  B("The system supports responders and does not replace field assessment; uncertain cases are routed to humans by design. Only public, research-licensed data is used, with no personal data."),
  H1("12. Conclusion and Future Work"),
  P("AfterShock shows that a database-centred pipeline can turn raw satellite pairs into damage predictions that are accurate, carry a measurable confidence, can be explained, and can be aggregated into ranked hotspots. The main technical finding is that seeing the before image helps only when the two images are fused at the feature-map level: the proposed feature-map fusion Siamese network achieved the best test macro-F1 (0.632) on 15,812 unseen buildings from 10 disasters, after a pilot showed that the standard late-fusion design underperforms. Storing probabilities and uncertainty in the database made selective prediction possible: 88.6% accuracy on the most confident half of the buildings."),
  P("Future work: a longer, controlled higher-resolution run (192 px did not help in 12 epochs, Section 8.7); tune the 'destroyed' threshold on validation to raise recall; repeat training with several seeds; test on a disaster held out entirely; deploy on PostgreSQL + PostGIS at full scale, with the AI layer ported to PL/Python and PL/pgSQL triggers."),
);

// References
children.push(
  H1("References"),
  N("R. Gupta et al., \"Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery,\" CVPR Workshops, 2019.", "refs"),
  N("K. He, X. Zhang, S. Ren, J. Sun, \"Deep Residual Learning for Image Recognition,\" CVPR, 2016.", "refs"),
  N("R. C. Daudt, B. Le Saux, A. Boulch, \"Fully Convolutional Siamese Networks for Change Detection,\" ICIP, 2018.", "refs"),
  N("T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, \"Focal Loss for Dense Object Detection,\" ICCV, 2017.", "refs"),
  N("Y. Gal, Z. Ghahramani, \"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,\" ICML, 2016.", "refs"),
  N("R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,\" ICCV, 2017.", "refs"),
  N("M. Ester, H.-P. Kriegel, J. Sander, X. Xu, \"A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,\" KDD, 1996.", "refs"),
  N("M. T. Ribeiro, S. Singh, C. Guestrin, \"'Why Should I Trust You?': Explaining the Predictions of Any Classifier,\" KDD, 2016.", "refs"),
  H1("Appendix A. How to Reproduce"),
  ...CODE([
    "pip install -r requirements.txt && pip install -e .",
    "python -m aftershock.ingest  --xbd data/xbd/train",
    "python -m aftershock.train   --name post_focal       --arch post       --loss focal --epochs 15",
    "python -m aftershock.train   --name siamese_focal    --arch siamese    --loss focal --epochs 15",
    "python -m aftershock.train   --name siamese_fm_focal --arch siamese_fm --loss focal --epochs 15",
    "python -m aftershock.predict --ckpt runs/siamese_fm_focal/best.pt --passes 20",
    "python -m aftershock.explain --ckpt runs/siamese_fm_focal/best.pt --most-uncertain 30",
    "python -m aftershock.hotspots --model-version siamese_fm_focal",
    "python -m aftershock.sqlai --demo --model-version siamese_fm_focal --ckpt runs/siamese_fm_focal/best.pt --out runs/sql_demo_full.md",
    "streamlit run app/dashboard.py",
  ]),
);

const doc = new Document({
  creator: "Daksh Chugh",
  title: "AfterShock Project Report",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri Light", size: 34, bold: true, color: DARK },
        paragraph: { spacing: { before: 360, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: "Calibri Light", size: 26, bold: true, color: ACCENT },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
      { reference: "nums", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
      { reference: "steps", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
      { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 600 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "AfterShock · Project Report", size: 18, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(process.argv[2], buf); console.log("written", process.argv[2]); });
