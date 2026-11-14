# AfterShock: Satellite-based Building Damage Assessment after Disasters

**Course:** Data Management and AI Integration · **Topic:** Deep Learning for Large-Scale Data Analysis (with Explainable AI, Probabilistic Querying and Data Mining)

**Student:** Daksh Chugh · Bennett University · **Roll No.:** [ ] · **Date:** [ ]

## 1. Abstract

After an earthquake, flood or cyclone, rescue teams first need to know which buildings are damaged and where. Analysts still mark damage by hand on satellite images, which takes days. AfterShock is a deep learning system that compares pre-disaster and post-disaster satellite images and classifies the damage level of every building (no damage, minor, major, destroyed). Building footprints, predictions and confidence values are stored in a spatial database (PostgreSQL + PostGIS). Analysts can therefore query results by location and confidence, see why the model decided (Grad-CAM), and send uncertain cases to a human. Damage hotspots are found with clustering to prioritise rescue. The system is trained and evaluated on the public xBD benchmark.

## 2. Problem Statement

- Manual damage mapping from satellite imagery is slow and does not scale to city-sized disasters.
- A model that is wrong without warning is risky in disaster response: missing a collapsed building costs lives.
- **Goal:** build an accurate, explainable and uncertainty-aware damage classifier on large-scale data stored in a database, and turn its output into rescue priorities.

## 3. Objectives

1. Store large-scale satellite image metadata and building polygons in a spatial database.
2. Train a deep learning model (Siamese CNN) to classify building damage from pre/post image pairs.
3. Estimate prediction uncertainty and flag low-confidence buildings for human review.
4. Explain predictions visually with Grad-CAM.
5. Discover damage hotspots with clustering and answer spatial queries for rescue planning.
6. Present results on an interactive map dashboard.

## 4. Alignment with Course Topics

This project combines four of the five prescribed topics, with **Topic 4 as the core**.

| Course topic | Requirement | How AfterShock meets it |
| --- | --- | --- |
| **4. Deep Learning for Large-Scale Data Analysis** (core) | Use PyTorch/TensorFlow on large datasets from a DB; image classification | Siamese CNN in PyTorch classifies damage for hundreds of thousands of building crops streamed from PostgreSQL/PostGIS |
| **1. Probabilistic and Uncertain Databases** | Store uncertain values; query with confidence levels | Class probabilities and MC-Dropout uncertainty stored per building; queries filter by confidence (Section 7) |
| **3. Data Mining and KDD** | Classification/clustering; knowledge discovery on DB data | DBSCAN finds damage hotspots from database coordinates; spatial queries rank rescue priorities |
| **5. Explainable AI (XAI)** | Interpret model results on database-stored data | Grad-CAM heatmaps (primary) and LIME (comparison) explain predictions for buildings stored in the DB |

Topic 5's time-series forecasting part is not used; only its XAI part is.

## 5. Dataset

**xBD (xView2 Challenge)**: a public benchmark for building damage assessment (Gupta et al., 2019).

| Property | Value |
| --- | --- |
| Imagery | High-resolution pre- and post-disaster satellite image pairs |
| Annotations | ~850,000 building polygons |
| Coverage | ~45,000 km², 19 disasters (earthquakes, floods, hurricanes, wildfires, etc.) |
| Labels | Joint Damage Scale: no damage, minor damage, major damage, destroyed |

**Scope used in this project:** all 10 disasters available in the downloaded xBD train set: 1,715 tiles and 107,441 labelled buildings. The split is made per tile (hashed), so neighbouring buildings never leak between splits. Train has 74,597 buildings, validation 17,032 and test 15,812.

| Disaster | Buildings | | Label | Buildings |
| --- | --- | --- | --- | --- |
| mexico-earthquake | 32,196 | | no-damage | 75,496 |
| hurricane-harvey | 22,725 | | minor-damage | 14,818 |
| hurricane-michael | 22,454 | | major-damage | 13,498 |
| hurricane-matthew | 13,155 | | destroyed | 3,629 |
| midwest-flooding | 8,471 | | | |
| hurricane-florence | 6,120 | | | |
| palu-tsunami | 979 | | | |
| guatemala-volcano | 682 | | | |
| socal-fire | 423 | | | |
| santa-rosa-wildfire | 236 | | | |

The data is heavily imbalanced: only 3.4% of buildings are "destroyed". The hashed split also happened to put every palu-tsunami tile in train, so that disaster has no test buildings.

## 6. System Architecture

```
xBD images + labels
   -> Preprocessing (building crops: pre + post, 128x128)
   -> PostgreSQL + PostGIS  (buildings, crops, predictions)
   -> PyTorch data loader (streams crops from the database)
   -> Siamese CNN  -> damage class + probabilities
   -> MC Dropout   -> uncertainty per building
   -> Grad-CAM     -> visual explanation
   -> DBSCAN + spatial SQL -> hotspots and rescue priorities
   -> Streamlit + Folium map dashboard
```

## 7. Database Design (PostgreSQL + PostGIS)

| Table | Key columns |
| --- | --- |
| `disasters` | name, disaster_type |
| `images` | image_id, disaster, split (train/val/test), width, height |
| `crops` | uid, pre_png, post_png (image crops stored in the database) |
| `buildings` | uid, image_id, disaster, split, label, footprint_wkt, lon, lat, footprint (POLYGON), geom (POINT) |
| `predictions` | uid, model_version, pred_idx, p_no, p_minor, p_major, p_destroyed, confidence, uncertainty, created_at |
| `facilities` | id, name, ftype (hospital, school), lon, lat |
| `hotspots` | model_version, cluster_id, disaster, n_buildings, lon, lat, mean_confidence |

Example query (probabilistic + spatial): *destroyed buildings within 2 km of a hospital where the model is at least 80% confident*:

```
SELECT b.uid, p.p_destroyed
FROM buildings b
JOIN predictions p USING (uid)
JOIN facilities f ON f.ftype = 'hospital'
WHERE p.pred_idx = 3 AND p.p_destroyed >= 0.8
  AND ST_DWithin(b.geom::geography, ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)::geography, 2000);
```

## 8. Methodology

**8.1 Preprocessing.** Crop every building from the pre- and post-disaster images using its polygon, resize to 128×128, normalise, and apply augmentation (flips, rotations, brightness).

**8.2 Baseline.** A ResNet-18 classifier on the post-disaster crop only.

**8.3 Main model: Siamese CNN.** Two weight-sharing ResNet encoders process the pre and post crops; their features and feature difference are combined by a classifier head. This captures *change*, which is what damage is.

**8.4 Class imbalance.** Destroyed and major-damage buildings are rare. Weighted cross-entropy or focal loss and class-balanced sampling are compared.

**8.5 Uncertainty (probabilistic).** Monte Carlo Dropout: dropout stays active at inference and each building is predicted N times. The mean gives the probability and the spread gives the uncertainty. Buildings above an uncertainty threshold are routed to human review.

**8.6 Explainability (XAI).** Grad-CAM heatmaps show which parts of the image drove the decision (for example, a collapsed roof or debris). LIME (superpixel perturbation) is applied to a sample of buildings to cross-check that both methods highlight the same evidence.

**8.7 Knowledge discovery.** DBSCAN on the coordinates of predicted major/destroyed buildings finds damage hotspots, which are ranked by count and distance to facilities.

## 9. Tools and Technologies

| Purpose | Tools |
| --- | --- |
| Deep learning | Python, PyTorch, torchvision / timm |
| Database | PostgreSQL, PostGIS |
| Geospatial | Shapely (polygons), Pillow (image crops), PostGIS (spatial queries) |
| XAI / uncertainty | pytorch-grad-cam, LIME, MC Dropout |
| Data mining | scikit-learn (DBSCAN), pandas |
| Dashboard | Streamlit, Folium |
| Compute / code | NVIDIA GeForce RTX 5070 Laptop GPU (8 GB), PyTorch 2.9.1 + CUDA 12.8; SQLite for the experiments (PostGIS schema in `sql/`); Git |

## 10. Evaluation

| Metric | Purpose |
| --- | --- |
| Per-class precision, recall, F1 | Damage classification quality per class |
| Macro F1 (harmonic mean over classes) | Overall score that does not hide rare classes |
| Confusion matrix | Which damage levels are confused |
| Recall on "destroyed" | Safety-critical: missed collapses |
| Selective accuracy vs coverage | Accuracy after routing uncertain buildings to humans |
| Inference time per km² | Practical speed |

### 10.1 Results

All three models were trained with the same settings: focal loss, 128×128 crops, 15 epochs, ImageNet-pretrained ResNet-18, and the same 74,597 training buildings. Only the architecture differs. **The final model was chosen by validation macro-F1.** The test set was used only once, for the numbers below. All numbers come from `runs/*/metrics.json`.

| Model | Architecture | Val macro F1 | Test macro F1 | Test accuracy | Recall (destroyed) | Precision (destroyed) | Recall (major) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| post_focal | Post-only ResNet-18 (baseline) | 0.609 | 0.600 | 0.710 | **0.716** | 0.410 | 0.503 |
| siamese_focal | Siamese, late fusion | 0.635 | 0.611 | 0.716 | 0.628 | 0.496 | 0.485 |
| **siamese_fm_focal** | **Siamese, feature-map fusion** | **0.647** | **0.632** | **0.739** | 0.661 | **0.514** | **0.521** |

**Findings:**
- The feature-map fusion Siamese (`SiameseFMNet`) is best overall: **+3.2 percentage points** test macro-F1 over the post-only baseline (0.632 vs 0.600, +5.3% relative). It compares pre and post features at every spatial position before pooling, so it keeps *where* the change happened. The plain late-fusion Siamese only gains about 1 point over the baseline.
- **Trade-off:** the post-only baseline has the highest recall on destroyed buildings (0.716 vs 0.661). For rescue, a missed collapse is the costliest error, so this result should not be hidden. The fusion model's advantage is fewer false alarms: destroyed precision is 0.514 vs 0.410.
- Destroyed precision is low for every model (≤ 0.51): about half of the "destroyed" calls are wrong. This is why low-confidence predictions are sent to human review (10.2).
- **Limitation:** each architecture was trained only once. Small gaps of 1–3 points could change on a re-run. The dashboard shows a bootstrap 95% interval from each test confusion matrix (about ±0.01 for macro-F1).

### 10.2 Uncertainty: selective prediction with MC Dropout

The best model was run 20 times per test building with dropout left on. Buildings are ranked by uncertainty, and the most uncertain ones are sent to a human (from `runs/siamese_fm_focal/mc_test.json`):

| Coverage (kept by model) | Accuracy | Macro F1 | Buildings sent to human review |
| --- | --- | --- | --- |
| 100% | 0.738 | 0.631 | 0 |
| 95% | 0.757 | 0.654 | 791 |
| 90% | 0.773 | 0.666 | 1,581 |
| 80% | 0.801 | 0.680 | 3,162 |
| 70% | 0.829 | 0.690 | 4,744 |
| 50% | 0.886 | 0.700 | 7,906 |

Accuracy goes up steadily as the most uncertain buildings are removed. This shows that the stored uncertainty value is meaningful: the model's least-sure predictions really are the ones most likely to be wrong.

### 10.3 Per-disaster test accuracy (best model)

| Disaster | Test buildings | Accuracy |
| --- | --- | --- |
| mexico-earthquake | 3,285 | 0.988 |
| midwest-flooding | 1,313 | 0.944 |
| guatemala-volcano | 168 | 0.935 |
| hurricane-florence | 1,002 | 0.872 |
| socal-fire | 88 | 0.875 |
| hurricane-harvey | 3,403 | 0.713 |
| hurricane-michael | 3,690 | 0.633 |
| hurricane-matthew | 2,862 | 0.461 |

The overall score hides large differences between disasters. Disasters made up mostly of undamaged buildings are easy. Hurricanes with mixed damage are much harder: hurricane-matthew is below 50%. (santa-rosa-wildfire has only 1 test building and palu-tsunami has none, so they are left out.)

### 10.4 Hotspots

DBSCAN on the predicted major/destroyed buildings found **37 hotspots containing 1,288 severe buildings**. These are shown on the dashboard map and listed per disaster.

### 10.5 Does higher resolution help? (128 px vs 192 px, same architecture)

The final architecture was retrained on 192×192 crops. The building data and split were the same. Because of time limits, it ran for 12 epochs instead of 15, with a batch size of 64.

| Run | Crop | Epochs | Val macro F1 | Test macro F1 | Test accuracy | Recall (destroyed) | Precision (destroyed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| siamese_fm_focal | 128 px | 15 | **0.647** | **0.632** | **0.739** | 0.661 | **0.514** |
| siamese_fm_focal_192 | 192 px | 12 | 0.643 | 0.616 | 0.731 | 0.663 | 0.453 |

**Higher resolution did not help in this setting.** The 192 px run is slightly worse on validation, so selection by validation keeps the 128 px model as final. Recall on destroyed buildings is unchanged, and destroyed precision is lower. Part of the gap may come from the shorter training (12 vs 15 epochs). The 192 px run's uncertainty ranking still works: accuracy rises from 0.731 at full coverage to 0.889 at 50% coverage.

### 10.6 The database as part of the AI system (SQL layer)

`src/aftershock/sqlai.py` adds an "AI inside the database" layer on top of the base schema. The live dashboard tab *Database & SQL* runs it read-only.

- **Model registry** `model_runs`, with CHECK constraints. The view `v_model_leaderboard` ranks the runs with `RANK() OVER (ORDER BY val_macro_f1 DESC)`. Only runs evaluated on this database are registered. The earlier `post_baseline` run used a different, 3-disaster database and test split, so it is excluded.
- **Trigger** `trg_queue_uncertain` (AFTER INSERT ON predictions). Every prediction with uncertainty ≥ 0.5 or confidence < 0.6 is automatically added to `review_queue`, which holds 17,251 rows for the two models with stored predictions.
- **Probabilistic views.**
  - `v_disaster_damage` puts the *expected* number of destroyed buildings (the sum of P(destroyed)) next to the hard count.
  - `v_calibration` gives a reliability table.
  - `v_uncertainty_rank` uses `ROW_NUMBER()` to compute selective accuracy in SQL. It reproduces the Python results in 10.2 (0.886 / 0.801 / 0.738 at 50 / 80 / 100% coverage).
- **Threshold query.** For the final model, 104 test buildings have P(destroyed) ≥ 0.8, and 97 of them really are destroyed (precision 0.933). The model's plain arg-max "destroyed" call has a precision of only 0.514. A confidence threshold in SQL makes the "destroyed" list much more reliable.
- **Index vs full scan.** `SELECT COUNT(*) FROM buildings WHERE split='test'` uses `buildings_split_idx` (a covering index search) and takes about 0.3 ms. With `NOT INDEXED` it falls back to a full table scan of about 60 ms, roughly 190× slower. Timings were measured on the training laptop.
- **In-database inference.** SQL functions `ai_predict()` and `ai_confidence()` run the final network on the PNG bytes stored in `crops`, directly inside a `SELECT`.

## 11. Expected Outcomes and Deliverables

- A trained Siamese damage-classification model and a comparison against the baseline.
- A PostGIS database of buildings, predictions, probabilities and uncertainty.
- Grad-CAM explanations and a human-review list of low-confidence buildings.
- Hotspot maps and rescue-priority queries.
- An interactive dashboard (before/after slider, coloured damage map, click-to-explain).
- Source code on GitHub, this report and a presentation.

## 12. Timeline

| Week | Work |
| --- | --- |
| 1 | Dataset subset, preprocessing, PostGIS database, baseline model |
| 2 | Siamese model, class-imbalance experiments, evaluation |
| 3 | Uncertainty, Grad-CAM, hotspots, dashboard, report and presentation |

## 13. Limitations and Ethics

- Results are on 10 xBD disasters. They may not generalise to unseen disaster types or regions, and per-disaster accuracy varies widely (46%–99%, Section 10.3).
- One training run per architecture, so small differences between models are not statistically certain.
- The models see 128×128 crops of imagery at about 0.5–0.8 m/pixel, so a house is only about 25 pixels wide. A 192 px retrain of the final architecture did not improve results (Section 10.5), but it ran for 12 epochs instead of 15, so this comparison is not fully controlled.
- Satellite resolution limits what can be seen: interior damage is invisible from above.
- The system supports human decision-makers; it does not replace field assessment. Low-confidence predictions are explicitly sent for human review.
- Only public, research-licensed data is used; no personal data is involved.

## 14. References

1. R. Gupta et al., "Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery," CVPR Workshops, 2019.
2. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR, 2016.
3. R. C. Daudt, B. Le Saux, A. Boulch, "Fully Convolutional Siamese Networks for Change Detection," ICIP, 2018.
4. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.
5. Y. Gal, Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," ICML, 2016.
6. T.-Y. Lin et al., "Focal Loss for Dense Object Detection," ICCV, 2017.
7. M. Ester et al., "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise," KDD, 1996.
8. M. T. Ribeiro, S. Singh, C. Guestrin, ""Why Should I Trust You?": Explaining the Predictions of Any Classifier," KDD, 2016.
