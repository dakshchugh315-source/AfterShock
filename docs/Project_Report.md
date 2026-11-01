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

**Scope for this project:** a subset of 2–3 disasters, to keep training feasible on free GPUs (Google Colab / Kaggle). The subset is fixed before training and documented in the final report.

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
| Compute / code | Google Colab or Kaggle GPU, Git + GitHub |

## 10. Evaluation

| Metric | Purpose |
| --- | --- |
| Per-class precision, recall, F1 | Damage classification quality per class |
| Macro F1 (harmonic mean over classes) | Overall score that does not hide rare classes |
| Confusion matrix | Which damage levels are confused |
| Recall on "destroyed" | Safety-critical: missed collapses |
| Selective accuracy vs coverage | Accuracy after routing uncertain buildings to humans |
| Inference time per km² | Practical speed |

**Results (to be filled after experiments):**

| Model | Macro F1 | Recall (destroyed) | Notes |
| --- | --- | --- | --- |
| Baseline ResNet-18 (post only) | [ ] | [ ] | |
| Siamese CNN | [ ] | [ ] | |
| Siamese CNN + focal loss | [ ] | [ ] | |

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

- Results on a 2–3 disaster subset may not generalise to unseen disaster types or regions.
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
