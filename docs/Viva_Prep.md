# AfterShock: Viva Preparation

Answer pattern: **direct answer → where it is in the code → a trade-off or limitation**.

## 1. Problem and data

**Q. What problem are you solving?**
After a disaster, responders need to know which buildings are damaged and where. Manual mapping from satellite images takes days. AfterShock classifies every building's damage from before/after images, gives a confidence, explains its decision and ranks hotspots.

**Q. Which dataset and why?**
xBD (xView2 challenge, Gupta et al. 2019): pre/post satellite pairs with building polygons labelled on the Joint Damage Scale (no damage, minor, major, destroyed) across 19 disasters. It is the standard public benchmark for this task. I used a 2–3 disaster subset because of GPU limits.

**Q. How did you split the data, and why not randomly by building?**
By tile, using a hash of the tile id (`ingest.assign_split`). Buildings in the same tile look alike (same lighting, same neighbourhood), so a random building split would leak information and inflate test scores.

**Q. What is the synthetic dataset?**
Drawn rectangles and rubble in xBD format, only for testing the pipeline end to end. No results are reported on it.

## 2. Database (topics 1 and data management)

**Q. Why a database instead of folders of images?**
Footprints, labels, crops, predictions and hotspots are linked by building id. SQL lets me filter by split, class and confidence, and PostGIS adds spatial queries (distance to a hospital) with a GiST index. The DataLoader streams crops from the `crops` table.

**Q. What makes it a probabilistic / uncertain database?**
Each building stores four class probabilities, a confidence and an uncertainty value instead of a single hard label. Queries then use confidence thresholds, for example *destroyed buildings within 2 km of a hospital with p ≥ 0.8*.

**Q. PostGIS functions you used?**
`ST_MakePoint`, `ST_SetSRID` (SRID 4326 = WGS84 lat/lon), `ST_GeomFromText` for footprints, `ST_DWithin` on `geography` for distances in metres, and a GiST index on the point geometry.

## 3. Deep learning (topic 4)

**Q. Why a Siamese network?**
Damage is *change*. Both crops go through the same ResNet-18 encoder (shared weights), and the head sees `[pre, post, |post − pre|]` features. The post-only baseline cannot tell an old ruin from new damage.

**Q. Why ResNet-18 and transfer learning?**
Small enough for a free GPU; ImageNet pre-training gives good edge and texture features, so it learns faster with limited data.

**Q. How did you handle class imbalance?**
Most buildings are undamaged. I compared plain cross-entropy, inverse-frequency weighted cross-entropy and focal loss (`model.FocalLoss`), which down-weights easy examples. I also report per-class metrics so the rare classes are visible.

**Q. Augmentation?**
Flips and 90° rotations applied identically to pre and post (otherwise the network would see fake "change"), plus brightness jitter on the post image to mimic different acquisition conditions.

## 4. Uncertainty (topic 1)

**Q. What is Monte Carlo Dropout?**
Dropout stays on at inference; I predict each building N times (default 20) and average the softmax outputs (Gal & Ghahramani, 2016). The spread approximates model uncertainty. BatchNorm stays in eval mode (`predict.enable_mc_dropout`).

**Q. How do you measure uncertainty?**
Predictive entropy of the mean probabilities, normalised to 0..1 by dividing by log(4).

**Q. How is it useful?**
Selective prediction: the model answers only its most certain buildings and sends the rest to human review. I report accuracy at 100/90/80% coverage (`metrics.selective_accuracy`).

## 5. Explainability (topic 5, XAI)

**Q. How does Grad-CAM work?**
Take the gradient of the predicted class score with respect to the last convolutional feature maps of the post branch, average it per channel to get weights, take the weighted sum of the maps, apply ReLU and upsample to the image size (`explain.gradcam`). Hot regions are what drove the decision.

**Q. Why Grad-CAM over SHAP for images?**
It is fast and works on convolutional features directly. LIME (superpixel perturbation) is available as a cross-check. SHAP on raw pixels is expensive and noisy.

**Q. Can explanations be wrong?**
Yes. They show what the model used, not the ground truth. If the heatmap highlights roads or shadows instead of the building, that reveals a shortcut the model learned.

## 6. Data mining (topic 3)

**Q. Why DBSCAN for hotspots?**
No need to fix the number of clusters, it finds arbitrary shapes, and it marks isolated buildings as noise. I use the haversine metric on latitude/longitude, with eps in metres divided by the Earth's radius.

**Q. How do you rank hotspots?**
By the number of major/destroyed buildings, with the mean confidence shown alongside.

## 7. Evaluation

**Q. Which metrics and why not just accuracy?**
Accuracy hides rare classes (predicting "no damage" everywhere scores high). I report per-class precision/recall/F1, macro F1, the harmonic mean of per-class F1 (xView2-style; one weak class drags it down), the confusion matrix, and **recall on "destroyed"** because missing a collapsed building is the costliest error.

**Q. Precision vs recall here?**
Recall on severe damage matters most (lives); precision matters for not wasting rescue teams. Uncertainty routing helps balance the two.

## 8. Limitations and ethics

- A subset of disasters may not generalise to new regions or disaster types.
- Satellites see roofs, not interiors; minor vs major damage is often ambiguous even for humans.
- The tool supports responders; uncertain cases go to people by design.
- Public research data only, no personal data.

## 9. Know your code (be ready to open these)

| File | Explain in one line |
|---|---|
| `ingest.py` | Parses xBD labels, crops each building, splits by tile, stores everything in the DB |
| `data.py` | Dataset that reads crops from the DB per item; paired augmentation; class weights |
| `model.py` | Siamese and post-only ResNet-18, focal loss, checkpoint loading |
| `train.py` | AdamW + cosine schedule, best model by validation macro F1, test evaluation |
| `predict.py` | MC Dropout, entropy, writes probabilities and uncertainty to the DB |
| `explain.py` | Grad-CAM and optional LIME, saves before/after/heatmap panels |
| `hotspots.py` | DBSCAN on severe predictions, ranked hotspots table |
