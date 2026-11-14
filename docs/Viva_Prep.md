# AfterShock: Viva Preparation

Answer pattern: **direct answer → where it is in the code → a trade-off or limitation**.

## 0. Numbers to know by heart (from `runs/*/metrics.json`)

| | post_focal (baseline) | siamese_focal | **siamese_fm_focal (final)** |
|---|---|---|---|
| Val macro F1 | 0.609 | 0.635 | **0.647** |
| Test macro F1 | 0.600 | 0.611 | **0.632** |
| Test accuracy | 0.710 | 0.716 | **0.739** |
| Recall (destroyed) | **0.716** | 0.628 | 0.661 |
| Precision (destroyed) | 0.410 | 0.496 | **0.514** |

- Data: 10 xBD disasters, 1,715 tiles, 107,441 buildings (train 74,597 / val 17,032 / test 15,812). Only 3.4% are "destroyed".
- MC Dropout (20 passes): accuracy 73.8% at full coverage, 80.1% at 80% coverage, 88.6% at 50% coverage.
- 37 DBSCAN hotspots containing 1,288 severe buildings.
- Trained on an RTX 5070 Laptop GPU (8 GB), 128×128 crops, 15 epochs, focal loss.

## 1. Problem and data

**Q. What problem are you solving?**
After a disaster, responders need to know which buildings are damaged and where. Manual mapping from satellite images takes days. AfterShock classifies every building's damage from before/after images, gives a confidence, explains its decision and ranks hotspots.

**Q. Which dataset and why?**
xBD (xView2 challenge, Gupta et al. 2019): pre/post satellite pairs with building polygons labelled on the Joint Damage Scale (no damage, minor, major, destroyed) across 19 disasters. It is the standard public benchmark for this task. I used all 10 disasters in the train set I downloaded: 107,441 buildings.

**Q. Why does palu-tsunami have no test results?**
The split is per tile, by hash. All of palu-tsunami's tiles happened to fall in train. I kept the split method rather than hand-moving tiles, because hand-picking the test set would bias it.

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

**Q. Show me the database doing real work, not just storing files.** (Dashboard → *Database & SQL* tab)
- **Leaderboard view:** `RANK() OVER (ORDER BY val_macro_f1 DESC)` over the `model_runs` registry.
- **Trigger:** `trg_queue_uncertain` fires AFTER INSERT on `predictions`. Uncertain rows land in `review_queue` automatically: 17,251 rows.
- **Expected counts:** `SUM(p_destroyed)` gives the *expected* number of destroyed buildings, next to the hard `SUM(pred_idx = 3)`. That is aggregation over uncertain data.
- **Selective accuracy:** computed in SQL with `ROW_NUMBER()`. It matches the Python numbers: 0.886 / 0.801 / 0.738.
- **Threshold query:** `p_destroyed >= 0.8` returns 104 buildings, 97 of them truly destroyed (93% precision), versus 51% precision for the plain arg-max.
- **Index:** `buildings_split_idx` turns a full scan (about 60 ms) into a covering-index search (about 0.3 ms). `EXPLAIN QUERY PLAN` shows `SEARCH ... USING COVERING INDEX` vs `SCAN buildings`.
- **AI inside SQL:** `ai_predict(pre_png, post_png)` is a Python function registered with `sqlite3.create_function`. SQLite calls the network for each row.

**Q. Why is `post_baseline` not on the leaderboard?**
It was trained and tested on the older 3-disaster database, which has a different test split. Putting it in the same ranking would compare scores on different exams.

**Q. Is the dashboard safe to show publicly?**
The database path is never shown, and it comes only from an environment variable. The SQL tab opens SQLite with `mode=ro` and only runs fixed, parameterised queries; there is no free-text SQL. The public tunnel is temporary and is stopped after the demo.

## 3. Deep learning (topic 4)

**Q. Why a Siamese network?**
Damage is *change*. Both crops go through the same ResNet-18 encoder (shared weights), and the head sees `[pre, post, |post − pre|]` features. The post-only baseline cannot tell an old ruin from new damage.

**Q. What is the feature-map fusion Siamese (`SiameseFMNet`) and why is it better?**
The plain Siamese pools each image to one 512-number vector *before* comparing them, so it loses *where* the change happened. `SiameseFMNet` (`model.py`) runs the shared encoder up to layer3 on both crops, then compares the two spatial feature maps position by position (`[pre, post, |post − pre|]` → 1×1 conv). layer4 then looks for change patterns in that fused map. Result: test macro-F1 0.632 vs 0.611 (late fusion) vs 0.600 (post-only).

**Q. Your earlier small experiment showed post-only winning. Why the change?**
That run used 6k training buildings at 64px for 5 epochs. With 75k buildings at 128px, the Siamese models have enough data to learn a useful "change" signal. Small-scale results don't always transfer to full scale.

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
Take the gradient of the predicted class score with respect to the last convolutional feature maps, average it per channel to get weights, take the weighted sum of the maps, apply ReLU and upsample to the image size (`explain.gradcam`). Hot regions are what drove the decision. For the final `siamese_fm` model, the feature maps are the fused pre/post *change* map (`fused_forward`), so the heatmap shows where the model saw change. For the other models it uses the post branch.

**Q. Why does the dashboard show two kinds of images?**
"What the model sees" is the actual 128×128 input. "Context" is a wider crop from the original tile, with the building outlined in red, for the human reviewer. At about 0.5–0.8 m/pixel a house is only about 25 pixels wide, so the model input looks small and soft; that is the real input, not a display bug.

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

**Q. How did you choose the final model? Did you pick it by test score?**
No. I chose it by **validation** macro-F1 (0.647). The test set was used once, at the end, to report numbers. Choosing by test score would let the test set influence the choice, and the reported score would then be optimistic. (`train.py` saves the best epoch by validation; the dashboard picks the best model by validation.)

**Q. Your best model has worse destroyed recall than the baseline. Isn't that a problem?**
Yes, and I report it rather than hide it. The baseline finds more collapsed buildings (0.716 vs 0.661 recall) but raises more false alarms (precision 0.410 vs 0.514). If missing a collapse is all that matters, the baseline is the safer choice. If rescue teams are limited, the fusion model wastes fewer trips. MC-Dropout review routing reduces the cost of either choice.

**Q. Is 0.632 vs 0.611 a real difference?**
Each architecture was trained once, so I can't fully rule out luck. Bootstrapping each model's test confusion matrix gives a 95% interval of about ±0.01 on macro-F1. So the 3.2-point gap over the baseline is likely real, and the 2.1-point gap over late fusion is borderline. The honest fix is 3 training seeds per model.

**Q. Why is accuracy so different between disasters (46% to 99%)?**
mexico-earthquake and midwest-flooding are mostly undamaged buildings, which are easy. Hurricanes like matthew have lots of minor/major damage, and the line between those two is ambiguous even for human labellers. The per-disaster table is in the report (10.3) and on the dashboard.

**Q. Did higher resolution (192 px) help?**
No. Test macro-F1 went 0.632 → 0.616 and validation 0.647 → 0.643, so the 128 px model stays final (selection by validation). Destroyed recall stayed the same (0.661 → 0.663), and precision dropped (0.514 → 0.453). Caveat: the 192 px run had 12 epochs instead of 15, so it isn't a perfectly controlled comparison.

**Q. "+3.2 points" or "+5.3%"?**
Both are right, but they mean different things. 0.632 − 0.600 = **3.2 percentage points** (absolute). 3.2 / 60.0 = **5.3% relative**. Always say which one you mean.

## 8. Limitations and ethics

- 10 xBD disasters may not generalise to new regions or disaster types; per-disaster accuracy already varies from 46% to 99%.
- One training run per architecture (no repeated seeds).
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
| `app/dashboard.py` | Streamlit app: one tab per course topic, validation-based model choice, calibration, live probabilistic query, Grad-CAM |
| `scripts/start_demo.ps1` | Starts the dashboard on the full 10-disaster DB (`-Public` adds a temporary public link) |
