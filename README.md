# 🛰️ AfterShock

**Satellite-based building damage assessment after disasters**: deep learning on pre/post satellite imagery, a spatial database of uncertain predictions, explainable AI and damage-hotspot discovery for rescue prioritisation.

> Course project · *Data Management and AI Integration* · Bennett University · Daksh Chugh

---

## Why

After an earthquake, flood or cyclone, responders first need to know **which buildings are damaged and where**. Analysts still mark damage by hand on satellite images, which takes days. AfterShock compares each building's *before* and *after* image, classifies its damage on the Joint Damage Scale (no damage, minor, major, destroyed), **says how sure it is**, **shows where it looked**, and **ranks the hardest-hit areas**. Low-confidence buildings go to a human instead of being guessed.

## What it does

| Step | How | Course topic |
|---|---|---|
| Store tiles, building footprints and image crops | PostgreSQL + **PostGIS** (SQLite for development) | Data management |
| Classify damage from pre/post crops streamed from the DB | **Siamese ResNet-18** in PyTorch vs a post-only baseline; focal loss for class imbalance | **4. Deep learning, large-scale** |
| Probabilities + uncertainty per building, queryable by confidence | **Monte Carlo Dropout** → `predictions` table; selective accuracy | **1. Probabilistic / uncertain DB** |
| Explain each decision | **Grad-CAM** heatmaps (LIME optional) | **5. Explainable AI** |
| Find and rank damage hotspots | **DBSCAN** (haversine) + spatial SQL (`ST_DWithin`) | **3. Data mining / KDD** |
| Present it | Streamlit + Folium map, human-review queue | — |

```
xBD images + labels → building crops (pre, post) → PostgreSQL/PostGIS
      → PyTorch DataLoader (streams crops from the DB) → Siamese CNN
      → MC Dropout → probabilities + uncertainty → predictions table
      → Grad-CAM explanations · DBSCAN hotspots · spatial queries → dashboard
```

## Dataset

[**xBD / xView2**](https://xview2.org/) (Gupta et al., 2019): pre/post-disaster satellite image pairs with building polygons labelled on the Joint Damage Scale, across 19 disasters. Download requires a free registration, so the data is **not** in this repository. This project uses a 2–3 disaster subset to fit free GPUs.

`scripts/make_synthetic_sample.py` creates a tiny **synthetic** dataset in xBD format (drawn rectangles and rubble) so the whole pipeline can be tested in minutes. It is for smoke tests only, never for reported results.

## Quick start (synthetic smoke test, CPU, ~2 minutes)

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pip install -e .

python scripts/make_synthetic_sample.py --out data/synthetic_xbd --tiles 16
python -m aftershock.ingest  --xbd data/synthetic_xbd --size 64
python -m aftershock.train   --name smoke --epochs 3 --img-size 64 --no-pretrained --num-workers 0
python -m aftershock.predict --ckpt runs/smoke/best.pt --passes 10
python -m aftershock.explain --ckpt runs/smoke/best.pt --most-uncertain 5
python -m aftershock.hotspots --model-version smoke --eps-m 60 --min-samples 3
streamlit run app/dashboard.py
pytest -q
```

## Real run on xBD

1. Download xBD (train set) from [xview2.org](https://xview2.org/) and unpack to `data/xbd/train` (`images/`, `labels/`).
2. *(Recommended)* PostgreSQL + PostGIS:
   ```bash
   createdb aftershock && psql -d aftershock -c "CREATE EXTENSION postgis;"
   set AFTERSHOCK_DB=postgresql://postgres:<password>@localhost:5432/aftershock   # export on Linux/macOS
   ```
3. Ingest a subset, then train (GPU: see [docs/Colab_Training.md](docs/Colab_Training.md)):
   ```bash
   python -m aftershock.ingest --xbd data/xbd/train --disasters hurricane-michael socal-fire
   python -m aftershock.train --name post_baseline --arch post --loss weighted --epochs 15
   python -m aftershock.train --name siamese_focal --arch siamese --loss focal --epochs 15
   python -m aftershock.predict --ckpt runs/siamese_focal/best.pt --passes 20
   python -m aftershock.explain --ckpt runs/siamese_focal/best.pt --most-uncertain 30
   python -m aftershock.hotspots --model-version siamese_focal
   ```

Example probabilistic + spatial query (PostGIS):

```sql
-- destroyed buildings within 2 km of a hospital, predicted with at least 80% confidence
SELECT b.uid, p.p_destroyed
FROM buildings b JOIN predictions p USING (uid) JOIN facilities f ON f.ftype = 'hospital'
WHERE p.model_version = 'siamese_focal' AND p.pred_idx = 3 AND p.p_destroyed >= 0.8
  AND ST_DWithin(b.geom::geography, ST_SetSRID(ST_MakePoint(f.lon, f.lat), 4326)::geography, 2000);
```

## Results

*To be filled from `runs/*/metrics.json` and `runs/*/mc_test.json` after training on xBD.*

| Model | Macro F1 | Harmonic F1 | Recall (destroyed) |
|---|---|---|---|
| Post-only ResNet-18 (weighted CE) | – | – | – |
| Siamese ResNet-18 (focal loss) | – | – | – |

| Coverage (rest sent to human review) | 100% | 90% | 80% |
|---|---|---|---|
| Accuracy (Siamese + MC Dropout) | – | – | – |

## Repository layout

```
src/aftershock/   config · db (PostGIS/SQLite) · xbd (labels, crops) · ingest · data (DB-streamed dataset)
                  model (Siamese, baseline, focal loss) · train · metrics · predict (MC Dropout)
                  explain (Grad-CAM, LIME) · hotspots (DBSCAN)
sql/              schema_postgis.sql
app/              dashboard.py (Streamlit + Folium)
scripts/          make_synthetic_sample.py
tests/            end-to-end pipeline tests (synthetic data)
docs/             project report, viva preparation, Colab training guide
```

## Limitations

Trained on a subset of disasters, so it may not generalise to unseen disaster types or regions. Satellites see roofs, not interiors. The tool supports responders and does not replace field assessment; uncertain buildings are routed to humans by design.

## References

1. R. Gupta et al., *Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery*, CVPR Workshops 2019.
2. K. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
3. R. C. Daudt, B. Le Saux, A. Boulch, *Fully Convolutional Siamese Networks for Change Detection*, ICIP 2018.
4. R. R. Selvaraju et al., *Grad-CAM*, ICCV 2017.
5. Y. Gal, Z. Ghahramani, *Dropout as a Bayesian Approximation*, ICML 2016.
6. T.-Y. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017.
7. M. Ester et al., *DBSCAN*, KDD 1996.
8. M. T. Ribeiro, S. Singh, C. Guestrin, *"Why Should I Trust You?" (LIME)*, KDD 2016.
