# Training on a free GPU (Google Colab or Kaggle)

The laptop is fine for the database, the dashboard and the synthetic smoke test. Train on xBD with a GPU.

## Google Colab

1. **Runtime → Change runtime type → T4 GPU.**
2. Upload a zip of your chosen xBD subset (`images/` + `labels/`) to Google Drive.
3. Run these cells:

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/<your-username>/AfterShock.git
%cd AfterShock
!pip install -q -e . shapely
!mkdir -p data/xbd && unzip -q /content/drive/MyDrive/xbd_subset.zip -d data/xbd
```

```python
# SQLite on Colab is fine for training; ingest once, then train both models
!python -m aftershock.ingest --xbd data/xbd/train --disasters hurricane-michael socal-fire
!python -m aftershock.train --name post_baseline --arch post --loss weighted --epochs 15 --batch-size 128
!python -m aftershock.train --name siamese_focal --arch siamese --loss focal --epochs 15 --batch-size 128
!python -m aftershock.predict --ckpt runs/siamese_focal/best.pt --passes 20
```

4. Save the results before the session ends:

```python
!cp -r runs data/aftershock.db /content/drive/MyDrive/aftershock_results/
```

5. Copy `runs/` and `data/aftershock.db` back into the project folder on the laptop, then run `streamlit run app/dashboard.py`.

## Tips

- Pick 2–3 disasters with enough *major* and *destroyed* buildings (check counts printed by `ingest`).
- Start with `--limit 2000 --epochs 2` to check everything runs before a long training.
- `--num-workers 2` on Colab; `0` on Windows if DataLoader workers cause issues.
- Record every run's `metrics.json` in the report's results table; never type numbers by hand.
