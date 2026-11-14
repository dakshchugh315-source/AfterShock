"""End-to-end smoke test on the SYNTHETIC sample: ingest -> train -> MC-Dropout predict -> Grad-CAM -> hotspots."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftershock.config import DAMAGE_CLASSES  # noqa: E402
from aftershock.db import DB, haversine_m  # noqa: E402
from aftershock.explain import explain_buildings, gradcam  # noqa: E402
from aftershock.hotspots import find_hotspots  # noqa: E402
from aftershock.ingest import assign_split, ingest  # noqa: E402
from aftershock.metrics import damage_metrics, selective_accuracy  # noqa: E402
from aftershock.model import FocalLoss, build_model  # noqa: E402
from aftershock.predict import predict, predictive_entropy  # noqa: E402
from aftershock.train import train  # noqa: E402
from aftershock.xbd import iter_tiles  # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "make_synthetic_sample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("aftershock")
    _load_generator().main(["--out", str(tmp / "xbd"), "--tiles", "16", "--size", "192", "--buildings", "16"])
    url = f"sqlite:///{(tmp / 'test.db').as_posix()}"
    summary = ingest(tmp / "xbd", DB(url), size=64, val_frac=0.25, test_frac=0.25, log=lambda *_: None)
    return {"tmp": tmp, "url": url, "summary": summary}


def test_xbd_parsing(world):
    tiles = list(iter_tiles(world["tmp"] / "xbd"))
    assert len(tiles) == 16
    b = tiles[0].buildings[0]
    assert b.label in DAMAGE_CLASSES and b.lon and b.lat and len(b.pixel_bounds) == 4


def test_ingest_splits_by_tile(world):
    s = world["summary"]
    assert s["tiles"] == 16 and s["train"] > 0 and s["train"] + s["val"] + s["test"] == sum(s["labels"].values())
    db = DB(world["url"])
    per_tile = db.execute("SELECT image_id, COUNT(DISTINCT split) FROM buildings GROUP BY image_id").fetchall()
    assert all(n == 1 for _, n in per_tile)                      # no tile is split across train/val/test
    assert assign_split("x", 0.15, 0.15) == assign_split("x", 0.15, 0.15)


def test_models_and_losses():
    pre, post = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    for arch in ("siamese", "siamese_fm", "post"):
        out = build_model(arch, pretrained=False)(pre, post)
        assert out.shape == (2, 4)
    loss = FocalLoss(torch.ones(4))(torch.randn(5, 4, requires_grad=True), torch.tensor([0, 1, 2, 3, 0]))
    loss.backward()
    assert loss.item() > 0


def test_metrics():
    m = damage_metrics([0, 1, 2, 3, 3], [0, 1, 2, 3, 2])
    assert m["per_class"]["destroyed"]["recall"] == 0.5 and 0 < m["harmonic_f1"] <= m["macro_f1"] <= 1
    sel = selective_accuracy([0, 1, 0, 1], [0, 1, 1, 0], uncertainty=[0.1, 0.2, 0.9, 0.8], coverages=(1.0, 0.5))
    assert sel[0]["accuracy"] == 0.5 and sel[1]["accuracy"] == 1.0 and sel[1]["to_human"] == 2
    p = np.array([[1.0, 0, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(predictive_entropy(p), [0, 1], atol=1e-6)


@pytest.fixture(scope="module")
def trained(world):
    result = train(world["url"], name="smoke", arch="siamese", epochs=2, batch_size=16, pretrained=False,
                   img_size=64, num_workers=0, runs_dir=world["tmp"] / "runs", log=lambda *_: None)
    return result, world["tmp"] / "runs" / "smoke" / "best.pt"


def test_train_writes_checkpoint_and_metrics(trained):
    result, ckpt = trained
    assert ckpt.exists() and (ckpt.parent / "metrics.json").exists() and (ckpt.parent / "history.csv").exists()
    assert set(result["test"]["per_class"]) == set(DAMAGE_CLASSES)


def test_mc_dropout_predictions_stored(world, trained):
    _, ckpt = trained
    report = predict(ckpt, world["url"], split="test", passes=4, batch_size=32, log=lambda *_: None)
    db = DB(world["url"])
    rows = db.execute("SELECT p_no + p_minor + p_major + p_destroyed, confidence, uncertainty FROM predictions WHERE model_version = 'smoke'").fetchall()
    assert len(rows) == report["n"] > 0
    assert all(abs(s - 1) < 1e-4 and 0 <= c <= 1 and 0 <= u <= 1 for s, c, u in rows)
    assert json.loads((ckpt.parent / "mc_test.json").read_text())["selective"][0]["coverage"] == 1.0


def test_gradcam_and_panel(world, trained):
    _, ckpt = trained
    uid = DB(world["url"]).execute("SELECT uid FROM predictions LIMIT 1").fetchone()[0]
    res = explain_buildings(ckpt, [uid], world["url"], out_dir=world["tmp"] / "cams", log=lambda *_: None)
    assert Path(res[0]["image"]).exists() and res[0]["pred"] in DAMAGE_CLASSES
    model = build_model("siamese", pretrained=False)
    cam, cls, probs = gradcam(model, torch.randn(3, 64, 64), torch.randn(3, 64, 64))
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 and abs(probs.sum() - 1) < 1e-5


def test_hotspots_and_spatial_query(world, trained):
    db = DB(world["url"])
    db.execute("UPDATE predictions SET pred_idx = 3, confidence = 0.9 WHERE model_version = 'smoke'")  # force severe for the test
    db.commit()
    hs = find_hotspots("smoke", world["url"], eps_m=60, min_samples=3, min_conf=0.5, log=lambda *_: None)
    assert hs and hs[0]["n_buildings"] >= hs[-1]["n_buildings"]
    lon, lat = hs[0]["lon"], hs[0]["lat"]
    near = db.buildings_near(lon, lat, 200, "smoke", pred_idx=3, min_conf=0.8)
    assert near and all(haversine_m(lon, lat, r[1], r[2]) <= 200 for r in near)


def test_sql_ai_layer(tmp_path):
    """Trigger queues uncertain predictions; views and SQL functions work."""
    from aftershock.db import DB
    from aftershock.sqlai import install, register_functions
    db = DB(f"sqlite:///{(tmp_path / 't.db').as_posix()}")
    db.init_schema()
    install(db)
    register_functions(db.conn)
    db.upsert("buildings", ["uid", "disaster", "split", "label_idx"], [("a", "d", "test", 3), ("b", "d", "test", 0)])
    db.upsert("predictions", ["uid", "model_version", "pred_idx", "p_no", "p_minor", "p_major", "p_destroyed", "confidence", "uncertainty"],
              [("a", "m", 3, 0.0, 0.0, 0.1, 0.9, 0.9, 0.2), ("b", "m", 1, 0.3, 0.4, 0.2, 0.1, 0.4, 0.9)])
    assert [r[0] for r in db.execute("SELECT uid FROM v_review_queue").fetchall()] == ["b"]
    assert db.execute("SELECT expected_destroyed, hard_destroyed FROM v_disaster_damage").fetchone() == (1.0, 1)
    db.execute("INSERT INTO human_reviews (uid, model_version, reviewer, label_idx) VALUES ('b', 'm', 'x', 0)")
    assert db.execute("SELECT COUNT(*) FROM v_review_queue").fetchone()[0] == 0
