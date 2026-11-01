"""Monte Carlo Dropout predictions -> probabilities + uncertainty stored in the database.

python -m aftershock.predict --ckpt runs/siamese_focal/best.pt --split test --passes 20
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DB_URL
from .data import CropPairDataset
from .db import DB
from .metrics import damage_metrics, selective_accuracy
from .model import load_checkpoint
from .train import pick_device


def enable_mc_dropout(model):
    """Eval mode everywhere (BatchNorm frozen) but dropout kept active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, loader, device, passes=20):
    """Returns (mean probabilities [N,4], labels [N])."""
    enable_mc_dropout(model)
    probs, labels = [], []
    for pre, post, y in loader:
        pre, post = pre.to(device), post.to(device)
        p = torch.stack([torch.softmax(model(pre, post), 1) for _ in range(passes)]).mean(0)
        probs.append(p.cpu()); labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    """Entropy of the mean prediction, scaled to 0 (certain) .. 1 (uniform)."""
    return -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1) / math.log(probs.shape[1])


def predict(ckpt_path, db_url=DB_URL, split="test", passes=20, batch_size=128, num_workers=0, log=print) -> dict:
    device = pick_device()
    model, ckpt = load_checkpoint(ckpt_path, device)
    db = DB(db_url)
    rows = db.split_rows(split)
    loader = DataLoader(CropPairDataset(db_url, rows, ckpt["img_size"]), batch_size=batch_size, num_workers=num_workers)
    probs, y = mc_predict(model, loader, device, passes)
    pred, unc = probs.argmax(1), predictive_entropy(probs)
    version = ckpt["model_version"]
    db.upsert("predictions", ["uid", "model_version", "pred_idx", "p_no", "p_minor", "p_major", "p_destroyed", "confidence", "uncertainty"],
              [(uid, version, int(pred[i]), *map(float, probs[i]), float(probs[i].max()), float(unc[i])) for i, (uid, _) in enumerate(rows)])
    report = {"model_version": version, "split": split, "passes": passes, "n": len(rows),
              "metrics": damage_metrics(y, pred), "selective": selective_accuracy(y, pred, unc),
              "mean_uncertainty": float(unc.mean())}
    (Path(ckpt_path).parent / f"mc_{split}.json").write_text(json.dumps(report, indent=2))
    log(f"{len(rows)} predictions stored for {version}; macro-F1={report['metrics']['macro_f1']:.4f}")
    for s in report["selective"]:
        log(f"  coverage {s['coverage']:.0%}: accuracy {s['accuracy']:.4f} ({s['to_human']} buildings sent to human review)")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--split", default="test")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args(argv)
    predict(a.ckpt, a.db, a.split, a.passes, a.batch_size)


if __name__ == "__main__":
    main()
