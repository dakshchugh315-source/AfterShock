"""Train the damage classifier on crops streamed from the database.

python -m aftershock.train --arch siamese --epochs 15 --loss focal --name siamese_focal
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DB_URL, RUNS_DIR
from .data import CropPairDataset, class_weights
from .db import DB
from .metrics import damage_metrics
from .model import FocalLoss, build_model


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for pre, post, y in loader:
        ps.append(model(pre.to(device), post.to(device)).argmax(1).cpu())
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return damage_metrics(y, p)


def train(db_url=DB_URL, name="run", arch="siamese", epochs=10, batch_size=64, lr=3e-4, loss="focal",
          pretrained=True, img_size=128, dropout=0.3, num_workers=2, limit=None, seed=42, runs_dir=None,
          max_per_class=None, log=print) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path(runs_dir or RUNS_DIR) / name
    out.mkdir(parents=True, exist_ok=True)
    db, device = DB(db_url), pick_device()
    rows = {s: db.split_rows(s, limit) for s in ("train", "val", "test")}
    if max_per_class:  # cap each damage class in TRAIN only (compute budget); val/test stay complete
        rng = random.Random(seed)
        capped = []
        for c in range(4):
            members = [r for r in rows["train"] if r[1] == c]
            capped += rng.sample(members, min(max_per_class, len(members)))
        rows["train"] = sorted(capped)
    log(f"device={device} train={len(rows['train'])} val={len(rows['val'])} test={len(rows['test'])}")
    if not rows["train"] or not rows["val"]:
        raise RuntimeError("Empty train/val split: run aftershock.ingest first")

    def loader(split, shuffle):
        ds = CropPairDataset(db_url, rows[split], img_size, augment=shuffle)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=device == "cuda", persistent_workers=num_workers > 0)

    train_dl, val_dl, test_dl = loader("train", True), loader("val", False), loader("test", False)
    weights = class_weights([y for _, y in rows["train"]]).to(device)
    criterion = {"ce": torch.nn.CrossEntropyLoss(), "weighted": torch.nn.CrossEntropyLoss(weight=weights),
                 "focal": FocalLoss(weight=weights)}[loss].to(device)
    model = build_model(arch, pretrained, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best, history = -1.0, []
    meta = {"arch": arch, "dropout": dropout, "img_size": img_size, "model_version": name, "loss": loss}
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for pre, post, y in train_dl:
            pre, post, y = pre.to(device), post.to(device), y.to(device)
            opt.zero_grad()
            batch_loss = criterion(model(pre, post), y)
            batch_loss.backward()
            opt.step()
            total, n = total + batch_loss.item() * len(y), n + len(y)
        sched.step()
        val = evaluate(model, val_dl, device)
        history.append({"epoch": ep, "train_loss": total / n, "val_macro_f1": val["macro_f1"],
                        "val_harmonic_f1": val["harmonic_f1"], "val_accuracy": val["accuracy"], "seconds": round(time.time() - t0, 1)})
        log(f"epoch {ep}/{epochs} loss={total / n:.4f} val_macroF1={val['macro_f1']:.4f} val_acc={val['accuracy']:.4f}")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({**meta, "state_dict": model.state_dict(), "epoch": ep, "val": val}, out / "best.pt")

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader(); w.writerows(history)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["state_dict"])
    test = evaluate(model, test_dl, device) if rows["test"] else {}
    result = {**meta, "best_val_macro_f1": best, "test": test, "epochs": epochs, "train_size": len(rows["train"]),
              "max_per_class": max_per_class, "pretrained": pretrained}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    log(f"best val macro-F1={best:.4f}  test macro-F1={test.get('macro_f1', float('nan')):.4f}  -> {out}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--name", default="run")
    ap.add_argument("--arch", choices=["siamese", "post"], default="siamese")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", choices=["ce", "weighted", "focal"], default="focal")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--limit", type=int, help="cap rows per split (quick experiments)")
    ap.add_argument("--max-per-class", type=int, help="cap each class in the TRAIN split (balanced, smaller compute)")
    a = ap.parse_args(argv)
    train(a.db, a.name, a.arch, a.epochs, a.batch_size, a.lr, a.loss, not a.no_pretrained, a.img_size, a.dropout,
          a.num_workers, a.limit, max_per_class=a.max_per_class)


if __name__ == "__main__":
    main()
