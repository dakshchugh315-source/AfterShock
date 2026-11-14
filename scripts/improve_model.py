"""Improve accuracy WITHOUT retraining, using only the validation set to decide.

Three standard, explainable tricks on top of already-trained checkpoints:
  1. Test-time augmentation (TTA): predict the original pair plus its flipped/rotated copies
     (same transform on pre and post) and average the probabilities.
  2. Ensemble: average the probabilities of two or more models (e.g. post-only + Siamese-FM,
     which make different mistakes: post-only has higher destroyed recall).
  3. Class-bias calibration: add a per-class offset to the log-probabilities, chosen on the
     VALIDATION set to maximise macro-F1. This corrects the "no-damage" bias of an imbalanced
     training set (like moving a decision threshold).

Every candidate is scored on validation; the best one by validation macro-F1 is chosen and
only then evaluated on test (one look). Optionally the chosen predictions are stored in the
`predictions` table under a new model_version, with runs/<name>/metrics.json, so the
dashboard can select it like any other model (it picks the best by validation macro-F1).

python scripts/improve_model.py --ckpt runs/post_focal/best.pt runs/siamese_fm_focal/best.pt \
    --name ensemble_tuned --passes 20 --store
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aftershock.config import DB_URL, RUNS_DIR  # noqa: E402
from aftershock.data import CropPairDataset  # noqa: E402
from aftershock.db import DB  # noqa: E402
from aftershock.metrics import damage_metrics, selective_accuracy  # noqa: E402
from aftershock.model import load_checkpoint  # noqa: E402
from aftershock.predict import enable_mc_dropout, predictive_entropy  # noqa: E402
from aftershock.train import pick_device  # noqa: E402

# identity + 3 geometric copies; the same transform is applied to pre and post
TRANSFORMS = {"id": lambda t: t, "hflip": lambda t: t.flip(-1), "vflip": lambda t: t.flip(-2),
              "rot180": lambda t: t.rot90(2, (-2, -1))}


@torch.no_grad()
def probs_per_transform(ckpt_path, db_url, split, passes, batch_size, device):
    """MC-Dropout mean probabilities for each transform: {name: [N,4]}, labels [N], uids."""
    model, ckpt = load_checkpoint(ckpt_path, device)
    enable_mc_dropout(model)
    rows = DB(db_url).split_rows(split)
    loader = DataLoader(CropPairDataset(db_url, rows, ckpt["img_size"]), batch_size=batch_size, num_workers=0)
    out = {k: [] for k in TRANSFORMS}
    labels = []
    for pre, post, y in loader:
        pre, post = pre.to(device), post.to(device)
        for name, f in TRANSFORMS.items():
            a, b = f(pre), f(post)
            p = torch.stack([torch.softmax(model(a, b), 1) for _ in range(passes)]).mean(0)
            out[name].append(p.cpu())
        labels.append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}, torch.cat(labels).numpy(), [r[0] for r in rows]


def cached(ckpt, db_url, split, passes, batch_size, device, cache_dir):
    tag = Path(ckpt).parent.name
    f = cache_dir / f"probs_{tag}_{split}_p{passes}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in TRANSFORMS}, z["y"], list(z["uids"])
    probs, y, uids = probs_per_transform(ckpt, db_url, split, passes, batch_size, device)
    np.savez_compressed(f, y=y, uids=np.array(uids), **probs)
    return probs, y, uids


def apply_bias(p, b):
    z = np.log(np.clip(p, 1e-12, 1)) + b
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def macro_f1(y, p):
    return damage_metrics(y, p.argmax(1))["macro_f1"]


def fit_bias(p_val, y_val, grid=np.round(np.arange(-1.5, 1.51, 0.1), 2), rounds=3):
    """Coordinate search for per-class log-offsets (class 0 fixed at 0), maximising val macro-F1."""
    b = np.zeros(4)
    best = macro_f1(y_val, p_val)
    for _ in range(rounds):
        for c in (1, 2, 3):
            for v in grid:
                trial = b.copy(); trial[c] = v
                s = macro_f1(y_val, apply_bias(p_val, trial))
                if s > best + 1e-9:
                    best, b = s, trial
    return b, best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more trained checkpoints")
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--name", default="ensemble_tuned", help="model_version for the stored result")
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--store", action="store_true", help="write the chosen test predictions to the DB")
    a = ap.parse_args(argv)

    device = pick_device()
    out_dir = Path(RUNS_DIR) / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for ck in a.ckpt:
        for split in ("val", "test"):
            print(f"probabilities: {Path(ck).parent.name} / {split} ...", flush=True)
            data[(ck, split)] = cached(ck, a.db, split, a.passes, a.batch_size, device, out_dir)
    y_val, y_test = data[(a.ckpt[0], "val")][1], data[(a.ckpt[0], "test")][1]
    uids_test = data[(a.ckpt[0], "test")][2]

    def combo(models, tta, split):
        ps = []
        for ck in models:
            t = data[(ck, split)][0]
            ps.append(np.mean([t[k] for k in TRANSFORMS], 0) if tta else t["id"])
        return np.mean(ps, 0)

    # every candidate: which models, TTA on/off, bias on/off - all scored on VALIDATION only
    candidates = []
    model_sets = [[ck] for ck in a.ckpt] + ([a.ckpt] if len(a.ckpt) > 1 else [])
    for models, tta in product(model_sets, (False, True)):
        pv, pt = combo(models, tta, "val"), combo(models, tta, "test")
        name = "+".join(Path(m).parent.name for m in models) + (" +TTA" if tta else "")
        candidates.append({"name": name, "bias": np.zeros(4), "val": macro_f1(y_val, pv), "pv": pv, "pt": pt})
        b, vb = fit_bias(pv, y_val)
        candidates.append({"name": name + " +bias", "bias": b, "val": vb, "pv": pv, "pt": pt})

    print("\nvalidation macro-F1 (selection) | test macro-F1 (reported once, for transparency)")
    rows = []
    for c in candidates:
        pt = apply_bias(c["pt"], c["bias"])
        m = damage_metrics(y_test, pt.argmax(1))
        c["test_metrics"] = m
        rows.append({"candidate": c["name"], "val_macro_f1": round(c["val"], 4), "test_macro_f1": round(m["macro_f1"], 4),
                     "test_accuracy": round(m["accuracy"], 4),
                     "test_recall_destroyed": round(m["per_class"]["destroyed"]["recall"], 4),
                     "bias": [round(float(v), 2) for v in c["bias"]]})
        print(f"  {c['name']:<55} val {c['val']:.4f} | test {m['macro_f1']:.4f}  acc {m['accuracy']:.4f}  "
              f"rec(destroyed) {m['per_class']['destroyed']['recall']:.3f}")
    best = max(candidates, key=lambda c: c["val"])
    print(f"\nchosen by validation: {best['name']} (val {best['val']:.4f}, test {best['test_metrics']['macro_f1']:.4f})")
    (out_dir / "candidates.json").write_text(json.dumps(rows, indent=2))

    p_final = apply_bias(best["pt"], best["bias"])
    pred, unc = p_final.argmax(1), predictive_entropy(p_final)
    first = load_checkpoint(a.ckpt[0], "cpu")[1]
    metrics = {"arch": best["name"].replace(" +", " + ").replace("+TTA", "TTA").replace("+bias", "bias"),
               "candidates_tried": [Path(c).parent.name for c in a.ckpt],
               "chosen": best["name"], "bias": best["bias"].tolist(), "img_size": first["img_size"],
               "model_version": a.name, "loss": "n/a (no retraining)", "best_val_macro_f1": best["val"],
               "test": best["test_metrics"], "selective_test": selective_accuracy(y_test, pred, unc),
               "passes": a.passes, "note": "selected on validation macro-F1; test evaluated once"}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir / 'metrics.json'} and candidates.json")

    if a.store:
        DB(a.db).upsert("predictions", ["uid", "model_version", "pred_idx", "p_no", "p_minor", "p_major", "p_destroyed",
                                        "confidence", "uncertainty"],
                        [(u, a.name, int(pred[i]), *map(float, p_final[i]), float(p_final[i].max()), float(unc[i]))
                         for i, u in enumerate(uids_test)])
        print(f"stored {len(uids_test)} test predictions as model_version '{a.name}'")


if __name__ == "__main__":
    main()
