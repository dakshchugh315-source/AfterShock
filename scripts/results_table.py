"""Build the README/report results tables straight from runs/*/metrics.json and mc_test.json (no hand-typed numbers).

python scripts/results_table.py post_baseline siamese_focal
"""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def load(name, file):
    p = RUNS / name / file
    return json.loads(p.read_text()) if p.exists() else None


def main(names):
    print("| Model | Train setting | Test macro F1 | Harmonic F1 | Accuracy | Recall (destroyed) | Recall (major) |")
    print("|---|---|---|---|---|---|---|")
    for n in names:
        m = load(n, "metrics.json")
        if not m:
            continue
        t, pc = m["test"], m["test"]["per_class"]
        setting = f"{m['arch']}, {m['loss']} loss, {m['img_size']}px, {m['train_size']} train" + \
                  (f" (≤{m['max_per_class']}/class)" if m.get("max_per_class") else "") + f", {m['epochs']} ep"
        print(f"| {n} | {setting} | {t['macro_f1']:.3f} | {t['harmonic_f1']:.3f} | {t['accuracy']:.3f} | "
              f"{pc['destroyed']['recall']:.3f} | {pc['major-damage']['recall']:.3f} |")
    for n in names:
        mc = load(n, "mc_test.json")
        if mc:
            print(f"\n**{n}: selective prediction with MC Dropout ({mc['passes']} passes)**\n")
            print("| Coverage | Accuracy | Macro F1 | Sent to human review |")
            print("|---|---|---|---|")
            for s in mc["selective"]:
                print(f"| {s['coverage']:.0%} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['to_human']} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["post_baseline", "siamese_focal"])
