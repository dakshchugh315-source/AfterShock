"""Explainability: Grad-CAM on the post-disaster branch (plus optional LIME cross-check).

python -m aftershock.explain --ckpt runs/siamese_focal/best.pt --most-uncertain 20
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DAMAGE_CLASSES, DB_URL
from .data import denormalize, png_to_tensor
from .db import DB
from .model import encoder_features, load_checkpoint


def gradcam(model, pre: torch.Tensor, post: torch.Tensor, target: int | None = None):
    """Grad-CAM (Selvaraju et al., 2017) w.r.t. the post image's last conv block.
    pre/post: [3,H,W] normalised tensors. Returns (cam HxW in 0..1, predicted class, probabilities)."""
    model.eval()
    pre, post = pre[None], post[None]
    if hasattr(model, "fused_forward"):      # siamese_fm: Grad-CAM on the fused (change) feature map
        act, logits = model.fused_forward(pre, post)
    else:
        with torch.no_grad():
            f_pre = encoder_features(model.encoder, pre)[1]
        act, f_post = encoder_features(model.encoder, post)
        logits = model.classify(f_pre, f_post)
    act.retain_grad()
    probs = torch.softmax(logits, 1)[0].detach()
    target = int(probs.argmax()) if target is None else target
    model.zero_grad()
    logits[0, target].backward()
    weights = act.grad.mean(dim=(2, 3), keepdim=True)               # importance of each feature map
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=post.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), target, probs.cpu().numpy()


def jet(x: np.ndarray) -> np.ndarray:
    """Small jet colormap (avoids a matplotlib dependency). x in 0..1 -> RGB uint8."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def explanation_panel(pre_t, post_t, cam, alpha=0.45) -> Image.Image:
    """[pre | post | post + heatmap] side by side."""
    pre_img, post_img = denormalize(pre_t), denormalize(post_t)
    overlay = (post_img * (1 - alpha) + jet(cam) * alpha).astype(np.uint8)
    return Image.fromarray(np.concatenate([pre_img, post_img, overlay], axis=1))


def lime_explain(model, pre_t, post_t, num_samples=300):
    """Optional LIME (Ribeiro et al., 2016): which superpixels of the post image drive the prediction."""
    try:
        from lime import lime_image
    except ImportError as e:
        raise RuntimeError("pip install lime  (optional dependency)") from e
    model.eval()

    def classify(batch_hwc):
        x = torch.from_numpy(batch_hwc.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        x = (x - torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)) / torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.no_grad():
            return torch.softmax(model(pre_t[None].expand(len(x), -1, -1, -1), x), 1).numpy()

    exp = lime_image.LimeImageExplainer().explain_instance(denormalize(post_t), classify, top_labels=1, num_samples=num_samples)
    _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    return mask


def explain_buildings(ckpt_path, uids, db_url=DB_URL, out_dir=None, log=print) -> list:
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    db = DB(db_url)
    out = Path(out_dir or Path(ckpt_path).parent / "explanations")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for uid in uids:
        pre_png, post_png = db.crop_bytes(uid)
        pre_t, post_t = png_to_tensor(pre_png, ckpt["img_size"]), png_to_tensor(post_png, ckpt["img_size"])
        cam, cls, probs = gradcam(model, pre_t, post_t)
        path = out / f"{uid}.png"
        explanation_panel(pre_t, post_t, cam).save(path)
        results.append({"uid": uid, "pred": DAMAGE_CLASSES[cls], "confidence": float(probs[cls]), "image": str(path)})
        log(f"{uid}: {DAMAGE_CLASSES[cls]} ({probs[cls]:.0%}) -> {path}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--db", default=DB_URL)
    ap.add_argument("--uid", nargs="*", default=[])
    ap.add_argument("--most-uncertain", type=int, default=0, help="explain the N most uncertain predictions")
    a = ap.parse_args(argv)
    uids = list(a.uid)
    if a.most_uncertain:
        version = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_version"]
        uids += [r[0] for r in DB(a.db).execute(
            "SELECT uid FROM predictions WHERE model_version = ? ORDER BY uncertainty DESC LIMIT ?", (version, a.most_uncertain)).fetchall()]
    explain_buildings(a.ckpt, uids, a.db)


if __name__ == "__main__":
    main()
