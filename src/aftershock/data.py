"""PyTorch dataset that streams pre/post crops straight from the database."""
import io
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import IMAGENET_MEAN, IMAGENET_STD
from .db import DB

_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def png_to_tensor(png: bytes, size: int) -> torch.Tensor:
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - _MEAN) / _STD


def denormalize(t: torch.Tensor) -> np.ndarray:
    """Normalised CHW tensor -> HWC uint8 image (for visualisation)."""
    img = (t.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


class CropPairDataset(Dataset):
    """Items: (pre, post, label). Each worker process opens its own DB connection."""

    def __init__(self, db_url: str, rows, size: int = 128, augment: bool = False):
        self.db_url, self.rows, self.size, self.augment = db_url, rows, size, augment
        self._db, self._pid = None, None

    def __len__(self):
        return len(self.rows)

    def _conn(self) -> DB:
        if self._db is None or self._pid != os.getpid():
            self._db, self._pid = DB(self.db_url), os.getpid()
        return self._db

    def __getitem__(self, i):
        uid, label = self.rows[i]
        pre_png, post_png = self._conn().crop_bytes(uid)
        pre, post = png_to_tensor(pre_png, self.size), png_to_tensor(post_png, self.size)
        if self.augment:  # same geometric transform on both, so the change between them is preserved
            if torch.rand(1) < 0.5:
                pre, post = pre.flip(-1), post.flip(-1)
            if torch.rand(1) < 0.5:
                pre, post = pre.flip(-2), post.flip(-2)
            k = int(torch.randint(0, 4, (1,)))
            pre, post = pre.rot90(k, (1, 2)), post.rot90(k, (1, 2))
            gain = 1 + 0.2 * (torch.rand(1) - 0.5)  # brightness jitter on the post image only
            post = post * gain
        return pre, post, int(label)


def class_weights(labels, n_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency weights (normalised to mean 1) for the imbalanced damage classes."""
    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
