"""Generate a synthetic surface-inspection dataset (offline, no downloads).

"Normal" images are a consistent woven/metal-like texture with small random
jitter. "Defect" images are the same texture with an injected scratch, blob, or
hole, plus a pixel ground-truth mask. Output mirrors the MVTec AD layout so the
real benchmark can be dropped in unchanged.

    python data/make_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic"
SIZE = 256


def normal_image(rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    phase = rng.uniform(0, 2 * np.pi)
    weave = (np.sin(xx * 0.35 + phase) + np.sin(yy * 0.35 + phase))
    weave = (weave - weave.min()) / (weave.max() - weave.min())
    noise = rng.normal(0, 0.04, (SIZE, SIZE))
    bright = rng.uniform(-0.04, 0.04)
    val = np.clip(0.55 + 0.18 * weave + noise + bright, 0, 1)
    rgb = np.stack([val * 0.88, val * 0.94, val], axis=-1)  # cool gray-blue
    return (rgb * 255).astype(np.uint8)


def defect_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    img = Image.fromarray(normal_image(rng))
    draw = ImageDraw.Draw(img)
    mask = Image.new("L", (SIZE, SIZE), 0)
    mdraw = ImageDraw.Draw(mask)

    kind = int(rng.integers(0, 3))
    if kind == 0:  # scratch
        p = rng.integers(30, SIZE - 30, size=4)
        width = int(rng.integers(2, 5))
        color = tuple(int(c) for c in rng.integers(0, 50, 3))
        draw.line([p[0], p[1], p[2], p[3]], fill=color, width=width)
        mdraw.line([p[0], p[1], p[2], p[3]], fill=255, width=width)
    elif kind == 1:  # bright blob (contamination)
        cx, cy = rng.integers(45, SIZE - 45, size=2)
        r = int(rng.integers(10, 22))
        color = tuple(int(c) for c in rng.integers(200, 256, 3))
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=color)
        mdraw.ellipse(box, fill=255)
    else:  # dark hole / dent
        cx, cy = rng.integers(45, SIZE - 45, size=2)
        r = int(rng.integers(12, 24))
        box = [cx - r, cy - r, cx + r, cy + r]
        draw.ellipse(box, fill=(20, 20, 25))
        mdraw.ellipse(box, fill=255)

    return np.array(img), np.array(mask)


def _save(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def build(n_train: int = 60, n_test_good: int = 20, n_test_defect: int = 30) -> None:
    rng = np.random.default_rng(42)
    for i in range(n_train):
        _save(normal_image(rng), OUT / "train" / "good" / f"{i:03d}.png")
    for i in range(n_test_good):
        _save(normal_image(rng), OUT / "test" / "good" / f"{i:03d}.png")
    for i in range(n_test_defect):
        img, mask = defect_image(rng)
        _save(img, OUT / "test" / "defect" / f"{i:03d}.png")
        _save(mask, OUT / "ground_truth" / "defect" / f"{i:03d}_mask.png")
    print(f"Built synthetic dataset at {OUT}")
    print(f"  train/good={n_train}  test/good={n_test_good}  test/defect={n_test_defect}")


if __name__ == "__main__":
    sys.exit(build())
