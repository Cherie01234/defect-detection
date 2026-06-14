"""Render anomaly maps as heatmap overlays."""
from __future__ import annotations

import matplotlib.cm as cm
import numpy as np
from PIL import Image


def _normalize(amap: np.ndarray, vmax: float | None = None) -> np.ndarray:
    a = amap - amap.min()
    denom = (vmax - amap.min()) if vmax is not None else a.max()
    return np.clip(a / (denom + 1e-8), 0, 1)


def heatmap(amap: np.ndarray, vmax: float | None = None) -> Image.Image:
    colored = cm.jet(_normalize(amap, vmax))[:, :, :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def overlay(image: Image.Image, amap: np.ndarray, alpha: float = 0.5,
            vmax: float | None = None) -> Image.Image:
    base = image.convert("RGB").resize((amap.shape[1], amap.shape[0]))
    heat = np.array(heatmap(amap, vmax)).astype(float)
    blended = np.array(base).astype(float) * (1 - alpha) + heat * alpha
    return Image.fromarray(blended.astype(np.uint8))


def triptych(image: Image.Image, amap: np.ndarray, vmax: float | None = None) -> Image.Image:
    """Side-by-side: original | heatmap | overlay."""
    size = (amap.shape[1], amap.shape[0])
    panels = [
        image.convert("RGB").resize(size),
        heatmap(amap, vmax),
        overlay(image, amap, vmax=vmax),
    ]
    w, h = size
    canvas = Image.new("RGB", (w * 3 + 16, h), "white")
    for i, p in enumerate(panels):
        canvas.paste(p, (i * (w + 8), 0))
    return canvas
