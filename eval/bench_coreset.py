"""Coreset efficiency benchmark: memory-bank size vs AUROC vs inference speed.

Compares random subsampling against k-center-greedy selection across a range of
memory-bank sizes, reusing precomputed features so the sweep is cheap. Saves a
trade-off plot to assets/coreset_tradeoff.png.

    python eval/bench_coreset.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image  # noqa: E402

from src import dataset  # noqa: E402
from src.coreset import greedy_coreset  # noqa: E402
from src.feature_extractor import FeatureExtractor  # noqa: E402

DATA = ROOT / "data" / "synthetic"
MASK_SIZE = 256
SIZES = [25, 50, 100, 250, 500, 1000, 2000]


def _patches(extractor, image):
    emb = extractor.embed(image)
    c, h, w = emb.shape
    return emb.reshape(c, h * w).T.numpy(), (h, w)


def main() -> None:
    extractor = FeatureExtractor()
    train = dataset.list_train_good(DATA)
    test = dataset.list_test(DATA)
    if not train or not test:
        raise SystemExit("Run `python data/make_synthetic.py` first.")

    print(f"Extracting features (train={len(train)}, test={len(test)})...")
    grid = None
    bank = []
    for p in train:
        feats, grid = _patches(extractor, dataset.load_image(p))
        bank.append(feats)
    bank = np.concatenate(bank, axis=0)

    test_feats, labels, masks = [], [], []
    for s in test:
        feats, grid = _patches(extractor, dataset.load_image(s.path))
        test_feats.append(feats)
        labels.append(s.label)
        if s.mask_path is not None:
            m = np.array(Image.open(s.mask_path).convert("L").resize((MASK_SIZE, MASK_SIZE)))
            masks.append((len(test_feats) - 1, (m.ravel() > 127).astype(int)))

    # Nested orderings: prefixes give every smaller coreset for free.
    max_size = SIZES[-1]
    rng = np.random.default_rng(0)
    orders = {
        "random": rng.permutation(len(bank))[:max_size],
        "greedy": greedy_coreset(bank, max_size, seed=0),
    }

    def evaluate(sub_bank):
        nn = NearestNeighbors(n_neighbors=1).fit(sub_bank)
        scores, amaps = [], {}
        for i, feats in enumerate(test_feats):
            dist, _ = nn.kneighbors(feats)
            amap = dist.reshape(grid)
            scores.append(float(amap.max()))
            amaps[i] = amap
        # Stable latency: one warmup, then median of repeated full-set passes.
        nn.kneighbors(test_feats[0])  # warmup
        timings = []
        for _ in range(3):
            t0 = time.time()
            for feats in test_feats:
                nn.kneighbors(feats)
            timings.append((time.time() - t0) / len(test_feats) * 1000)
        infer_ms = float(np.median(timings))
        img_auroc = roc_auc_score(labels, scores)
        # pixel AUROC over masked images (upsample small grid map to mask size)
        import torch
        import torch.nn.functional as F
        pv, pl = [], []
        for i, mlabel in masks:
            t = torch.tensor(amaps[i], dtype=torch.float32)[None, None]
            up = F.interpolate(t, size=(MASK_SIZE, MASK_SIZE), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
            pv.append(up.ravel())
            pl.append(mlabel)
        pix_auroc = roc_auc_score(np.concatenate(pl), np.concatenate(pv))
        return img_auroc, pix_auroc, infer_ms

    results = {m: {"img": [], "pix": [], "ms": []} for m in orders}
    print(f"\nFull bank: {len(bank)} patches\n")
    print(f"{'method':>7} {'size':>6} {'imgAUROC':>9} {'pixAUROC':>9} {'infer/img':>10}")
    for method, order in orders.items():
        for k in SIZES:
            img, pix, ms = evaluate(bank[order[:k]])
            results[method]["img"].append(img)
            results[method]["pix"].append(pix)
            results[method]["ms"].append(ms)
            print(f"{method:>7} {k:>6} {img:>9.3f} {pix:>9.3f} {ms:>9.2f}ms")

    # Two panels: accuracy stays flat while latency grows with bank size.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for method in orders:
        ax1.plot(SIZES, results[method]["img"], marker="o", label=f"{method} image")
        ax1.plot(SIZES, results[method]["pix"], marker="s", linestyle="--",
                 label=f"{method} pixel")
        ax2.plot(SIZES, results[method]["ms"], marker="o", label=method)
    ax1.set_xscale("log")
    ax1.set_xlabel("memory-bank size (patches, log)")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.9, 1.005)
    ax1.set_title("Accuracy is preserved when compressing the bank")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xscale("log")
    ax2.set_xlabel("memory-bank size (patches, log)")
    ax2.set_ylabel("inference time / image (ms)")
    ax2.set_title("Smaller bank → faster inference")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Coreset trade-off (full bank = {len(bank)} patches)")
    fig.tight_layout()
    out = ROOT / "assets" / "coreset_tradeoff.png"
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
