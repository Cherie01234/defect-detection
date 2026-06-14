"""Evaluate the detector: Image-level and Pixel-level AUROC.

    python eval/run_eval.py
    python eval/run_eval.py --root path/to/mvtec/bottle

Image AUROC : separates defect vs good images by their anomaly score.
Pixel AUROC : separates defect vs normal pixels by their anomaly value
              (only over images that have a ground-truth mask).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import dataset  # noqa: E402
from src.detector import AnomalyDetector  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "synthetic"
MASK_SIZE = 256


def evaluate(root: Path) -> dict:
    train_paths = dataset.list_train_good(root)
    test = dataset.list_test(root)
    if not train_paths or not test:
        raise SystemExit(f"No data under {root}. Run `python data/make_synthetic.py` first.")

    print(f"Fitting on {len(train_paths)} normal images...")
    t0 = time.time()
    detector = AnomalyDetector()
    detector.fit([dataset.load_image(p) for p in train_paths])

    img_scores, img_labels = [], []
    pixel_vals, pixel_labels = [], []
    for s in test:
        score, amap = detector.predict(dataset.load_image(s.path), out_size=MASK_SIZE)
        img_scores.append(score)
        img_labels.append(s.label)
        if s.mask_path is not None:
            mask = np.array(Image.open(s.mask_path).convert("L").resize((MASK_SIZE, MASK_SIZE)))
            pixel_vals.append(amap.ravel())
            pixel_labels.append((mask.ravel() > 127).astype(int))

    image_auroc = roc_auc_score(img_labels, img_scores)
    result = {"image_auroc": image_auroc, "n_test": len(test)}

    if pixel_labels:
        result["pixel_auroc"] = roc_auc_score(
            np.concatenate(pixel_labels), np.concatenate(pixel_vals)
        )

    elapsed = time.time() - t0
    print(f"\nImage-level AUROC : {image_auroc:.3f}")
    if "pixel_auroc" in result:
        print(f"Pixel-level AUROC : {result['pixel_auroc']:.3f}")
    print(f"({len(test)} test images, {elapsed:.1f}s on CPU)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    evaluate(Path(args.root))
