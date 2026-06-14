"""Compare two anomaly-detection paradigms on the same dataset.

  - PatchCore  : feature-distance to normal patch memory bank (training-free)
  - Autoencoder: reconstruction error from an AE trained on normal images

Reports Image/Pixel AUROC and timing, and saves a qualitative side-by-side
heatmap comparison to assets/method_comparison.png.

    python eval/compare_methods.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import dataset, visualize  # noqa: E402
from src.autoencoder import AEDetector  # noqa: E402
from src.detector import AnomalyDetector  # noqa: E402

DATA = ROOT / "data" / "synthetic"
MASK_SIZE = 256


def evaluate(detector, train_imgs, test) -> dict:
    t0 = time.time()
    detector.fit(train_imgs)
    fit_s = time.time() - t0

    scores, labels, pv, pl, amaps = [], [], [], [], {}
    t0 = time.time()
    for i, s in enumerate(test):
        score, amap = detector.predict(dataset.load_image(s.path), out_size=MASK_SIZE)
        scores.append(score)
        labels.append(s.label)
        amaps[i] = amap
        if s.mask_path is not None:
            m = np.array(Image.open(s.mask_path).convert("L").resize((MASK_SIZE, MASK_SIZE)))
            pv.append(amap.ravel())
            pl.append((m.ravel() > 127).astype(int))
    predict_s = time.time() - t0

    return {
        "image_auroc": roc_auc_score(labels, scores),
        "pixel_auroc": roc_auc_score(np.concatenate(pl), np.concatenate(pv)),
        "fit_s": fit_s,
        "predict_s": predict_s,
        "amaps": amaps,
    }


def main() -> None:
    train = [dataset.load_image(p) for p in dataset.list_train_good(DATA)]
    test = dataset.list_test(DATA)
    if not train or not test:
        raise SystemExit("Run `python data/make_synthetic.py` first.")

    print(f"PatchCore: fitting on {len(train)} normal images...")
    pc = evaluate(AnomalyDetector(), train, test)
    print("Autoencoder: training on normal images...")
    ae = evaluate(AEDetector(epochs=40), train, test)

    print(f"\n{'method':>12} {'imgAUROC':>9} {'pixAUROC':>9} {'fit':>8} {'predict':>9}")
    for name, r in [("PatchCore", pc), ("Autoencoder", ae)]:
        print(f"{name:>12} {r['image_auroc']:>9.3f} {r['pixel_auroc']:>9.3f} "
              f"{r['fit_s']:>7.1f}s {r['predict_s']:>8.1f}s")

    # Qualitative comparison on the clearest defect.
    defects = [(i, s) for i, s in enumerate(test) if s.label == 1]
    best_i = max(defects, key=lambda t: pc["amaps"][t[0]].max())[0]
    img = dataset.load_image(test[best_i].path)
    panels = [
        img.convert("RGB").resize((MASK_SIZE, MASK_SIZE)),
        visualize.overlay(img, pc["amaps"][best_i]),
        visualize.overlay(img, ae["amaps"][best_i]),
    ]
    gap = 8
    canvas = Image.new("RGB", (MASK_SIZE * 3 + gap * 2, MASK_SIZE), "white")
    for j, p in enumerate(panels):
        canvas.paste(p, (j * (MASK_SIZE + gap), 0))
    out = ROOT / "assets" / "method_comparison.png"
    canvas.save(out)
    print(f"\nSaved {out}  (left: input | center: PatchCore | right: Autoencoder)")


if __name__ == "__main__":
    main()
