"""Render README hero images: original | heatmap | overlay for a few defects.

    python assets/make_hero.py
    -> assets/hero.png (one striking example) and assets/examples.png (a grid)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import dataset, visualize  # noqa: E402
from src.detector import AnomalyDetector  # noqa: E402

DATA = ROOT / "data" / "synthetic"


def main() -> None:
    train = dataset.list_train_good(DATA)
    if not train:
        raise SystemExit("Run `python data/make_synthetic.py` first.")
    det = AnomalyDetector().fit([dataset.load_image(p) for p in train])

    defects = [s for s in dataset.list_test(DATA) if s.label == 1]
    scored = []
    for s in defects:
        img = dataset.load_image(s.path)
        score, amap = det.predict(img, out_size=256)
        scored.append((score, img, amap))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Hero: the clearest defect.
    _, img, amap = scored[len(scored) // 4]
    visualize.triptych(img, amap).save(ROOT / "assets" / "hero.png")

    # Examples grid: 3 different defects stacked vertically.
    picks = [scored[0], scored[len(scored) // 2], scored[-1]]
    panels = [visualize.triptych(im, am) for _, im, am in picks]
    w = panels[0].width
    gap = 10
    grid = Image.new("RGB", (w, sum(p.height for p in panels) + gap * 2), "white")
    y = 0
    for p in panels:
        grid.paste(p, (0, y))
        y += p.height + gap
    grid.save(ROOT / "assets" / "examples.png")
    print("Wrote assets/hero.png and assets/examples.png")


if __name__ == "__main__":
    main()
