"""Dataset loading in MVTec-AD folder layout.

    <root>/train/good/*.png
    <root>/test/good/*.png
    <root>/test/<defect_type>/*.png
    <root>/ground_truth/<defect_type>/<name>_mask.png

This is exactly MVTec AD's structure, so swapping the synthetic data for a real
MVTec category is just pointing `root` at it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class TestSample:
    path: Path
    label: int            # 0 = good, 1 = defect
    mask_path: Path | None  # ground-truth mask for defects


def _images_in(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def list_train_good(root: str | Path) -> list[Path]:
    return _images_in(Path(root) / "train" / "good")


def list_test(root: str | Path) -> list[TestSample]:
    root = Path(root)
    test_dir = root / "test"
    gt_dir = root / "ground_truth"
    samples: list[TestSample] = []
    for sub in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        label = 0 if sub.name == "good" else 1
        for img in _images_in(sub):
            mask = None
            if label == 1:
                cand = gt_dir / sub.name / f"{img.stem}_mask.png"
                mask = cand if cand.exists() else None
            samples.append(TestSample(img, label, mask))
    return samples


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
