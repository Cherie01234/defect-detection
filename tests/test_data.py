"""Lightweight tests (no torch / no model download) — these run in CI."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.make_synthetic import SIZE, defect_image, normal_image  # noqa: E402
from src import dataset  # noqa: E402


def test_normal_image_shape_and_range():
    img = normal_image(np.random.default_rng(0))
    assert img.shape == (SIZE, SIZE, 3)
    assert img.dtype == np.uint8


def test_defect_has_nonempty_binary_mask():
    img, mask = defect_image(np.random.default_rng(1))
    assert img.shape == (SIZE, SIZE, 3)
    assert mask.shape == (SIZE, SIZE)
    assert mask.max() == 255 and mask.min() == 0  # binary
    assert mask.sum() > 0                          # a defect was drawn


def test_dataset_layout_parsing(tmp_path):
    # Build a minimal MVTec-style tree and check the loader reads it.
    (tmp_path / "train" / "good").mkdir(parents=True)
    (tmp_path / "test" / "good").mkdir(parents=True)
    (tmp_path / "test" / "defect").mkdir(parents=True)
    (tmp_path / "ground_truth" / "defect").mkdir(parents=True)

    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "train" / "good" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "test" / "good" / "g.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "test" / "defect" / "d.png")
    Image.new("L", (8, 8)).save(tmp_path / "ground_truth" / "defect" / "d_mask.png")

    assert len(dataset.list_train_good(tmp_path)) == 1
    samples = dataset.list_test(tmp_path)
    assert {s.label for s in samples} == {0, 1}
    defect = next(s for s in samples if s.label == 1)
    assert defect.mask_path is not None and defect.mask_path.exists()
    good = next(s for s in samples if s.label == 0)
    assert good.mask_path is None
