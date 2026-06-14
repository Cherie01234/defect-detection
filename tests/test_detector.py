"""Integration test for the detector.

Needs torch/torchvision and downloads ResNet18 weights on first run, so it is
skipped unless RUN_HEAVY=1. Run locally with:

    RUN_HEAVY=1 pytest tests/test_detector.py        # bash
    $env:RUN_HEAVY=1; pytest tests/test_detector.py  # PowerShell
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_HEAVY") != "1", reason="set RUN_HEAVY=1 to run model tests"
)


def test_defect_scores_higher_than_normal():
    from data.make_synthetic import defect_image, normal_image
    from src.detector import AnomalyDetector

    rng = np.random.default_rng(0)
    train = [Image.fromarray(normal_image(rng)) for _ in range(8)]
    detector = AnomalyDetector().fit(train)

    good_score, amap = detector.predict(Image.fromarray(normal_image(rng)))
    defect_img, _ = defect_image(rng)
    defect_score, _ = detector.predict(Image.fromarray(defect_img))

    assert amap.shape == (256, 256)
    assert defect_score > good_score  # a defect must be more anomalous
