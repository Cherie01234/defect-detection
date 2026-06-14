"""PatchCore-style anomaly detector.

fit()     : build a memory bank of normal patch embeddings.
predict() : for a query image, score each patch by distance to its nearest
            normal patch -> anomaly map; the image score is the map's maximum.

There is no gradient training: "fitting" is just storing (a coreset of) normal
features. This runs comfortably on CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from .coreset import greedy_coreset
from .feature_extractor import FeatureExtractor


class AnomalyDetector:
    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        coreset_size: int = 30000,
        sampler: str = "random",  # "random" or "greedy" (k-center)
        seed: int = 0,
    ) -> None:
        self.extractor = extractor or FeatureExtractor()
        self.coreset_size = coreset_size
        self.sampler = sampler
        self.seed = seed
        self.nn: NearestNeighbors | None = None
        self.grid: tuple[int, int] | None = None
        self.threshold: float | None = None

    # --- training (memory bank) -------------------------------------------
    def _patches(self, image: Image.Image) -> np.ndarray:
        emb = self.extractor.embed(image)          # (C, H, W)
        c, h, w = emb.shape
        self.grid = (h, w)
        return emb.reshape(c, h * w).T.numpy()      # (H*W, C)

    def fit(self, images: list[Image.Image]) -> "AnomalyDetector":
        bank = np.concatenate([self._patches(img) for img in images], axis=0)
        if len(bank) > self.coreset_size:
            if self.sampler == "greedy":
                idx = greedy_coreset(bank, self.coreset_size, seed=self.seed)
            else:
                idx = np.random.default_rng(self.seed).choice(
                    len(bank), self.coreset_size, replace=False
                )
            bank = bank[idx]
        self.nn = NearestNeighbors(n_neighbors=1).fit(bank)
        return self

    # --- inference ---------------------------------------------------------
    def predict(self, image: Image.Image, out_size: int = 256) -> tuple[float, np.ndarray]:
        if self.nn is None or self.grid is None:
            raise RuntimeError("Detector is not fitted. Call fit() first.")
        patches = self._patches(image)
        dist, _ = self.nn.kneighbors(patches)       # (H*W, 1)
        amap = dist.reshape(self.grid)              # (H, W)
        score = float(amap.max())

        t = torch.tensor(amap, dtype=torch.float32)[None, None]
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        t = TF.gaussian_blur(t, kernel_size=33, sigma=4.0)
        return score, t[0, 0].numpy()

    def image_scores(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([self.predict(img)[0] for img in images])

    def calibrate(self, good_images: list[Image.Image], margin: float = 1.1) -> float:
        """Set a decision threshold from known-good images (held out from fit)."""
        scores = self.image_scores(good_images)
        self.threshold = float(scores.max() * margin)
        return self.threshold
