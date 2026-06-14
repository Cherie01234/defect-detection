"""Convolutional autoencoder anomaly detector (reconstruction-error based).

A second, *trained* paradigm to contrast with the (training-free) PatchCore
detector: the AE learns to reconstruct normal images; regions it cannot
reconstruct well (defects, unseen during training) yield high reconstruction
error, which we use as the anomaly map.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


class ConvAutoencoder(nn.Module):
    def __init__(self, ch: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.ReLU(True),        # 128 -> 64
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.ReLU(True),   # 64 -> 32
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.ReLU(True),  # 32 -> 16
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.ReLU(True),  # 16 -> 8
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, 2, 1), nn.ReLU(True),  # 8 -> 16
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.ReLU(True),  # 16 -> 32
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.ReLU(True),      # 32 -> 64
            nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Sigmoid(),           # 64 -> 128
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AEDetector:
    def __init__(
        self,
        image_size: int = 128,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 16,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        torch.manual_seed(seed)
        self.image_size = image_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = ConvAutoencoder().to(device)
        self.threshold: float | None = None
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _batch(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(im.convert("RGB")) for im in images])

    def fit(self, images: list[Image.Image], verbose: bool = False) -> "AEDetector":
        x = self._batch(images).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        n = len(x)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                batch = x[perm[i:i + self.batch_size]]
                opt.zero_grad()
                loss = F.mse_loss(self.model(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * len(batch)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1}/{self.epochs}  loss={total / n:.5f}")
        self.model.eval()
        return self

    @torch.no_grad()
    def predict(self, image: Image.Image, out_size: int = 256) -> tuple[float, np.ndarray]:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        recon = self.model(x)
        err = ((x - recon) ** 2).mean(dim=1, keepdim=True)  # (1,1,H,W)
        err = F.interpolate(err, size=(out_size, out_size), mode="bilinear", align_corners=False)
        err = TF.gaussian_blur(err, kernel_size=33, sigma=4.0)
        amap = err[0, 0].cpu().numpy()
        score = float(np.percentile(amap, 99))  # robust high-error summary
        return score, amap

    def image_scores(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([self.predict(im)[0] for im in images])

    def calibrate(self, good_images: list[Image.Image], margin: float = 1.1) -> float:
        self.threshold = float(self.image_scores(good_images).max() * margin)
        return self.threshold
