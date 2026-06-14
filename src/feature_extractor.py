"""Pretrained-CNN patch feature extractor.

We take intermediate feature maps from an ImageNet-pretrained ResNet (layer2 +
layer3), upsample layer3 to layer2's resolution and concatenate along the
channel axis. Each spatial location of the resulting map is one "patch
embedding". No training is involved — this is the locally-aware feature space
that PatchCore-style anomaly detection operates in.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FeatureExtractor:
    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.layers = layers
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._features: dict[str, torch.Tensor] = {}
        for name in layers:
            getattr(self.model, name).register_forward_hook(self._make_hook(name))

        self.transform = transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def _make_hook(self, name: str):
        def hook(_module, _inp, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        """Return a (C, H, W) tensor of concatenated patch embeddings."""
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        self._features = {}
        self.model(x)

        feats = [self._features[name] for name in self.layers]
        ref_hw = feats[0].shape[-2:]
        resized = [
            F.interpolate(f, size=ref_hw, mode="bilinear", align_corners=False)
            for f in feats
        ]
        emb = torch.cat(resized, dim=1)  # (1, C, H, W)
        return emb.squeeze(0).cpu()
