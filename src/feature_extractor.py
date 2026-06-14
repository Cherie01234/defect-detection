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
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    resnet50,
    wide_resnet50_2,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# All ResNet-family backbones expose .layer2 / .layer3, so the hook logic is
# shared; only channel widths differ (richer features on the larger nets).
BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.IMAGENET1K_V1),
}


class FeatureExtractor:
    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}. Choose from {list(BACKBONES)}.")
        self.device = device
        self.backbone = backbone
        self.layers = layers
        ctor, weights = BACKBONES[backbone]
        self.model = ctor(weights=weights)
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
