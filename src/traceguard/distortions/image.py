from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F


def _round_ste(value: torch.Tensor) -> torch.Tensor:
    return value + (value.round() - value).detach()


class DifferentiableDistortions(nn.Module):
    def __init__(
        self,
        probability: float = 0.5,
        jpeg_quality: tuple[int, int] = (55, 95),
        resize_scale: tuple[float, float] = (0.75, 1.0),
        crop_scale: tuple[float, float] = (0.85, 1.0),
    ) -> None:
        super().__init__()
        self.probability = probability
        self.jpeg_quality = jpeg_quality
        self.resize_scale = resize_scale
        self.crop_scale = crop_scale

    def _jpeg_approximation(self, images: torch.Tensor) -> torch.Tensor:
        quality = random.uniform(*self.jpeg_quality)
        levels = max(8.0, 2.0 + quality * 2.5)
        return _round_ste(images * levels) / levels

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        scale = random.uniform(*self.resize_scale)
        small = F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)

    def _crop(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        scale = random.uniform(*self.crop_scale)
        crop_h, crop_w = max(1, int(height * scale)), max(1, int(width * scale))
        top = random.randint(0, height - crop_h)
        left = random.randint(0, width - crop_w)
        cropped = images[..., top : top + crop_h, left : left + crop_w]
        return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.probability:
            return images
        operations = [self._jpeg_approximation, self._resize, self._crop]
        random.shuffle(operations)
        output = images
        for operation in operations[: random.randint(1, len(operations))]:
            output = operation(output)
        return output.clamp(0, 1)
