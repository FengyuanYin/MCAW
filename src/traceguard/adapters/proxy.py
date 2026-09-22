from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from traceguard.errors import DependencyError, WeightNotFoundError
from traceguard.types import ProxyOutput


class MockTalkingHeadProxy(nn.Module):
    """Small frozen feature proxy for CPU engineering checks."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        generator = torch.Generator().manual_seed(2025)
        for parameter in self.parameters():
            parameter.data.normal_(generator=generator, std=0.02)
            parameter.requires_grad_(False)

    def extract_reference_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)

    def forward(self, clean: torch.Tensor, protected: torch.Tensor) -> ProxyOutput:
        with torch.no_grad():
            clean_features = self.extract_reference_features(clean)
        protected_features = self.extract_reference_features(protected)
        distance = F.mse_loss(protected_features, clean_features)
        return ProxyOutput(-distance, clean_features, protected_features, {"feature_mse": float(distance.detach())})


class SilencerLatentProxy(nn.Module):
    """Single-GPU adaptation of Silencer's latent/reference nullification objective.

    The pinned Silencer implementation exposes this objective through `g_mode=latent`
    but constructs a training DataLoader and hard-codes CUDA during model creation.
    This adapter reuses the same frozen VAE reference feature idea without importing
    or editing that executable training wrapper.
    """

    def __init__(self, pretrained_root: Path, dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise DependencyError("Hallo proxy requires diffusers; install traceguard[hallo]") from exc
        candidates = [pretrained_root / "sd-vae-ft-mse", pretrained_root / "vae", pretrained_root]
        model_path = next((path for path in candidates if (path / "config.json").exists()), None)
        if model_path is None:
            raise WeightNotFoundError(
                f"Cannot find a Diffusers VAE under {pretrained_root}; expected sd-vae-ft-mse/config.json"
            )
        self.vae = AutoencoderKL.from_pretrained(str(model_path), local_files_only=True)
        self.vae.to(dtype=dtype)
        self.vae.eval()
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)

    def extract_reference_features(self, images: torch.Tensor) -> torch.Tensor:
        vae_input = images.mul(2).sub(1).to(dtype=self.vae.dtype)
        return self.vae.encode(vae_input).latent_dist.mean * 0.18215

    def forward(self, clean: torch.Tensor, protected: torch.Tensor) -> ProxyOutput:
        with torch.no_grad():
            clean_features = self.extract_reference_features(clean)
        protected_features = self.extract_reference_features(protected)
        distance = F.mse_loss(protected_features.float(), clean_features.float())
        return ProxyOutput(-distance, clean_features, protected_features, {"latent_mse": float(distance.detach())})
