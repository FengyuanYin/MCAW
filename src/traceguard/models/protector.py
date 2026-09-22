from __future__ import annotations

import torch
from torch import nn

from traceguard.types import ProtectionOutput


class MessageResidualAdapter(nn.Module):
    def __init__(self, nbits: int, channels: int, max_residual: float) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.image_stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.message = nn.Sequential(nn.Linear(nbits, channels * 2), nn.SiLU(), nn.Linear(channels * 2, channels * 2))
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        features = self.image_stem(images)
        scale, shift = self.message(messages.mul(2).sub(1)).chunk(2, dim=1)
        features = features * (1 + 0.1 * scale[:, :, None, None]) + 0.1 * shift[:, :, None, None]
        return self.max_residual * torch.tanh(self.output(features))


class MessageConditionedProtector(nn.Module):
    def __init__(
        self,
        wam: nn.Module,
        nbits: int = 32,
        channels: int = 48,
        max_linf: float = 16 / 255,
    ) -> None:
        super().__init__()
        self.wam = wam
        self.max_linf = max_linf
        self.adapter = MessageResidualAdapter(nbits, channels, max_linf)

    def forward(self, images: torch.Tensor, messages: torch.Tensor, decode: bool = True) -> ProtectionOutput:
        wam_images, base_residual = self.wam.encode(images, messages)
        adapter_residual = self.adapter(images, messages)
        total_residual = (wam_images - images + adapter_residual).clamp(-self.max_linf, self.max_linf)
        protected = (images + total_residual).clamp(0, 1)
        decoded = self.wam.decode(protected) if decode else None
        return ProtectionOutput(
            protected_images=protected,
            residuals=protected - images,
            base_residuals=base_residual,
            adapter_residuals=adapter_residual,
            decoded=decoded,
        )

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

