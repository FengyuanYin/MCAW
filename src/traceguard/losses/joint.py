from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from traceguard.types import ProtectionOutput, ProxyOutput


def bounded_direction_alignment(
    residual: torch.Tensor, direction: torch.Tensor, rms_floor: float = 1 / 255,
) -> torch.Tensor:
    """Smooth alignment with a pixel-scale floor at the zero-residual start.

    Ordinary cosine normalization has a 1/eps derivative at zero. Here a
    one-pixel-level RMS floor bounds that derivative, independently of image
    resolution. Compute normalization in float32 for mixed precision training.
    This is a revised objective, not an exact cosine near zero.
    """
    if rms_floor <= 0:
        raise ValueError("Alignment RMS floor must be positive")
    residual = residual.flatten(1).float()
    direction = direction.detach().flatten(1).float()
    unit_direction = F.normalize(direction, dim=1, eps=1e-12)
    denominator = (residual.square().sum(1) + residual.shape[1] * rms_floor**2).sqrt()
    alignment = (residual * unit_direction).sum(1) / denominator
    return (1 - alignment).mean()


@dataclass
class JointLossOutput:
    total: torch.Tensor
    values: dict[str, torch.Tensor]

    def detached_metrics(self) -> dict[str, float]:
        return {name: float(value.detach()) for name, value in self.values.items()}


class JointLoss(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights

    def forward(
        self,
        clean: torch.Tensor,
        messages: torch.Tensor,
        protection: ProtectionOutput,
        proxy: ProxyOutput | None = None,
        adversarial_direction: torch.Tensor | None = None,
        robust_decode_logits: torch.Tensor | None = None,
    ) -> JointLossOutput:
        if protection.decoded is None or protection.decoded.raw_logits is None:
            raise ValueError("Joint training requires differentiable WAM decode logits")
        values: dict[str, torch.Tensor] = {}
        values["watermark"] = F.binary_cross_entropy_with_logits(protection.decoded.raw_logits, messages)
        values["detection"] = -torch.log(protection.decoded.detection_scores.clamp_min(1e-6)).mean()
        values["visual_l1"] = F.l1_loss(protection.protected_images, clean)
        values["adversarial"] = proxy.loss if proxy is not None else clean.new_zeros(())
        if adversarial_direction is not None:
            values["coupling"] = bounded_direction_alignment(
                protection.adapter_residuals, adversarial_direction,
            )
        else:
            values["coupling"] = clean.new_zeros(())
        if robust_decode_logits is not None:
            values["robustness"] = F.binary_cross_entropy_with_logits(robust_decode_logits, messages)
        else:
            values["robustness"] = clean.new_zeros(())
        total = sum(self.weights.get(name, 0.0) * value for name, value in values.items())
        values["total"] = total
        return JointLossOutput(total, values)

