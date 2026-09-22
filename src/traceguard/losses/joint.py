from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from traceguard.types import ProtectionOutput, ProxyOutput


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
            residual = protection.adapter_residuals.flatten(1)
            direction = adversarial_direction.flatten(1)
            values["coupling"] = (1 - F.cosine_similarity(residual, direction, dim=1, eps=1e-8)).mean()
        else:
            values["coupling"] = clean.new_zeros(())
        if robust_decode_logits is not None:
            values["robustness"] = F.binary_cross_entropy_with_logits(robust_decode_logits, messages)
        else:
            values["robustness"] = clean.new_zeros(())
        total = sum(self.weights.get(name, 0.0) * value for name, value in values.items())
        values["total"] = total
        return JointLossOutput(total, values)

