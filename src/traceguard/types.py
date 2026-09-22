from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class DecodeOutput:
    bits: torch.Tensor
    bit_probabilities: torch.Tensor
    bit_confidences: torch.Tensor
    detection_scores: torch.Tensor
    raw_logits: torch.Tensor | None = None


@dataclass
class ProtectionOutput:
    protected_images: torch.Tensor
    residuals: torch.Tensor
    base_residuals: torch.Tensor
    adapter_residuals: torch.Tensor
    decoded: DecodeOutput | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProxyOutput:
    loss: torch.Tensor
    clean_features: torch.Tensor
    protected_features: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoArtifact:
    path: Path
    frames: int | None = None
    fps: float | None = None

