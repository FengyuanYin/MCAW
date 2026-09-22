from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from traceguard.errors import DependencyError, WeightNotFoundError
from traceguard.types import DecodeOutput


class MockWAMAdapter(nn.Module):
    """Deterministic differentiable backend for engineering smoke runs only."""

    def __init__(self, nbits: int = 32, strength: float = 0.03) -> None:
        super().__init__()
        self.nbits = nbits
        self.strength = strength

    def _basis(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.linspace(0, 1, height, device=device, dtype=dtype)
        x = torch.linspace(0, 1, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        patterns = []
        for index in range(self.nbits):
            fx, fy = index % 8 + 1, index // 8 + 1
            pattern = torch.cos(torch.pi * fx * xx) * torch.cos(torch.pi * fy * yy)
            patterns.append(pattern / pattern.square().mean().sqrt().clamp_min(1e-6))
        return torch.stack(patterns).unsqueeze(1)

    def encode(self, images: torch.Tensor, messages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        basis = self._basis(images.shape[-2], images.shape[-1], images.device, images.dtype)
        signs = messages.mul(2).sub(1)
        pattern = torch.einsum("bk,kchw->bchw", signs, basis) / self.nbits**0.5
        residual = self.strength * pattern.repeat(1, 3, 1, 1)
        watermarked = (images + residual).clamp(0, 1)
        return watermarked, watermarked - images

    def decode(self, images: torch.Tensor) -> DecodeOutput:
        basis = self._basis(images.shape[-2], images.shape[-1], images.device, images.dtype)
        gray = images.mean(dim=1, keepdim=True) - images.mean(dim=(1, 2, 3), keepdim=True)
        logits = torch.einsum("bchw,kchw->bk", gray, basis) / (images.shape[-2] * images.shape[-1])
        probabilities = torch.sigmoid(logits * 80)
        detection = torch.sigmoid(logits.abs().mean(dim=1) * 80)
        return DecodeOutput(
            bits=probabilities >= 0.5,
            bit_probabilities=probabilities,
            bit_confidences=(probabilities - 0.5).abs() * 2,
            detection_scores=detection,
            raw_logits=logits,
        )

    def set_trainable_policy(self, policy: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class WAMAdapter(nn.Module):
    """Stable wrapper around the pinned upstream Watermark Anything model."""

    def __init__(self, root: Path, params_path: Path, checkpoint_path: Path) -> None:
        super().__init__()
        if not checkpoint_path.is_file():
            raise WeightNotFoundError(
                f"WAM checkpoint not found: {checkpoint_path}. Download it explicitly as documented."
            )
        if not params_path.is_file():
            raise WeightNotFoundError(f"WAM params file not found: {params_path}")
        self.root = root.resolve()
        self.model = self._load(params_path.resolve(), checkpoint_path.resolve())
        self.nbits = 32
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load(self, params_path: Path, checkpoint_path: Path) -> nn.Module:
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise DependencyError("WAM requires omegaconf; install traceguard[wam]") from exc
        sys.path.insert(0, str(self.root))
        try:
            from watermark_anything.augmentation.augmenter import Augmenter
            from watermark_anything.data.transforms import normalize_img, unnormalize_img
            from watermark_anything.models import Wam, build_embedder, build_extractor
            from watermark_anything.modules.jnd import JND
        except Exception as exc:
            raise DependencyError(f"Cannot import pinned WAM source from {self.root}: {exc}") from exc
        with params_path.open("r", encoding="utf-8") as handle:
            args = argparse.Namespace(**json.load(handle))

        def cfg(relative: str):
            return OmegaConf.load(str((self.root / relative).resolve()))

        embedder_cfg = cfg(args.embedder_config)
        extractor_cfg = cfg(args.extractor_config)
        augmenter_cfg = cfg(args.augmentation_config)
        attenuation_cfg = cfg(args.attenuation_config)
        embedder = build_embedder(args.embedder_model, embedder_cfg[args.embedder_model], args.nbits)
        extractor = build_extractor(extractor_cfg.model, extractor_cfg[args.extractor_model], args.img_size, args.nbits)
        augmenter = Augmenter(**augmenter_cfg)
        try:
            attenuation = JND(
                **attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img
            )
        except (KeyError, TypeError):
            attenuation = None
        model = Wam(
            embedder,
            extractor,
            augmenter,
            attenuation,
            args.scaling_w,
            args.scaling_i,
            img_size_extractor=getattr(args, "img_size_extractor", 256),
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        return images * self.std + self.mean

    def encode(self, images: torch.Tensor, messages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self._normalize(images)
        output = self.model.embed(normalized, messages)
        watermarked = self._denormalize(output["imgs_w"]).clamp(0, 1)
        return watermarked, watermarked - images

    def decode(self, images: torch.Tensor) -> DecodeOutput:
        logits = self.model.detect(self._normalize(images))["preds"]
        mask_prob = logits[:, :1].sigmoid()
        bit_logits_map = logits[:, 1:]
        weights = mask_prob / mask_prob.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        bit_logits = (bit_logits_map * weights).sum(dim=(2, 3))
        probabilities = bit_logits.sigmoid()
        return DecodeOutput(
            bits=probabilities >= 0.5,
            bit_probabilities=probabilities,
            bit_confidences=(probabilities - 0.5).abs() * 2,
            detection_scores=mask_prob.mean(dim=(1, 2, 3)),
            raw_logits=bit_logits,
        )

    def set_trainable_policy(self, policy: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if policy == "wam_tail":
            for parameter in self.model.embedder.decoder.conv_out.parameters():
                parameter.requires_grad_(True)
        elif policy not in {"adapter_only", "frozen"}:
            raise ValueError(f"Unknown WAM trainable policy: {policy}")

