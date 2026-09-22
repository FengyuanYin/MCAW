from __future__ import annotations

from pathlib import Path

import torch

from .adapters import MockTalkingHeadProxy, MockWAMAdapter, SilencerLatentProxy, WAMAdapter
from .config import LoadedConfig
from .models import MessageConditionedProtector


def resolve_device(config: LoadedConfig) -> torch.device:
    requested = str(config.get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if config.get("backend") == "mock":
            return torch.device("cpu")
        raise RuntimeError("CUDA was requested but is unavailable. Run traceguard preflight for details.")
    return torch.device(requested)


def build_wam(config: LoadedConfig, device: torch.device):
    if config.get("backend") == "mock":
        wam = MockWAMAdapter(int(config.require("model.message_bits")))
    else:
        wam = WAMAdapter(
            config.resolve("model.wam.root"),
            config.resolve("model.wam.params"),
            config.resolve("model.wam.checkpoint"),
        )
    wam.set_trainable_policy(str(config.get("model.wam.trainable", "adapter_only")))
    return wam.to(device)


def build_protector(config: LoadedConfig, device: torch.device) -> MessageConditionedProtector:
    return MessageConditionedProtector(
        build_wam(config, device),
        nbits=int(config.require("model.message_bits")),
        channels=int(config.get("model.adapter_channels", 48)),
        max_linf=float(config.require("model.max_linf")),
    ).to(device)


def build_proxy(config: LoadedConfig, device: torch.device):
    if config.get("backend") == "mock":
        return MockTalkingHeadProxy().to(device)
    kind = str(config.get("model.proxy.kind", "silencer_latent"))
    if kind != "silencer_latent":
        raise ValueError(f"Unsupported real proxy kind: {kind}")
    precision = str(config.get("precision", "fp16"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    return SilencerLatentProxy(config.resolve("model.proxy.pretrained_root"), dtype=dtype).to(device)


def load_protector_checkpoint(protector: MessageConditionedProtector, path: str | Path | None) -> None:
    if path is None:
        return
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    protector.load_state_dict(state.get("model", state), strict=True)
