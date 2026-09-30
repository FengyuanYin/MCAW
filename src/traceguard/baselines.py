from __future__ import annotations

import torch

from .types import ProtectionOutput


BASELINES = (
    "wam",
    "silencer_latent_proxy",
    "wam_then_silencer_latent_proxy",
    "silencer_latent_proxy_then_wam",
    "naive_joint",
    "message_coupled",
)


def _pgd_proxy(images: torch.Tensor, proxy, epsilon: float, steps: int = 10, start: torch.Tensor | None = None) -> torch.Tensor:
    origin = images.detach()
    if start is None:
        # The latent-distance objective has zero gradient at the clean reference.
        # A reproducible random PGD start avoids a no-op baseline.
        noise = torch.empty_like(origin).uniform_(-epsilon, epsilon)
        protected = (origin + noise).clamp(0, 1)
    else:
        protected = start.detach().clone()
    step_size = epsilon / max(steps // 2, 1)
    for _ in range(steps):
        protected.requires_grad_(True)
        loss = proxy(origin, protected).loss
        gradient = torch.autograd.grad(loss, protected)[0]
        protected = protected.detach() - step_size * gradient.sign()
        protected = torch.max(torch.min(protected, origin + epsilon), origin - epsilon).clamp(0, 1)
    return protected


def generate_baseline(
    name, images, messages, protector, proxy=None, pgd_steps: int = 10,
    joint_models: dict | None = None,
) -> ProtectionOutput:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name}; expected one of {', '.join(BASELINES)}")
    if name in {"naive_joint", "message_coupled"}:
        if joint_models is None or name not in joint_models:
            raise ValueError(f"{name} requires its own trained checkpoint")
        return joint_models[name](images, messages)
    wam_images = None
    wam_residual = torch.zeros_like(images)
    if name != "silencer_latent_proxy":
        wam_images, wam_residual = protector.wam.encode(images, messages)
    if name == "wam":
        final = wam_images
    elif name == "silencer_latent_proxy":
        if proxy is None:
            raise ValueError("Silencer latent proxy baseline requires a talking-head proxy")
        final = _pgd_proxy(images, proxy, protector.max_linf, pgd_steps)
    elif name == "wam_then_silencer_latent_proxy":
        if proxy is None:
            raise ValueError("Serial baseline requires a talking-head proxy")
        final = _pgd_proxy(images, proxy, protector.max_linf, pgd_steps, start=wam_images)
    else:
        if proxy is None:
            raise ValueError("Serial baseline requires a talking-head proxy")
        silenced = _pgd_proxy(images, proxy, protector.max_linf, pgd_steps)
        final, _ = protector.wam.encode(silenced, messages)
    final = torch.max(torch.min(final, images + protector.max_linf), images - protector.max_linf).clamp(0, 1)
    decoded = protector.wam.decode(final)
    return ProtectionOutput(final, final - images, wam_residual, torch.zeros_like(final), decoded)
