from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def image_metrics(clean: torch.Tensor, protected: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(protected.float(), clean.float()).item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    linf = (protected - clean).abs().amax().item()
    result = {"mse": mse, "psnr": psnr, "linf": linf}
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(clean.device).eval()
        with torch.no_grad():
            score = model(clean.mul(2).sub(1), protected.mul(2).sub(1)).mean().item()
        result["lpips"] = score
    except (ImportError, RuntimeError, OSError):
        result["lpips"] = float("nan")
    return result


def bit_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted.bool() == target.bool()).float().mean())

