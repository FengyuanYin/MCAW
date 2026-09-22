from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def load_image(path: str | Path, size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    tensor = pil_to_tensor(image).float().div(255.0)
    return tensor.unsqueeze(0)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    image = image.clamp(0, 1)
    to_pil_image(image).save(destination)


def parse_message(message: str, bits: int, device: torch.device | str) -> torch.Tensor:
    compact = message.strip().replace(" ", "")
    if len(compact) != bits or any(char not in "01" for char in compact):
        raise ValueError(f"message must contain exactly {bits} binary digits")
    return torch.tensor([[int(char) for char in compact]], dtype=torch.float32, device=device)


def message_to_string(bits: torch.Tensor) -> str:
    return "".join("1" if bool(value) else "0" for value in bits.flatten().tolist())
