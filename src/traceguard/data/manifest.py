from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from traceguard.utils.images import load_image, parse_message


@dataclass(frozen=True)
class ManifestItem:
    sample_id: str
    image: Path
    audio: Path | None
    message: str


def read_manifest(path: str | Path) -> list[ManifestItem]:
    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")
    items: list[ManifestItem] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            image = Path(record["image"]).expanduser()
            audio = Path(record["audio"]).expanduser() if record.get("audio") else None
            if not image.is_file():
                raise FileNotFoundError(f"Line {line_number}: image not found: {image}")
            if audio is not None and not audio.is_file():
                raise FileNotFoundError(f"Line {line_number}: audio not found: {audio}")
            items.append(ManifestItem(str(record["sample_id"]), image, audio, str(record["message"])))
    if not items:
        raise ValueError(f"Manifest is empty: {manifest}")
    return items


class AuthorizedManifestDataset(Dataset):
    def __init__(self, path: str | Path, image_size: int, message_bits: int = 32) -> None:
        self.items = read_manifest(path)
        self.image_size = image_size
        self.message_bits = message_bits

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.items[index]
        return {
            "image": load_image(item.image, self.image_size).squeeze(0),
            "message": parse_message(item.message, self.message_bits, "cpu").squeeze(0),
            "sample_id": item.sample_id,
            "audio": str(item.audio) if item.audio else "",
        }

