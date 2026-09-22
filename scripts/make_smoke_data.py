from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    image = np.stack(
        [
            80 + 120 * xx / width,
            70 + 100 * yy / height,
            100 + 80 * (xx + yy) / (height + width),
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)
    image_path = DATA / "smoke.png"
    Image.fromarray(image).save(image_path)
    record = {
        "sample_id": "synthetic-smoke",
        "image": str(image_path.resolve()),
        "message": "01010101010101010101010101010101",
    }
    (DATA / "smoke.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    print(DATA / "smoke.jsonl")


if __name__ == "__main__":
    main()
