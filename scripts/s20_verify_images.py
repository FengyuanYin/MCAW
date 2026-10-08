"""Independently verify saved S20 PNGs against clean PNGs and the fixed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixed-manifest", type=Path, required=True)
    parser.add_argument("--epsilon-codes", type=int, default=16)
    args = parser.parse_args()
    fixed = [json.loads(line) for line in args.fixed_manifest.read_text().splitlines() if line.strip()]
    expected = {row["sample_id"] for row in fixed}
    metrics = [json.loads(line) for line in (args.root / "image_metrics.jsonl").read_text().splitlines() if line.strip()]
    methods = {row["method"] for row in metrics}
    grouped = defaultdict(set)
    max_delta = 0
    distinct = defaultdict(set)
    for row in metrics:
        sample_id, domain, method = row["sample_id"], row["domain"], row["method"]
        assert sample_id in expected
        assert (domain == "celeba" and sample_id.startswith("celeba-")) or (domain == "th1kh" and sample_id.startswith("th1kh-"))
        clean_path = args.root / "clean" / domain / f"{sample_id}.png"
        method_path = args.root / method / domain / f"{sample_id}.png"
        clean, image = load(clean_path), load(method_path)
        assert clean.shape == image.shape == (256, 256, 3), method_path
        delta = int(np.abs(image - clean).max())
        assert delta <= args.epsilon_codes, (method_path, delta)
        assert delta == row["linf_codes"], (method_path, delta, row["linf_codes"])
        max_delta = max(max_delta, delta)
        grouped[method].add(sample_id)
        distinct[sample_id].add(hashlib.sha256(method_path.read_bytes()).hexdigest())
    assert len(metrics) == len(expected) * len(methods)
    assert all(ids == expected for ids in grouped.values())
    assert all(len(images) == len(methods) for images in distinct.values())
    print(json.dumps({"result": "PASS", "samples": len(expected), "methods": len(methods),
                      "protected_pngs": len(metrics), "max_linf_codes": max_delta}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
