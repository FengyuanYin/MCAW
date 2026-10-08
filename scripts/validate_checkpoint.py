"""Evaluate a trained protector on a held-out image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from traceguard.config import load_config
from traceguard.data.manifest import read_manifest
from traceguard.factory import build_protector, build_proxy, resolve_device
from traceguard.utils.images import load_image, parse_message


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = resolve_device(config)
    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" not in state or "step" not in state or "config" not in state:
        raise ValueError("Expected a Trainer checkpoint with model, step and config")
    protector = build_protector(config, device)
    protector.load_state_dict(state["model"], strict=True)
    protector.eval()
    proxy = build_proxy(config, device).eval()
    for parameter in proxy.parameters():
        parameter.requires_grad_(False)
    amp = device.type == "cuda" and config.get("precision") in {"fp16", "bf16"}
    dtype = torch.float16 if config.get("precision") == "fp16" else torch.bfloat16
    items = read_manifest(args.manifest)
    if args.limit:
        items = items[:args.limit]
    records = []
    output = args.output_dir / "samples.jsonl"
    with output.open("w", encoding="utf-8") as handle, torch.no_grad():
        for item in items:
            image = load_image(item.image, int(config.require("model.image_size"))).to(device)
            message = parse_message(item.message, int(config.require("model.message_bits")), device)
            with torch.autocast(device.type, dtype=dtype, enabled=amp):
                protected = protector(image, message)
                proxy_result = proxy(image, protected.protected_images)
            delta = (protected.protected_images - image).float()
            mse = float(delta.square().mean())
            record = {
                "sample_id": item.sample_id,
                "bit_accuracy": float((protected.decoded.bits == message.bool()).float().mean()),
                "detection_score": float(protected.decoded.detection_scores.float().mean()),
                "psnr": -10 * math.log10(mse) if mse > 0 else float("inf"),
                "linf": float(delta.abs().amax()),
                "latent_mse": float(proxy_result.metrics["latent_mse"]),
            }
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            if len(records) % 25 == 0:
                print(f"validated {len(records)}/{len(items)}", flush=True)
    if not records:
        raise ValueError("No validation items")
    keys = ("bit_accuracy", "detection_score", "psnr", "linf", "latent_mse")
    summary = {
        "count": len(records), "step": int(state["step"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "training_coupling_weight": state["config"].get("loss", {}).get("coupling"),
        "mean": {key: sum(row[key] for row in records) / len(records) for key in keys},
        "linf_violations": sum(row["linf"] > float(config.require("model.max_linf")) + 1e-5
                               for row in records),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 1 if summary["linf_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
