"""Save comparable protected PNGs and video manifests for the S20 pilot.

The three Silencer-related names here use the local frozen VAE latent proxy.
They are not the full upstream Silencer-Hallo attack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from traceguard.baselines import BASELINES, generate_baseline
from traceguard.config import load_config
from traceguard.data.manifest import read_manifest
from traceguard.factory import build_protector, build_proxy, resolve_device
from traceguard.utils.images import load_image, parse_message


def checkpoint_info(path: Path, expected_coupling: float) -> tuple[dict, str]:
    path = path.resolve()
    state = torch.load(path, map_location="cpu", weights_only=False)
    actual = float(state.get("config", {}).get("loss", {}).get("coupling", float("nan")))
    if not math.isfinite(actual) or abs(actual - expected_coupling) > 1e-8:
        raise ValueError(f"{path}: coupling weight {actual}; expected {expected_coupling}")
    if "model" not in state or "step" not in state:
        raise ValueError(f"{path}: not a Trainer checkpoint")
    return state, hashlib.sha256(path.read_bytes()).hexdigest()


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image[0].detach().float().cpu().permute(1, 2, 0).mul(255)
            .round().clamp(0, 255).to(torch.uint8).numpy())


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("data/formal_eval/fixed_40x2.jsonl"))
    parser.add_argument("--naive-checkpoint", type=Path, required=True)
    parser.add_argument("--coupled-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit-per-domain", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=10)
    args = parser.parse_args()
    if args.limit_per_domain < 0 or args.pgd_steps < 1:
        raise ValueError("Invalid limit or PGD steps")
    if args.naive_checkpoint.resolve() == args.coupled_checkpoint.resolve():
        raise ValueError("Joint baselines cannot share a checkpoint")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty: {args.output_root}")
    config = load_config(args.config)
    device = resolve_device(config)
    torch.manual_seed(int(config.get("project.seed", 42)))
    protector = build_protector(config, device).eval()
    proxy = build_proxy(config, device).eval()
    for parameter in proxy.parameters():
        parameter.requires_grad_(False)
    joint_models = {}
    identities = {}
    for name, path, weight in (
        ("naive_joint", args.naive_checkpoint, 0.0),
        ("message_coupled", args.coupled_checkpoint, 0.1),
    ):
        state, digest = checkpoint_info(path, weight)
        model = build_protector(config, device)
        model.load_state_dict(state["model"], strict=True)
        joint_models[name] = model.eval()
        identities[name] = {"checkpoint": str(path.resolve()), "sha256": digest,
                            "step": int(state["step"]), "coupling_weight": weight}
        del state
    if identities["naive_joint"]["step"] != identities["message_coupled"]["step"]:
        raise ValueError("Joint checkpoints must have the same training step")

    items = read_manifest(args.manifest)
    selected = []
    counts = {}
    for item in items:
        if item.sample_id.startswith("celeba-"):
            domain = "celeba"
        elif item.sample_id.startswith("th1kh-"):
            domain = "th1kh"
        else:
            raise ValueError(f"Unknown sample domain: {item.sample_id}")
        if args.limit_per_domain and counts.get(domain, 0) >= args.limit_per_domain:
            continue
        counts[domain] = counts.get(domain, 0) + 1
        selected.append((domain, item))
    if not selected:
        raise ValueError("No selected samples")

    args.output_root.mkdir(parents=True)
    (args.output_root / "manifests").mkdir()
    manifest_rows = {method: [] for method in ("clean", *BASELINES)}
    metrics = []
    epsilon_codes = round(float(config.require("model.max_linf")) * 255)
    if epsilon_codes != 16:
        raise ValueError(f"Expected 16/255 budget, got {epsilon_codes}/255")
    for domain, item in selected:
        image = load_image(item.image, int(config.require("model.image_size"))).to(device)
        message = parse_message(item.message, int(config.require("model.message_bits")), device)
        clean = to_uint8(image)
        clean_path = args.output_root / "clean" / domain / f"{item.sample_id}.png"
        save_png(clean, clean_path)
        if item.audio is None:
            raise ValueError(f"Missing fixed audio: {item.sample_id}")
        common = {"sample_id": item.sample_id, "domain": domain, "audio": str(item.audio),
                  "message": item.message, "source_image": str(item.image)}
        manifest_rows["clean"].append({**common, "image": str(clean_path.resolve()), "method": "clean"})
        for method in BASELINES:
            context = torch.enable_grad() if "proxy" in method else torch.no_grad()
            with context:
                output = generate_baseline(method, image, message, protector, proxy,
                                           args.pgd_steps, joint_models)
            raw = to_uint8(output.protected_images)
            bounded = np.clip(raw.astype(np.int16), clean.astype(np.int16) - epsilon_codes,
                              clean.astype(np.int16) + epsilon_codes).clip(0, 255).astype(np.uint8)
            path = args.output_root / method / domain / f"{item.sample_id}.png"
            save_png(bounded, path)
            final = torch.from_numpy(bounded.copy()).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255
            with torch.no_grad():
                decoded = protector.wam.decode(final)
            delta = bounded.astype(np.int16) - clean.astype(np.int16)
            mse = float(np.mean(delta.astype(np.float64) ** 2)) / 255**2
            linf_codes = int(np.max(np.abs(delta)))
            if linf_codes > epsilon_codes:
                raise AssertionError(f"{method}/{item.sample_id}: budget violation")
            record = {**common, "method": method, "image": str(path.resolve()),
                      "psnr": -10 * math.log10(mse) if mse else None,
                      "linf_codes": linf_codes,
                      "bit_accuracy": float((decoded.bits == message.bool()).float().mean()),
                      "detection_score": float(decoded.detection_scores.float().mean()),
                      "checkpoint": identities.get(method, {}).get("checkpoint"),
                      "checkpoint_sha256": identities.get(method, {}).get("sha256"),
                      "implementation": ("silencer_latent_proxy_v1" if "proxy" in method else
                                         "joint_adapter_v1" if method in joint_models else "wam_mit"),
            }
            metrics.append(record)
            manifest_rows[method].append({key: record[key] for key in
                                          ("sample_id", "domain", "audio", "message", "source_image", "method", "image")})
        print(f"finished {domain}/{item.sample_id}", flush=True)
    for method, rows in manifest_rows.items():
        (args.output_root / "manifests" / f"{method}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (args.output_root / "image_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics), encoding="utf-8")
    summary = {"samples": len(selected), "methods": list(BASELINES),
               "message": "Silencer-related methods are VAE latent proxy adaptations, not upstream Silencer-Hallo",
               "resolution": int(config.require("model.image_size")), "epsilon_codes": epsilon_codes,
               "pgd_steps": args.pgd_steps, "joint_checkpoints": identities,
               "fixed_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
