"""Decode matched S20 Hallo videos with one frozen WAM reader."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch

from traceguard.config import load_config
from traceguard.evaluation.temporal import TemporalDecoder
from traceguard.factory import build_wam, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--video-name", default="protected.mp4")
    parser.add_argument("--allow-generation-failures", action="store_true",
                        help="Report recorded generation failures separately; never assign them video metrics")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Manifest is empty")
    keys = [(row["method"], row["domain"], row["sample_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate method/domain/sample_id")
    methods = {method for method, _, _ in keys}
    samples = {(domain, sample_id) for _, domain, sample_id in keys}
    if len(rows) != len(methods) * len(samples):
        raise ValueError("The method/sample grid is incomplete")
    generation_failures = []
    successful_rows = []
    for row in rows:
        folder = args.video_root / row["method"] / row["domain"] / row["sample_id"]
        status_path = folder / "status.json"
        if not status_path.is_file():
            raise FileNotFoundError(status_path)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        video = folder / args.video_name
        if status.get("image") != row["image"] or status.get("audio") != row["audio"]:
            raise ValueError(f"Video input differs from manifest: {video}")
        if status.get("generated") is False and args.allow_generation_failures:
            log_path = folder / "run.log"
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
            reason = "face_preprocessing_failed" if "#face is invalid: 0" in log else "generation_failed_unclassified"
            failure = {"method": row["method"], "domain": row["domain"],
                       "sample_id": row["sample_id"], "generated": False,
                       "exit_code": status.get("exit_code"), "reason": reason,
                       "status": str(status_path.resolve()), "log": str(log_path.resolve())}
            generation_failures.append(failure)
            continue
        if not status.get("generated") or not video.is_file() or video.stat().st_size == 0:
            raise ValueError(f"Missing successful video: {video}")
        successful_rows.append(row)

    config = load_config(args.config)
    device = resolve_device(config)
    wam = build_wam(config, device).eval()
    decoder = TemporalDecoder(wam, float(config.get("evaluation.detection_threshold", 0.5)))
    stride = int(config.get("evaluation.frame_stride", 4))
    max_frames = int(config.get("evaluation.max_frames", 32))
    records = []
    grouped = defaultdict(list)
    with torch.inference_mode():
        for row in successful_rows:
            video = args.video_root / row["method"] / row["domain"] / row["sample_id"] / args.video_name
            result = decoder.decode_video(video, stride, max_frames, str(device))
            target = row["message"]
            if len(target) != len(result.message):
                raise ValueError(f"Message length mismatch: {video}")
            accuracy = sum(a == b for a, b in zip(result.message, target)) / len(target)
            frame_detection = mean(float(frame["detection_score"]) for frame in result.frames)
            record = {
                "method": row["method"], "domain": row["domain"], "sample_id": row["sample_id"],
                "video": str(video.resolve()), "video_bytes": video.stat().st_size,
                "target_message": target, "decoded_message": result.message,
                "bit_accuracy": accuracy, "detection_score": result.detection_score,
                "mean_frame_detection_score": frame_detection,
                "detected_video": result.valid_frame_ratio > 0,
                "valid_frame_ratio": result.valid_frame_ratio, "sampled_frames": len(result.frames),
            }
            records.append(record)
            grouped[(row["method"], row["domain"])].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "total_videos": len(records), "expected_jobs": len(rows),
        "generation_failures": len(generation_failures),
        "generation_success_rate": len(records) / len(rows),
        "metric_population": "successfully generated videos only",
        "generation_by_method_domain": [
            {"method": method, "domain": domain,
             "expected": sum(row["method"] == method and row["domain"] == domain for row in rows),
             "generated": sum(row["method"] == method and row["domain"] == domain for row in successful_rows),
             "failed": sum(row["method"] == method and row["domain"] == domain for row in generation_failures)}
            for method, domain in sorted({(row["method"], row["domain"]) for row in rows})
        ],
        "samples": len(samples), "methods": sorted(methods),
        "frame_stride": stride, "max_frames": max_frames,
        "by_method_domain": [
            {"method": method, "domain": domain, "videos": len(group),
             "mean_bit_accuracy": mean(item["bit_accuracy"] for item in group),
             "mean_detection_score": mean(item["detection_score"] for item in group),
             "mean_frame_detection_score": mean(item["mean_frame_detection_score"] for item in group),
             "detected_videos": sum(item["detected_video"] for item in group),
             "mean_valid_frame_ratio": mean(item["valid_frame_ratio"] for item in group)}
            for (method, domain), group in sorted(grouped.items())
        ],
        "method_note": "Silencer-related methods in this evaluation use a VAE latent proxy, not upstream Silencer-Hallo",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "video_metrics.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    (args.output_root / "generation_failures.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in generation_failures), encoding="utf-8")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
