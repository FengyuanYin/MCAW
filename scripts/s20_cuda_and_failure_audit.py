"""Audit S20 video failures and test selected CUDA devices before a resume."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


def category(log: str) -> str:
    for needle, name in (
        ("No CUDA GPUs are available", "cuda_unavailable"),
        ("CUDA out of memory", "cuda_oom"),
        ("Broken pipe", "ffmpeg_pipe"),
        ("No such file or directory", "missing_file"),
    ):
        if needle in log:
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--gpus", default="3,4,5,6")
    args = parser.parse_args()

    failures = []
    for status_path in args.video_root.rglob("status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("generated"):
            continue
        log_path = status_path.with_name("run.log")
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        failures.append({**status, "category": category(log), "log_tail": log[-1200:]})
    print(json.dumps({
        "failed": len(failures),
        "by_category": dict(Counter(row["category"] for row in failures)),
        "by_gpu": dict(Counter(row.get("gpu") for row in failures)),
        "first_failures": [{key: row.get(key) for key in ("method", "sample_id", "gpu", "category")}
                           for row in failures[:8]],
        "category_examples": {kind: next(row["log_tail"] for row in failures if row["category"] == kind)
                              for kind in {row["category"] for row in failures}},
    }, indent=2), flush=True)

    gpu_ids = [int(value) for value in args.gpus.split(",")]
    devices = []
    for gpu in gpu_ids:
        try:
            with torch.cuda.device(gpu):
                value = torch.ones(1, device=f"cuda:{gpu}")
                torch.cuda.synchronize(gpu)
                devices.append({"gpu": gpu, "ok": bool(value.item() == 1)})
        except Exception as exc:
            devices.append({"gpu": gpu, "ok": False, "error": str(exc)})
    print(json.dumps({"torch_cuda_available": torch.cuda.is_available(),
                      "torch_cuda_device_count": torch.cuda.device_count(),
                      "device_checks": devices}, indent=2), flush=True)


if __name__ == "__main__":
    main()
