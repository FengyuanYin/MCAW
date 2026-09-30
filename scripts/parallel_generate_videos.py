"""Run independent Hallo video generations across several GPUs.

Each GPU runs one subprocess at a time. Existing successful outputs are skipped,
so an interrupted batch can be restarted with the same arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices, e.g. 3,4,5")
    parser.add_argument("--audio", type=Path, help="Fixed audio; otherwise use each row's audio field")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluate.yaml"))
    parser.add_argument("--traceguard", default="traceguard")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--video-name", default="clean.mp4")
    parser.add_argument("--min-free-mib", type=int, default=0,
                        help="Stop a GPU worker before launching a job if free VRAM is below this")
    return parser.parse_args()


def load_jobs(args: argparse.Namespace) -> list[tuple[str, str, str, Path, Path]]:
    jobs = []
    seen = set()
    for manifest in args.manifest:
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method") or "")
            domain = str(row.get("domain") or manifest.stem.split("_")[0])
            sample_id = str(row["sample_id"])
            if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) and part not in {".", ".."}
                       for part in (domain, sample_id) + ((method,) if method else ())):
                raise ValueError(f"Unsafe method/domain/sample_id at {manifest}:{number}")
            key = (method, domain, sample_id)
            if key in seen:
                raise ValueError(f"Duplicate job: {key}")
            seen.add(key)
            image = Path(row["image"]).resolve()
            audio = (args.audio or Path(row["audio"])).resolve()
            if not image.is_file() or not audio.is_file():
                raise FileNotFoundError(f"Missing image/audio at {manifest}:{number}: {image}, {audio}")
            jobs.append((method, domain, sample_id, image, audio))
    return jobs


def valid_existing(status: Path, video: Path, image: Path, audio: Path, method: str) -> bool:
    try:
        record = json.loads(status.read_text(encoding="utf-8"))
        return (record.get("generated") is True
                and record.get("method", "") == method
                and record.get("image") == str(image)
                and record.get("audio") == str(audio)
                and video.is_file() and video.stat().st_size > 0)
    except (OSError, ValueError):
        return False


def free_gpu_mib(gpu: str) -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi GPU {gpu}: {result.stderr.strip()}")
    return int(result.stdout.strip().splitlines()[0])


def run_one(args: argparse.Namespace, gpu: str, job: tuple[str, str, str, Path, Path]) -> str:
    method, domain, sample_id, image, audio = job
    folder = args.output_root / method / domain / sample_id if method else args.output_root / domain / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    video = folder / args.video_name
    status = folder / "status.json"
    if valid_existing(status, video, image, audio, method):
        return "SKIP"
    # MoviePy derives its temporary audio filename from the video basename.
    # A unique staging name prevents parallel Hallo workers from colliding.
    staged_video = folder / f"{video.stem}_{uuid.uuid4().hex}.mp4"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [args.traceguard, "generate-video", "--config", str(args.config.resolve()),
               "--image", str(image), "--audio", str(audio), "--output", str(staged_video.resolve())]
    log_path = folder / "run.log"
    if log_path.is_file() and status.is_file():
        log_path.replace(folder / f"run_previous_{uuid.uuid4().hex}.log")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("GPU=" + gpu + "\n")
        log.flush()
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    probe = subprocess.run(
        [args.ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(staged_video)],
        capture_output=True, text=True, check=False,
    ) if staged_video.is_file() else None
    try:
        duration = float(probe.stdout.strip()) if probe and probe.returncode == 0 else 0.0
    except ValueError:
        duration = 0.0
    generated = result.returncode == 0 and duration > 0
    if generated:
        staged_video.replace(video)
    record = {
        "sample_id": sample_id, "domain": domain, "method": method,
        "image": str(image), "audio": str(audio),
        "video": str(video.resolve()), "gpu": gpu, "exit_code": result.returncode,
        "bytes": video.stat().st_size if generated else staged_video.stat().st_size if staged_video.is_file() else 0,
        "duration_seconds": duration, "generated": generated,
        "staged_video_on_failure": str(staged_video) if not generated else None,
    }
    temporary = status.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(status)
    return "PASS" if record["generated"] else "FAIL"


def main() -> int:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("--gpus must be a comma-separated list of distinct GPU indices")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", args.video_name):
        raise ValueError("--video-name must be a simple .mp4 filename")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if args.min_free_mib < 0:
        raise ValueError("--min-free-mib must be non-negative")
    jobs = load_jobs(args)
    if not jobs:
        raise ValueError("No jobs found in the input manifests")
    pending: queue.Queue[tuple[str, str, str, Path, Path]] = queue.Queue()
    for job in jobs:
        pending.put(job)
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            if args.min_free_mib:
                try:
                    free = free_gpu_mib(gpu)
                except (OSError, RuntimeError, ValueError, IndexError) as exc:
                    print(f"GPU {gpu} unavailable: {exc}; worker stopped", file=sys.stderr, flush=True)
                    return
                if free < args.min_free_mib:
                    print(f"GPU {gpu} free VRAM {free} MiB < {args.min_free_mib} MiB; worker stopped",
                          file=sys.stderr, flush=True)
                    return
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                outcome = run_one(args, gpu, job)
            except Exception as exc:
                outcome = "FAIL"
                print(f"GPU {gpu} {job[0]}/{job[1]}/{job[2]} ERROR {exc}", file=sys.stderr, flush=True)
            with lock:
                counts[outcome] += 1
                print(f"GPU {gpu} {job[0]}/{job[1]}/{job[2]} {outcome} {counts}", flush=True)
            pending.task_done()
            if outcome == "FAIL":
                method, domain, sample_id, _, _ = job
                folder = args.output_root / method / domain / sample_id if method else args.output_root / domain / sample_id
                log_path = folder / "run.log"
                log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
                if any(message in log for message in (
                    "CUDA out of memory", "CUDA driver initialization failed", "No CUDA GPUs are available",
                )):
                    print(f"GPU {gpu} CUDA failure; worker stopped", file=sys.stderr, flush=True)
                    return

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    remaining = pending.qsize()
    print(json.dumps({"total": len(jobs), "counts": counts, "remaining": remaining,
                      "gpus": gpus}, sort_keys=True))
    return 1 if counts["FAIL"] or remaining else 0


if __name__ == "__main__":
    raise SystemExit(main())
