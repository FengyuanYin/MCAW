"""Verify S18 outputs and prepare an unbiased manual clean-video review."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/formal_eval"
CLEAN = ROOT / "outputs/formal_eval/clean"
DOMAINS = ("celeba", "th1kh")
FFPROBE = "/root/miniconda3/bin/ffprobe"


def probe(video: Path) -> tuple[float, int, int]:
    result = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height", "-of", "json", str(video)],
        capture_output=True, text=True, timeout=30, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"ffprobe exit {result.returncode}")
    details = json.loads(result.stdout)
    streams = [stream for stream in details.get("streams", [])
               if stream.get("codec_type") == "video"]
    if not streams:
        raise ValueError("no video stream")
    stream = streams[0]
    return (float(details["format"]["duration"]), int(stream["width"]),
            int(stream["height"]))


def inspect(item: tuple[str, dict]) -> dict:
    domain, row = item
    sample_id = row["sample_id"]
    folder = CLEAN / domain / sample_id
    video = folder / "clean.mp4"
    status_file = folder / "status.json"
    result = {
        "domain": domain, "sample_id": sample_id, "image": row["image"],
        "video": str(video), "generated": False, "duration_seconds": "",
        "width": "", "height": "", "qc_ok": False, "qc_reason": "",
    }
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status.get("sample_id") != sample_id or status.get("domain") != domain:
            raise ValueError("status identity mismatch")
        if not status.get("generated"):
            raise ValueError("status reports generation failure")
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError("video missing or empty")
        duration, width, height = probe(video)
        if duration <= 0 or width <= 0 or height <= 0:
            raise ValueError("invalid video metadata")
        result.update(generated=True, duration_seconds=f"{duration:.3f}",
                      width=width, height=height, qc_ok=True)
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        result["qc_reason"] = str(exc)
    return result


def main() -> int:
    items = []
    for domain in DOMAINS:
        manifest = DATA / f"{domain}_candidates_100.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if len(rows) != 100 or len({row["sample_id"] for row in rows}) != 100:
            raise ValueError(f"{domain}: expected 100 unique candidates")
        items.extend((domain, row) for row in rows)
    with ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(inspect, items))

    review = DATA / "clean_review.csv"
    if review.exists():
        print(f"Preserved existing manual review: {review}")
    else:
        with review.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "domain", "sample_id", "image", "video", "generated",
                "duration_seconds", "manual_ok", "reason",
            ])
            writer.writeheader()
            for item in checked:
                writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
        print(f"Created review table: {review}")

    order = DATA / "clean_review_order.csv"
    if not order.exists():
        with order.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "domain", "sample_id", "video", "duration_seconds",
                "width", "height", "qc_ok", "qc_reason",
            ])
            writer.writeheader()
            for domain, seed in (("celeba", 2026), ("th1kh", 2027)):
                domain_rows = [item for item in checked if item["domain"] == domain]
                random.Random(seed).shuffle(domain_rows)
                for item in domain_rows:
                    writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
        print(f"Created randomized review order: {order}")

    counts = {domain: sum(item["qc_ok"] for item in checked if item["domain"] == domain)
              for domain in DOMAINS}
    failures = [item for item in checked if not item["qc_ok"]]
    report = {"total": len(checked), "metadata_pass": counts,
              "metadata_fail": len(failures), "failures": failures,
              "manual_review_complete": False}
    path = ROOT / "outputs/formal_eval/s19_preflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata_pass": counts, "metadata_fail": len(failures)},
                     ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
