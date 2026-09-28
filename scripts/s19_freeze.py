"""Freeze 40 manually approved clean videos from each evaluation domain."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/formal_eval"
AUDIO = Path("/data/MCAW-data/hallo_samples/driving_audios/1.wav")
OUTPUTS = {
    "celeba": DATA / "celeba_fixed_40.jsonl",
    "th1kh": DATA / "th1kh_fixed_40.jsonl",
    "combined": DATA / "fixed_40x2.jsonl",
    "metadata": DATA / "fixed_40x2.meta.json",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError("Frozen output already exists: " + ", ".join(existing))
    if not AUDIO.is_file() or AUDIO.stat().st_size == 0:
        raise FileNotFoundError(AUDIO)
    with (DATA / "clean_review.csv").open(newline="", encoding="utf-8") as handle:
        review = list(csv.DictReader(handle))
    if len(review) != 200:
        raise ValueError(f"Expected 200 review rows, found {len(review)}")
    review_by_id = {(row["domain"], row["sample_id"]): row for row in review}
    if len(review_by_id) != 200:
        raise ValueError("Duplicate review domain/sample_id")

    held_out = set()
    for name in ("train_expanded_v2.jsonl", "val_expanded_v2.jsonl"):
        held_out.update(Path(row["image"]).resolve() for row in read_jsonl(ROOT / "data" / name))

    selected = {}
    audit = {}
    for domain, seed in (("celeba", 2026), ("th1kh", 2027)):
        manifest = DATA / f"{domain}_candidates_100.jsonl"
        candidates = read_jsonl(manifest)
        if len(candidates) != 100:
            raise ValueError(f"{domain}: expected 100 candidate rows")
        by_id = {row["sample_id"]: row for row in candidates}
        if len(by_id) != 100:
            raise ValueError(f"{domain}: duplicate sample IDs")
        if {(domain, sample_id) for sample_id in by_id} - set(review_by_id):
            raise ValueError(f"{domain}: candidate missing from review")
        approved = []
        for sample_id, candidate in by_id.items():
            row = review_by_id[(domain, sample_id)]
            if row["manual_ok"].strip().lower() != "yes":
                continue
            video = ROOT / "outputs/formal_eval/clean" / domain / sample_id / "clean.mp4"
            status = json.loads((video.parent / "status.json").read_text(encoding="utf-8"))
            if not status.get("generated") or not video.is_file() or video.stat().st_size == 0:
                raise ValueError(f"{domain}/{sample_id}: approved but video is not valid")
            image = Path(candidate["image"]).resolve()
            if not image.is_file() or image in held_out:
                raise ValueError(f"{domain}/{sample_id}: missing image or train/val overlap")
            message = str(candidate["message"])
            if len(message) != 32 or set(message) - {"0", "1"}:
                raise ValueError(f"{domain}/{sample_id}: invalid message")
            approved.append(sample_id)
        if len(approved) < 40:
            raise ValueError(f"{domain}: only {len(approved)} manual approvals; need 40")
        chosen_ids = random.Random(seed).sample(sorted(approved), 40)
        rows = []
        for sample_id in chosen_ids:
            row = dict(by_id[sample_id])
            row.update(image=str(Path(row["image"]).resolve()), audio=str(AUDIO), domain=domain)
            rows.append(row)
        selected[domain] = rows
        audit[domain] = {
            "candidate_count": 100, "manual_approved_count": len(approved),
            "selected_count": 40, "selection_seed": seed,
            "candidate_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }

    combined = selected["celeba"] + selected["th1kh"]
    images = [Path(row["image"]).resolve() for row in combined]
    if len(images) != len(set(images)):
        raise ValueError("Duplicate images in frozen evaluation set")
    payloads = {
        OUTPUTS[domain]: "".join(json.dumps(row, ensure_ascii=False) + "\n"
                                 for row in selected[domain]).encode("utf-8")
        for domain in ("celeba", "th1kh")
    }
    payloads[OUTPUTS["combined"]] = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in combined).encode("utf-8")
    metadata = {
        "audio": str(AUDIO), "review_sha256": hashlib.sha256((DATA / "clean_review.csv").read_bytes()).hexdigest(),
        "domains": audit, "files_sha256": {path.name: hashlib.sha256(data).hexdigest()
                                            for path, data in payloads.items()},
    }
    payloads[OUTPUTS["metadata"]] = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    for path, data in payloads.items():
        path.write_bytes(data)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
