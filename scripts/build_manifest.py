from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a")


def message_for(relative_path: str, seed: int, bits: int = 32) -> str:
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).digest()
    binary = "".join(f"{byte:08b}" for byte in digest)
    return binary[:bits]


def find_audio(audio_dir: Path | None, stem: str) -> Path | None:
    if audio_dir is None:
        return None
    for suffix in AUDIO_SUFFIXES:
        candidate = audio_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic TraceGuard JSONL manifest")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio-dir")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="sample")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    if not image_dir.is_dir():
        parser.error(f"image directory not found: {image_dir}")
    if audio_dir is not None and not audio_dir.is_dir():
        parser.error(f"audio directory not found: {audio_dir}")
    iterator = image_dir.rglob("*") if args.recursive else image_dir.glob("*")
    images = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    selected = images[args.offset : None if args.limit is None else args.offset + args.limit]
    if not selected:
        parser.error("no images selected; check --image-dir, --offset and --limit")

    output.parent.mkdir(parents=True, exist_ok=True)
    audio_matches = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, image in enumerate(selected):
            relative = image.relative_to(image_dir).as_posix()
            record = {
                "sample_id": f"{args.prefix}-{args.offset + index:06d}",
                "image": str(image.resolve()),
                "message": message_for(relative, args.seed),
            }
            audio = find_audio(audio_dir, image.stem)
            if audio is not None:
                record["audio"] = str(audio)
                audio_matches += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output),
        "records": len(selected),
        "audio_matches": audio_matches,
        "offset": args.offset,
        "seed": args.seed,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
