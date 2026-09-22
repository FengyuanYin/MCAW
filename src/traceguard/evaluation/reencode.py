from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def reencode_h264(source: str | Path, destination: str | Path, crf: int = 23) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for video re-encoding but was not found on PATH")
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source video not found: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path), "-c:v", "libx264", "-crf", str(crf),
            "-c:a", "aac", str(destination_path),
        ],
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg re-encoding failed with exit code {result.returncode}")
    return destination_path
