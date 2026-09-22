from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "third_party/watermark-anything": "2c08af04d037d5667c02f6ddebbda9ff04581c3e",
    "third_party/Silencer": "78b1c5dc50548f4659944d9694d18a311eeac7c5",
    "third_party/hallo": "8fd7c572a3d43c2a9c1a5473219ce4fc1b6e3ed2",
}


def git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> int:
    failed = False
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.exists():
            print(f"MISSING  {relative}")
            failed = True
            continue
        try:
            actual = git(path, "rev-parse", "HEAD")
            dirty = git(path, "status", "--porcelain")
        except RuntimeError as exc:
            print(f"ERROR    {relative}: {exc}")
            failed = True
            continue
        ok = actual == expected and not dirty
        state = "OK" if ok else "FAILED"
        print(f"{state:7} {relative} commit={actual} dirty={bool(dirty)}")
        failed |= not ok
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
