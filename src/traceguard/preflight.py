from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import LoadedConfig


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _git_state(path: Path, expected: str) -> list[Check]:
    if not path.is_dir():
        return [Check(path.name, False, f"missing upstream directory: {path}")]
    checks: list[Check] = []
    for name, args in (
        ("commit", ["rev-parse", "HEAD"]),
        ("dirty", ["status", "--porcelain"]),
    ):
        result = subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
        )
        if result.returncode:
            checks.append(Check(f"{path.name}.{name}", False, result.stderr.strip()))
            continue
        value = result.stdout.strip()
        ok = value == expected if name == "commit" else value == ""
        detail = value or "clean"
        checks.append(Check(f"{path.name}.{name}", ok, detail))
    return checks


def run_preflight(config: LoadedConfig) -> list[Check]:
    is_mock = config.get("backend") == "mock"
    checks = [Check("python", sys.version_info[:2] == (3, 10), sys.version.split()[0], required=not is_mock)]
    checks.append(
        Check(
            "ffmpeg",
            shutil.which("ffmpeg") is not None,
            shutil.which("ffmpeg") or "not found",
            required=not is_mock,
        )
    )
    for package in ("torch", "torchvision", "yaml"):
        found = importlib.util.find_spec(package) is not None
        checks.append(Check(f"package.{package}", found, "found" if found else "not installed"))
    try:
        import torch

        cuda = torch.cuda.is_available()
        checks.append(Check("cuda", cuda, torch.version.cuda or "unavailable", required=not is_mock))
        if cuda:
            props = torch.cuda.get_device_properties(0)
            gib = props.total_memory / 1024**3
            checks.append(Check("gpu_memory", gib >= 23.0, f"{props.name}: {gib:.1f} GiB"))
    except (ImportError, OSError) as exc:
        checks.append(Check("torch_runtime", False, str(exc), required=True))
    upstreams = {
        "model.wam.root": "2c08af04d037d5667c02f6ddebbda9ff04581c3e",
        "model.proxy.silencer_root": "78b1c5dc50548f4659944d9694d18a311eeac7c5",
        "model.proxy.hallo_root": "8fd7c572a3d43c2a9c1a5473219ce4fc1b6e3ed2",
    }
    for key, commit in upstreams.items():
        checks.extend(_git_state(config.resolve(key), commit))
    for key in ("model.wam.params", "model.wam.checkpoint", "model.proxy.pretrained_root"):
        path = config.resolve(key)
        checks.append(Check(key, path.exists(), str(path), required=not is_mock))
    return checks


def checks_as_dicts(checks: list[Check]) -> list[dict[str, object]]:
    return [asdict(check) for check in checks]


def preflight_ok(checks: list[Check]) -> bool:
    return all(check.ok for check in checks if check.required)
