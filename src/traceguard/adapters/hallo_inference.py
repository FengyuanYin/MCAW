from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from traceguard.config import LoadedConfig
from traceguard.errors import WeightNotFoundError


def _runtime_config(config: LoadedConfig, destination: Path) -> Path:
    hallo_root = config.resolve("model.proxy.hallo_root")
    template = hallo_root / "configs" / "inference" / "default.yaml"
    pretrained = config.resolve("model.proxy.pretrained_root")
    if not template.is_file():
        raise FileNotFoundError(f"Hallo inference config not found: {template}")
    if not pretrained.is_dir():
        raise WeightNotFoundError(f"Hallo pretrained model root not found: {pretrained}")
    with template.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values.update(
        {
            "audio_ckpt_dir": str(pretrained / "hallo"),
            "base_model_path": str(pretrained / "stable-diffusion-v1-5"),
            "motion_module_path": str(pretrained / "motion_module" / "mm_sd_v15_v2.ckpt"),
        }
    )
    values["face_analysis"]["model_path"] = str(pretrained / "face_analysis")
    values["wav2vec"]["model_path"] = str(pretrained / "wav2vec" / "wav2vec2-base-960h")
    values["audio_separator"]["model_path"] = str(pretrained / "audio_separator" / "Kim_Vocal_2.onnx")
    values["vae"]["model_path"] = str(pretrained / "sd-vae-ft-mse")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    return destination


def generate_hallo_video(config: LoadedConfig, source_image: Path, driving_audio: Path, output: Path) -> None:
    hallo_root = config.resolve("model.proxy.hallo_root")
    runtime = _runtime_config(config, output.parent / "hallo_runtime.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/inference.py",
        "--config", str(runtime),
        "--source_image", str(source_image.resolve()),
        "--driving_audio", str(driving_audio.resolve()),
        "--output", str(output.resolve()),
    ]
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(hallo_root) + (os.pathsep + existing_python_path if existing_python_path else "")
    result = subprocess.run(command, cwd=hallo_root, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Hallo inference failed with exit code {result.returncode}")
