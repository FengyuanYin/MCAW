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
    # 模板里是 `./.cache`，会相对工作目录解析。显式写成运行配置所在目录下的子目录
    # （即调用方传入的 output.parent），既避免把中间产物写进 third_party/hallo，
    # 也让每次运行的缓存跟输出放在一起。
    values["save_path"] = str(destination.parent / "hallo_cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(values, handle, sort_keys=False)
    return destination


def generate_hallo_video(config: LoadedConfig, source_image: Path, driving_audio: Path, output: Path) -> None:
    # 调用方传进来的 output 可能是相对路径，而 hallo 子进程的工作目录不是本进程的工作目录，
    # 若原样把 output.parent/"hallo_runtime.yaml" 交给子进程的 --config，它会在错误的工作目录下
    # 解析这个相对路径（曾报 FileNotFoundError: .../third_party/hallo/outputs/video_check/hallo_runtime.yaml）。
    # 这里统一解析成绝对路径，后面的工作目录变化就不会影响任何路径。
    output = output.resolve()
    hallo_root = config.resolve("model.proxy.hallo_root")
    runtime = _runtime_config(config, output.parent / "hallo_runtime.yaml")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        # 必须用绝对路径：下面把工作目录设成项目根目录，而不是 hallo_root。
        str(hallo_root / "scripts" / "inference.py"),
        "--config", str(runtime),
        "--source_image", str(source_image.resolve()),
        "--driving_audio", str(driving_audio.resolve()),
        "--output", str(output.resolve()),
    ]
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(hallo_root) + (os.pathsep + existing_python_path if existing_python_path else "")
    # hallo/utils/util.py 的 get_landmark() 把 mediapipe 的 landmark 模型写成了相对路径
    # "pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task"，
    # 而本项目的 pretrained_models 在仓库根目录、不在 third_party/hallo 下。
    # 因此用 config.root 作为工作目录，让这个相对路径命中项目根的 pretrained_models。
    # 配置里其余路径都由 _runtime_config 写成了绝对路径，不受工作目录影响。
    result = subprocess.run(command, cwd=config.root, env=environment, check=False)
    if result.returncode:
        raise RuntimeError(f"Hallo inference failed with exit code {result.returncode}")
