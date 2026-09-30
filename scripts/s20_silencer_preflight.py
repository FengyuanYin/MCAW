"""Read-only audit of upstream Silencer-Hallo dependencies and model files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path


PACKAGES = (
    "advertorch", "hydra-core", "mlflow", "diffusers", "transformers", "torch",
    "torchvision", "mediapipe", "insightface", "onnxruntime-gpu", "xformers",
    "accelerate", "audio-separator", "omegaconf",
)
MODEL_PATHS = (
    "stable-diffusion-v1-5", "sd-vae-ft-mse", "face_analysis",
    "motion_module/mm_sd_v15_v2.ckpt", "wav2vec/wav2vec2-base-960h",
    "audio_separator/Kim_Vocal_2.onnx", "hallo",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--silencer-root", type=Path,
                        default=Path("third_party/Silencer/Silencer-I"))
    args = parser.parse_args()
    project = args.project_root.resolve()
    silencer = (project / args.silencer_root).resolve()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = {name: str(project / "pretrained_models" / name)
             if (project / "pretrained_models" / name).exists() else None
             for name in MODEL_PATHS}
    result = {
        "python_packages": packages,
        "model_paths": paths,
        "upstream_entrypoint": str(silencer / "protect" / "protect_hallo.py")
            if (silencer / "protect" / "protect_hallo.py").is_file() else None,
        "upstream_hallo_config": str(silencer / "hallo" / "configs" / "train" / "stage2.yaml")
            if (silencer / "hallo" / "configs" / "train" / "stage2.yaml").is_file() else None,
        "upstream_target": str(silencer / "protect" / "test_images" / "target" / "MIST.png")
            if (silencer / "protect" / "test_images" / "target" / "MIST.png").is_file() else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
