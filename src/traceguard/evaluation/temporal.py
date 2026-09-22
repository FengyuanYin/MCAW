from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from traceguard.errors import DependencyError
from traceguard.utils.images import message_to_string


@dataclass
class TemporalDecodeResult:
    message: str
    bit_probabilities: list[float]
    bit_confidences: list[float]
    detection_score: float
    valid_frame_ratio: float
    frames: list[dict[str, object]]

    def as_dict(self) -> dict[str, object]:
        return {
            "message": self.message,
            "bit_probabilities": self.bit_probabilities,
            "bit_confidences": self.bit_confidences,
            "detection_score": self.detection_score,
            "valid_frame_ratio": self.valid_frame_ratio,
            "frames": self.frames,
        }


class TemporalDecoder:
    def __init__(self, wam, detection_threshold: float = 0.5) -> None:
        self.wam = wam
        self.detection_threshold = detection_threshold

    def decode_frames(self, frames: torch.Tensor) -> TemporalDecodeResult:
        output = self.wam.decode(frames)
        valid = output.detection_scores >= self.detection_threshold
        confidences = output.bit_confidences * output.detection_scores[:, None]
        if valid.any():
            probabilities = (output.bit_probabilities[valid] * confidences[valid]).sum(0) / confidences[valid].sum(0).clamp_min(1e-6)
            detection = float(output.detection_scores[valid].mean())
        else:
            probabilities = output.bit_probabilities.mean(0)
            detection = float(output.detection_scores.mean())
        bits = probabilities >= 0.5
        per_frame = []
        for index in range(frames.shape[0]):
            per_frame.append(
                {
                    "frame": index,
                    "message": message_to_string(output.bits[index]),
                    "detection_score": float(output.detection_scores[index]),
                    "valid": bool(valid[index]),
                }
            )
        return TemporalDecodeResult(
            message=message_to_string(bits),
            bit_probabilities=probabilities.detach().cpu().tolist(),
            bit_confidences=((probabilities - 0.5).abs() * 2).detach().cpu().tolist(),
            detection_score=detection,
            valid_frame_ratio=float(valid.float().mean()),
            frames=per_frame,
        )

    def decode_video(self, path: str | Path, stride: int = 4, max_frames: int = 32, device: str = "cpu") -> TemporalDecodeResult:
        try:
            import cv2
        except ImportError as exc:
            raise DependencyError("Video decoding requires opencv-python-headless; install traceguard[metrics]") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        frames = []
        frame_index = 0
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0))
            frame_index += 1
        capture.release()
        if not frames:
            raise ValueError(f"No decodable frames found in {path}")
        return self.decode_frames(torch.stack(frames).to(device))
