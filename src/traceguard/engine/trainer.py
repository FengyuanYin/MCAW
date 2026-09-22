from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from traceguard.config import LoadedConfig
from traceguard.data import AuthorizedManifestDataset
from traceguard.distortions import DifferentiableDistortions
from traceguard.factory import build_protector, build_proxy, resolve_device
from traceguard.losses import JointLoss
from traceguard.utils.runtime import append_jsonl, atomic_torch_save, seed_everything


class Trainer:
    def __init__(self, config: LoadedConfig) -> None:
        self.config = config
        self.device = resolve_device(config)
        seed_everything(int(config.get("project.seed", 42)))
        self.output_dir = config.resolve("project.output_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.protector = build_protector(config, self.device)
        self.proxy = build_proxy(config, self.device)
        self.proxy.eval()
        for parameter in self.proxy.parameters():
            parameter.requires_grad_(False)
        self.objective = JointLoss(dict(config.get("loss", {}))).to(self.device)
        distortion = config.get("distortions", {})
        self.distortions = DifferentiableDistortions(
            probability=float(distortion.get("probability", 0.5)),
            jpeg_quality=tuple(distortion.get("jpeg_quality", [55, 95])),
            resize_scale=tuple(distortion.get("resize_scale", [0.75, 1.0])),
            crop_scale=tuple(distortion.get("crop_scale", [0.85, 1.0])),
        ).to(self.device)
        parameters = list(self.protector.trainable_parameters())
        if not parameters:
            raise RuntimeError("No trainable parameters selected")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.get("train.learning_rate", 1e-4)),
            weight_decay=float(config.get("train.weight_decay", 0.01)),
        )
        self.use_amp = self.device.type == "cuda" and config.get("precision") in {"fp16", "bf16"}
        self.amp_dtype = torch.float16 if config.get("precision") == "fp16" else torch.bfloat16
        scaler_enabled = self.use_amp and self.amp_dtype == torch.float16
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.step = 0

    def _loader(self) -> DataLoader:
        dataset = AuthorizedManifestDataset(
            self.config.resolve("train.manifest"),
            int(self.config.require("model.image_size")),
            int(self.config.require("model.message_bits")),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("train.batch_size", 1)),
            shuffle=True,
            num_workers=int(self.config.get("train.num_workers", 2)),
            pin_memory=self.device.type == "cuda",
        )

    def train_step(self, batch: dict[str, object]) -> dict[str, float]:
        images = batch["image"].to(self.device)
        messages = batch["message"].to(self.device)
        self.protector.train()
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            protection = self.protector(images, messages, decode=True)
            proxy_output = self.proxy(images, protection.protected_images)
            adversarial_gradient = torch.autograd.grad(
                proxy_output.loss,
                protection.protected_images,
                retain_graph=True,
                create_graph=False,
            )[0]
            adversarial_direction = -adversarial_gradient.detach()
            distorted = self.distortions(protection.protected_images)
            robust_logits = self.protector.wam.decode(distorted).raw_logits
            losses = self.objective(
                images,
                messages,
                protection,
                proxy_output,
                adversarial_direction,
                robust_logits,
            )
            scaled_loss = losses.total / int(self.config.get("train.gradient_accumulation", 1))
        self.scaler.scale(scaled_loss).backward()
        accumulation = int(self.config.get("train.gradient_accumulation", 1))
        if (self.step + 1) % accumulation == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.protector.trainable_parameters()), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        metrics = losses.detached_metrics()
        metrics.update(proxy_output.metrics)
        metrics["step"] = self.step
        if self.device.type == "cuda":
            metrics["cuda_peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        self.step += 1
        return metrics

    def save_checkpoint(self, name: str | None = None) -> Path:
        path = self.output_dir / "checkpoints" / (name or f"step-{self.step:07d}.pt")
        state = {
            "step": self.step,
            "model": self.protector.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "python_random": random.getstate(),
            "torch_random": torch.get_rng_state(),
            "config": self.config.values,
        }
        if torch.cuda.is_available():
            state["cuda_random"] = torch.cuda.get_rng_state_all()
        atomic_torch_save(state, path)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.protector.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.step = int(state["step"])
        random.setstate(state["python_random"])
        torch.set_rng_state(state["torch_random"])
        if "cuda_random" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_random"])

    def run(self) -> None:
        loader = self._loader()
        maximum = int(self.config.get("train.steps", 10000))
        checkpoint_every = int(self.config.get("train.checkpoint_every", 500))
        iterator = iter(loader)
        self.optimizer.zero_grad(set_to_none=True)
        while self.step < maximum:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            try:
                metrics = self.train_step(batch)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    "CUDA OOM. Reduce model.image_size, keep batch_size=1, increase gradient accumulation, "
                    "or use the silencer latent proxy."
                ) from exc
            append_jsonl(self.output_dir / "metrics.jsonl", metrics)
            print(json.dumps(metrics, sort_keys=True))
            if self.step % checkpoint_every == 0:
                self.save_checkpoint()
        self.save_checkpoint("final.pt")
