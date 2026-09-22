from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    root: Path
    values: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.values
        for part in dotted.split("."):
            if not isinstance(value, Mapping) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise ConfigurationError(f"Missing required configuration key: {dotted}")
        return value

    def resolve(self, dotted: str) -> Path:
        raw = self.require(dotted)
        path = Path(str(raw)).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def _set_dotted(values: dict[str, Any], dotted: str, value: Any) -> None:
    node = values
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigurationError(f"Cannot override non-mapping key: {part}")
        node = child
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> LoadedConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ConfigurationError("Top-level YAML value must be a mapping")
    for item in overrides or []:
        if "=" not in item:
            raise ConfigurationError(f"Override must use key=value syntax: {item}")
        key, raw = item.split("=", 1)
        _set_dotted(values, key, yaml.safe_load(raw))
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    config = LoadedConfig(config_path, root.resolve(), values)
    validate_config(config)
    return config


def validate_config(config: LoadedConfig) -> None:
    bits = int(config.require("model.message_bits"))
    if bits != 32:
        raise ConfigurationError(f"WAM checkpoint expects 32 message bits; received {bits}")
    size = int(config.require("model.image_size"))
    if size <= 0 or size % 8:
        raise ConfigurationError("model.image_size must be positive and divisible by 8")
    epsilon = float(config.require("model.max_linf"))
    if not 0 < epsilon <= 1:
        raise ConfigurationError("model.max_linf must be in (0, 1]")
    if config.get("train.batch_size", 1) < 1:
        raise ConfigurationError("train.batch_size must be at least 1")

