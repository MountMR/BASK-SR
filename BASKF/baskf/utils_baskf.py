from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class Config(dict):
    """Dictionary with recursive attribute access."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _as_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _as_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_config(v) for v in value]
    return value


def load_config(path: str | Path, data_root: str | None = None, work_dir: str | None = None) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = _as_config(yaml.safe_load(handle))
    if data_root:
        cfg.paths.data_root = data_root
    if work_dir:
        cfg.paths.work_dir = work_dir
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def save_json(value: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def checkpoint_payload(model: torch.nn.Module, cfg: Config, step: int, score: float) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "step": int(step),
        "score": float(score),
        "config": dict(cfg),
        "torch_version": torch.__version__,
    }


def atomic_torch_save(value: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, target)


