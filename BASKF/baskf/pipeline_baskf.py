from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .signal_baskf import CausalSTFT
from .baselines_baskf import run_tf_nlms, run_tfdkf
from .data_baskf import AECCatalog, AECChunkDataset, read_split_index
from .models_baskf import ETActKF
from .utils_baskf import Config, ensure_dir


def work_paths(cfg: Config) -> dict[str, Path]:
    root = Path(cfg.paths.work_dir)
    paths = {
        "root": root,
        "manifests": root / "manifests",
        "checkpoints": root / "checkpoints",
        "results": root / "results",
        "audio": root / "audio",
        "logs": root / "logs",
    }
    for path in paths.values():
        ensure_dir(path)
    paths["splits"] = paths["manifests"] / "splits.json"
    paths["fir_bank"] = paths["manifests"] / "fir_bank.pt"
    paths["backbone"] = paths["checkpoints"] / "backbone_best.pt"
    paths["full"] = paths["backbone"]
    return paths


def load_assets(cfg: Config) -> tuple[AECCatalog, Any, dict[str, Any]]:
    paths = work_paths(cfg)
    if not paths["splits"].exists() or not paths["fir_bank"].exists():
        raise FileNotFoundError("Prepared manifests are missing; run scripts/prepare_data.py first")
    catalog = AECCatalog(cfg.paths.data_root, int(cfg.audio.sample_rate))
    splits = read_split_index(paths["splits"])
    fir_bank = torch.load(paths["fir_bank"], map_location="cpu", weights_only=True)
    return catalog, splits, fir_bank


def make_dataset(
    cfg: Config,
    split: str,
    purpose: str,
    length: int,
    seed_offset: int = 0,
) -> AECChunkDataset:
    catalog, splits, fir_bank = load_assets(cfg)
    return AECChunkDataset(
        catalog,
        list(getattr(splits, split)),
        cfg,
        length=length,
        purpose=purpose,
        fir_bank=fir_bank,
        fir_split=split,
        seed_offset=seed_offset,
    )


def make_loader(dataset: AECChunkDataset, cfg: Config, batch_size: int, shuffle: bool) -> DataLoader:
    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    workers = min(int(cfg.data.num_workers), available_cpus, 16)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def load_model(cfg: Config, checkpoint: str | Path, device: torch.device) -> ETActKF:
    model = ETActKF(cfg).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"] if "model" in payload else payload)
    return model


def transform_batch(batch: dict[str, Any], transform: CausalSTFT, device: torch.device) -> dict[str, torch.Tensor]:
    names = ("x", "mic", "echo", "near", "target")
    waves = {name: batch[name].to(device, non_blocking=True) for name in names}
    spectra = {name.upper(): transform.analysis(waves[name]) for name in names}
    return {**waves, **spectra}


@torch.inference_mode()
def run_method(
    method: str,
    x: torch.Tensor,
    mic: torch.Tensor,
    model: ETActKF | None,
    cfg: Config,
    x3: torch.Tensor | None = None,
    x5: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if method == "nlms":
        error, echo = run_tf_nlms(x, mic, taps=int(cfg.audio.taps), step_size=float(cfg.baselines.nlms_step))
        empty = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        return error, echo, {"actions": empty, "invoked": empty.bool()}
    if method == "tfdkf":
        error, echo = run_tfdkf(
            x, mic, taps=int(cfg.audio.taps), process_noise=float(cfg.baselines.tfdkf_process_noise),
            covariance_init=float(cfg.baselines.tfdkf_covariance_init),
        )
        empty = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
        return error, echo, {"actions": empty, "invoked": empty.bool()}
    if model is None:
        raise ValueError(f"Method {method} requires a checkpoint")
    return model.run(x, mic, mode=method)




