from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import fftconvolve
from torch.utils.data import Dataset

from .signal_baskf import load_audio
from .utils_baskf import Config, atomic_torch_save, load_json, save_json


@dataclass(frozen=True)
class SplitIndex:
    train: list[int]
    val: list[int]
    test: list[int]


class AECCatalog:
    def __init__(self, root: str | Path, sample_rate: int = 16000):
        self.root = Path(root)
        self.sample_rate = sample_rate
        meta_path = self.root / "meta.csv"
        if not meta_path.exists():
            candidates = [path for path in self.root.rglob("meta.csv") if (path.parent / "synthetic_farend").is_dir()]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"Expected one AEC Challenge synthetic root below {self.root}; found {len(candidates)}"
                )
            self.root, meta_path = candidates[0].parent, candidates[0]
        self.meta = pd.read_csv(meta_path).set_index("fileid", drop=False)

    def path(self, kind: str, file_id: int) -> Path:
        templates = {
            "farend": ("synthetic_farend", "farend_speech_fileid_{}.flac"),
            "echo": ("synthetic_echo", "echo_fileid_{}.flac"),
            "near": ("synthetic_nearend_speech", "nearend_speech_fileid_{}.flac"),
            "mic": ("synthetic_nearend_mic", "nearend_mic_fileid_{}.flac"),
        }
        directory, filename = templates[kind]
        return self.root / directory / filename.format(file_id)

    def load(self, file_id: int) -> dict[str, np.ndarray]:
        values = {key: load_audio(self.path(key, file_id), self.sample_rate) for key in ("farend", "echo", "near", "mic")}
        size = min(array.size for array in values.values())
        values = {key: array[:size] for key, array in values.items()}
        # Estimate the actual near-end scale by projection; this is robust whether the
        # stored clean near-end file is pre-scaled or not.
        residual = values["mic"] - values["echo"]
        denominator = float(np.dot(values["near"], values["near"])) + 1.0e-8
        alpha = float(np.clip(np.dot(residual, values["near"]) / denominator, 0.05, 2.0))
        values["near"] = values["near"] * alpha
        values["noise"] = values["mic"] - values["echo"] - values["near"]
        return values

    def ids(self, split: str) -> list[int]:
        rows = self.meta[self.meta["split"] == split]
        ids = [int(value) for value in rows["fileid"].tolist()]
        return [
            file_id for file_id in ids
            if all(self.path(kind, file_id).exists() for kind in ("farend", "echo", "near", "mic"))
        ]

    def nonlinear(self, file_id: int) -> bool:
        return bool(int(self.meta.loc[file_id, "is_farend_nonlinear"]))


def build_split_index(catalog: AECCatalog, val_size: int, seed: int) -> SplitIndex:
    train_ids = np.asarray(catalog.ids("train"), dtype=np.int64)
    if train_ids.size < max(20, 2 * val_size):
        raise RuntimeError(f"Only {train_ids.size} complete training examples found below {catalog.root}")
    test_ids = catalog.ids("test")
    rng = np.random.default_rng(seed)
    # Stratify by nonlinearity, noise flags and a coarse SER bin.
    rows = catalog.meta.loc[train_ids].copy()
    rows["ser_bin"] = (rows["ser"].astype(float) // 5).astype(int)
    groups = rows.groupby(["is_farend_nonlinear", "is_farend_noisy", "is_nearend_noisy", "ser_bin"])
    val: list[int] = []
    for _, group in groups:
        ids = group["fileid"].astype(int).to_numpy().copy()
        rng.shuffle(ids)
        take = max(1, int(round(len(ids) * val_size / len(train_ids))))
        val.extend(ids[:take].tolist())
    if len(val) > val_size:
        rng.shuffle(val)
        val = val[:val_size]
    elif len(val) < val_size:
        remaining = list(set(train_ids.tolist()) - set(val))
        rng.shuffle(remaining)
        val.extend(remaining[: val_size - len(val)])
    val_set = set(val)
    train = [int(value) for value in train_ids if int(value) not in val_set]
    if not test_ids:
        rng.shuffle(train)
        holdout = min(val_size, max(2, len(train) // 10))
        test_ids, train = train[:holdout], train[holdout:]
    return SplitIndex(train=train, val=sorted(val), test=sorted(test_ids))


def save_split_index(index: SplitIndex, path: str | Path) -> None:
    save_json({"train": index.train, "val": index.val, "test": index.test}, path)


def read_split_index(path: str | Path) -> SplitIndex:
    value = load_json(path)
    return SplitIndex(train=value["train"], val=value["val"], test=value["test"])


def _estimate_bulk_delay(
    x: np.ndarray,
    d: np.ndarray,
    max_delay: int = 8000,
    guard: int = 64,
) -> int:
    """Estimate the non-negative far-end-to-echo bulk delay."""
    size = min(x.size, d.size)
    probe = min(size, 80000)
    if probe < 2:
        return 0
    x_probe = x[:probe].astype(np.float64, copy=False)
    d_probe = d[:probe].astype(np.float64, copy=False)
    x_probe = x_probe - x_probe.mean()
    d_probe = d_probe - d_probe.mean()
    fft_size = 1 << (2 * probe - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(d_probe, fft_size) * np.conj(np.fft.rfft(x_probe, fft_size)),
        fft_size,
    )
    upper = min(int(max_delay), probe - 1)
    delay = int(np.argmax(np.abs(correlation[: upper + 1])))
    return max(0, delay - int(guard))


def _estimate_fir(x: np.ndarray, d: np.ndarray, length: int, n_fft: int, hop: int, ridge: float) -> tuple[np.ndarray, float]:
    """Estimate a delay-aligned FIR and its full-signal fit ERLE."""
    size = min(x.size, d.size)
    x = np.asarray(x[:size], dtype=np.float64)
    d = np.asarray(d[:size], dtype=np.float64)
    bulk_delay = _estimate_bulk_delay(x, d)
    aligned_size = size - bulk_delay
    if aligned_size < n_fft:
        return np.zeros(length, dtype=np.float32), float("-inf")

    xa = x[:aligned_size]
    da = d[bulk_delay : bulk_delay + aligned_size]
    window = np.sqrt(np.maximum(np.hanning(n_fft), 1.0e-8))
    numerator = np.zeros(n_fft // 2 + 1, dtype=np.complex128)
    denominator = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    frames = 1 + (aligned_size - n_fft) // hop
    for index in range(frames):
        frame_start = index * hop
        xf = np.fft.rfft(xa[frame_start : frame_start + n_fft] * window)
        df = np.fft.rfft(da[frame_start : frame_start + n_fft] * window)
        numerator += df * xf.conj()
        denominator += np.abs(xf) ** 2

    regularizer = max(ridge * float(np.mean(denominator)), np.finfo(np.float64).eps)
    impulse = np.fft.irfft(numerator / (denominator + regularizer), n=n_fft).real
    search_length = min(length, n_fft // 2)
    peak = int(np.argmax(np.abs(impulse[:search_length])))
    crop_start = max(0, peak - 64)
    fir = np.zeros(length, dtype=np.float64)
    available = impulse[crop_start : crop_start + length]
    fir[: available.size] = available

    total_delay = bulk_delay + crop_start
    convolution = fftconvolve(x, fir, mode="full")
    estimate = np.zeros(size, dtype=np.float64)
    available_samples = min(max(0, size - total_delay), convolution.size)
    if available_samples:
        estimate[total_delay : total_delay + available_samples] = convolution[:available_samples]

    scale = float(np.dot(d, estimate) / (np.dot(estimate, estimate) + np.finfo(np.float64).eps))
    fir *= scale
    estimate *= scale
    fit = 10.0 * math.log10(
        (float(np.dot(d, d)) + 1.0e-12)
        / (float(np.dot(d - estimate, d - estimate)) + 1.0e-12)
    )
    return fir.astype(np.float32), fit


def build_fir_bank(catalog: AECCatalog, splits: SplitIndex, cfg: Config, output_path: str | Path) -> dict[str, Any]:
    settings = cfg.data.fir
    rng = np.random.default_rng(cfg.seed)
    result: dict[str, Any] = {}
    requested = {"train": settings.train_count, "val": settings.val_count, "test": settings.test_count}
    total = sum(int(count) for count in requested.values())

    # Build all acoustic paths from the large training partition, then assign
    # disjoint paths to the three banks. Evaluation speech remains strictly in
    # its original split, while scarce validation/test paths no longer force a
    # relaxation of the FIR-quality threshold.
    linear = [file_id for file_id in splits.train if not catalog.nonlinear(file_id)]
    rng.shuffle(linear)
    good: list[tuple[float, int, np.ndarray]] = []
    scan_limit = min(len(linear), max(total * 20, total))
    for file_id in linear[:scan_limit]:
        values = catalog.load(file_id)
        fir, fit = _estimate_fir(
            values["farend"], values["echo"], int(settings.length), int(settings.n_fft),
            int(settings.hop), float(settings.ridge),
        )
        if fit >= float(settings.min_fit_erle_db):
            good.append((fit, file_id, fir))
            if len(good) >= total:
                break

    if len(good) < total:
        best = max((item[0] for item in good), default=float("-inf"))
        raise RuntimeError(
            f"Only {len(good)}/{total} disjoint FIRs satisfy the "
            f"{float(settings.min_fit_erle_db):.1f} dB threshold; best={best:.2f} dB"
        )

    good.sort(key=lambda item: item[0], reverse=True)
    offset = 0
    for split_name, count in requested.items():
        count = int(count)
        selected = good[offset : offset + count]
        offset += count
        result[split_name] = {
            "fir": torch.from_numpy(np.stack([item[2] for item in selected])),
            "file_id": torch.tensor([item[1] for item in selected], dtype=torch.long),
            "fit_erle_db": torch.tensor([item[0] for item in selected], dtype=torch.float32),
        }
    atomic_torch_save(result, output_path)
    return result


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal**2) + 1.0e-10))


def _scale_to(reference: np.ndarray, signal: np.ndarray, ratio_db: float) -> np.ndarray:
    target = _rms(reference) * 10.0 ** (ratio_db / 20.0)
    return signal * (target / max(_rms(signal), 1.0e-8))


class AECChunkDataset(Dataset):
    """Deterministic indexed sampler for static and path-change AEC chunks."""

    def __init__(
        self,
        catalog: AECCatalog,
        file_ids: list[int],
        cfg: Config,
        length: int,
        purpose: str = "train",
        fir_bank: dict[str, Any] | None = None,
        fir_split: str = "train",
        seed_offset: int = 0,
    ):
        self.catalog = catalog
        self.file_ids = list(file_ids)
        self.cfg = cfg
        self.length = int(length)
        self.purpose = purpose
        self.seed_offset = seed_offset
        seconds = cfg.audio.dynamic_seconds if purpose == "dynamic" else cfg.audio.chunk_seconds
        self.samples = int(round(seconds * cfg.audio.sample_rate))
        self.firs = None if fir_bank is None else fir_bank[fir_split]["fir"].cpu().numpy()
        self.linear_ids = [value for value in self.file_ids if not catalog.nonlinear(value)]
        self.nonlinear_ids = [value for value in self.file_ids if catalog.nonlinear(value)]

    def __len__(self) -> int:
        return self.length

    def _choose_id(self, rng: np.random.Generator) -> int:
        nonlinear = rng.random() < float(self.cfg.data.nonlinear_probability)
        pool = self.nonlinear_ids if nonlinear and self.nonlinear_ids else self.linear_ids
        if not pool:
            pool = self.file_ids
        return int(pool[int(rng.integers(0, len(pool)))])

    def _crop(self, values: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        size = min(array.size for array in values.values())
        if size < self.samples:
            return {key: np.pad(array, (0, self.samples - size)) for key, array in values.items()}
        start = int(rng.integers(0, size - self.samples + 1))
        return {key: array[start : start + self.samples].copy() for key, array in values.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(self.cfg.seed + self.seed_offset + index * 104729))
        file_id = self._choose_id(rng)
        values = self._crop(self.catalog.load(file_id), rng)
        x, d, near, noise = values["farend"], values["echo"], values["near"], values["noise"]
        is_nonlinear = self.catalog.nonlinear(file_id)
        fir_target: np.ndarray | None = None

        controlled = self.purpose in {
            "backbone_static", "backbone_mixed", "backbone_val",
            "backbone_v7_teacher", "backbone_v7_fst", "backbone_v7_mixed",
            "backbone_v8_gate", "backbone_v8_repair", "backbone_v8_joint",
            "backbone_test", "capacity"
        }
        if self.purpose == "dynamic":
            scenario = "epc"
        elif self.purpose == "backbone_static":
            scenario = str(rng.choice(["fst", "dt"], p=[0.65, 0.35]))
        elif self.purpose == "backbone_mixed":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.45, 0.35, 0.20]))
        elif self.purpose == "backbone_v7_teacher":
            scenario = str(rng.choice(["fst", "dt"], p=[0.70, 0.30]))
        elif self.purpose == "backbone_v7_fst":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.60, 0.30, 0.10]))
        elif self.purpose == "backbone_v7_mixed":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.40, 0.30, 0.30]))
        elif self.purpose == "backbone_v8_gate":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.20, 0.55, 0.25]))
        elif self.purpose == "backbone_v8_repair":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.35, 0.40, 0.25]))
        elif self.purpose == "backbone_v8_joint":
            scenario = str(rng.choice(["fst", "dt", "epc"], p=[0.45, 0.30, 0.25]))
        elif self.purpose == "backbone_val":
            scenario = ("fst", "dt", "epc", "epc")[index % 4]
        elif self.purpose == "backbone_test":
            scenario = ("fst", "dt")[index % 2]
        elif self.purpose == "capacity":
            scenario = "fst"
        elif self.purpose == "static":
            scenarios = ["fst", "dt", "nst"]
            weights = np.asarray(
                [self.cfg.data.scenario_probabilities[name] for name in scenarios],
                dtype=float,
            )
            weights /= weights.sum()
            scenario = str(rng.choice(scenarios, p=weights))
        else:
            scenarios = list(self.cfg.data.scenario_probabilities.keys())
            weights = np.asarray(
                list(self.cfg.data.scenario_probabilities.values()), dtype=float
            )
            weights /= weights.sum()
            scenario = str(rng.choice(scenarios, p=weights))

        switch_sample = -1
        if controlled or scenario == "epc":
            if self.firs is None or len(self.firs) < 2:
                raise RuntimeError("Controlled backbone sampling requires a FIR bank")
            if self.purpose in {"backbone_mixed", "backbone_v7_fst", "backbone_v7_mixed", "backbone_v8_gate", "backbone_v8_repair", "backbone_v8_joint", "backbone_test"}:
                nonlinear_probability = float(self.cfg.data.backbone_nonlinear_probability)
            elif self.purpose == "dynamic":
                nonlinear_probability = float(self.cfg.data.nonlinear_probability)
            else:
                nonlinear_probability = 0.0
            is_nonlinear = rng.random() < nonlinear_probability
            x_effective = x
            if is_nonlinear:
                gain = float(rng.uniform(1.5, 3.0))
                x_effective = np.tanh(gain * x) / math.tanh(gain)
            first = int(rng.integers(0, len(self.firs)))
            if self.purpose == "capacity":
                # The FIR preflight must use the exact impulse response that
                # generated this deterministic clean-FST echo. Returning it
                # only for capacity samples avoids adding an unused tensor to
                # every regular training batch.
                fir_target = np.asarray(self.firs[first], dtype=np.float32).copy()
            d1 = fftconvolve(x_effective, self.firs[first], mode="full")[: self.samples]
            if scenario == "epc":
                second = int(rng.integers(0, len(self.firs) - 1))
                if second >= first:
                    second += 1
                d2 = fftconvolve(x_effective, self.firs[second], mode="full")[: self.samples]
                d2 *= 10.0 ** (float(rng.uniform(-3.0, 3.0)) / 20.0)
                lo, hi = self.cfg.data.switch_seconds
                switch_sample = int(rng.uniform(lo, hi) * self.cfg.audio.sample_rate)
                d = d1.astype(np.float32)
                d[switch_sample:] = d2[switch_sample:]
                if self.purpose == "dynamic" and rng.random() >= 0.6:
                    near.fill(0.0)
                elif self.purpose == "backbone_val" and index % 4 == 2:
                    near.fill(0.0)
                elif self.purpose in {"backbone_v7_fst", "backbone_v7_mixed", "backbone_v8_gate", "backbone_v8_repair", "backbone_v8_joint"} and rng.random() < 0.40:
                    near.fill(0.0)
            else:
                d = d1.astype(np.float32)

        if scenario == "fst":
            near.fill(0.0)
        elif scenario == "nst":
            x.fill(0.0)
            d.fill(0.0)

        noise_probability = 0.0 if self.purpose == "capacity" else float(self.cfg.data.noise_probability)
        if rng.random() >= noise_probability:
            noise.fill(0.0)

        if _rms(d) > 1.0e-6 and _rms(near) > 1.0e-6:
            ser = float(rng.uniform(*self.cfg.data.ser_db))
            near = _scale_to(d, near, ser)
        reference_for_noise = near if _rms(near) > 1.0e-6 else d
        if _rms(reference_for_noise) > 1.0e-6 and _rms(noise) > 1.0e-6:
            enr = float(rng.uniform(*self.cfg.data.enr_db))
            noise = _scale_to(reference_for_noise, noise, -enr)

        mic = d + near + noise
        peak = max(float(np.max(np.abs(mic))), float(np.max(np.abs(x))), 1.0)
        x, d, near, noise, mic = (array / peak for array in (x, d, near, noise, mic))
        result = {
            "x": torch.from_numpy(np.asarray(x, dtype=np.float32)),
            "mic": torch.from_numpy(np.asarray(mic, dtype=np.float32)),
            "echo": torch.from_numpy(np.asarray(d, dtype=np.float32)),
            "near": torch.from_numpy(np.asarray(near, dtype=np.float32)),
            "target": torch.from_numpy(np.asarray(near + noise, dtype=np.float32)),
            "switch_sample": torch.tensor(switch_sample, dtype=torch.long),
            "scenario": scenario,
            "file_id": torch.tensor(file_id, dtype=torch.long),
            "is_nonlinear": torch.tensor(is_nonlinear, dtype=torch.bool),
        }
        if fir_target is not None:
            result["fir_target"] = torch.from_numpy(fir_target)
        return result



