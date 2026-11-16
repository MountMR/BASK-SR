from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import yaml


class Settings(dict):
    def __getattr__(self, key: str) -> Any:
        value = self[key]
        return _wrap(value)


def _wrap(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, Settings):
        return Settings({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def load_settings(path: str | Path) -> Settings:
    with Path(path).open("r", encoding="utf-8") as handle:
        return _wrap(yaml.safe_load(handle))


def load_v17_runtime(settings: Settings, device: torch.device):
    project = Path(settings.paths.v17_project).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from baskf.pipeline_baskf import load_model
    from baskf.signal_baskf import CausalSTFT
    from baskf.utils_baskf import load_config

    v17_cfg = load_config(
        settings.paths.v17_config,
        settings.paths.get("data_root") or None,
        settings.paths.get("v17_work_dir") or None,
    )
    backbone = load_model(v17_cfg, settings.paths.v17_checkpoint, device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    transform = CausalSTFT(
        int(v17_cfg.audio.n_fft), int(v17_cfg.audio.win_length),
        int(v17_cfg.audio.hop_length),
    )
    return v17_cfg, backbone, transform


def run_system(
    backbone,
    postfilter,
    x_spec: torch.Tensor,
    mic_spec: torch.Tensor,
    mix: float = 1.0,
    detach_backbone: bool = True,
    diagnostics: bool = False,
):
    backbone_state = backbone.initial_state(x_spec.shape[0], x_spec.device)
    post_state = postfilter.initial_state(x_spec.shape[0], x_spec.device)
    backbone_errors, outputs, echoes, gains = [], [], [], []
    diagnostic_frames = {
        "near_probability": [], "near_probability_band": [],
        "near_safe_probability": [], "fst_extra_gain_db": [],
        "fst_gate": [], "linear_gate": [],
    } if diagnostics else None
    context = torch.no_grad if detach_backbone else torch.enable_grad
    for frame in range(x_spec.shape[1]):
        with context():
            error, echo, backbone_state, aux = backbone.step(
                x_spec[:, frame], mic_spec[:, frame], backbone_state, mode="track"
            )
        output, post_state, post_aux = postfilter.step(
            x_spec[:, frame], mic_spec[:, frame], error.detach() if detach_backbone else error,
            echo.detach() if detach_backbone else echo, aux, post_state, mix,
        )
        backbone_errors.append(error)
        outputs.append(output)
        echoes.append(echo)
        gains.append(post_aux["gain_bins"])
        if diagnostics:
            for name in diagnostic_frames: diagnostic_frames[name].append(post_aux[name])
    result = (
        torch.stack(outputs, dim=1), torch.stack(backbone_errors, dim=1),
        torch.stack(echoes, dim=1), torch.stack(gains, dim=1),
    )
    if diagnostics:
        return (*result, {name: torch.stack(values, dim=1) for name, values in diagnostic_frames.items()})
    return result


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return payload["postfilter"] if "postfilter" in payload else payload["model"]


def artifact_paths(settings: Settings) -> dict[str, Path]:
    root = Path(settings.paths.work_dir)
    paths = {"root": root, "checkpoints": root / "checkpoints", "results": root / "results", "logs": root / "logs", "audio": root / "audio"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def validation_score(values: dict[str, float], prefix: str = "system") -> float:
    def get(name: str) -> float:
        value = float(values.get(f"{prefix}_{name}_erle_db", float("nan")))
        return value if value == value else -20.0
    return 0.45 * get("fst") + 0.15 * get("dt") + 0.20 * get("epc_st") + 0.20 * get("epc_dt")


@torch.inference_mode()
def validate_spectral(settings, v17_cfg, backbone, postfilter, transform, device, mix: float = 1.0, seed_offset: int = 20_000_000):
    from collections import defaultdict
    from baskf.pipeline_baskf import make_dataset, make_loader, transform_batch

    dataset = make_dataset(v17_cfg, "val", "backbone_val", int(settings.train.validation_items), seed_offset=seed_offset)
    loader = make_loader(dataset, v17_cfg, int(settings.evaluation.batch_size), shuffle=False)
    groups = {"backbone": defaultdict(list), "system": defaultdict(list)}
    gain_means: list[float] = []
    scenario_diagnostics = defaultdict(list)
    postfilter.eval()
    for batch in loader:
        data = transform_batch(batch, transform, device)
        output, backbone_error, _, gain, diagnostics = run_system(backbone, postfilter, data["X"], data["MIC"], mix=mix, diagnostics=True)
        near_active = batch["near"].square().mean(dim=1).gt(1.0e-8).tolist()
        for kind, candidate in (("backbone", backbone_error), ("system", output)):
            residual = candidate - data["TARGET"]
            echo_energy = data["ECHO"].abs().square().sum(dim=(1, 2))
            residual_energy = residual.abs().square().sum(dim=(1, 2))
            erle = 10.0 * torch.log10(echo_energy / residual_energy.clamp_min(1.0e-10))
            for index, value in enumerate(erle.tolist()):
                if echo_energy[index].item() <= 1.0e-8:
                    continue
                scenario = str(batch["scenario"][index])
                names = ["overall", scenario]
                if scenario == "epc":
                    names.append("epc_dt" if near_active[index] else "epc_st")
                for name in names:
                    groups[kind][name].append(float(value))
        per_item_gain = gain.mean(dim=(1, 2)).tolist()
        gain_means.extend(per_item_gain)
        near_power = data["NEAR"].abs().square() @ postfilter.membership.T
        near_threshold = (
            near_power.amax((1, 2), keepdim=True) * float(settings.train.near_label_relative_floor)
        ).clamp_min(1.0e-9)
        near_mask = near_power > near_threshold
        for index, scenario_value in enumerate(batch["scenario"]):
            scenario = str(scenario_value)
            name = "epc_dt" if scenario == "epc" and near_active[index] else "epc_st" if scenario == "epc" else scenario
            scenario_diagnostics[f"gain_mean_{name}"].append(float(per_item_gain[index]))
            for key, values in diagnostics.items():
                scenario_diagnostics[f"{key}_{name}"].append(float(values[index].mean().item()))
            if name in ("dt", "epc_dt") and bool(near_mask[index].any()):
                active = near_mask[index]
                scenario_diagnostics[f"near_band_probability_{name}_active"].append(
                    float(diagnostics["near_probability_band"][index][active].mean().item())
                )
                scenario_diagnostics[f"near_safe_probability_{name}_active"].append(
                    float(diagnostics["near_safe_probability"][index][active].mean().item())
                )
                scenario_diagnostics[f"fst_gate_{name}_active"].append(
                    float(diagnostics["fst_gate"][index][active].mean().item())
                )
    result: dict[str, float] = {}
    for kind, values in groups.items():
        for name in ("overall", "fst", "dt", "epc", "epc_st", "epc_dt"):
            items = values.get(name, [])
            result[f"{kind}_{name}_erle_db"] = sum(items) / len(items) if items else float("nan")
    result["mean_gain"] = sum(gain_means) / len(gain_means) if gain_means else 1.0
    for name, values in scenario_diagnostics.items():
        result[name] = sum(values) / len(values) if values else float("nan")
    for name in ("fst", "dt", "epc_st", "epc_dt"):
        system = groups["system"].get(name, [])
        reference = groups["backbone"].get(name, [])
        paired = sorted(a - b for a, b in zip(system, reference))
        if paired:
            result[f"{name}_gain_p10_db"] = paired[min(len(paired) - 1, int(0.1 * len(paired)))]
            result[f"{name}_regression_fraction"] = sum(value < 0.0 for value in paired) / len(paired)
    result["backbone_score"] = validation_score(result, "backbone")
    result["system_score"] = validation_score(result, "system")
    postfilter.train()
    return result
