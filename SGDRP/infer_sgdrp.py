from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SGDRP"))
sys.path.insert(0, str(ROOT / "BASKF"))
import torch

from sgdrp.core_sgdrp import UGSRPF
from sgdrp.runtime_sgdrp import checkpoint_state, load_settings, load_v17_runtime, run_system


def main():
    parser = argparse.ArgumentParser(description="Offline BASK-SR inference")
    parser.add_argument("--config", default=str(ROOT / "SGDRP/config_sgdrp.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "SGDRP/weights/sgdrp.pt"))
    parser.add_argument("--farend", required=True); parser.add_argument("--mic", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto"); parser.add_argument("--bypass", action="store_true")
    args = parser.parse_args(); settings = load_settings(args.config)
    # Repository-relative paths keep the bundled configuration portable.
    for key in ("v17_project", "v17_config", "v17_checkpoint", "v17_work_dir", "work_dir", "warm_start_checkpoint"):
        value = Path(settings.paths[key])
        settings.paths[key] = str(value if value.is_absolute() else ROOT / value)
    requested = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device; device = torch.device(requested)
    v17_cfg, backbone, transform = load_v17_runtime(settings, device)
    from baskf.signal_baskf import load_audio, save_audio
    far = load_audio(args.farend, int(v17_cfg.audio.sample_rate)); mic = load_audio(args.mic, int(v17_cfg.audio.sample_rate))
    length = min(far.size, mic.size); far = torch.from_numpy(far[:length]).to(device)[None]; mic = torch.from_numpy(mic[:length]).to(device)[None]
    model = UGSRPF(backbone.membership, backbone.band_centers, settings.postfilter).to(device).eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("version") != "ug_srpf_v9_r1":
        raise RuntimeError("Checkpoint is not compatible with SGDRP")
    model.load_state_dict(checkpoint_state(payload))
    model.dt_common_correction_scale = float(payload.get("dt_correction_scale", model.dt_common_correction_scale))
    model.near_gate_blend = float(payload.get("near_gate_blend", 0.0))
    model.routing_blend = float(payload.get("routing_blend", model.routing_blend))
    model.near_safe_threshold = float(payload.get("near_safe_threshold", model.near_safe_threshold))
    model.near_safe_temperature = float(payload.get("near_safe_temperature", model.near_safe_temperature))
    mix = 0.0 if args.bypass or not payload.get("enabled", True) else float(payload.get("deploy_mix", settings.train.max_mix))
    with torch.inference_mode(): output, _, _, _ = run_system(backbone, model, transform.analysis(far), transform.analysis(mic), mix=mix)
    save_audio(args.output, transform.synthesis(output, length)[0].cpu().numpy(), int(v17_cfg.audio.sample_rate))
    print(f"output={args.output} postfilter={'BYPASS' if mix == 0 else 'ENABLED'} mix={mix:.2f} routing_blend={model.routing_blend:.2f}")


if __name__ == "__main__": main()
