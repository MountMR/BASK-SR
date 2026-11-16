from __future__ import annotations

import torch


@torch.no_grad()
def run_tf_nlms(x: torch.Tensor, y: torch.Tensor, taps: int = 8, step_size: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, frequencies = x.shape
    history = torch.zeros(batch, frequencies, taps, dtype=x.dtype, device=x.device)
    weights = torch.zeros_like(history)
    errors, echoes = [], []
    for frame in range(frames):
        history = torch.cat([x[:, frame].unsqueeze(-1), history[..., :-1]], dim=-1)
        echo = (history * weights).sum(dim=-1)
        error = y[:, frame] - echo
        denominator = history.abs().square().sum(dim=-1).clamp_min(1.0e-6)
        weights = weights + step_size * history.conj() * error.unsqueeze(-1) / denominator.unsqueeze(-1)
        errors.append(error)
        echoes.append(echo)
    return torch.stack(errors, 1), torch.stack(echoes, 1)


@torch.no_grad()
def run_tfdkf(
    x: torch.Tensor,
    y: torch.Tensor,
    taps: int = 8,
    process_noise: float = 1.0e-4,
    covariance_init: float = 1.0e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, frequencies = x.shape
    history = torch.zeros(batch, frequencies, taps, dtype=x.dtype, device=x.device)
    weights = torch.zeros_like(history)
    covariance = torch.full((batch, frequencies, taps), covariance_init, device=x.device)
    observation = torch.full((batch, frequencies), 1.0e-3, device=x.device)
    errors, echoes = [], []
    for frame in range(frames):
        history = torch.cat([x[:, frame].unsqueeze(-1), history[..., :-1]], dim=-1)
        prior = (covariance + process_noise).clamp(1.0e-8, 1.0)
        echo = (history * weights).sum(dim=-1)
        error = y[:, frame] - echo
        observation = 0.95 * observation + 0.05 * error.abs().square()
        denominator = (prior * history.abs().square()).sum(dim=-1) + observation
        gain = prior * history.conj() / denominator.clamp_min(1.0e-8).unsqueeze(-1)
        mu = (gain * history).real.sum(dim=-1).clamp_min(0.0)
        scale = torch.minimum(torch.ones_like(mu), 0.7 / mu.clamp_min(1.0e-8))
        gain = gain * scale.unsqueeze(-1)
        weights = weights + gain * error.unsqueeze(-1)
        response = (gain * history).real
        covariance = ((1.0 - response).square() * prior + gain.abs().square() * observation.unsqueeze(-1)).clamp(1.0e-8, 1.0)
        errors.append(error)
        echoes.append(echo)
    return torch.stack(errors, 1), torch.stack(echoes, 1)



