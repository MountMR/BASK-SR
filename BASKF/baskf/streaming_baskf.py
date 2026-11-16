from __future__ import annotations

import torch

from .models_baskf import ETActKF, KFState


class StreamingAEC:
    """Hop-by-hop, strictly causal WOLA wrapper for a batch-one ET-ActKF."""

    def __init__(self, model: ETActKF, mode: str = "track", device: torch.device | None = None):
        if mode != "track":
            raise ValueError("Bounded_muq is controller-free and only supports mode='track'")
        self.model = model.eval()
        self.cfg = model.cfg
        self.device = device or next(model.parameters()).device
        self.mode = mode
        self.win = int(self.cfg.audio.win_length)
        self.hop = int(self.cfg.audio.hop_length)
        self.n_fft = int(self.cfg.audio.n_fft)
        self.window = torch.hann_window(self.win, periodic=True, device=self.device).sqrt()
        self.reset()

    def reset(self) -> None:
        self.x_buffer = torch.zeros(self.win, device=self.device)
        self.y_buffer = torch.zeros(self.win, device=self.device)
        self.overlap = torch.zeros(self.win - self.hop, device=self.device)
        self.weight = torch.zeros(self.win - self.hop, device=self.device)
        self.state: KFState = self.model.initial_state(1, self.device)
        self.received = 0

    @torch.inference_mode()
    def push(self, farend_hop: torch.Tensor, mic_hop: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        farend_hop = torch.as_tensor(farend_hop, dtype=torch.float32, device=self.device).flatten()
        mic_hop = torch.as_tensor(mic_hop, dtype=torch.float32, device=self.device).flatten()
        if farend_hop.numel() != self.hop or mic_hop.numel() != self.hop:
            raise ValueError(f"push expects exactly {self.hop} samples per channel")
        self.x_buffer = torch.roll(self.x_buffer, -self.hop)
        self.y_buffer = torch.roll(self.y_buffer, -self.hop)
        self.x_buffer[-self.hop :] = farend_hop
        self.y_buffer[-self.hop :] = mic_hop
        self.received += self.hop
        if self.received < self.win:
            return torch.zeros(self.hop, device=self.device), None

        x = torch.fft.rfft(self.x_buffer * self.window, n=self.n_fft).unsqueeze(0)
        y = torch.fft.rfft(self.y_buffer * self.window, n=self.n_fft).unsqueeze(0)
        error, _, self.state, aux = self.model.step(x, y, self.state, mode=self.mode)
        frame = torch.fft.irfft(error.squeeze(0), n=self.n_fft)[: self.win] * self.window
        numerator = frame.clone()
        denominator = self.window.square().clone()
        tail = self.win - self.hop
        numerator[:tail] += self.overlap
        denominator[:tail] += self.weight
        output = numerator[: self.hop] / denominator[: self.hop].clamp_min(1.0e-8)
        self.overlap = numerator[self.hop :]
        self.weight = denominator[self.hop :]
        return output, aux

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        output, _ = self.push(torch.zeros(self.hop, device=self.device), torch.zeros(self.hop, device=self.device))
        return output



