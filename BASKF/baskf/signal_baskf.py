from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if rate != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {rate} Hz: {path}")
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)
    return np.asarray(audio, dtype=np.float32)


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


class CausalSTFT:
    """Non-centered WOLA transform with one-window algorithmic delay."""

    def __init__(self, n_fft: int = 512, win_length: int = 320, hop_length: int = 160):
        if n_fft < win_length:
            raise ValueError("n_fft must be greater than or equal to win_length")
        if win_length % hop_length != 0:
            raise ValueError(
                "v17 requires win_length to be an integer multiple of hop_length "
                "so offline and streaming WOLA remain sample-aligned"
            )
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(win_length, periodic=True).sqrt()

    def _window(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.window.to(device=tensor.device, dtype=tensor.dtype)

    def analysis(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        length = waveform.shape[-1]
        frames = (length + self.hop_length - 1) // self.hop_length
        covered = self.win_length + (frames - 1) * self.hop_length
        padded = F.pad(waveform, (0, max(0, covered - length)))
        framed = padded.unfold(-1, self.win_length, self.hop_length)
        framed = framed * self._window(waveform)
        return torch.fft.rfft(framed, n=self.n_fft, dim=-1)

    def synthesis(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        if spectrum.ndim != 3:
            raise ValueError("Expected [batch, frames, frequency]")
        frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-1)[..., : self.win_length]
        window = self._window(frames)
        frames = frames * window
        count = frames.shape[1]
        covered = self.win_length + (count - 1) * self.hop_length
        columns = frames.transpose(1, 2).contiguous()
        output = F.fold(
            columns,
            output_size=(1, covered),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)
        weights = window.square().view(1, self.win_length, 1).expand(1, -1, count).contiguous()
        denominator = F.fold(
            weights,
            output_size=(1, covered),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze(1).squeeze(1)
        output = output / denominator.clamp_min(1.0e-8)
        return output[..., :length]


