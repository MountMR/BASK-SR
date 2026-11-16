from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils_baskf import Config


class Action(IntEnum):
    TRACK = 0
    HOLD = 1
    REACQUIRE = 2


class StatisticsNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # v10: compact v7-compatible recurrent core. The shared GRU
        # uses 25 units; analytic Kalman recursion and feature geometry stay fixed.
        self.hidden_size = 25
        self.input = nn.Linear(20, 24)
        self.global_input = nn.Linear(4, 24, bias=False)
        self.norm1 = nn.LayerNorm(24)
        self.gru = nn.GRU(24, self.hidden_size, batch_first=True)
        self.output = nn.Linear(self.hidden_size, 16)
        self.norm2 = nn.LayerNorm(16)
        self.head = nn.Linear(16, 4)
        nn.init.zeros_(self.global_input.weight)
        nn.init.zeros_(self.head.weight[2:])
        nn.init.zeros_(self.head.bias[2:])
        nn.init.constant_(self.head.bias[3], 4.0)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, bands, _ = features.shape
        global_features = torch.stack(
            [features[..., 0].mean(-1), features[..., 1].mean(-1),
             features[..., 2].mean(-1), features[..., 6].mean(-1)], dim=-1
        )
        context = self.global_input(global_features)[:, None, :].expand(-1, bands, -1)
        value = F.silu(self.norm1(self.input(features) + context)).reshape(batch * bands, 1, 24)
        state = hidden.reshape(1, batch * bands, self.hidden_size).contiguous()
        value, state = self.gru(value, state)
        value = F.silu(self.norm2(self.output(value[:, 0])))
        logits = self.head(value).reshape(batch, bands, 4)
        return logits, state[0].reshape(batch, bands, self.hidden_size)


@dataclass(frozen=True)
class ControlDecision:
    """Parameter-free boundary between the v17 backbone and a future controller."""

    action: torch.Tensor
    invoked: torch.Tensor


class ControllerInterface(Protocol):
    """A future controller may implement this without changing the backbone."""

    def decide(
        self,
        base: torch.Tensor,
        flags: torch.Tensor,
        far_active: torch.Tensor,
        strong: torch.Tensor,
    ) -> ControlDecision: ...


class TrackController:
    """Default v17 policy: always run the controller-free TRACK path."""

    def decide(
        self,
        base: torch.Tensor,
        flags: torch.Tensor,
        far_active: torch.Tensor,
        strong: torch.Tensor,
    ) -> ControlDecision:
        batch = base.shape[0]
        return ControlDecision(
            action=torch.full((batch,), int(Action.TRACK), device=base.device, dtype=torch.long),
            invoked=torch.zeros(batch, device=base.device, dtype=torch.bool),
        )


@dataclass
class KFState:
    h: torch.Tensor
    covariance: torch.Tensor
    process: torch.Tensor
    x_history: torch.Tensor
    powers_short: torch.Tensor
    powers_long: torch.Tensor
    x_power_taps: torch.Tensor
    cross_y: torch.Tensor
    cross_e: torch.Tensor
    stat_hidden: torch.Tensor
    previous_doubletalk: torch.Tensor
    previous_power_event: torch.Tensor
    previous_mu: torch.Tensor
    previous_update: torch.Tensor
    previous_delta: torch.Tensor
    frame_index: int = 0

    def detach(self) -> "KFState":
        values: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.detach() if torch.is_tensor(value) else value
        return KFState(**values)


def _band_edges(sample_rate: int, bands: int) -> torch.Tensor:
    if sample_rate == 16000 and bands == 24:
        return torch.tensor(
            [0, 125, 250, 375, 500, 625, 750, 1000, 1250, 1500, 1750, 2000, 2250,
             2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8001],
            dtype=torch.float32,
        )
    return torch.linspace(0, sample_rate / 2 + 1, bands + 1)


class ETActKF(nn.Module):
    """Controller-free v17 backbone with an injectable control interface."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.freqs = cfg.audio.n_fft // 2 + 1
        self.taps = int(cfg.audio.taps)
        self.bands = int(cfg.audio.bands)
        self.stats_net = StatisticsNet()
        self.control_policy: ControllerInterface = TrackController()

        frequency = torch.arange(self.freqs) * cfg.audio.sample_rate / cfg.audio.n_fft
        edges = _band_edges(cfg.audio.sample_rate, self.bands)
        membership = torch.zeros(self.bands, self.freqs)
        bin_to_band = torch.zeros(self.freqs, dtype=torch.long)
        for band in range(self.bands):
            if band == self.bands - 1:
                mask = (frequency >= edges[band]) & (frequency <= edges[band + 1])
            else:
                mask = (frequency >= edges[band]) & (frequency < edges[band + 1])
            if not mask.any():
                mask[torch.argmin(torch.abs(frequency - (edges[band] + edges[band + 1]) / 2))] = True
            membership[band, mask] = 1.0 / mask.sum()
            bin_to_band[mask] = band
        centers = ((edges[:-1] + edges[1:]) * 0.5) / (cfg.audio.sample_rate * 0.5)
        taper = 0.8 + 0.2 * torch.exp(-torch.arange(self.taps, dtype=torch.float32) / 4.0)
        self.register_buffer("membership", membership)
        self.register_buffer("bin_to_band", bin_to_band)
        self.register_buffer("band_centers", centers)
        self.register_buffer("tap_taper", taper)

    def initial_state(self, batch: int, device: torch.device) -> KFState:
        complex_zeros = torch.zeros(batch, self.freqs, self.taps, device=device, dtype=torch.complex64)
        band_taps = (batch, self.bands, self.taps)
        epsilon = 1.0e-6
        return KFState(
            h=complex_zeros.clone(),
            covariance=torch.full((batch, self.freqs, self.taps), float(self.cfg.model.covariance_init), device=device),
            process=torch.full((batch, self.freqs, self.taps), float(self.cfg.model.process_init), device=device),
            x_history=complex_zeros.clone(),
            powers_short=torch.full((batch, self.bands, 4), epsilon, device=device),
            powers_long=torch.full((batch, self.bands, 4), epsilon, device=device),
            x_power_taps=torch.full(band_taps, epsilon, device=device),
            cross_y=torch.zeros(band_taps, device=device, dtype=torch.complex64),
            cross_e=torch.zeros(band_taps, device=device, dtype=torch.complex64),
            stat_hidden=torch.zeros(batch, self.bands, self.stats_net.hidden_size, device=device),
            previous_doubletalk=torch.zeros(batch, device=device, dtype=torch.bool),
            previous_power_event=torch.zeros(batch, device=device, dtype=torch.bool),
            previous_mu=torch.zeros(batch, self.bands, device=device),
            previous_update=torch.zeros(batch, self.bands, device=device),
            previous_delta=complex_zeros.clone(),
        )

    def set_controller(self, controller: ControllerInterface) -> None:
        """Attach a future controller through the stable v17 interface."""
        self.control_policy = controller

    def band_mean(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.membership.T.to(value.dtype)

    def interpolate_bands(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value.unsqueeze(1), size=self.freqs, mode="linear", align_corners=True)[:, 0]

    def _statistics(self, x: torch.Tensor, y: torch.Tensor, error: torch.Tensor, echo: torch.Tensor, state: KFState) -> dict[str, torch.Tensor]:
        raw = torch.stack(
            [self.band_mean(x.abs().square()), self.band_mean(y.abs().square()),
             self.band_mean(error.abs().square()), self.band_mean(echo.abs().square())],
            dim=-1,
        )
        beta_short, beta_long = float(self.cfg.model.short_smoothing), float(self.cfg.model.long_smoothing)
        state.powers_short = beta_short * state.powers_short + (1.0 - beta_short) * raw
        state.powers_long = beta_long * state.powers_long + (1.0 - beta_long) * raw

        x_power = torch.einsum("bfp,kf->bkp", state.x_history.abs().square(), self.membership)
        complex_membership = self.membership.to(state.x_history.dtype)
        cross_y = torch.einsum("bfp,bf,kf->bkp", state.x_history.conj(), y, complex_membership)
        cross_e = torch.einsum("bfp,bf,kf->bkp", state.x_history.conj(), error, complex_membership)
        state.x_power_taps = beta_short * state.x_power_taps + (1.0 - beta_short) * x_power
        state.cross_y = beta_short * state.cross_y + (1.0 - beta_short) * cross_y
        state.cross_e = beta_short * state.cross_e + (1.0 - beta_short) * cross_e

        py = state.powers_short[..., 1].unsqueeze(-1)
        pe = state.powers_short[..., 2].unsqueeze(-1)
        gamma_y_taps = state.cross_y.abs().square() / (state.x_power_taps * py + 1.0e-10)
        gamma_e_taps = state.cross_e.abs().square() / (state.x_power_taps * pe + 1.0e-10)
        gamma_y = gamma_y_taps.clamp(0.0, 1.0).amax(dim=-1)
        gamma_e = gamma_e_taps.clamp(0.0, 1.0).amax(dim=-1)
        return {"gamma_y": gamma_y, "gamma_e": gamma_e, "gamma_e_taps": gamma_e_taps.clamp(0.0, 1.0)}

    def _network_features(
        self,
        stats: dict[str, torch.Tensor],
        state: KFState,
        innovation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        short, long = state.powers_short, state.powers_long
        log_short = (
            (torch.log10(short.clamp_min(1.0e-10)).clamp(-10.0, 2.0) + 4.0)
            / 6.0
        ).clamp(-1.0, 1.0)
        log_px_ratio = (
            torch.log10((short[..., 0] / long[..., 0]).clamp_min(1.0e-6))
            .clamp(-3.0, 3.0) / 3.0
        )
        log_pe_ratio_raw = torch.log10(
            (short[..., 2] / long[..., 2]).clamp_min(1.0e-6)
        ).clamp(-3.0, 3.0)
        log_pe_ratio = log_pe_ratio_raw / 3.0
        covariance = self.band_mean(state.covariance.mean(dim=-1))
        process = self.band_mean(state.process.mean(dim=-1))
        state_energy = self.band_mean(state.h.abs().square().sum(dim=-1))
        tap_power = state.x_power_taps.mean(dim=-1)
        state_echo = state_energy * tap_power
        predicted = self.band_mean(
            (state.covariance * state.x_history.abs().square()).sum(dim=-1)
        ) + float(self.cfg.model.observation_floor_ratio) * short[..., 1]
        previous_echo_update = (state.x_history * state.previous_delta).sum(dim=-1)
        previous_echo_update_power = self.band_mean(previous_echo_update.abs().square())
        innovation_power = self.band_mean(innovation.abs().square())
        update_to_innovation = (
            torch.log10(
                (previous_echo_update_power / innovation_power.clamp_min(1.0e-10))
                .clamp(1.0e-6, 1.0e6)
            )
            / 6.0
        )
        update_alignment = self.band_mean(
            (innovation * previous_echo_update.conj()).real
        ) / (innovation_power * previous_echo_update_power).clamp_min(1.0e-12).sqrt()
        update_alignment = update_alignment.clamp(-1.0, 1.0)
        features = torch.stack(
            [log_short[..., 0], log_short[..., 1], log_short[..., 2], log_short[..., 3],
             log_px_ratio, log_pe_ratio, stats["gamma_y"], stats["gamma_e"],
             torch.log10((short[..., 2] / short[..., 1]).clamp(1.0e-6, 1.0e6)) / 6.0,
             state.previous_mu,
             torch.log1p(state.previous_update) / math.log(11.0),
             self.band_centers.view(1, -1).expand(short.shape[0], -1),
             ((torch.log10(covariance.clamp(1.0e-8, 1.0)) + 4.0) / 4.0).clamp(-1.0, 1.0),
             ((torch.log10((process / covariance.clamp_min(1.0e-8)).clamp(1.0e-6, 1.0e2)) + 2.0) / 4.0).clamp(-1.0, 1.0),
             torch.log10((state_echo / short[..., 3].clamp_min(1.0e-10)).clamp(1.0e-6, 1.0e6)) / 6.0,
             torch.log10((short[..., 2] / predicted.clamp_min(1.0e-10)).clamp(1.0e-6, 1.0e6)) / 6.0,
             update_to_innovation,
             update_alignment,
             (stats["gamma_e"] - stats["gamma_y"]).clamp(-1.0, 1.0),
             (torch.log10((short[..., 1] / short[..., 3]).clamp(1.0e-6, 1.0e6)) / 6.0).clamp(-1.0, 1.0)],
            dim=-1,
        )
        return features, log_pe_ratio_raw

    def _control_features(
        self, stats: dict[str, torch.Tensor], state: KFState, pe_ratio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        short, long = state.powers_short, state.powers_long
        px_s, py_s, pe_s = (short[..., index].mean(dim=-1) for index in range(3))
        px_l, py_l, pe_l = (long[..., index].mean(dim=-1) for index in range(3))
        gamma_y = stats["gamma_y"].mean(dim=-1)
        gamma_e = stats["gamma_e"].mean(dim=-1)
        burst = 10.0 * torch.log10((pe_s / pe_l).clamp_min(1.0e-8))
        erle_short = 10.0 * torch.log10((py_s / pe_s).clamp_min(1.0e-8))
        erle_long = 10.0 * torch.log10((py_l / pe_l).clamp_min(1.0e-8))
        erle_drop = erle_long - erle_short
        noncoherent = 1.0 - gamma_y
        px_change = 10.0 * torch.log10((px_s / px_l).clamp_min(1.0e-8))
        py_change = 10.0 * torch.log10((py_s / py_l).clamp_min(1.0e-8))
        far_db = 10.0 * torch.log10(px_s.clamp_min(1.0e-12))
        far_active = far_db > float(self.cfg.control.farend_dbfs)
        path = far_active & (burst > self.cfg.control.innovation_burst_db) & (gamma_e > self.cfg.control.echo_coherence) & (erle_drop > self.cfg.control.erle_drop_db)
        strong = far_active & (burst > self.cfg.control.strong_burst_db) & (gamma_e > self.cfg.control.strong_coherence)
        doubletalk = (noncoherent > self.cfg.control.doubletalk_enter) & (gamma_e < self.cfg.control.doubletalk_exit)
        doubletalk_event = doubletalk != state.previous_doubletalk
        power_condition = (px_change.abs() > self.cfg.control.power_change_db) | (py_change.abs() > self.cfg.control.power_change_db)
        power_event = power_condition & ~state.previous_power_event
        # The inherited v15 TRACK path keeps its timer at zero, so expiry is
        # true on every frame. Preserve that feature without timer state.
        expiry = torch.ones_like(path, dtype=torch.bool)
        flags = torch.stack([path, doubletalk_event, power_event, expiry], dim=-1).float()
        base = torch.stack(
            [far_db.clamp(-100, 10) / 50.0, burst.clamp(-20, 20) / 10.0,
             gamma_y, gamma_e, erle_short.clamp(-20, 40) / 20.0,
             erle_drop.clamp(-20, 20) / 10.0, noncoherent,
             state.previous_update.mean(dim=-1).clamp(0, 10) / 10.0],
            dim=-1,
        )
        state.previous_doubletalk = doubletalk
        state.previous_power_event = power_condition
        return base, flags, far_active, strong

    def _decide_control(
        self,
        base: torch.Tensor,
        flags: torch.Tensor,
        far_active: torch.Tensor,
        strong: torch.Tensor,
        mode: str,
    ) -> ControlDecision:
        if mode != "track":
            raise ValueError(
                "Bounded_muq is a backbone-only model; use mode='track'. "
                "Attach a future policy through set_controller() before adding new modes."
            )
        decision = self.control_policy.decide(base, flags, far_active, strong)
        if decision.action.shape != far_active.shape or decision.invoked.shape != far_active.shape:
            raise ValueError("ControllerInterface returned an invalid batch shape")
        return decision

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        state: KFState,
        mode: str = "track",
        fixed_q: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, KFState, dict[str, torch.Tensor]]:
        state.x_history = torch.cat([x.unsqueeze(-1), state.x_history[..., :-1]], dim=-1)
        h_minus = state.h
        echo_prior = (state.x_history * h_minus).sum(dim=-1)
        innovation = y - echo_prior

        stats = self._statistics(x, y, innovation, echo_prior, state)
        features, pe_ratio = self._network_features(stats, state, innovation)
        logits, state.stat_hidden = self.stats_net(features, state.stat_hidden)
        bounded = logits[..., :2].sigmoid()
        reliability = torch.sigmoid(logits[..., 3]).clamp(0.02, 1.0)

        mu_band = (
            float(self.cfg.model.mu_min)
            + (float(self.cfg.model.mu_max) - float(self.cfg.model.mu_min)) * bounded[..., 0]
        )
        calibration = torch.exp(
            math.log(float(self.cfg.model.mu_calibration_max))
            * torch.tanh(logits[..., 2])
        )
        mu_band = mu_band * calibration
        mu_band = float(self.cfg.model.mu_min) + reliability * (mu_band - float(self.cfg.model.mu_min))
        log_q_min = math.log(float(self.cfg.model.q_min))
        log_q_max = math.log(float(self.cfg.model.q_max))
        rho_q = torch.exp(log_q_min + bounded[..., 1] * (log_q_max - log_q_min))
        rho_bins = self.interpolate_bands(rho_q)
        process_next = rho_bins.unsqueeze(-1) * (
            state.covariance
            + float(self.cfg.model.q_delta_scale) * state.previous_delta.abs().square()
        )
        process_next = process_next + float(self.cfg.model.covariance_floor)
        if fixed_q:
            process_next = torch.full_like(process_next, float(self.cfg.model.process_init))
        covariance_minus = (state.covariance + process_next).clamp(
            float(self.cfg.model.covariance_floor),
            float(self.cfg.model.covariance_cap),
        )

        base, flags, far_active, strong = self._control_features(stats, state, pe_ratio)
        decision = self._decide_control(base, flags, far_active, strong, mode)
        action, invoked = decision.action, decision.invoked

        sx = (covariance_minus * state.x_history.abs().square()).sum(dim=-1)
        requested_mu = self.interpolate_bands(mu_band).clamp_min(float(self.cfg.model.mu_min))
        requested_mu = torch.minimum(
            requested_mu,
            torch.full_like(requested_mu, float(self.cfg.model.track_mu_cap)),
        )
        observation_floor = (
            float(self.cfg.model.observation_floor_ratio)
            * self.interpolate_bands(state.powers_short[..., 1])
        ).clamp_min(float(self.cfg.model.observation_floor))
        observation = torch.maximum(
            sx * (1.0 - requested_mu) / requested_mu.clamp_min(1.0e-6),
            observation_floor,
        )
        denominator = (sx + observation).clamp_min(1.0e-8)
        gain = covariance_minus * state.x_history.conj() / denominator.unsqueeze(-1)
        effective_gain = gain
        delta = effective_gain * innovation.unsqueeze(-1)
        h_plus = h_minus + delta

        response = (effective_gain * state.x_history).real
        covariance_plus = (
            (1.0 - response).square() * covariance_minus
            + effective_gain.abs().square() * observation.unsqueeze(-1)
        ).clamp(
            float(self.cfg.model.covariance_floor),
            float(self.cfg.model.covariance_cap),
        )
        h_norm = (h_plus.abs().square().sum(dim=-1) + 1.0e-12).sqrt()
        h_limit_ratio = (
            state.powers_long[..., 1]
            / state.x_power_taps.mean(dim=-1).clamp_min(1.0e-8)
        )
        h_limit_band = (
            float(self.cfg.model.coefficient_norm_cap)
            * h_limit_ratio.clamp_min(1.0e-12).sqrt()
        )
        h_limit = self.interpolate_bands(h_limit_band).clamp_min(1.0e-3)
        projection = torch.minimum(
            torch.ones_like(h_norm), h_limit / h_norm.clamp_min(1.0e-8)
        )
        h_plus = torch.where(
            far_active[:, None, None],
            h_plus * projection.unsqueeze(-1),
            h_plus,
        )

        echo_posterior = (state.x_history * h_plus).sum(dim=-1)
        error_posterior = y - echo_posterior
        actual_mu = (effective_gain * state.x_history).real.sum(dim=-1).clamp(0.0, 1.0)
        update_power = self.band_mean(delta.abs().square().sum(dim=-1))
        state_power = self.band_mean(h_plus.abs().square().sum(dim=-1))
        state.previous_mu = self.band_mean(actual_mu)
        state.previous_update = (
            update_power / (state_power + 1.0e-8)
        ).clamp(0.0, 10.0)
        state.h = h_plus
        state.covariance = covariance_plus
        state.process = process_next.clamp(
            float(self.cfg.model.covariance_floor),
            float(self.cfg.model.covariance_cap),
        )
        state.previous_delta = delta
        state.frame_index += 1

        auxiliary = {
            "mu": actual_mu,
            "requested_mu": requested_mu,
            "mu_calibration": calibration,
            "reliability": reliability,
            "rho_q": rho_q,
            "xi": denominator,
            "innovation": innovation,
            "prior_echo": echo_prior,
            "delta": delta,
            "action": action,
            "invoked": invoked,
            "control_features": base,
            "event_flags": flags,
            "gamma_y": stats["gamma_y"],
            "gamma_e": stats["gamma_e"],
        }
        return error_posterior, echo_posterior, state, auxiliary

    def run(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: str = "track",
        fixed_q: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        state = self.initial_state(x.shape[0], x.device)
        errors, echoes, actions, invoked = [], [], [], []
        for frame in range(x.shape[1]):
            error, echo, state, auxiliary = self.step(
                x[:, frame], y[:, frame], state, mode=mode, fixed_q=fixed_q
            )
            errors.append(error)
            echoes.append(echo)
            actions.append(auxiliary["action"])
            invoked.append(auxiliary["invoked"])
        return (
            torch.stack(errors, dim=1),
            torch.stack(echoes, dim=1),
            {
                "actions": torch.stack(actions, dim=1),
                "invoked": torch.stack(invoked, dim=1),
                "controller_calls": torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
                "switches": torch.zeros(x.shape[0], device=x.device, dtype=torch.long),
            },
        )



