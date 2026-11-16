from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PostFilterState:
    band_hidden: torch.Tensor
    fast_hidden: torch.Tensor
    slow_hidden: torch.Tensor
    powers: torch.Tensor
    cross_xe: torch.Tensor
    cross_he: torch.Tensor
    feature_history: torch.Tensor
    echo_history: torch.Tensor
    reference_history: torch.Tensor
    previous_gain: torch.Tensor
    previous_echo_coefficients: torch.Tensor
    previous_reference_coefficients: torch.Tensor
    previous_dt_echo_coefficients: torch.Tensor
    previous_dt_reference_coefficients: torch.Tensor
    previous_scene: torch.Tensor
    previous_near_band: torch.Tensor

    def detach(self) -> "PostFilterState":
        return PostFilterState(*[value.detach() for value in self.__dict__.values()])


class UGSRPF(nn.Module):
    """Causal v17 residual-echo estimator with temporal and nonlinear bases."""

    def __init__(self, membership: torch.Tensor, band_centers: torch.Tensor, cfg: Any):
        super().__init__()
        if membership.ndim != 2:
            raise ValueError("membership must have shape [bands, frequencies]")
        self.bands, self.freqs = membership.shape
        self.band_hidden_size = 20
        self.scene_hidden_size = 10
        self.temporal_taps = int(cfg.temporal_taps)
        if self.temporal_taps != 4:
            raise ValueError("v9 requires temporal_taps=4")
        self.smoothing = float(cfg.smoothing)
        self.attack = float(cfg.attack)
        self.release = float(cfg.release)
        self.correction_smoothing = float(cfg.correction_smoothing)
        self.gain_residual_limit_db = float(cfg.gain_residual_limit_db)
        self.fst_min_gain = float(cfg.fst_min_gain)
        self.dt_min_gain = float(cfg.dt_min_gain)
        self.epc_transition_min_gain = float(cfg.epc_transition_min_gain)
        self.gate_floor = float(cfg.gate_floor)
        self.near_attack = float(cfg.near_attack)
        self.near_release = float(cfg.near_release)
        self.near_gate_power = float(cfg.near_gate_power)
        self.near_safe_threshold = float(cfg.near_safe_threshold)
        self.near_safe_temperature = float(cfg.near_safe_temperature)
        self.fst_extra_gain_limit_db = float(cfg.fst_extra_gain_limit_db)
        # routing_blend=0 executes the transferred v8 weights exactly.
        self.routing_blend = 0.0
        self.near_gate_blend = 0.0

        self.register_buffer("echo_basis_limits", torch.tensor(cfg.echo_basis_limits).float())
        self.register_buffer("reference_basis_limits", torch.tensor(cfg.reference_basis_limits).float())
        self.register_buffer("dt_echo_basis_limits", torch.tensor(cfg.dt_echo_basis_limits).float())
        self.register_buffer("dt_reference_basis_limits", torch.tensor(cfg.dt_reference_basis_limits).float())
        self.nonlinear_order2_limit = float(cfg.nonlinear_order2_limit)
        self.nonlinear_order3_limit = float(cfg.nonlinear_order3_limit)
        self.dt_nonlinear_limit = float(cfg.dt_nonlinear_limit)
        self.dt_common_correction_scale = float(cfg.dt_common_correction_scale)

        # Causal depthwise temporal convolution over the last four feature frames.
        self.temporal_kernel = nn.Parameter(torch.zeros(14, self.temporal_taps))
        nn.init.constant_(self.temporal_kernel[:, 0], 1.0)
        self.input = nn.Linear(14, 16)
        self.norm1 = nn.LayerNorm(16)
        self.band_gru = nn.GRU(16, self.band_hidden_size, batch_first=True)
        self.band_projection = nn.Linear(self.band_hidden_size, 12)
        self.band_norm = nn.LayerNorm(12)
        self.frequency_depthwise = nn.Conv1d(12, 12, 3, padding=1, groups=12)
        self.frequency_pointwise = nn.Conv1d(12, 12, 1)
        self.frequency_norm = nn.LayerNorm(12)

        self.global_projection = nn.Linear(8, 10)
        self.fast_gru = nn.GRU(10, self.scene_hidden_size, batch_first=True)
        self.slow_gru = nn.GRU(10, self.scene_hidden_size, batch_first=True)
        self.fast_norm = nn.LayerNorm(10)
        self.slow_norm = nn.LayerNorm(10)

        # 3 complex echo coefficients and 3 complex reference coefficients.
        self.temporal_basis_head = nn.Linear(32, 4 * self.temporal_taps)
        self.dt_temporal_basis_head = nn.Linear(32, 4 * self.temporal_taps)
        self.nonlinear_basis_head = nn.Linear(32, 4)
        self.gain_residual_head = nn.Linear(32, 1)
        self.fst_strength_head = nn.Linear(32, 1)
        self.scene_head = nn.Linear(20, 4)
        self.near_adapter = nn.Sequential(nn.Linear(46, 16), nn.SiLU(), nn.Linear(16, 1))

        self.register_buffer("membership", membership.detach().float().clone())
        self.register_buffer("band_centers", band_centers.detach().float().clone())

        for head in (self.temporal_basis_head, self.dt_temporal_basis_head, self.nonlinear_basis_head, self.gain_residual_head, self.scene_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        with torch.no_grad():
            self.scene_head.bias.copy_(torch.tensor([-2.2, 2.2, -3.0, 2.2]))
            nn.init.zeros_(self.near_adapter[-1].weight)
            nn.init.constant_(self.near_adapter[-1].bias, -2.2)
            nn.init.zeros_(self.fst_strength_head.weight)
            nn.init.constant_(self.fst_strength_head.bias, -2.2)

    def initial_state(self, batch: int, device: torch.device) -> PostFilterState:
        complex_zeros = lambda *shape: torch.zeros(*shape, device=device, dtype=torch.complex64)
        return PostFilterState(
            band_hidden=torch.zeros(batch, self.bands, self.band_hidden_size, device=device),
            fast_hidden=torch.zeros(batch, self.scene_hidden_size, device=device),
            slow_hidden=torch.zeros(batch, self.scene_hidden_size, device=device),
            powers=torch.full((batch, self.bands, 4), 1.0e-6, device=device),
            cross_xe=complex_zeros(batch, self.bands),
            cross_he=complex_zeros(batch, self.bands),
            feature_history=torch.zeros(batch, self.bands, 14, self.temporal_taps, device=device),
            echo_history=complex_zeros(batch, self.temporal_taps, self.freqs),
            reference_history=complex_zeros(batch, self.temporal_taps, self.freqs),
            previous_gain=torch.ones(batch, self.bands, device=device),
            previous_echo_coefficients=complex_zeros(batch, self.bands, self.temporal_taps),
            previous_reference_coefficients=complex_zeros(batch, self.bands, self.temporal_taps),
            previous_dt_echo_coefficients=complex_zeros(batch, self.bands, self.temporal_taps),
            previous_dt_reference_coefficients=complex_zeros(batch, self.bands, self.temporal_taps),
            previous_scene=torch.zeros(batch, 4, device=device),
            previous_near_band=torch.zeros(batch, self.bands, device=device),
        )

    def band_mean(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.membership.T.to(value.dtype)

    def interpolate(self, value: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value.unsqueeze(1), size=self.freqs, mode="linear", align_corners=True)[:, 0]

    @staticmethod
    def _log_feature(value: torch.Tensor) -> torch.Tensor:
        return ((torch.log10(value.clamp_min(1.0e-10)) + 4.0) / 6.0).clamp(-1.0, 1.0)

    @staticmethod
    def _bounded_complex(real: torch.Tensor, imag: torch.Tensor, limit: torch.Tensor | float) -> torch.Tensor:
        value = torch.complex(torch.tanh(real), torch.tanh(imag))
        value = value / value.abs().clamp_min(1.0).detach()
        return value * limit

    def _features(self, x, mic, error, echo_hat, aux, state):
        raw = torch.stack([
            self.band_mean(x.abs().square()), self.band_mean(mic.abs().square()),
            self.band_mean(error.abs().square()), self.band_mean(echo_hat.abs().square()),
        ], dim=-1)
        beta = self.smoothing
        state.powers = beta * state.powers + (1.0 - beta) * raw
        membership = self.membership.to(error.dtype)
        cross_xe = torch.einsum("bf,bf,kf->bk", x.conj(), error, membership)
        cross_he = torch.einsum("bf,bf,kf->bk", echo_hat.conj(), error, membership)
        state.cross_xe = beta * state.cross_xe + (1.0 - beta) * cross_xe
        state.cross_he = beta * state.cross_he + (1.0 - beta) * cross_he
        px, py, pe, ph = (state.powers[..., index] for index in range(4))
        coh_xe = (state.cross_xe.abs().square() / (px * pe + 1.0e-10)).clamp(0.0, 1.0)
        coh_he = (state.cross_he.abs().square() / (ph * pe + 1.0e-10)).clamp(0.0, 1.0)
        reliability = aux["reliability"].clamp(0.0, 1.0)
        actual_mu = self.band_mean(aux["mu"]).clamp(0.0, 1.0)
        rho_q = ((torch.log10(aux["rho_q"].clamp_min(1.0e-8)) + 5.0) / 4.0).clamp(-1.0, 1.0)
        xi = self.band_mean(aux["xi"].clamp_min(1.0e-10))
        features = torch.stack([
            self._log_feature(px), self._log_feature(py), self._log_feature(pe), self._log_feature(ph),
            torch.log10((pe / ph.clamp_min(1.0e-10)).clamp(1.0e-6, 1.0e6)) / 6.0,
            coh_xe, coh_he, torch.log10((pe / xi).clamp(1.0e-6, 1.0e6)) / 6.0,
            reliability, actual_mu, rho_q, self._log_feature(xi), state.previous_gain,
            self.band_centers.view(1, -1).expand(x.shape[0], -1),
        ], dim=-1)
        return features, coh_he

    def step(self, x, mic, error, echo_hat, aux, state, mix):
        features, coh_he = self._features(x, mic, error, echo_hat, aux, state)
        state.feature_history = torch.cat([features.unsqueeze(-1), state.feature_history[..., :-1]], dim=-1)
        temporal_features = (state.feature_history * self.temporal_kernel.view(1, 1, 14, -1)).sum(-1)
        batch = features.shape[0]
        band_input = F.silu(self.norm1(self.input(temporal_features)))
        band_value, band_hidden = self.band_gru(
            band_input.reshape(batch * self.bands, 1, 16),
            state.band_hidden.reshape(1, batch * self.bands, self.band_hidden_size).contiguous(),
        )
        state.band_hidden = band_hidden[0].reshape(batch, self.bands, self.band_hidden_size)
        band_value = F.silu(self.band_norm(self.band_projection(band_value[:, 0]))).reshape(batch, self.bands, 12)
        context = self.frequency_pointwise(self.frequency_depthwise(band_value.transpose(1, 2))).transpose(1, 2)
        band_value = F.silu(self.frequency_norm(band_value + context))

        global_features = torch.stack([
            features[..., 0].mean(-1), features[..., 2].mean(-1), features[..., 4].mean(-1),
            features[..., 5].mean(-1), features[..., 6].mean(-1), features[..., 8].mean(-1),
            (features[..., 8] < 0.35).float().mean(-1), state.previous_scene[:, 0],
        ], dim=-1)
        global_input = F.silu(self.global_projection(global_features))[:, None]
        fast, fast_hidden = self.fast_gru(global_input, state.fast_hidden[None].contiguous())
        # Slow state sees a leaky version of the same causal observation.
        slow_input = (0.25 * global_input + 0.75 * state.slow_hidden[:, None]).contiguous()
        slow, slow_hidden = self.slow_gru(slow_input, state.slow_hidden[None].contiguous())
        state.fast_hidden, state.slow_hidden = fast_hidden[0], slow_hidden[0]
        global_value = torch.cat([F.silu(self.fast_norm(fast[:, 0])), F.silu(self.slow_norm(slow[:, 0]))], -1)
        scene = torch.sigmoid(self.scene_head(global_value))
        state.previous_scene = scene.detach()
        # Route with calibrated probabilities; only explicit scene labels train the classifier.
        near_probability, far_probability, path_probability, echo_confidence = scene.detach().unbind(-1)

        head_input = torch.cat([band_value, global_value[:, None].expand(-1, self.bands, -1)], -1)
        near_band_probability = torch.sigmoid(self.near_adapter(torch.cat([head_input, features], -1))[..., 0])
        near_alpha = torch.where(
            near_band_probability > state.previous_near_band,
            torch.full_like(near_band_probability, self.near_attack),
            torch.full_like(near_band_probability, self.near_release),
        )
        near_band_smoothed = near_alpha * state.previous_near_band + (1.0 - near_alpha) * near_band_probability
        state.previous_near_band = near_band_smoothed.detach()
        # Separate heads stop the FST objective from rewriting the DT estimator.
        fst_logits = self.temporal_basis_head(head_input).view(batch, self.bands, self.temporal_taps, 4)
        dt_logits = self.dt_temporal_basis_head(head_input).view(batch, self.bands, self.temporal_taps, 4)
        near_band = near_probability.detach()[:, None, None]
        fst_echo = self._bounded_complex(fst_logits[..., 0], fst_logits[..., 1], self.echo_basis_limits.view(1, 1, -1))
        fst_ref = self._bounded_complex(fst_logits[..., 2], fst_logits[..., 3], self.reference_basis_limits.view(1, 1, -1))
        dt_echo = self._bounded_complex(dt_logits[..., 0], dt_logits[..., 1], self.dt_echo_basis_limits.view(1, 1, -1))
        dt_ref = self._bounded_complex(dt_logits[..., 2], dt_logits[..., 3], self.dt_reference_basis_limits.view(1, 1, -1))
        # DT has its own causal coefficient history; FST smoothing cannot leak into it.
        dt_echo = self.correction_smoothing * state.previous_dt_echo_coefficients + (1.0 - self.correction_smoothing) * dt_echo
        dt_ref = self.correction_smoothing * state.previous_dt_reference_coefficients + (1.0 - self.correction_smoothing) * dt_ref
        state.previous_dt_echo_coefficients, state.previous_dt_reference_coefficients = dt_echo, dt_ref
        # Keep FST and DT estimators separate until their signal predictions are routed.
        echo_coeff = fst_echo
        ref_coeff = fst_ref
        beta = self.correction_smoothing
        echo_coeff = beta * state.previous_echo_coefficients + (1.0 - beta) * echo_coeff
        ref_coeff = beta * state.previous_reference_coefficients + (1.0 - beta) * ref_coeff
        state.previous_echo_coefficients, state.previous_reference_coefficients = echo_coeff, ref_coeff

        echo_bases = torch.cat([echo_hat[:, None], state.echo_history[:, :-1]], 1)
        ref_bases = torch.cat([x[:, None], state.reference_history[:, :-1]], 1)
        state.echo_history, state.reference_history = echo_bases, ref_bases
        echo_bins = torch.stack([torch.complex(self.interpolate(echo_coeff[..., k].real), self.interpolate(echo_coeff[..., k].imag)) for k in range(self.temporal_taps)], 1)
        ref_bins = torch.stack([torch.complex(self.interpolate(ref_coeff[..., k].real), self.interpolate(ref_coeff[..., k].imag)) for k in range(self.temporal_taps)], 1)
        linear_gate = torch.stack([far_probability, echo_confidence, 1.0 - 0.5 * path_probability], -1).amin(-1).clamp(0.0, 1.0)
        common_scale = 1.0 - near_probability.detach() * (1.0 - self.dt_common_correction_scale)
        fst_linear_prediction = linear_gate[:, None] * ((echo_bins * echo_bases).sum(1) + (ref_bins * ref_bases).sum(1))
        dt_echo_bins = torch.stack([torch.complex(self.interpolate(dt_echo[..., k].real), self.interpolate(dt_echo[..., k].imag)) for k in range(self.temporal_taps)], 1)
        dt_ref_bins = torch.stack([torch.complex(self.interpolate(dt_ref[..., k].real), self.interpolate(dt_ref[..., k].imag)) for k in range(self.temporal_taps)], 1)
        dt_basis_prediction = (dt_echo_bins * echo_bases).sum(1) + (dt_ref_bins * ref_bases).sum(1)
        dt_linear_prediction = common_scale[:, None] * linear_gate[:, None] * dt_basis_prediction

        nonlinear_logits = self.nonlinear_basis_head(head_input)
        nonlinear_limit2 = self.nonlinear_order2_limit * (1.0 - near_band[..., 0]) + self.dt_nonlinear_limit * near_band[..., 0]
        nonlinear_limit3 = self.nonlinear_order3_limit * (1.0 - near_band[..., 0]) + self.dt_nonlinear_limit * near_band[..., 0]
        coefficient2 = self._bounded_complex(nonlinear_logits[..., 0], nonlinear_logits[..., 1], nonlinear_limit2)
        coefficient3 = self._bounded_complex(nonlinear_logits[..., 2], nonlinear_logits[..., 3], nonlinear_limit3)
        coefficient2 = torch.complex(self.interpolate(coefficient2.real), self.interpolate(coefficient2.imag))
        coefficient3 = torch.complex(self.interpolate(coefficient3.real), self.interpolate(coefficient3.imag))
        normalized_x = x / x.abs().square().mean(-1, keepdim=True).sqrt().clamp_min(1.0e-4)
        nonlinear_prediction = coefficient2 * normalized_x * normalized_x.abs() + coefficient3 * normalized_x * normalized_x.abs().square()

        fst_gate = torch.stack([far_probability, 1.0 - near_probability, echo_confidence, 1.0 - path_probability], -1).amin(-1)
        fst_gate = ((fst_gate - self.gate_floor) / max(1.0e-6, 1.0 - self.gate_floor)).clamp(0.0, 1.0)
        fst_gate = fst_gate * torch.sigmoid((0.50 - near_probability) * 12.0)
        if self.near_gate_blend <= 0.0:
            # Preserve v7's exact arithmetic until the new classifier is calibrated.
            execution_gate = fst_gate[:, None]
            gate_band = fst_gate[:, None].expand(-1, self.bands)
        else:
            far_confidence = (far_probability * echo_confidence * (1.0 - path_probability)).clamp(0.0, 1.0).pow(0.25)
            protected_band = far_confidence[:, None] * (1.0 - near_band_smoothed.detach()).clamp(0.0, 1.0).pow(self.near_gate_power)
            gate_band = (1.0 - self.near_gate_blend) * fst_gate[:, None] + self.near_gate_blend * protected_band
            execution_gate = self.interpolate(gate_band).clamp(0.0, 1.0)
        old_gate_bins = execution_gate
        old_prediction = old_gate_bins * (fst_linear_prediction + nonlinear_prediction) + (1.0 - old_gate_bins) * dt_linear_prediction
        safe_probability = torch.sigmoid(
            (near_band_smoothed - self.near_safe_threshold)
            / max(self.near_safe_temperature, 1.0e-3)
        )
        far_confidence = (far_probability * echo_confidence * (1.0 - path_probability)).clamp(0.0, 1.0).pow(0.25)
        safe_gate_band = far_confidence[:, None] * (1.0 - safe_probability)
        safe_gate_bins = self.interpolate(safe_gate_band).clamp(0.0, 1.0)
        # Unlike v8's global-near multiplier, zero DT scale now really bypasses
        # the DT correction. Spectral gain is routed independently below.
        safe_dt_prediction = self.dt_common_correction_scale * linear_gate[:, None] * dt_basis_prediction
        safe_prediction = safe_gate_bins * (fst_linear_prediction + nonlinear_prediction) + (1.0 - safe_gate_bins) * safe_dt_prediction
        residual_prediction = (1.0 - self.routing_blend) * old_prediction + self.routing_blend * safe_prediction
        corrected = error - torch.as_tensor(mix, device=error.device, dtype=error.real.dtype) * residual_prediction

        analytic_gain = (1.0 - 0.90 * coh_he * echo_confidence[:, None]).clamp(self.fst_min_gain ** 2, 1.0).sqrt()
        residual_db = torch.tanh(self.gain_residual_head(head_input)[..., 0]) * self.gain_residual_limit_db
        raw_gain = (analytic_gain * torch.pow(10.0, residual_db / 20.0)).clamp(self.fst_min_gain, 1.0)
        floor = self.dt_min_gain * near_probability[:, None] + self.epc_transition_min_gain * path_probability[:, None]
        raw_gain = torch.maximum(raw_gain, floor.clamp(max=1.0))
        coefficient = torch.where(raw_gain < state.previous_gain, torch.full_like(raw_gain, self.attack), torch.full_like(raw_gain, self.release))
        gain_band = coefficient * state.previous_gain + (1.0 - coefficient) * raw_gain
        state.previous_gain = gain_band
        gain_bins = self.interpolate(gain_band).clamp(self.fst_min_gain, 1.0)
        mix_tensor = torch.as_tensor(mix, dtype=gain_bins.dtype, device=gain_bins.device).clamp(0.0, 1.0)
        old_executed_gain = 1.0 - mix_tensor * old_gate_bins * (1.0 - gain_bins)
        extra_gain_db = self.fst_extra_gain_limit_db * torch.sigmoid(
            self.fst_strength_head(head_input)[..., 0]
        )
        strong_gain_bins = self.interpolate(
            (gain_band * torch.pow(10.0, -extra_gain_db / 20.0)).clamp(self.fst_min_gain, 1.0)
        )
        safe_executed_gain = 1.0 - mix_tensor * safe_gate_bins * (1.0 - strong_gain_bins)
        executed_gain = (1.0 - self.routing_blend) * old_executed_gain + self.routing_blend * safe_executed_gain
        output = corrected * executed_gain
        basis_regularization = echo_coeff.abs().mean((1, 2)) + ref_coeff.abs().mean((1, 2)) + coefficient2.abs().mean(1) + coefficient3.abs().mean(1)
        return output, state, {
            "gain_bins": executed_gain, "residual_prediction": residual_prediction,
            "scene_probabilities": scene, "basis_regularization": basis_regularization,
            "near_probability": near_probability, "far_probability": far_probability,
            "path_probability": path_probability, "echo_confidence": echo_confidence,
            "fst_gate": (1.0 - self.routing_blend) * gate_band + self.routing_blend * safe_gate_band,
            "linear_gate": linear_gate,
            "near_probability_band": near_band_probability,
            "near_probability_band_smoothed": near_band_smoothed,
            "near_safe_probability": safe_probability,
            "fst_extra_gain_db": extra_gain_db,
        }


def postfilter_loss(
    output: torch.Tensor, backbone_error: torch.Tensor, target: torch.Tensor,
    near: torch.Tensor, echo: torch.Tensor, residual_prediction: torch.Tensor,
    gain: torch.Tensor, scene_probabilities: torch.Tensor, path_label: torch.Tensor,
    scenarios: list[str], cfg: Any, sample_weights: torch.Tensor,
    lagrange_weights: dict[str, float], target_scale: float = 1.0,
    basis_regularization: torch.Tensor | None = None,
    near_probability_band: torch.Tensor | None = None,
    band_membership: torch.Tensor | None = None,
):
    eps = 1.0e-8
    target_residual = (backbone_error - target).detach()
    output_residual = output - target
    near_energy = near.abs().square().mean((1, 2))
    backbone_energy = target_residual.abs().square().mean((1, 2)).clamp_min(eps)
    output_energy = output_residual.abs().square().mean((1, 2))
    prediction_error = residual_prediction - target_residual
    prediction_nmse = prediction_error.abs().square().mean((1, 2)) / backbone_energy
    complex_loss_sample = prediction_error.abs().mean((1, 2)) / target_residual.abs().mean((1, 2)).clamp_min(eps)
    output_ratio = output_energy / backbone_energy
    achieved_gain_db = -10.0 * torch.log10(output_ratio.clamp_min(eps))

    fst = torch.tensor([name == "fst" for name in scenarios], device=output.device)
    dt = torch.tensor([name == "dt" for name in scenarios], device=output.device)
    epc = torch.tensor([name == "epc" for name in scenarios], device=output.device)
    near_active = near_energy > eps
    masks = {"fst": fst, "dt": dt, "epc_st": epc & ~near_active, "epc_dt": epc & near_active}
    targets = {name: float(getattr(cfg, f"{name}_target_gain_db")) * float(target_scale) for name in masks}

    def weighted(values, mask=None):
        active = torch.ones_like(sample_weights, dtype=torch.bool) if mask is None else mask
        if not bool(active.any()):
            return values.sum() * 0.0
        weights = sample_weights[active]
        return (values[active] * weights).sum() / weights.sum().clamp_min(eps)

    margin_sample = torch.zeros_like(output_ratio)
    scenario_losses = []
    for name, mask in masks.items():
        margin = F.softplus((targets[name] - achieved_gain_db) / float(cfg.margin_temperature_db)) * float(cfg.margin_temperature_db)
        margin_sample = margin_sample + mask.to(margin.dtype) * margin
        if bool(mask.any()):
            mass = sample_weights[mask].sum()
            scenario_loss = weighted(
                float(cfg.loss_residual_prediction) * prediction_nmse
                + float(cfg.loss_complex) * complex_loss_sample
                + float(cfg.loss_scenario_margin) * margin,
                mask,
            )
            scenario_losses.append((name, mass, scenario_loss))

    near_frame = near.abs().square().mean(-1)
    echo_frame = echo.abs().square().mean(-1)
    near_floor = 1.0e-4 * near_frame.amax(1, keepdim=True).clamp_min(eps)
    echo_floor = 1.0e-4 * echo_frame.amax(1, keepdim=True).clamp_min(eps)
    near_label = (near_frame > near_floor).to(scene_probabilities.dtype)
    far_label = (echo_frame > echo_floor).to(scene_probabilities.dtype)
    confidence_label = (far_label * (1.0 - path_label)).clamp(0.0, 1.0)
    labels = torch.stack([near_label, far_label, path_label, confidence_label], -1)
    scene_loss = F.binary_cross_entropy(scene_probabilities.clamp(1.0e-4, 1.0 - 1.0e-4), labels)
    if near_probability_band is not None and band_membership is not None:
        near_band_power = near.abs().square() @ band_membership.T.to(near.real.dtype)
        threshold = (near_band_power.amax((1, 2), keepdim=True) * float(cfg.near_label_relative_floor)).clamp_min(1.0e-9)
        near_band_label = (near_band_power > threshold).to(near_probability_band.dtype)
        band_bce = F.binary_cross_entropy(
            near_probability_band.clamp(1.0e-4, 1.0 - 1.0e-4), near_band_label, reduction="none",
        )
        band_weight = 1.0 + near_band_label * (float(cfg.near_positive_weight) - 1.0)
        near_band_loss = (band_bce * band_weight).mean()
    else:
        near_band_loss = output.real.sum() * 0.0
    near_preserve = weighted(output_residual.abs().square().mean((1, 2)) / near_energy.clamp_min(eps), near_active)
    near_gain = ((gain - 1.0).square() * near_label[..., None]).sum() / (near_label.sum() * gain.shape[-1]).clamp_min(1.0)
    dt_samples = dt | (epc & near_active)
    if bool(dt_samples.any()):
        tolerance = float(cfg.dt_tail_tolerance_db)
        temperature = float(cfg.margin_temperature_db)
        regressions = F.softplus(
            (tolerance - achieved_gain_db[dt_samples]) / temperature
        ) * temperature
        tail_count = max(1, int((regressions.numel() + 3) // 4))
        dt_tail_loss = regressions.topk(tail_count).values.mean()
    else:
        dt_tail_loss = output.real.sum() * 0.0
    time_smooth = (residual_prediction[:, 1:] - residual_prediction[:, :-1]).abs().mean() if output.shape[1] > 1 else output.real.sum() * 0.0
    frequency_smooth = (residual_prediction[..., 1:] - residual_prediction[..., :-1]).abs().mean()
    basis_loss = basis_regularization.mean() if basis_regularization is not None else residual_prediction.real.sum() * 0.0
    scenario_weight = torch.zeros_like(output_ratio)
    for name, mask in masks.items():
        scenario_weight = scenario_weight + mask.to(output_ratio.dtype) * float(lagrange_weights[name])
    primary = weighted(scenario_weight * (
        float(cfg.loss_residual_prediction) * prediction_nmse
        + float(cfg.loss_complex) * complex_loss_sample
        + float(cfg.loss_scenario_margin) * margin_sample
    ))
    common = (
        float(cfg.loss_near_preserve) * near_preserve + float(cfg.loss_scene) * scene_loss
        + float(cfg.loss_near_band) * near_band_loss
        + float(cfg.loss_dt_tail) * dt_tail_loss
        + float(cfg.loss_gain_target) * near_gain + float(cfg.loss_time_smooth) * time_smooth
        + float(cfg.loss_frequency_smooth) * frequency_smooth
        + float(cfg.loss_basis_regularization) * basis_loss
    )
    total = primary + common
    # Before projection, these tasks sum to the ordinary weighted objective.
    tasks = [mass * (float(lagrange_weights[name]) * loss + common) for name, mass, loss in scenario_losses]
    parts = {
        "prediction": weighted(prediction_nmse).detach(), "complex": weighted(complex_loss_sample).detach(),
        "margin": weighted(margin_sample).detach(), "scene": scene_loss.detach(),
        "near_band": near_band_loss.detach(),
        "dt_tail": dt_tail_loss.detach(),
        "near_preserve": near_preserve.detach(), "near_gain": near_gain.detach(),
        "smooth": (time_smooth + frequency_smooth).detach(), "basis": basis_loss.detach(),
    }
    return total, parts, tasks
