"""Rain- and state-conditioned mass-conserving runoff LSTM.

The module has two explicit runoff stores and one bounded unobserved-loss
flux. Rainfall is conserved in physical units, while the LSTM receives both
the current rainfall feature and the current physical stores when deciding
partitioning and recession. Observed Q/Z never enters this state update.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse_softplus value必须>0")
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError("logit value必须在(0,1)")
    return math.log(value / (1.0 - value))


def continuous_release_fraction(
    rate_per_hour: torch.Tensor, seconds: float
) -> torch.Tensor:
    """Convert a non-negative continuous release rate to a step fraction."""

    dt_hours = float(seconds) / 3600.0
    if not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("seconds必须为有限正数")
    if not torch.isfinite(rate_per_hour).all() or (rate_per_hour < 0).any():
        raise ValueError("rate_per_hour必须为有限非负值")
    return -torch.expm1(-rate_per_hour * dt_hours)


class MassConservingRunoffLSTMCell(nn.Module):
    """One physical step with fast/slow stores and a bounded loss flux."""

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int,
        *,
        max_unobserved_loss_fraction: float,
        initial_unobserved_loss_fraction: float,
    ) -> None:
        super().__init__()
        if not 0.0 < max_unobserved_loss_fraction < 1.0:
            raise ValueError("max_unobserved_loss_fraction必须在(0,1)")
        if not 0.0 <= initial_unobserved_loss_fraction < max_unobserved_loss_fraction:
            raise ValueError("initial_unobserved_loss_fraction必须小于最大值")
        self.max_unobserved_loss_fraction = float(max_unobserved_loss_fraction)
        # static attributes + normalized log rain + log fast/slow stores
        self.cell = nn.LSTMCell(int(static_dim) + 3, int(hidden_dim))
        self.controls = nn.Linear(int(hidden_dim), 4)
        with torch.no_grad():
            self.controls.bias.zero_()
            if initial_unobserved_loss_fraction > 0.0:
                ratio = initial_unobserved_loss_fraction / max_unobserved_loss_fraction
                self.controls.bias[1] = _logit(ratio)
            else:
                self.controls.bias[1] = -8.0
            self.controls.bias[2:].fill_(_inverse_softplus(0.1))

    def forward(
        self,
        static: torch.Tensor,
        rain_mm: torch.Tensor,
        rain_feature: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        seconds: float,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        h, c, fast_storage, slow_storage = state
        if (
            static.ndim != 2
            or rain_mm.ndim != 1
            or rain_feature.ndim != 1
            or static.shape[0] != rain_mm.shape[0]
            or rain_mm.shape != rain_feature.shape
        ):
            raise ValueError("water-balance cell输入形状错误")
        if (rain_mm < 0).any() or not torch.isfinite(rain_mm).all():
            raise ValueError("water-balance rain必须为有限非负mm")
        controls_input = torch.cat(
            [
                static,
                rain_feature.unsqueeze(-1),
                torch.log1p(fast_storage).unsqueeze(-1),
                torch.log1p(slow_storage).unsqueeze(-1),
            ],
            dim=-1,
        )
        h, c = self.cell(controls_input, (h, c))
        raw = self.controls(h)
        fast_partition = torch.sigmoid(raw[:, 0])
        loss_fraction = self.max_unobserved_loss_fraction * torch.sigmoid(raw[:, 1])
        fast_rate = torch.nn.functional.softplus(raw[:, 2])
        slow_rate = torch.nn.functional.softplus(raw[:, 3])

        loss = loss_fraction * rain_mm
        available_rain = rain_mm - loss
        fast_available = fast_storage + fast_partition * available_rain
        slow_available = slow_storage + (1.0 - fast_partition) * available_rain
        fast_release = continuous_release_fraction(fast_rate, seconds)
        slow_release = continuous_release_fraction(slow_rate, seconds)
        fast_runoff = fast_release * fast_available
        slow_runoff = slow_release * slow_available
        fast_next = fast_available - fast_runoff
        slow_next = slow_available - slow_runoff
        runoff = fast_runoff + slow_runoff

        total_before = fast_storage + slow_storage + rain_mm
        total_after = fast_next + slow_next + runoff + loss
        residual = total_before - total_after
        return runoff, (h, c, fast_next, slow_next), {
            "rain_partition_fast": fast_partition,
            "unobserved_loss_fraction": loss_fraction,
            "unobserved_loss_mm": loss,
            "release_rate_fast_per_hour": fast_rate,
            "release_rate_slow_per_hour": slow_rate,
            "storage_fast_mm": fast_next,
            "storage_slow_mm": slow_next,
            "residual_mm": residual,
        }


class MassConservingRunoffLSTM(nn.Module):
    """Sequential mass-conserving runoff model for the physical ablation."""

    def __init__(
        self,
        static_dim: int,
        hidden_dim: int,
        *,
        max_unobserved_loss_fraction: float = 0.90,
        initial_unobserved_loss_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cell = MassConservingRunoffLSTMCell(
            static_dim,
            hidden_dim,
            max_unobserved_loss_fraction=max_unobserved_loss_fraction,
            initial_unobserved_loss_fraction=initial_unobserved_loss_fraction,
        )

    def forward(
        self,
        static_features: torch.Tensor,
        rain: torch.Tensor,
        rain_features: torch.Tensor,
        area_km2: torch.Tensor,
        *,
        seconds: float = 3600.0,
        initial_state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=static_features.device.type, enabled=False):
            static = static_features.float()
            physical_rain = rain.float().squeeze(-1)
            rain_context = rain_features.float().squeeze(-1)
            area = area_km2.float()
            if static.ndim != 4 or physical_rain.ndim != 3 or rain_context.ndim != 3:
                raise ValueError("runoff要求static=[B,T,N,D], rain/rain_features=[B,T,N,1]")
            batch, steps, nodes, _ = static.shape
            if physical_rain.shape != (batch, steps, nodes) or rain_context.shape != physical_rain.shape:
                raise ValueError("water-balance rainfall形状不一致")
            if area.shape != (nodes,):
                raise ValueError("area_km2必须为[N]")
            if initial_state is None:
                zero_hidden = torch.zeros(
                    batch * nodes, self.hidden_dim, device=static.device, dtype=static.dtype
                )
                state = (
                    zero_hidden,
                    zero_hidden.clone(),
                    torch.zeros(batch * nodes, device=static.device, dtype=static.dtype),
                    torch.zeros(batch * nodes, device=static.device, dtype=static.dtype),
                )
            else:
                h0, c0, fast0, slow0 = initial_state
                if h0.shape != (batch, nodes, self.hidden_dim) or c0.shape != h0.shape:
                    raise ValueError("runoff initial h/c必须为[B,N,hidden_dim]")
                if fast0.shape != (batch, nodes) or slow0.shape != (batch, nodes):
                    raise ValueError("runoff initial storage必须为[B,N]")
                if (
                    not torch.isfinite(h0).all()
                    or not torch.isfinite(c0).all()
                    or not torch.isfinite(fast0).all()
                    or not torch.isfinite(slow0).all()
                    or (fast0 < 0).any()
                    or (slow0 < 0).any()
                ):
                    raise ValueError("runoff initial state非法")
                state = (
                    h0.float().reshape(batch * nodes, self.hidden_dim),
                    c0.float().reshape(batch * nodes, self.hidden_dim),
                    fast0.float().reshape(batch * nodes),
                    slow0.float().reshape(batch * nodes),
                )

            diagnostics: dict[str, list[torch.Tensor]] = {
                "runoff_water_balance_residual": [],
                "storage_fast_mm": [],
                "storage_slow_mm": [],
                "unobserved_loss_mm": [],
                "unobserved_loss_fraction": [],
                "rain_partition_fast": [],
                "release_rate_fast_per_hour": [],
                "release_rate_slow_per_hour": [],
            }
            runoff_series: list[torch.Tensor] = []
            for index in range(steps):
                runoff_mm, state, step = self.cell(
                    static[:, index].reshape(batch * nodes, -1),
                    physical_rain[:, index].reshape(-1),
                    rain_context[:, index].reshape(-1),
                    state,
                    seconds=seconds,
                )
                runoff_series.append(
                    runoff_mm.reshape(batch, nodes)
                    * area.unsqueeze(0)
                    * 1000.0
                    / float(seconds)
                )
                diagnostics["runoff_water_balance_residual"].append(
                    step["residual_mm"].reshape(batch, nodes)
                )
                for name in (
                    "storage_fast_mm",
                    "storage_slow_mm",
                    "unobserved_loss_mm",
                    "unobserved_loss_fraction",
                    "rain_partition_fast",
                    "release_rate_fast_per_hour",
                    "release_rate_slow_per_hour",
                ):
                    diagnostics[name].append(step[name].reshape(batch, nodes))

            h, c, fast, slow = state
            result = {
                name: torch.stack(values, dim=1) for name, values in diagnostics.items()
            }
            result.update(
                {
                    "storage_mm": result["storage_fast_mm"] + result["storage_slow_mm"],
                    "final_h": h.reshape(batch, nodes, self.hidden_dim),
                    "final_c": c.reshape(batch, nodes, self.hidden_dim),
                    "final_storage_fast_mm": fast.reshape(batch, nodes),
                    "final_storage_slow_mm": slow.reshape(batch, nodes),
                }
            )
            return torch.stack(runoff_series, dim=1), result


# Keep the public symbol stable for external imports. New code uses the explicit
# MassConservingRunoffLSTM name.
ContinuousTimeWaterBalanceLSTM = MassConservingRunoffLSTM
