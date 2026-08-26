"""Continuous-time water-balanced two-reservoir rainfall-runoff module.

The release controls are rates in 1/hour.  A physical time step ``dt`` is
converted to a release fraction ``1-exp(-lambda*dt)`` so the same learned
recession rate keeps its meaning when forcing changes from hourly to minute
resolution during transfer learning.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("inverse softplus value必须>0")
    return math.log(math.expm1(value))


def continuous_release_fraction(
    rate_per_hour: torch.Tensor, seconds: float
) -> torch.Tensor:
    """Convert a positive continuous release rate to a step-specific fraction."""

    dt_hours = float(seconds) / 3600.0
    if not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("seconds必须为有限正数")
    if not torch.isfinite(rate_per_hour).all() or (rate_per_hour < 0).any():
        raise ValueError("rate_per_hour必须为有限非负值")
    return -torch.expm1(-rate_per_hour * dt_hours)


class ContinuousTimeWaterBalanceLSTMCell(nn.Module):
    """Mass-conserving two-reservoir cell with resolution-invariant recession."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cell = nn.LSTMCell(int(input_dim), int(hidden_dim))
        self.controls = nn.Linear(int(hidden_dim), 3)
        with torch.no_grad():
            # Start near a neutral rainfall partition and ~0.1 h^-1 recession.
            self.controls.bias.zero_()
            self.controls.bias[1:].fill_(_inverse_softplus(0.1))

    def forward(
        self,
        x: torch.Tensor,
        rain_mm: torch.Tensor,
        state: tuple[torch.Tensor, ...],
        *,
        seconds: float,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        h, c, sf, ss = state
        h, c = self.cell(x, (h, c))
        raw = self.controls(h)
        partition = torch.sigmoid(raw[:, 0])
        rate_fast = torch.nn.functional.softplus(raw[:, 1])
        rate_slow = torch.nn.functional.softplus(raw[:, 2])
        k_fast = continuous_release_fraction(rate_fast, seconds)
        k_slow = continuous_release_fraction(rate_slow, seconds)

        fast_available = sf + partition * rain_mm
        slow_available = ss + (1.0 - partition) * rain_mm
        fast_next = fast_available * (1.0 - k_fast)
        slow_next = slow_available * (1.0 - k_slow)

        total = (sf + ss) + rain_mm
        runoff = total - (fast_next + slow_next)
        residual = total - ((fast_next + slow_next) + runoff)
        return runoff, (h, c, fast_next, slow_next), {
            "rain_partition_fast": partition,
            "release_rate_fast_per_hour": rate_fast,
            "release_rate_slow_per_hour": rate_slow,
            "release_fraction_fast": k_fast,
            "release_fraction_slow": k_slow,
            "storage_fast": fast_next,
            "storage_slow": slow_next,
            "residual": residual,
        }


class ContinuousTimeWaterBalanceLSTM(nn.Module):
    """Sequential runoff model whose learned recession parameters survive dt changes."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cell = ContinuousTimeWaterBalanceLSTMCell(input_dim, hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        rain: torch.Tensor,
        area_km2: torch.Tensor,
        seconds: float = 3600.0,
        initial_state: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]
        | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=features.device.type, enabled=False):
            x = features.float()
            p = rain.float().squeeze(-1)
            area = area_km2.float()
            if x.ndim != 4 or p.ndim != 3:
                raise ValueError("runoff要求features=[B,T,N,D], rain=[B,T,N,1]")
            b, t, n, _ = x.shape
            if tuple(p.shape) != (b, t, n):
                raise ValueError("rain与features形状不一致")
            if tuple(area.shape) != (n,):
                raise ValueError("area_km2必须为[N]")

            if initial_state is None:
                zero_hidden = torch.zeros(b * n, self.hidden_dim, device=x.device)
                state = (
                    zero_hidden,
                    zero_hidden.clone(),
                    torch.zeros(b * n, device=x.device),
                    torch.zeros(b * n, device=x.device),
                )
            else:
                h0, c0, sf0, ss0 = initial_state
                if h0.shape != (b, n, self.hidden_dim) or c0.shape != h0.shape:
                    raise ValueError("runoff initial h/c必须为[B,N,hidden_dim]")
                if sf0.shape != (b, n) or ss0.shape != (b, n):
                    raise ValueError("runoff initial storage必须为[B,N]")
                if (
                    not torch.isfinite(h0).all()
                    or not torch.isfinite(c0).all()
                    or not torch.isfinite(sf0).all()
                    or not torch.isfinite(ss0).all()
                    or (sf0 < 0).any()
                    or (ss0 < 0).any()
                ):
                    raise ValueError("runoff initial state非法")
                state = (
                    h0.float().reshape(b * n, self.hidden_dim),
                    c0.float().reshape(b * n, self.hidden_dim),
                    sf0.float().reshape(b * n),
                    ss0.float().reshape(b * n),
                )

            runoff_series: list[torch.Tensor] = []
            residuals: list[torch.Tensor] = []
            fast_storage: list[torch.Tensor] = []
            slow_storage: list[torch.Tensor] = []
            rate_fast: list[torch.Tensor] = []
            rate_slow: list[torch.Tensor] = []
            partition: list[torch.Tensor] = []
            for index in range(t):
                runoff_mm, state, diagnostics = self.cell(
                    x[:, index].reshape(b * n, -1),
                    p[:, index].reshape(-1),
                    state,
                    seconds=seconds,
                )
                runoff_series.append(
                    runoff_mm.reshape(b, n) * area[None] * 1000.0 / float(seconds)
                )
                residuals.append(diagnostics["residual"].reshape(b, n))
                fast_storage.append(diagnostics["storage_fast"].reshape(b, n))
                slow_storage.append(diagnostics["storage_slow"].reshape(b, n))
                rate_fast.append(
                    diagnostics["release_rate_fast_per_hour"].reshape(b, n)
                )
                rate_slow.append(
                    diagnostics["release_rate_slow_per_hour"].reshape(b, n)
                )
                partition.append(diagnostics["rain_partition_fast"].reshape(b, n))

            h, c, sf, ss = state
            q = torch.stack(runoff_series, dim=1)
            return q, {
                "runoff_water_balance_residual": torch.stack(residuals, dim=1),
                "storage_fast": torch.stack(fast_storage, dim=1),
                "storage_slow": torch.stack(slow_storage, dim=1),
                "storage": torch.stack(fast_storage, dim=1)
                + torch.stack(slow_storage, dim=1),
                "release_rate_fast_per_hour": torch.stack(rate_fast, dim=1),
                "release_rate_slow_per_hour": torch.stack(rate_slow, dim=1),
                "rain_partition_fast": torch.stack(partition, dim=1),
                "final_h": h.reshape(b, n, self.hidden_dim),
                "final_c": c.reshape(b, n, self.hidden_dim),
                "final_storage_fast_mm": sf.reshape(b, n),
                "final_storage_slow_mm": ss.reshape(b, n),
            }
