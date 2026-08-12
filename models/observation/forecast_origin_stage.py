"""Forecast-origin dynamic stage memory for P3 delta-Z prediction."""
from __future__ import annotations

import torch
from torch import nn


class ForecastOriginStageMemory(nn.Module):
    """Predict a causal delta-Z residual from history state and future hydraulics.

    The monotone Q-Z observation module remains the physical backbone.  This
    module only predicts a cumulative residual that restores the stage-specific
    memory lost when P3 replaced the original 24 h recurrent warm-up with a
    forecast-origin state initializer.  Its hidden state is initialized from a
    dedicated stage context encoded from the full history window.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim必须大于0")
        # stage_context + compressed observed Z0 + compressed recent dZ/dt
        self.initial_state = nn.Linear(self.hidden_dim + 2, self.hidden_dim)
        # log(Qh), signed-log(Qh-Q0), log(1+channel storage), horizon fraction,
        # and whether Q0 was observed.
        self.forecast_gru = nn.GRU(5, self.hidden_dim, batch_first=True)
        self.increment_head = nn.Linear(self.hidden_dim, 1)
        # Start close to the existing monotone Q-Z backbone, but do not make the
        # residual branch gradient-dead on its first optimization step.
        nn.init.normal_(self.increment_head.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.increment_head.bias)

    @staticmethod
    def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(value.abs())

    def forward(
        self,
        stage_context: torch.Tensor,
        q_future: torch.Tensor,
        q_origin: torch.Tensor,
        q_origin_available: torch.Tensor,
        channel_future: torch.Tensor | None,
        z_origin: torch.Tensor,
        z_origin_available: torch.Tensor,
        recent_z_trend: torch.Tensor,
    ) -> torch.Tensor:
        if q_future.ndim != 3:
            raise ValueError("q_future必须为[B,T,N]")
        batch, steps, nodes = q_future.shape
        if stage_context.shape != (batch, nodes, self.hidden_dim):
            raise ValueError("stage_context必须为[B,N,hidden_dim]")
        for name, value in (
            ("q_origin", q_origin),
            ("q_origin_available", q_origin_available),
            ("z_origin", z_origin),
            ("z_origin_available", z_origin_available),
            ("recent_z_trend", recent_z_trend),
        ):
            if value.shape != (batch, nodes):
                raise ValueError(f"{name}必须为[B,N]")
        if channel_future is None:
            channel = torch.zeros_like(q_future)
        else:
            if channel_future.shape != q_future.shape:
                raise ValueError("channel_future必须与q_future同形状")
            channel = channel_future.float()
        if not torch.isfinite(q_future).all() or (q_future < 0).any():
            raise ValueError("q_future必须为有限非负值")
        if not torch.isfinite(q_origin).all():
            raise ValueError("q_origin包含NaN/Inf")
        if not torch.isfinite(z_origin).all() or not torch.isfinite(recent_z_trend).all():
            raise ValueError("Z forecast-origin状态包含NaN/Inf")

        z0_feature = self._signed_log1p(z_origin)
        trend_feature = self._signed_log1p(recent_z_trend)
        initial_input = torch.cat(
            [stage_context, z0_feature.unsqueeze(-1), trend_feature.unsqueeze(-1)],
            dim=-1,
        )
        hidden0 = torch.tanh(self.initial_state(initial_input))
        hidden0 = hidden0.reshape(batch * nodes, self.hidden_dim).unsqueeze(0)

        q0 = q_origin.unsqueeze(1).expand(-1, steps, -1)
        q_delta = q_future - q0
        horizon = torch.arange(
            1, steps + 1, device=q_future.device, dtype=q_future.dtype
        ).view(1, steps, 1)
        horizon = horizon.expand(batch, -1, nodes) / float(max(steps, 1))
        q0_available = q_origin_available.to(q_future.dtype).unsqueeze(1).expand(-1, steps, -1)
        sequence = torch.stack(
            [
                torch.log1p(q_future.clamp_min(0)),
                self._signed_log1p(q_delta),
                torch.log1p(channel.clamp_min(0)),
                horizon,
                q0_available,
            ],
            dim=-1,
        )
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * nodes, steps, 5)
        encoded, _ = self.forecast_gru(sequence, hidden0)
        increments = self.increment_head(encoded).squeeze(-1)
        increments = increments.reshape(batch, nodes, steps).permute(0, 2, 1)
        residual = torch.cumsum(increments, dim=1)
        # Delta-Z supervision is only meaningful where a forecast-origin stage
        # exists.  Other nodes retain the physical backbone without an
        # unconstrained residual branch.
        return torch.where(
            z_origin_available.bool().unsqueeze(1), residual, torch.zeros_like(residual)
        )
