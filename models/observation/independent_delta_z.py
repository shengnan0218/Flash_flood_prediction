"""Independent forecast-origin delta-Z prediction head for P3."""
from __future__ import annotations

import torch
from torch import nn


class IndependentDeltaZHead(nn.Module):
    """Predict the complete future delta-Z trajectory with a feed-forward head.

    The head is deliberately independent from the monotone Q-Z observation
    equation.  It receives a stage-specific history context, forecast-origin
    stage/trend, station identity, and detached future hydraulic predictions as
    explanatory features, then predicts all forecast horizons jointly.  The
    detached hydraulic features let Z use Q information without allowing the
    primary Z supervision to rewrite the Q transition model; Q-Z interaction is
    handled separately by a weak consistency loss.
    """

    def __init__(
        self,
        history_context_dim: int,
        hidden_dim: int,
        horizon: int,
        stations: int,
    ) -> None:
        super().__init__()
        self.history_context_dim = int(history_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        if self.history_context_dim <= 0 or self.hidden_dim <= 0 or self.horizon <= 0:
            raise ValueError("independent delta-Z head维度必须为正")
        if int(stations) <= 0:
            raise ValueError("stations必须为正")

        self.station_embedding = nn.Embedding(int(stations), self.hidden_dim)
        # history context + station embedding + 6 forecast-origin scalars +
        # [Q_h, Q_h-Q0, channel_h] for every forecast horizon.
        input_dim = (
            self.history_context_dim
            + self.hidden_dim
            + 6
            + 3 * self.horizon
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon),
        )
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(final.bias)

    @staticmethod
    def _signed_log1p(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(value.abs())

    def forward(
        self,
        history_context: torch.Tensor,
        q_future: torch.Tensor,
        q_origin: torch.Tensor,
        q_origin_available: torch.Tensor,
        channel_future: torch.Tensor | None,
        initial_channel: torch.Tensor,
        z_origin: torch.Tensor,
        z_origin_available: torch.Tensor,
        recent_z_trend: torch.Tensor,
        recent_z_trend_available: torch.Tensor,
        station_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if q_future.ndim != 3:
            raise ValueError("q_future必须为[B,F,N]")
        batch, steps, nodes = q_future.shape
        if steps != self.horizon:
            raise ValueError(
                f"q_future horizon应为{self.horizon}，实际={steps}"
            )
        if history_context.shape != (batch, nodes, self.history_context_dim):
            raise ValueError("history_context必须为[B,N,history_context_dim]")
        for name, value in (
            ("q_origin", q_origin),
            ("q_origin_available", q_origin_available),
            ("initial_channel", initial_channel),
            ("z_origin", z_origin),
            ("z_origin_available", z_origin_available),
            ("recent_z_trend", recent_z_trend),
            ("recent_z_trend_available", recent_z_trend_available),
        ):
            if value.shape != (batch, nodes):
                raise ValueError(f"{name}必须为[B,N]")
        if station_index is None:
            station_index = torch.arange(nodes, device=q_future.device, dtype=torch.long)
        if station_index.dtype != torch.long or tuple(station_index.shape) != (nodes,):
            raise ValueError("station_index必须为[N] LongTensor")
        if (station_index < 0).any() or (station_index >= self.station_embedding.num_embeddings).any():
            raise ValueError("station_index超出independent delta-Z station embedding范围")

        if channel_future is None:
            channel = torch.zeros_like(q_future)
        else:
            if channel_future.shape != q_future.shape:
                raise ValueError("channel_future必须与q_future同形状")
            channel = channel_future.float()
        if not torch.isfinite(q_future).all() or (q_future < 0).any():
            raise ValueError("q_future必须为有限非负值")
        if not torch.isfinite(q_origin).all() or not torch.isfinite(initial_channel).all():
            raise ValueError("forecast-origin Q/channel状态含NaN/Inf")
        if not torch.isfinite(z_origin).all() or not torch.isfinite(recent_z_trend).all():
            raise ValueError("forecast-origin Z状态含NaN/Inf")

        # Primary Z supervision must not directly alter the Q/routing model.
        # The weak Q-Z consistency loss is the only gradient bridge between the
        # independent heads.
        q_feature = q_future.detach()
        q0_feature = q_origin.detach()
        channel_feature = channel.detach()
        initial_channel_feature = initial_channel.detach()

        station = self.station_embedding(station_index.to(q_future.device))
        station = station.unsqueeze(0).expand(batch, -1, -1)
        q0 = q0_feature.unsqueeze(1).expand(-1, steps, -1)
        scalar_features = torch.stack(
            [
                self._signed_log1p(z_origin),
                self._signed_log1p(recent_z_trend),
                torch.log1p(q0_feature.clamp_min(0)),
                torch.log1p(initial_channel_feature.clamp_min(0)),
                q_origin_available.to(q_future.dtype),
                recent_z_trend_available.to(q_future.dtype),
            ],
            dim=-1,
        )
        q_features = torch.cat(
            [
                torch.log1p(q_feature.clamp_min(0)),
                self._signed_log1p(q_feature - q0),
                torch.log1p(channel_feature.clamp_min(0)),
            ],
            dim=1,
        )
        # [B,3F,N] -> [B,N,3F]
        q_features = q_features.permute(0, 2, 1)
        features = torch.cat(
            [history_context, station, scalar_features, q_features], dim=-1
        )
        delta = self.network(features)  # [B,N,F]
        delta = delta.permute(0, 2, 1).contiguous()
        return torch.where(
            z_origin_available.bool().unsqueeze(1),
            delta,
            torch.zeros_like(delta),
        )
