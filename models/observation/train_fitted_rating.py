"""Differentiable station-wise linear Q-Z rating curves fitted on TRAIN only."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class TrainFittedLinearRating(nn.Module):
    """Apply fixed TRAIN-only station rating curves while preserving dZ/dQ.

    The fitted slope/intercept are data facts, not trainable parameters.  They
    are registered as buffers so they travel with checkpoints and devices while
    gradients still flow from Z loss through the affine transformation into Q.
    """

    def __init__(self, stations: int) -> None:
        super().__init__()
        if stations <= 0:
            raise ValueError("stations必须为正")
        self.num_stations = int(stations)
        self.register_buffer("slope", torch.zeros(stations, dtype=torch.float32))
        self.register_buffer("intercept", torch.zeros(stations, dtype=torch.float32))
        self.register_buffer("available", torch.zeros(stations, dtype=torch.bool))

    def configure(
        self,
        statistics: Mapping[str, Any],
        station_to_index: Mapping[str, int],
    ) -> None:
        station_stats = statistics.get("stations")
        if not isinstance(station_stats, Mapping):
            raise ValueError("rating statistics缺少stations")
        slope = torch.zeros_like(self.slope)
        intercept = torch.zeros_like(self.intercept)
        available = torch.zeros_like(self.available)
        for station_id, values in station_stats.items():
            if station_id not in station_to_index:
                continue
            if not isinstance(values, Mapping) or not bool(values.get("usable_linear", False)):
                continue
            index = int(station_to_index[station_id])
            if not 0 <= index < self.num_stations:
                raise ValueError(f"rating station index越界: {station_id} -> {index}")
            a = float(values["slope_m_per_m3s"])
            b = float(values["intercept_m"])
            if not (a > 0.0):
                raise ValueError(f"STATION_ID={station_id}: rating slope必须>0")
            slope[index] = a
            intercept[index] = b
            available[index] = True
        self.slope.copy_(slope)
        self.intercept.copy_(intercept)
        self.available.copy_(available)

    def _indices(
        self, nodes: int, station_index: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if station_index is None:
            if nodes > self.num_stations:
                raise ValueError("当前图节点数超过rating站点目录")
            return torch.arange(nodes, device=device, dtype=torch.long)
        if station_index.dtype != torch.long or tuple(station_index.shape) != (nodes,):
            raise ValueError("station_index必须为[N] LongTensor")
        indices = station_index.to(device=device)
        if (indices < 0).any() or (indices >= self.num_stations).any():
            raise ValueError("station_index超出rating站点目录")
        return indices

    def forward(
        self,
        q: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 3:
            raise ValueError("rating q必须为[B,T,N]")
        if not torch.isfinite(q).all():
            raise FloatingPointError("rating q含NaN/Inf")
        _, _, nodes = q.shape
        indices = self._indices(nodes, station_index, q.device)
        a = self.slope[indices].view(1, 1, nodes)
        b = self.intercept[indices].view(1, 1, nodes)
        level = a * q.clamp_min(0) + b
        available = self.available[indices]
        return level, available

    def inverse_from_z(
        self,
        z: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("rating inverse z必须为[B,N]")
        batch, nodes = z.shape
        indices = self._indices(nodes, station_index, z.device)
        a = self.slope[indices].view(1, nodes)
        b = self.intercept[indices].view(1, nodes)
        available = self.available[indices].view(1, nodes).expand(batch, -1)
        safe_a = torch.where(available, a, torch.ones_like(a))
        q = ((z - b) / safe_a).clamp_min(0)
        return torch.where(available, q, torch.zeros_like(q)), available
