"""Differentiable station-wise Q-Z rating functions fitted on TRAIN only."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


class _StationIndexMixin:
    num_stations: int

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


class TrainFittedLinearRating(_StationIndexMixin, nn.Module):
    """Apply fixed TRAIN-only station linear rating curves preserving dZ/dQ."""

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


class TrainFittedMonotoneRating(_StationIndexMixin, nn.Module):
    """Frozen low-DOF TRAIN-only monotone station rating functions.

    Interior knots come from aggregated TRAIN Q/Z pairs and are not trainable.
    Interpolation is piecewise linear and differentiable with respect to Q. For
    Q outside the calibrated TRAIN range, the derivative is the station-wide
    TRAIN OLS Q-Z slope, not the potentially noisy first/last local segment.
    This keeps the physical Z->Q gradient data-derived and stable without a
    manually imposed slope cap.
    """

    def __init__(self, stations: int) -> None:
        super().__init__()
        if stations <= 0:
            raise ValueError("stations必须为正")
        self.num_stations = int(stations)
        self.register_buffer("available", torch.zeros(stations, dtype=torch.bool))
        self.register_buffer("knot_count", torch.zeros(stations, dtype=torch.long))
        self.register_buffer(
            "extrapolation_slope", torch.zeros(stations, dtype=torch.float32)
        )
        self.register_buffer("q_knots", torch.zeros((stations, 2), dtype=torch.float32))
        self.register_buffer("z_knots", torch.zeros((stations, 2), dtype=torch.float32))

    def configure(
        self,
        statistics: Mapping[str, Any],
        station_to_index: Mapping[str, int],
    ) -> None:
        station_stats = statistics.get("stations")
        if not isinstance(station_stats, Mapping):
            raise ValueError("calibrated rating statistics缺少stations")
        prepared: dict[int, tuple[list[float], list[float], float]] = {}
        max_knots = 2
        for station_id, values in station_stats.items():
            if station_id not in station_to_index:
                continue
            if not isinstance(values, Mapping) or not bool(values.get("usable_calibrated", False)):
                continue
            q_values = [float(value) for value in values.get("calibrated_q_knots_m3s", [])]
            z_values = [float(value) for value in values.get("calibrated_z_knots_m", [])]
            external_slope = float(values.get("extrapolation_slope_m_per_m3s", float("nan")))
            if len(q_values) != len(z_values) or len(q_values) < 2:
                raise ValueError(f"STATION_ID={station_id}: calibrated knots无效")
            if any(b <= a for a, b in zip(q_values, q_values[1:])):
                raise ValueError(f"STATION_ID={station_id}: calibrated Q knots必须严格递增")
            if any(b <= a for a, b in zip(z_values, z_values[1:])):
                raise ValueError(f"STATION_ID={station_id}: calibrated Z knots必须严格递增")
            if not torch.isfinite(torch.tensor(external_slope)) or external_slope <= 0.0:
                raise ValueError(
                    f"STATION_ID={station_id}: TRAIN全局rating外推斜率必须为有限正数"
                )
            index = int(station_to_index[station_id])
            if not 0 <= index < self.num_stations:
                raise ValueError(f"rating station index越界: {station_id} -> {index}")
            prepared[index] = (q_values, z_values, external_slope)
            max_knots = max(max_knots, len(q_values))

        device = self.available.device
        q_knots = torch.zeros(
            (self.num_stations, max_knots), dtype=torch.float32, device=device
        )
        z_knots = torch.zeros(
            (self.num_stations, max_knots), dtype=torch.float32, device=device
        )
        counts = torch.zeros(self.num_stations, dtype=torch.long, device=device)
        available = torch.zeros(self.num_stations, dtype=torch.bool, device=device)
        external = torch.zeros(self.num_stations, dtype=torch.float32, device=device)
        for index, (q_values, z_values, external_slope) in prepared.items():
            count = len(q_values)
            q_tensor = torch.tensor(q_values, dtype=torch.float32, device=device)
            z_tensor = torch.tensor(z_values, dtype=torch.float32, device=device)
            q_knots[index, :count] = q_tensor
            z_knots[index, :count] = z_tensor
            q_knots[index, count:] = q_tensor[-1]
            z_knots[index, count:] = z_tensor[-1]
            counts[index] = count
            external[index] = external_slope
            available[index] = True
        self.q_knots = q_knots
        self.z_knots = z_knots
        self.knot_count.copy_(counts)
        self.extrapolation_slope.copy_(external)
        self.available.copy_(available)

    @staticmethod
    def _interp_with_global_extrapolation(
        value: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        external_slope: torch.Tensor,
    ) -> torch.Tensor:
        """PWL interpolation; station-global TRAIN slope outside knot range."""
        right = torch.searchsorted(
            x.contiguous(), value.contiguous()
        ).clamp(1, x.numel() - 1)
        left = right - 1
        x0 = x[left]
        x1 = x[right]
        y0 = y[left]
        y1 = y[right]
        local_slope = (y1 - y0) / (x1 - x0)
        inside = y0 + local_slope * (value - x0)
        below = y[0] + external_slope * (value - x[0])
        above = y[-1] + external_slope * (value - x[-1])
        return torch.where(
            value < x[0], below, torch.where(value > x[-1], above, inside)
        )

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
        output = torch.zeros_like(q)
        station_available = self.available[indices]
        for local_node, global_index in enumerate(indices.tolist()):
            if not bool(self.available[global_index]):
                continue
            count = int(self.knot_count[global_index].item())
            x = self.q_knots[global_index, :count]
            y = self.z_knots[global_index, :count]
            output[..., local_node] = self._interp_with_global_extrapolation(
                q[..., local_node].clamp_min(0),
                x,
                y,
                self.extrapolation_slope[global_index],
            )
        return output, station_available

    def inverse_from_z(
        self,
        z: torch.Tensor,
        station_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z.ndim != 2:
            raise ValueError("rating inverse z必须为[B,N]")
        batch, nodes = z.shape
        indices = self._indices(nodes, station_index, z.device)
        output = torch.zeros_like(z)
        station_available = self.available[indices]
        available = station_available.view(1, nodes).expand(batch, -1)
        for local_node, global_index in enumerate(indices.tolist()):
            if not bool(self.available[global_index]):
                continue
            count = int(self.knot_count[global_index].item())
            # Strict monotonicity makes the inverse unique. Outside the observed
            # Z range, use the reciprocal of the same TRAIN-global Q->Z slope.
            x = self.z_knots[global_index, :count]
            y = self.q_knots[global_index, :count]
            reciprocal = 1.0 / self.extrapolation_slope[global_index]
            output[..., local_node] = self._interp_with_global_extrapolation(
                z[..., local_node], x, y, reciprocal
            ).clamp_min(0)
        return torch.where(available, output, torch.zeros_like(output)), available
