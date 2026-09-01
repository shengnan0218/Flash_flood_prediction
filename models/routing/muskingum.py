"""Low-dimensional differentiable Muskingum routing on a directed river graph.

The router intentionally learns only a bounded correction to a travel-time
prior.  It does not infer channel width, depth, or Manning roughness from
outlet discharge, because those hydraulic quantities are not identifiable from
the available supervision.  Each reach therefore has one effective routing
parameter (travel time), regionalized from observed reach and node attributes.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from data.schema import topological_levels


class EdgeTravelTimeNetwork(nn.Module):
    """Predict a deliberately small log correction around a physical K prior."""

    def __init__(
        self, input_dim: int, hidden_dim: int, max_log_adjustment: float
    ) -> None:
        super().__init__()
        if not math.isfinite(max_log_adjustment) or max_log_adjustment <= 0:
            raise ValueError("max_log_adjustment必须为有限正数")
        self.max_log_adjustment = float(max_log_adjustment)
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        # Start exactly at the hand-specified travel-time prior.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.max_log_adjustment * torch.tanh(
            self.network(features.float()).squeeze(-1)
        )


class MuskingumGraphRouter(nn.Module):
    """Mass-conservative, stable Muskingum routing on a converging DAG.

    ``q_lat`` is the lateral discharge generated at each computational node.
    The graph operation routes it in topological order, conserving mass within
    the trapezoidal continuity relation used by the Muskingum recurrence.
    No observed Q/Z is accepted by this module.
    """

    def __init__(
        self,
        node_static_dim: int,
        edge_static_dim: int,
        hidden_dim: int,
        cfg: dict[str, Any],
        *,
        seconds_per_step: float,
    ) -> None:
        super().__init__()
        if edge_static_dim != 2:
            raise ValueError(
                "muskingum_gnn要求两个边属性：reach_length_m和reach_slope_m_per_m"
            )
        self.node_static_dim = int(node_static_dim)
        self.edge_static_dim = int(edge_static_dim)
        self.dt = float(seconds_per_step)
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("seconds_per_step必须为有限正数")

        self.reference_slope = float(cfg["reference_slope_m_per_m"])
        self.reference_velocity = float(cfg["reference_velocity_mps"])
        self.slope_velocity_exponent = float(cfg["slope_velocity_exponent"])
        velocity_bounds = tuple(float(value) for value in cfg["velocity_bounds_mps"])
        travel_bounds = tuple(float(value) for value in cfg["travel_time_bounds_hours"])
        self.muskingum_x = float(cfg["muskingum_x"])
        self.minimum_length = float(cfg["minimum_length_m"])
        self.minimum_slope = float(cfg["minimum_slope_m_per_m"])

        if (
            not math.isfinite(self.reference_slope)
            or self.reference_slope <= 0
            or not math.isfinite(self.reference_velocity)
            or self.reference_velocity <= 0
            or not math.isfinite(self.slope_velocity_exponent)
            or velocity_bounds[0] <= 0
            or velocity_bounds[0] >= velocity_bounds[1]
            or travel_bounds[0] <= 0
            or travel_bounds[0] >= travel_bounds[1]
            or not 0.0 <= self.muskingum_x < 0.5
            or self.minimum_length <= 0
            or self.minimum_slope <= 0
        ):
            raise ValueError("Muskingum路由配置非法")

        stability_minimum_hours = self.dt / (
            2.0 * (1.0 - self.muskingum_x) * 3600.0
        )
        if travel_bounds[0] < stability_minimum_hours:
            raise ValueError(
                "travel_time_bounds_hours下界必须满足Muskingum非负系数稳定条件："
                f">={stability_minimum_hours:.6g} h"
            )
        # For X>0, a fixed external time step also sets an upper K bound for
        # non-negative C0.  Check the complete configured range at startup,
        # rather than failing only when the first long reach is routed.
        # X=0 is the linear-reservoir Muskingum special case and has no upper
        # K bound, which makes it appropriate for the hourly forecasts here.
        if self.muskingum_x > 0.0:
            stability_maximum_hours = self.dt / (
                2.0 * self.muskingum_x * 3600.0
            )
            if travel_bounds[1] > stability_maximum_hours:
                raise ValueError(
                    "travel_time_bounds_hours上界与muskingum_x/时间步不兼容；"
                    f"X={self.muskingum_x:.6g}时必须<={stability_maximum_hours:.6g} h，"
                    "或使用X=0的线性蓄水库Muskingum特例"
                )
        self.velocity_low, self.velocity_high = velocity_bounds
        self.travel_time_low_h, self.travel_time_high_h = travel_bounds
        self.travel_time_network = EdgeTravelTimeNetwork(
            self.edge_static_dim + 2 * self.node_static_dim,
            hidden_dim,
            float(cfg["max_log_travel_time_adjustment"]),
        )
        self._topology_cache: dict[
            tuple[int, tuple[int, ...]], list[torch.Tensor]
        ] = {}
        self._device_topology_cache: dict[
            tuple[tuple[int, tuple[int, ...]], str], list[torch.Tensor]
        ] = {}

    def _edge_levels(
        self, edge_index: torch.Tensor, nodes: int, device: torch.device
    ) -> list[torch.Tensor]:
        edge_cpu = edge_index.detach().cpu()
        key = (nodes, tuple(edge_cpu.flatten().tolist()))
        cached = self._topology_cache.get(key)
        if cached is None:
            levels, _ = topological_levels(edge_cpu, nodes)
            source = edge_cpu[0]
            outdegree = torch.bincount(source, minlength=nodes)
            divergent = torch.where(outdegree > 1)[0].tolist()
            if divergent:
                raise ValueError(
                    "muskingum_gnn不支持未提供分流权重的出度>1节点，"
                    f"节点索引={divergent}"
                )
            source_list = source.tolist()
            cached = [
                torch.tensor(
                    [
                        edge
                        for edge, source_index in enumerate(source_list)
                        if source_index in level
                    ],
                    dtype=torch.long,
                )
                for level in levels
            ]
            self._topology_cache[key] = cached
        device_key = (key, str(device))
        active = self._device_topology_cache.get(device_key)
        if active is None:
            active = [edge_ids.to(device) for edge_ids in cached if edge_ids.numel()]
            self._device_topology_cache[device_key] = active
        return active

    def _prior_travel_time_hours(
        self, length_m: torch.Tensor, slope: torch.Tensor
    ) -> torch.Tensor:
        slope_factor = (
            (slope / self.reference_slope).clamp(0.05, 20.0)
            .pow(self.slope_velocity_exponent)
        )
        velocity = (
            self.reference_velocity * slope_factor
        ).clamp(self.velocity_low, self.velocity_high)
        return (length_m / velocity / 3600.0).clamp(
            self.travel_time_low_h, self.travel_time_high_h
        )

    def _route_sequence(
        self, inflow: torch.Tensor, travel_time_hours: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route independent reach series with exact trapezoidal continuity."""

        if inflow.ndim != 3:
            raise ValueError("Muskingum inflow必须为[B,T,E]")
        batch, steps, edges = inflow.shape
        if travel_time_hours.shape != (edges,):
            raise ValueError("travel_time_hours必须为[E]")
        k_seconds = travel_time_hours * 3600.0
        denominator = 2.0 * k_seconds * (1.0 - self.muskingum_x) + self.dt
        c0 = (self.dt - 2.0 * k_seconds * self.muskingum_x) / denominator
        c1 = (self.dt + 2.0 * k_seconds * self.muskingum_x) / denominator
        c2 = (2.0 * k_seconds * (1.0 - self.muskingum_x) - self.dt) / denominator
        if (c0 < -1.0e-7).any() or (c1 < -1.0e-7).any() or (c2 < -1.0e-7).any():
            raise FloatingPointError("Muskingum系数出现负值；请检查K下界或时间步")

        previous_inflow = torch.zeros(batch, edges, device=inflow.device, dtype=inflow.dtype)
        previous_outflow = torch.zeros_like(previous_inflow)
        previous_storage = torch.zeros_like(previous_inflow)
        outflow_steps: list[torch.Tensor] = []
        storage_steps: list[torch.Tensor] = []
        residual_steps: list[torch.Tensor] = []
        for input_t in inflow.unbind(dim=1):
            output_t = (
                c0.unsqueeze(0) * input_t
                + c1.unsqueeze(0) * previous_inflow
                + c2.unsqueeze(0) * previous_outflow
            ).clamp_min(0.0)
            storage_t = k_seconds.unsqueeze(0) * (
                self.muskingum_x * input_t
                + (1.0 - self.muskingum_x) * output_t
            )
            residual_t = (
                storage_t
                - previous_storage
                - 0.5
                * self.dt
                * ((previous_inflow + input_t) - (previous_outflow + output_t))
            )
            outflow_steps.append(output_t)
            storage_steps.append(storage_t)
            residual_steps.append(residual_t)
            previous_inflow, previous_outflow, previous_storage = (
                input_t,
                output_t,
                storage_t,
            )
        return (
            torch.stack(outflow_steps, dim=1),
            torch.stack(storage_steps, dim=1),
            torch.stack(residual_steps, dim=1),
        )

    def forward(
        self,
        q_lat: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
        *,
        neural_edge_static: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=q_lat.device.type, enabled=False):
            q_lat = q_lat.float()
            node_static = node_static.float()
            edge_static = edge_static.float()
            neural_edge_static = neural_edge_static.float()
            edge_index = edge_index.long()
            if q_lat.ndim != 3:
                raise ValueError("q_lat必须为[B,T,N]")
            batch, steps, nodes = q_lat.shape
            edges = int(edge_index.shape[1])
            if steps <= 0 or node_static.shape != (nodes, self.node_static_dim):
                raise ValueError("Muskingum节点输入形状错误")
            if edge_static.shape != (edges, self.edge_static_dim):
                raise ValueError("Muskingum边属性形状错误")
            if neural_edge_static.shape != edge_static.shape:
                raise ValueError("neural_edge_static必须与edge_static同形")
            if (
                not torch.isfinite(q_lat).all()
                or (q_lat < 0).any()
                or not torch.isfinite(edge_static).all()
                or not torch.isfinite(neural_edge_static).all()
            ):
                raise ValueError("Muskingum输入必须为有限值，且q_lat必须非负")
            if edges == 0:
                zeros = torch.zeros(batch, steps, device=q_lat.device, dtype=q_lat.dtype)
                return q_lat, {
                    "routing_mass_balance_residual_m3": zeros,
                    "routing_storage_m3": zeros,
                    "routing_travel_time_hours": torch.empty(0, device=q_lat.device),
                    "routing_travel_time_prior_hours": torch.empty(0, device=q_lat.device),
                    "routing_log_travel_time_adjustment": torch.empty(0, device=q_lat.device),
                }

            length = edge_static[:, 0]
            slope = edge_static[:, 1]
            if (length < self.minimum_length).any() or (slope < 0).any():
                raise ValueError("Muskingum河段长度或坡降非法")
            slope = slope.clamp_min(self.minimum_slope)
            source, destination = edge_index
            features = torch.cat(
                [
                    neural_edge_static,
                    node_static.index_select(0, source),
                    node_static.index_select(0, destination),
                ],
                dim=-1,
            )
            prior_hours = self._prior_travel_time_hours(length, slope)
            log_adjustment = self.travel_time_network(features)
            travel_hours = (prior_hours * torch.exp(log_adjustment)).clamp(
                self.travel_time_low_h, self.travel_time_high_h
            )

            routed = q_lat.clone()
            all_storage: list[torch.Tensor] = []
            all_residual: list[torch.Tensor] = []
            for edge_ids in self._edge_levels(edge_index, nodes, q_lat.device):
                sources = source.index_select(0, edge_ids)
                destinations = destination.index_select(0, edge_ids)
                outflow, storage, residual = self._route_sequence(
                    routed.index_select(2, sources),
                    travel_hours.index_select(0, edge_ids),
                )
                routed = routed.index_add(2, destinations, outflow)
                all_storage.append(storage.sum(dim=-1))
                all_residual.append(residual.sum(dim=-1))
            storage_total = torch.stack(all_storage, dim=0).sum(dim=0)
            residual_total = torch.stack(all_residual, dim=0).sum(dim=0)
            return routed, {
                "routing_mass_balance_residual_m3": residual_total,
                "routing_storage_m3": storage_total,
                "routing_travel_time_hours": travel_hours,
                "routing_travel_time_prior_hours": prior_hours,
                "routing_log_travel_time_adjustment": log_adjustment,
                "routing_muskingum_x": torch.full_like(travel_hours, self.muskingum_x),
            }
