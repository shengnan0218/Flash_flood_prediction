"""Execution-optimized differentiable conservative kinematic-wave routing.

This module intentionally preserves the physical equations, solver settings,
state semantics and diagnostics of :mod:`models.routing.kinematic_wave`.
Only the execution order is changed: each topological edge level is routed over
its full time sequence before its complete outflow series is added downstream.
For a converging DAG this is algebraically equivalent to the original
``time -> topological level`` loop because every edge depends only on its own
previous storage and on already-completed upstream levels.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from data.schema import topological_levels
from .kinematic_wave import EdgeParameterNetwork


@torch.jit.script
def _route_level_sequence(
    inflow_series: torch.Tensor,
    initial_volume: torch.Tensor,
    length: torch.Tensor,
    alpha: torch.Tensor,
    cell_length: torch.Tensor,
    dt: float,
    cfl_limit: float,
    implicit_iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route one independent topological edge level over all time steps.

    The recurrent loop lives in TorchScript rather than Python.  The eight
    backward-Euler/Newton updates are unchanged from the reference solver.
    """
    exponent = 5.0 / 3.0
    volume = initial_volume
    outflow_steps = torch.jit.annotate(List[torch.Tensor], [])
    volume_steps = torch.jit.annotate(List[torch.Tensor], [])
    explicit_steps = torch.jit.annotate(List[torch.Tensor], [])
    celerity_steps = torch.jit.annotate(List[torch.Tensor], [])
    implicit_residual_steps = torch.jit.annotate(List[torch.Tensor], [])

    for time_index in range(inflow_series.size(1)):
        inflow = inflow_series[:, time_index]

        # CFL is diagnostic only.  Detaching it avoids building a large
        # autograd graph that never contributes to the training objective.
        current_area = (
            volume.detach() / length.detach().unsqueeze(0)
        ).clamp_min(0.0)
        equilibrium_area = (
            inflow.detach() / alpha.detach().unsqueeze(0)
        ).clamp_min(0.0).pow(1.0 / exponent)
        cfl_area = torch.maximum(current_area, equilibrium_area).clamp_min(1.0e-8)
        celerity = (
            exponent
            * alpha.detach().unsqueeze(0)
            * cfl_area.pow(exponent - 1.0)
        )
        cfl_ratio = (
            celerity
            * dt
            / cell_length.detach().unsqueeze(0)
            / cfl_limit
        )
        explicit_steps.append(torch.ceil(cfl_ratio.amax()).clamp_min(1.0))
        celerity_steps.append(celerity.amax())

        available = volume + inflow * dt
        wet_area_for_power = (
            available / length.unsqueeze(0)
        ).clamp_min(1.0e-12)
        beta = (
            dt
            * alpha.unsqueeze(0)
            / length.unsqueeze(0)
            * wet_area_for_power.pow(exponent - 1.0)
        )
        storage_fraction = (1.0 + beta).pow(-1.0 / exponent)
        for _ in range(implicit_iterations):
            fraction_power = storage_fraction.pow(exponent)
            storage_fraction = (
                1.0
                + (exponent - 1.0) * beta * fraction_power
            ) / (
                1.0
                + exponent
                * beta
                * storage_fraction.pow(exponent - 1.0)
            )
        implicit_relative_residual = (
            storage_fraction
            + beta * storage_fraction.pow(exponent)
            - 1.0
        ).abs()
        implicit_residual_steps.append(
            implicit_relative_residual.detach().amax()
        )

        updated_volume = available * storage_fraction
        mean_outflow = (available - updated_volume) / dt
        volume = updated_volume
        outflow_steps.append(mean_outflow)
        volume_steps.append(volume)

    return (
        torch.stack(outflow_steps, dim=1),
        torch.stack(volume_steps, dim=1),
        torch.stack(explicit_steps),
        torch.stack(celerity_steps),
        torch.stack(implicit_residual_steps),
    )


class KinematicWaveGNN(nn.Module):
    """Execution-optimized version of the formal conservative river router."""

    def __init__(
        self,
        node_static_dim: int,
        edge_static_dim: int,
        hidden_dim: int,
        bounds: dict,
        solver: dict,
    ) -> None:
        super().__init__()
        if edge_static_dim != 2:
            raise ValueError(
                "kinematic_wave_gnn要求恰好两个边静态特征："
                "reach_length_m和reach_slope_m_per_m"
            )
        self.edge_static_dim = int(edge_static_dim)
        self.edge_net = EdgeParameterNetwork(
            edge_static_dim + 2 * node_static_dim,
            hidden_dim,
            tuple(bounds["width"]),
            tuple(bounds["manning_n"]),
        )
        self.solver = solver

        # The training sampler shuffles same-graph mini-batches across 33
        # graphs.  The reference implementation kept only the most recent
        # topology, forcing repeated CPU DAG reconstruction.  Cache every
        # topology encountered; these entries are tiny and are not state_dict
        # parameters/buffers.
        self._topology_cache: dict[
            tuple[int, tuple[int, ...]], tuple[list[torch.Tensor], torch.Tensor]
        ] = {}
        self._device_topology_cache: dict[
            tuple[tuple[int, tuple[int, ...]], str],
            tuple[list[torch.Tensor], torch.Tensor],
        ] = {}
        self._active_topology_key: tuple[int, tuple[int, ...]] | None = None

    def _prepare(
        self, edge_index: torch.Tensor, nodes: int, device: torch.device
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        edge_cpu = edge_index.detach().cpu()
        key = (nodes, tuple(edge_cpu.flatten().tolist()))
        cached = self._topology_cache.get(key)
        if cached is None:
            levels, _ = topological_levels(edge_cpu, nodes)
            source_cpu = edge_cpu[0]
            outdegree = torch.bincount(source_cpu, minlength=nodes)
            divergent = torch.where(outdegree > 1)[0].tolist()
            if divergent:
                raise ValueError(
                    "kinematic_wave_gnn不允许没有分流权重的出度>1节点，"
                    f"节点索引={divergent}"
                )
            source_list = source_cpu.tolist()
            edge_levels = [
                torch.tensor(
                    [
                        edge
                        for edge, source in enumerate(source_list)
                        if source in level
                    ],
                    dtype=torch.long,
                )
                for level in levels
            ]
            sink_index = torch.where(outdegree == 0)[0].long()
            cached = (edge_levels, sink_index)
            self._topology_cache[key] = cached

        device_key = (key, str(device))
        device_cached = self._device_topology_cache.get(device_key)
        if device_cached is None:
            edge_levels_cpu, sink_cpu = cached
            device_cached = (
                [ids.to(device=device) for ids in edge_levels_cpu if ids.numel()],
                sink_cpu.to(device=device),
            )
            self._device_topology_cache[device_key] = device_cached
        self._active_topology_key = key
        return device_cached

    @staticmethod
    def _raise_if_invalid_inputs(
        q_lat: torch.Tensor,
        edge_static: torch.Tensor,
        length: torch.Tensor,
        slope: torch.Tensor,
        minimum_length: float,
    ) -> None:
        bad_q = (~torch.isfinite(q_lat)).any() | (q_lat < 0).any()
        bad_edge = (
            (~torch.isfinite(edge_static)).any()
            | (length < minimum_length).any()
            | (slope < 0).any()
        )
        status = torch.stack(
            [bad_q.to(torch.int8), bad_edge.to(torch.int8)]
        ).detach().cpu()
        if bool(status[0].item()):
            raise ValueError("q_lat必须为有限非负流量")
        if bool(status[1].item()):
            if not bool(torch.isfinite(edge_static).all().detach().cpu().item()):
                raise ValueError("边属性含缺失/非有限值；运动波不允许隐式填补")
            if bool((length < minimum_length).any().detach().cpu().item()):
                raise ValueError(
                    f"存在极短河段(<{minimum_length} m)，请合并河段或修正km/m单位"
                )
            raise ValueError("坡降为负：可能src/dst颠倒")

    def forward(
        self,
        q_lat: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
        initial_edge_discharge: torch.Tensor | None = None,
        initial_edge_storage: torch.Tensor | None = None,
        neural_edge_static: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=q_lat.device.type, enabled=False):
            q_lat = q_lat.float()
            node_static = node_static.float()
            edge_static = edge_static.float()
            edge_index = edge_index.long()
            if q_lat.ndim != 3:
                raise ValueError("q_lat必须为[B,T,N]")
            batch, steps, nodes = q_lat.shape
            if steps <= 0:
                raise ValueError("q_lat时间维必须>0")
            edges = int(edge_index.shape[1])
            if edge_static.ndim != 2 or edge_static.shape != (
                edges,
                self.edge_static_dim,
            ):
                raise ValueError(
                    "edge_static必须为[E,2]，依次是"
                    "reach_length_m和reach_slope_m_per_m"
                )
            if initial_edge_discharge is not None and initial_edge_storage is not None:
                raise ValueError("initial_edge_discharge与initial_edge_storage不能同时提供")

            edge_levels, sink_index = self._prepare(
                edge_index, nodes, q_lat.device
            )
            length = edge_static[:, 0]
            slope_raw = edge_static[:, 1]
            minimum_length = float(self.solver["minimum_length"])
            minimum_slope = float(self.solver["minimum_slope"])
            self._raise_if_invalid_inputs(
                q_lat, edge_static, length, slope_raw, minimum_length
            )
            slope = slope_raw.clamp_min(minimum_slope)
            source, destination = edge_index

            edge_ml = (
                torch.sign(edge_static) * torch.log1p(edge_static.abs())
                if neural_edge_static is None
                else neural_edge_static.float()
            )
            if edge_ml.shape != edge_static.shape or not torch.isfinite(edge_ml).all():
                raise ValueError("neural_edge_static必须与edge_static同形且有限")
            node_ml = node_static
            parameters = torch.cat(
                [edge_ml, node_ml[source], node_ml[destination]], dim=-1
            )
            width, manning = self.edge_net(parameters)
            alpha = torch.sqrt(slope) / (manning * width.pow(2.0 / 3.0))

            exponent = 5.0 / 3.0
            dt = float(self.solver["seconds_per_step"])
            dx = float(self.solver["dx"])
            cfl_limit = float(self.solver["cfl"])
            integration_scheme = self.solver["integration_scheme"]
            implicit_iterations = int(self.solver["implicit_iterations"])
            implicit_tolerance = float(
                self.solver["implicit_residual_tolerance"]
            )
            if integration_scheme != "backward_euler":
                raise ValueError("kinematic_wave_gnn仅支持backward_euler积分")
            if (
                dt <= 0
                or dx <= 0
                or not 0 < cfl_limit <= 1
                or implicit_iterations < 1
                or implicit_tolerance <= 0
            ):
                raise ValueError("solver的时间步、CFL诊断或隐式求解参数无效")
            alpha_bad = (~torch.isfinite(alpha)).any() | (alpha <= 0).any()
            if bool(alpha_bad.detach().cpu().item()):
                raise FloatingPointError("运动波参数alpha必须为有限正数")

            if initial_edge_storage is not None:
                initial_volume = initial_edge_storage.float()
                if initial_volume.shape != (batch, edges):
                    raise ValueError("initial_edge_storage必须为[B,E]")
                initial_bad = (
                    (~torch.isfinite(initial_volume)).any()
                    | (initial_volume < 0).any()
                )
                if bool(initial_bad.detach().cpu().item()):
                    raise ValueError("initial_edge_storage必须为有限非负m3")
                if edges:
                    initial_area = (
                        initial_volume / length.unsqueeze(0)
                    ).clamp_min(0.0)
                    initial_q = alpha.unsqueeze(0) * initial_area.pow(exponent)
                else:
                    initial_q = initial_volume.clone()
            elif initial_edge_discharge is not None:
                initial_q = initial_edge_discharge.float()
                if initial_q.shape != (batch, edges):
                    raise ValueError("initial_edge_discharge必须为[B,E]")
                initial_bad = (
                    (~torch.isfinite(initial_q)).any() | (initial_q < 0).any()
                )
                if bool(initial_bad.detach().cpu().item()):
                    raise ValueError("initial_edge_discharge必须为有限非负m3/s")
                if edges:
                    initial_area = (
                        initial_q / alpha.unsqueeze(0)
                    ).clamp_min(0.0).pow(1.0 / exponent)
                    initial_volume = initial_area * length.unsqueeze(0)
                else:
                    initial_volume = initial_q.clone()
            else:
                initial_volume = torch.zeros(
                    batch, edges, device=q_lat.device, dtype=q_lat.dtype
                )
                initial_q = torch.zeros_like(initial_volume)

            # No channel edges: preserve the reference solver's identity route
            # and diagnostic shapes without entering any recurrent loop.
            if edges == 0:
                routed = q_lat.clone()
                explicit_equivalent = torch.ones(
                    steps, device=q_lat.device, dtype=torch.float32
                )
                maximum_celerity = torch.zeros_like(explicit_equivalent)
                implicit_residual = torch.zeros_like(explicit_equivalent)
                node_storage = torch.zeros(
                    batch, steps, nodes, device=q_lat.device, dtype=q_lat.dtype
                )
                residual = torch.zeros(
                    batch, steps, device=q_lat.device, dtype=q_lat.dtype
                )
                return routed, {
                    "routing_mass_balance_residual": residual,
                    "explicit_equivalent_substeps": explicit_equivalent,
                    "maximum_celerity_m_per_s": maximum_celerity,
                    "implicit_relative_residual": implicit_residual,
                    "implicit_iterations": torch.tensor(
                        implicit_iterations, device=q_lat.device, dtype=torch.int64
                    ),
                    "explicit_cfl_exceedance_count": explicit_equivalent.gt(1).sum(),
                    "learned_effective_width": width,
                    "learned_effective_manning_n": manning,
                    "initial_edge_discharge_m3s": initial_q,
                    "initial_edge_storage_m3": initial_volume,
                    "edge_storage": initial_volume,
                    "node_channel_storage": node_storage,
                }

            cell_length = torch.minimum(
                length, torch.full_like(length, dx)
            )
            node_q_series = q_lat.clone()
            level_ids_parts: list[torch.Tensor] = []
            level_volume_parts: list[torch.Tensor] = []
            explicit_parts: list[torch.Tensor] = []
            celerity_parts: list[torch.Tensor] = []
            implicit_parts: list[torch.Tensor] = []

            # Topological-level outer loop: once upstream levels have completed,
            # the whole inflow sequence for this level is known.  Route that
            # sequence recurrently in TorchScript and add all time-step outflows
            # downstream in one index_add operation.
            for edge_ids in edge_levels:
                sources = source.index_select(0, edge_ids)
                destinations = destination.index_select(0, edge_ids)
                inflow_series = node_q_series.index_select(2, sources)
                (
                    mean_outflow_series,
                    volume_series,
                    explicit_level,
                    celerity_level,
                    implicit_level,
                ) = _route_level_sequence(
                    inflow_series,
                    initial_volume.index_select(1, edge_ids),
                    length.index_select(0, edge_ids),
                    alpha.index_select(0, edge_ids),
                    cell_length.index_select(0, edge_ids),
                    dt,
                    cfl_limit,
                    implicit_iterations,
                )
                node_q_series = node_q_series.index_add(
                    2, destinations, mean_outflow_series
                )
                level_ids_parts.append(edge_ids)
                level_volume_parts.append(volume_series)
                explicit_parts.append(explicit_level)
                celerity_parts.append(celerity_level)
                implicit_parts.append(implicit_level)

            edge_order = torch.cat(level_ids_parts)
            if int(edge_order.numel()) != edges:
                raise RuntimeError("运动波拓扑level没有覆盖全部河段")
            restore_order = torch.argsort(edge_order)
            edge_volume_trajectory = torch.cat(
                level_volume_parts, dim=2
            ).index_select(2, restore_order)
            routed = node_q_series
            volume = edge_volume_trajectory[:, -1]

            explicit_equivalent = torch.maximum(
                torch.stack(explicit_parts, dim=0).amax(dim=0),
                torch.ones(steps, device=q_lat.device, dtype=torch.float32),
            )
            maximum_celerity = torch.stack(
                celerity_parts, dim=0
            ).amax(dim=0)
            implicit_residual = torch.stack(
                implicit_parts, dim=0
            ).amax(dim=0)

            node_storage = torch.zeros(
                batch,
                steps,
                nodes,
                device=q_lat.device,
                dtype=q_lat.dtype,
            ).index_add(2, destination, edge_volume_trajectory)

            storage_after = edge_volume_trajectory.sum(dim=-1)
            storage_before = torch.cat(
                [initial_volume.sum(dim=-1, keepdim=True), storage_after[:, :-1]],
                dim=1,
            )
            external_inflow = q_lat.sum(dim=-1)
            domain_outflow = routed.index_select(2, sink_index).sum(dim=-1)
            residual = (
                storage_before
                + external_inflow * dt
                - storage_after
                - domain_outflow * dt
            )

            invalid = (
                (~torch.isfinite(routed)).any()
                | (~torch.isfinite(volume)).any()
                | (~torch.isfinite(explicit_equivalent)).any()
                | (~torch.isfinite(maximum_celerity)).any()
                | (~torch.isfinite(implicit_residual)).any()
            )
            negative = (routed < 0).any() | (volume < 0).any()
            maximum_implicit_residual_tensor = implicit_residual.amax()
            summary = torch.stack(
                [
                    invalid.to(torch.float32),
                    negative.to(torch.float32),
                    maximum_implicit_residual_tensor,
                ]
            ).detach().cpu()
            if bool(summary[0].item()) or bool(summary[1].item()):
                raise FloatingPointError(
                    "隐式运动波产生非有限值或负蓄量/负流量；请检查输入与物理参数"
                )
            maximum_implicit_residual = float(summary[2].item())
            if maximum_implicit_residual > implicit_tolerance:
                raise RuntimeError(
                    "隐式后向Euler求解未收敛：最大无量纲残差="
                    f"{maximum_implicit_residual:.6g}，容差="
                    f"{implicit_tolerance:.6g}；请检查输入或增加implicit_iterations"
                )

            return routed, {
                "routing_mass_balance_residual": residual,
                "explicit_equivalent_substeps": explicit_equivalent,
                "maximum_celerity_m_per_s": maximum_celerity,
                "implicit_relative_residual": implicit_residual,
                "implicit_iterations": torch.tensor(
                    implicit_iterations, device=q_lat.device, dtype=torch.int64
                ),
                "explicit_cfl_exceedance_count": explicit_equivalent.gt(1).sum(),
                "learned_effective_width": width,
                "learned_effective_manning_n": manning,
                "initial_edge_discharge_m3s": initial_q,
                "initial_edge_storage_m3": initial_volume,
                "edge_storage": volume,
                "node_channel_storage": node_storage,
            }
