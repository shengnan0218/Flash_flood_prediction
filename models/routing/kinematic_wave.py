"""Directed neural and differentiable conservative river-routing modules."""
from __future__ import annotations

import torch
from torch import nn

from data.schema import topological_levels


class EdgeParameterNetwork(nn.Module):
    """Estimate bounded effective width and Manning roughness per reach.

    Both quantities are latent, differentiable routing parameters inferred
    from the available reach and endpoint-node attributes.  In particular,
    ``effective width`` is not presented as a surveyed channel measurement:
    it is constrained to the configured physical interval and learned only
    through the routing objective.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        width_bounds: tuple[float, float],
        n_bounds: tuple[float, float],
    ) -> None:
        super().__init__()
        if width_bounds[0] <= 0 or width_bounds[0] >= width_bounds[1]:
            raise ValueError("width_bounds必须满足0<下界<上界")
        if n_bounds[0] <= 0 or n_bounds[0] >= n_bounds[1]:
            raise ValueError("n_bounds必须满足0<下界<上界")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        # A zero output is a neutral midpoint prior in bounded logit space.
        # It is not a fixed pseudo-observation: gradients update both outputs
        # and subsequently make them reach-specific functions of ``x``.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.bounds = (width_bounds, n_bounds)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(x.float())
        (width_low, width_high), (n_low, n_high) = self.bounds
        width = width_low + (width_high - width_low) * torch.sigmoid(raw[:, 0])
        manning = n_low + (n_high - n_low) * torch.sigmoid(raw[:, 1])
        return width, manning


class KinematicWaveGNN(nn.Module):
    """Route lateral discharge through a converging DAG in physical units.

    Each reach is a finite-volume reservoir advanced with a differentiable
    backward-Euler solve under ``Q = alpha * A ** (5/3)``.  The routing is
    mass conservative, only sends water from ``src`` to ``dst``, and explicitly
    rejects divergent nodes: no split fraction exists in the formal input
    contract, so duplicating flow at a branch would be scientifically invalid.
    """

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
        self.edge_static_dim = edge_static_dim
        self.edge_net = EdgeParameterNetwork(
            edge_static_dim + 2 * node_static_dim,
            hidden_dim,
            tuple(bounds["width"]),
            tuple(bounds["manning_n"]),
        )
        self.solver = solver
        self._cache_key: tuple[int, tuple[int, ...]] | None = None
        self._edge_levels: list[torch.Tensor] = []

    def _prepare(self, edge_index: torch.Tensor, nodes: int) -> None:
        key = (nodes, tuple(edge_index.detach().cpu().flatten().tolist()))
        if key == self._cache_key:
            return
        levels, _ = topological_levels(edge_index, nodes)
        source_cpu = edge_index[0].detach().cpu()
        if source_cpu.numel():
            outdegree = torch.bincount(source_cpu, minlength=nodes)
            divergent = torch.where(outdegree > 1)[0].tolist()
            if divergent:
                raise ValueError(
                    "kinematic_wave_gnn不允许没有分流权重的出度>1节点，"
                    f"节点索引={divergent}"
                )
        self._edge_levels = [
            torch.tensor(
                [edge for edge, source in enumerate(source_cpu.tolist()) if source in level],
                dtype=torch.long,
            )
            for level in levels
        ]
        self._cache_key = key

    def forward(
        self,
        q_lat: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(device_type=q_lat.device.type, enabled=False):
            q_lat = q_lat.float()
            node_static = node_static.float()
            edge_static = edge_static.float()
            edge_index = edge_index.long()
            if q_lat.ndim != 3:
                raise ValueError("q_lat必须为[B,T,N]")
            batch, steps, nodes = q_lat.shape
            edges = edge_index.shape[1]
            if edge_static.ndim != 2 or edge_static.shape != (edges, self.edge_static_dim):
                raise ValueError(
                    "edge_static必须为[E,2]，依次是"
                    "reach_length_m和reach_slope_m_per_m"
                )
            self._prepare(edge_index, nodes)
            if not torch.isfinite(q_lat).all() or (q_lat < 0).any():
                raise ValueError("q_lat必须为有限非负流量")
            if not torch.isfinite(edge_static).all():
                raise ValueError("边属性含缺失/非有限值；运动波不允许隐式填补")

            length = edge_static[:, 0]
            slope = edge_static[:, 1]
            minimum_length = float(self.solver["minimum_length"])
            minimum_slope = float(self.solver["minimum_slope"])
            if (length < minimum_length).any():
                raise ValueError(
                    f"存在极短河段(<{minimum_length} m)，请合并河段或修正km/m单位"
                )
            if (slope < 0).any():
                raise ValueError("坡降为负：可能src/dst颠倒")
            slope = slope.clamp_min(minimum_slope)
            source, destination = edge_index

            # Compress heterogeneous physical scales only for the parameter MLP.
            edge_ml = torch.sign(edge_static) * torch.log1p(edge_static.abs())
            node_ml = torch.sign(node_static) * torch.log1p(node_static.abs())
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
                raise ValueError(
                    "solver的时间步、CFL诊断或隐式求解参数无效"
                )
            if not torch.isfinite(alpha).all() or (alpha <= 0).any():
                raise FloatingPointError("运动波参数alpha必须为有限正数")

            volume = torch.zeros(batch, edges, device=q_lat.device)
            node_outputs: list[torch.Tensor] = []
            node_storages: list[torch.Tensor] = []
            residuals: list[torch.Tensor] = []
            explicit_equivalent_stats: list[torch.Tensor] = []
            maximum_celerity_stats: list[torch.Tensor] = []
            implicit_residual_stats: list[torch.Tensor] = []
            for time_index in range(steps):
                node_q = q_lat[:, time_index].clone()
                storage_before = volume.sum(dim=-1)
                external_inflow = q_lat[:, time_index].sum(dim=-1)
                time_explicit_equivalent = [
                    torch.ones((), device=q_lat.device, dtype=torch.float32)
                ]
                time_maximum_celerity = [
                    torch.zeros((), device=q_lat.device, dtype=torch.float32)
                ]
                time_implicit_residual = [
                    torch.zeros((), device=q_lat.device, dtype=torch.float32)
                ]
                for edge_ids_cpu in self._edge_levels:
                    if edge_ids_cpu.numel() == 0:
                        continue
                    edge_ids = edge_ids_cpu.to(q_lat.device)
                    sources = source[edge_ids]
                    destinations = destination[edge_ids]
                    inflow = node_q[:, sources]

                    # CFL is retained as a stiffness diagnostic for an
                    # equivalent explicit discretisation.  It does not control
                    # the backward-Euler update and therefore cannot abort a
                    # physically valid short-reach/high-flow sample.
                    with torch.no_grad():
                        current_area = (
                            volume[:, edge_ids] / length[edge_ids]
                        ).clamp_min(0.0)
                        equilibrium_area = (
                            inflow / alpha[edge_ids]
                        ).clamp_min(0.0).pow(1.0 / exponent)
                        cfl_area = torch.maximum(
                            current_area, equilibrium_area
                        ).clamp_min(1.0e-8)
                        celerity = (
                            exponent
                            * alpha[edge_ids]
                            * cfl_area.pow(exponent - 1.0)
                        )
                        cell_length = torch.minimum(
                            length[edge_ids],
                            torch.full_like(length[edge_ids], dx),
                        )
                        cfl_ratio = celerity * dt / cell_length / cfl_limit
                        time_explicit_equivalent.append(
                            torch.ceil(cfl_ratio.amax()).clamp_min(1.0)
                        )
                        time_maximum_celerity.append(celerity.amax())

                    # Backward Euler for a reach with constant mean inflow:
                    #   V1 + dt * alpha * (V1 / L) ** p = V0 + Qin * dt.
                    # With V1=available*y this becomes y+beta*y**p=1 on
                    # 0<=y<=1.  The algebraic Newton form below avoids the
                    # cancellation in x-f/df, stays positive, and uses a fixed
                    # iteration count so the complete gradient path is retained.
                    available = volume[:, edge_ids] + inflow * dt
                    # ``available ** (2/3)`` has an infinite derivative at an
                    # exactly dry reach even though the physical storage/outflow
                    # derivative is finite there.  Protect only this fractional
                    # power with a tiny dimensionless wet-area floor.  This is
                    # deliberately well above float32 subnormal range so CUDA
                    # flush-to-zero cannot recreate an inf*0 gradient.
                    # ``available`` itself remains untouched, so a dry reach
                    # still has exactly zero storage and zero outflow.
                    wet_area_for_power = (
                        available / length[edge_ids]
                    ).clamp_min(1.0e-12)
                    beta = (
                        dt
                        * alpha[edge_ids]
                        / length[edge_ids]
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
                    time_implicit_residual.append(
                        implicit_relative_residual.detach().amax()
                    )
                    updated_volume = available * storage_fraction
                    mean_outflow = (available - updated_volume) / dt
                    next_volume = volume.clone()
                    next_volume[:, edge_ids] = updated_volume
                    volume = next_volume
                    node_q = node_q.index_add(1, destinations, mean_outflow)

                sinks = torch.ones(nodes, dtype=torch.bool, device=q_lat.device)
                sinks[source] = False
                domain_outflow = node_q[:, sinks].sum(dim=-1)
                residuals.append(
                    storage_before
                    + external_inflow * dt
                    - volume.sum(dim=-1)
                    - domain_outflow * dt
                )
                node_storage = torch.zeros(batch, nodes, device=q_lat.device)
                if edges:
                    node_storage.index_add_(1, destination, volume.clone())
                node_outputs.append(node_q)
                node_storages.append(node_storage)
                explicit_equivalent_stats.append(
                    torch.stack(time_explicit_equivalent).amax()
                )
                maximum_celerity_stats.append(
                    torch.stack(time_maximum_celerity).amax()
                )
                implicit_residual_stats.append(
                    torch.stack(time_implicit_residual).amax()
                )

            routed = torch.stack(node_outputs, dim=1)
            explicit_equivalent = torch.stack(explicit_equivalent_stats)
            maximum_celerity = torch.stack(maximum_celerity_stats)
            implicit_residual = torch.stack(implicit_residual_stats)
            if (
                not torch.isfinite(routed).all()
                or not torch.isfinite(volume).all()
                or not torch.isfinite(explicit_equivalent).all()
                or not torch.isfinite(maximum_celerity).all()
                or not torch.isfinite(implicit_residual).all()
                or (routed < 0).any()
                or (volume < 0).any()
            ):
                raise FloatingPointError(
                    "隐式运动波产生非有限值或负蓄量/负流量；请检查输入与物理参数"
                )
            maximum_implicit_residual = float(implicit_residual.amax().cpu())
            if maximum_implicit_residual > implicit_tolerance:
                raise RuntimeError(
                    "隐式后向Euler求解未收敛：最大无量纲残差="
                    f"{maximum_implicit_residual:.6g}，容差="
                    f"{implicit_tolerance:.6g}；请检查输入或增加implicit_iterations"
                )
            return routed, {
                "routing_mass_balance_residual": torch.stack(residuals, dim=1),
                "explicit_equivalent_substeps": explicit_equivalent,
                "maximum_celerity_m_per_s": maximum_celerity,
                "implicit_relative_residual": implicit_residual,
                "implicit_iterations": torch.tensor(
                    implicit_iterations, device=q_lat.device, dtype=torch.int64
                ),
                "explicit_cfl_exceedance_count": explicit_equivalent.gt(1).sum(),
                "learned_effective_width": width,
                "learned_effective_manning_n": manning,
                "edge_storage": volume,
                "node_channel_storage": torch.stack(node_storages, dim=1),
            }


class PureDirectedGNN(nn.Module):
    """A non-physical directed message-passing routing ablation."""

    def __init__(
        self, node_static_dim: int, edge_static_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(1 + edge_static_dim + 2 * node_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self._order: list[int] = []
        self._key: tuple[int, tuple[int, ...]] | None = None

    def forward(
        self,
        q_lat: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, steps, nodes = q_lat.shape
        key = (nodes, tuple(edge_index.detach().cpu().flatten().tolist()))
        if key != self._key:
            _, self._order = topological_levels(edge_index, nodes)
            self._key = key
        source, destination = edge_index
        edge_features = torch.sign(edge_static) * torch.log1p(edge_static.abs())
        node_features = torch.sign(node_static) * torch.log1p(node_static.abs())
        outputs: list[torch.Tensor] = []
        for time_index in range(steps):
            q = q_lat[:, time_index].clone()
            for node in self._order:
                edge_ids = torch.where(source == node)[0]
                if edge_ids.numel():
                    message_input = torch.cat(
                        [
                            q[:, node, None]
                            .expand(-1, edge_ids.numel())
                            .unsqueeze(-1),
                            edge_features[edge_ids][None].expand(batch, -1, -1),
                            node_features[source[edge_ids]][None].expand(
                                batch, -1, -1
                            ),
                            node_features[destination[edge_ids]][None].expand(
                                batch, -1, -1
                            ),
                        ],
                        dim=-1,
                    )
                    q.index_add_(
                        1, destination[edge_ids], self.msg(message_input).squeeze(-1)
                    )
            outputs.append(q)
        routed = torch.stack(outputs, dim=1)
        return routed, {
            "routing_mass_balance_residual": torch.full(
                (batch, steps), float("nan"), device=routed.device
            ),
        }
