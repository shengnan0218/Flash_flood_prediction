"""Active v9 model with mass-aware forecast-origin state assimilation.

This module keeps the frozen v9 runoff/routing/Z-head architecture but replaces
its forecast-origin observation correction with a stronger, physically
interpretable analysis step:

* Q/Z history context is assigned to the nearest downstream observation and
  propagated upstream through the drainage tree;
* observed-minus-warmup Q at the origin is distributed upstream by incremental
  area, so a basin-wide discharge residual becomes a uniform equivalent-depth
  storage correction over its contributing area;
* runoff storages receive both bounded multiplicative adjustment and an
  additive mass correction, allowing observations to restore water that entered
  the catchment before the 24 h warm-up window;
* physical routing edge storage receives the same additive residual-volume
  correction and is then carried exactly into the forecast;
* the Z head uses an analysed origin discharge: observed Q0 when available,
  otherwise the discharge implied by the corrected physical routing state.

No additional recurrent network is introduced.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from data.v8_schema import HydrologicGraphBatch
from models.hydrologic_graph_v9 import (
    HydrologicGraphV9Model as _BaseHydrologicGraphV9Model,
    _validate_v9_batch,
)


class MassAwareObservationStateCorrectorV9(nn.Module):
    """Residual forecast-origin analysis with additive, Q-anchored storage mass."""

    def __init__(
        self,
        node_static_dim: int,
        hidden_dim: int,
        *,
        hidden_residual_scale: float,
        storage_log_scale: float,
        max_additive_storage_hours: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_residual_scale = float(hidden_residual_scale)
        self.storage_log_scale = float(storage_log_scale)
        self.max_additive_storage_hours = float(max_additive_storage_hours)
        if self.max_additive_storage_hours <= 0:
            raise ValueError("max_additive_storage_hours必须>0")

        input_dim = 3 * self.hidden_dim + int(node_static_dim) + 3
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.h_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.storage_log_head = nn.Linear(self.hidden_dim, 2)
        self.storage_partition_head = nn.Linear(self.hidden_dim, 1)
        self.assimilation_hours_head = nn.Linear(self.hidden_dim, 1)
        self.edge_storage_log_head = nn.Linear(self.hidden_dim, 1)

        for head in (
            self.h_head,
            self.c_head,
            self.storage_log_head,
            self.storage_partition_head,
            self.edge_storage_log_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.assimilation_hours_head.weight)
        # Start at one hour of residual-volume memory instead of an arbitrary
        # large mass injection. The bounded head can learn up to the configured
        # maximum from TRAIN only.
        initial_fraction = min(1.0 / self.max_additive_storage_hours, 0.999)
        initial_fraction = max(initial_fraction, 1.0e-3)
        initial_logit = torch.log(
            torch.tensor(initial_fraction / (1.0 - initial_fraction))
        ).item()
        nn.init.constant_(self.assimilation_hours_head.bias, initial_logit)

    def forward(
        self,
        *,
        state: dict[str, torch.Tensor],
        node_observation_context: torch.Tensor,
        node_observation_available: torch.Tensor,
        node_q0_residual_norm: torch.Tensor,
        node_q0_residual_m3s: torch.Tensor,
        node_q0_residual_available: torch.Tensor,
        node_static_norm: torch.Tensor,
        incremental_area_km2: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch, nodes, hidden = state["h"].shape
        if hidden != self.hidden_dim or state["c"].shape != state["h"].shape:
            raise ValueError("v9 analysis corrector h/c形状错误")
        if node_observation_context.shape != (batch, nodes, self.hidden_dim):
            raise ValueError("v9 analysis observation context形状错误")
        gate_bool = node_observation_available.bool()
        q_gate_bool = node_q0_residual_available.bool()
        if gate_bool.shape != (batch, nodes, 1) or q_gate_bool.shape != gate_bool.shape:
            raise ValueError("v9 analysis availability形状错误")
        if node_q0_residual_norm.shape != (batch, nodes, 1):
            raise ValueError("v9 analysis normalized Q residual形状错误")
        if node_q0_residual_m3s.shape != (batch, nodes, 1):
            raise ValueError("v9 analysis physical Q residual形状错误")
        if incremental_area_km2.shape != (nodes,):
            raise ValueError("v9 analysis incremental area形状错误")

        dtype = state["h"].dtype
        gate = gate_bool.to(dtype)
        q_gate = q_gate_bool.to(dtype)
        static = node_static_norm.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat(
            [
                state["h"],
                state["c"],
                node_observation_context,
                gate,
                node_q0_residual_norm,
                q_gate,
                static,
            ],
            dim=-1,
        )
        context = self.fusion(features)

        delta_h = self.hidden_residual_scale * torch.tanh(self.h_head(context)) * gate
        delta_c = self.hidden_residual_scale * torch.tanh(self.c_head(context)) * gate
        storage_log = (
            self.storage_log_scale
            * torch.tanh(self.storage_log_head(context))
            * gate
        )
        fast_fraction = torch.sigmoid(self.storage_partition_head(context))[..., 0]
        assimilation_hours = (
            self.max_additive_storage_hours
            * torch.sigmoid(self.assimilation_hours_head(context))[..., 0]
            * q_gate[..., 0]
        )

        # q[m3/s] * hours * 3600 / (area[km2] * 1000) = equivalent depth [mm].
        area = incremental_area_km2.to(
            device=state["h"].device, dtype=dtype
        ).clamp_min(1.0e-8)
        additive_total_mm = (
            node_q0_residual_m3s[..., 0]
            * assimilation_hours
            * 3600.0
            / (area.unsqueeze(0) * 1000.0)
        )
        additive_fast_mm = additive_total_mm * fast_fraction
        additive_slow_mm = additive_total_mm * (1.0 - fast_fraction)

        fast_candidate = (
            state["storage_fast_mm"] * torch.exp(storage_log[..., 0])
            + additive_fast_mm
        ).clamp_min(0.0)
        slow_candidate = (
            state["storage_slow_mm"] * torch.exp(storage_log[..., 1])
            + additive_slow_mm
        ).clamp_min(0.0)
        node_gate = gate_bool[..., 0]
        storage_fast = torch.where(
            node_gate, fast_candidate, state["storage_fast_mm"]
        )
        storage_slow = torch.where(
            node_gate, slow_candidate, state["storage_slow_mm"]
        )

        corrected = {
            "h": state["h"] + delta_h,
            "c": state["c"] + delta_c,
            "storage_fast_mm": storage_fast,
            "storage_slow_mm": storage_slow,
        }
        edge_node_log_factor = (
            self.storage_log_scale
            * torch.tanh(self.edge_storage_log_head(context))
            * gate
        )
        return corrected, {
            "context": context,
            "delta_h": delta_h,
            "delta_c": delta_c,
            "storage_log_factor": storage_log,
            "storage_additive_total_mm": additive_total_mm,
            "storage_additive_fast_mm": additive_fast_mm,
            "storage_additive_slow_mm": additive_slow_mm,
            "assimilation_hours": assimilation_hours,
            "edge_node_log_factor": edge_node_log_factor,
            "node_observation_available": node_observation_available,
            "node_q0_residual_norm": node_q0_residual_norm,
            "node_q0_residual_m3s": node_q0_residual_m3s,
            "node_q0_residual_available": node_q0_residual_available,
        }


class HydrologicGraphV9Model(_BaseHydrologicGraphV9Model):
    """Formal v9 model with upstream, mass-aware forecast-origin assimilation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__(cfg)
        correction_cfg = cfg.get("state_correction", {})
        self.state_corrector = MassAwareObservationStateCorrectorV9(
            self.node_static_dim,
            self.hidden_dim,
            hidden_residual_scale=float(
                correction_cfg.get("hidden_residual_scale", 0.25)
            ),
            storage_log_scale=float(correction_cfg.get("storage_log_scale", 0.35)),
            max_additive_storage_hours=float(
                correction_cfg.get("max_additive_storage_hours", 6.0)
            ),
        )
        self._analysis_topology_key: tuple[Any, ...] | None = None
        self._analysis_distance_cpu: torch.Tensor | None = None

    def _upstream_distances(
        self,
        edge_index: torch.Tensor,
        obs_node_index: torch.Tensor,
        nodes: int,
    ) -> torch.Tensor:
        key = (
            nodes,
            tuple(edge_index.detach().cpu().flatten().tolist()),
            tuple(obs_node_index.detach().cpu().tolist()),
        )
        if key != self._analysis_topology_key:
            source = edge_index[0].detach().cpu().tolist()
            destination = edge_index[1].detach().cpu().tolist()
            parents: list[list[int]] = [[] for _ in range(nodes)]
            for src, dst in zip(source, destination):
                parents[int(dst)].append(int(src))
            obs_nodes = obs_node_index.detach().cpu().tolist()
            distance = torch.full(
                (len(obs_nodes), nodes), float("inf"), dtype=torch.float32
            )
            for obs_position, node in enumerate(obs_nodes):
                queue: list[tuple[int, int]] = [(int(node), 0)]
                visited: set[int] = set()
                while queue:
                    current, hops = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    distance[obs_position, current] = float(hops)
                    queue.extend((parent, hops + 1) for parent in parents[current])
            self._analysis_topology_key = key
            self._analysis_distance_cpu = distance
        assert self._analysis_distance_cpu is not None
        return self._analysis_distance_cpu.to(edge_index.device)

    @staticmethod
    def _nearest_downstream_values(
        values: torch.Tensor,
        available: torch.Tensor,
        distances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign each computational node to its nearest available downstream gauge."""
        if values.ndim not in (2, 3):
            raise ValueError("analysis values必须为[B,Nobs]或[B,Nobs,D]")
        batch, obs = values.shape[:2]
        if available.shape != (batch, obs) or distances.shape[0] != obs:
            raise ValueError("analysis nearest-observation形状错误")
        masked_distance = distances.unsqueeze(0).expand(batch, -1, -1).clone()
        masked_distance = torch.where(
            available.unsqueeze(-1),
            masked_distance,
            torch.full_like(masked_distance, float("inf")),
        )
        minimum, selected = masked_distance.min(dim=1)
        node_available = torch.isfinite(minimum)
        if values.ndim == 2:
            gathered = values.gather(1, selected)
            gathered = torch.where(node_available, gathered, torch.zeros_like(gathered))
        else:
            gathered = values.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, values.shape[-1])
            )
            gathered = torch.where(
                node_available.unsqueeze(-1), gathered, torch.zeros_like(gathered)
            )
        return gathered, node_available, selected

    def _upstream_analysis_fields(
        self,
        *,
        observation_context: torch.Tensor,
        observation_available: torch.Tensor,
        q0_residual_norm: torch.Tensor,
        q0_residual_m3s: torch.Tensor,
        q0_available: torch.Tensor,
        obs_node_index: torch.Tensor,
        edge_index: torch.Tensor,
        incremental_area_km2: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distances = self._upstream_distances(edge_index, obs_node_index, nodes)
        node_context, node_obs_available, _ = self._nearest_downstream_values(
            observation_context, observation_available, distances
        )
        node_q_norm, node_q_available, selected_q = self._nearest_downstream_values(
            q0_residual_norm, q0_available, distances
        )
        selected_q_physical, _, _ = self._nearest_downstream_values(
            q0_residual_m3s, q0_available, distances
        )

        # Each selected gauge residual is distributed over its upstream area.
        # Dividing the resulting node discharge by local area in the corrector
        # gives a uniform equivalent-depth storage adjustment over that gauge's
        # contributing catchment, without duplicating the whole outlet residual
        # at every node.
        membership = torch.isfinite(distances).to(incremental_area_km2.dtype)
        upstream_area = membership @ incremental_area_km2
        selected_area = upstream_area[selected_q].clamp_min(1.0e-8)
        area_fraction = incremental_area_km2.unsqueeze(0) / selected_area
        node_q_physical = selected_q_physical * area_fraction
        node_q_physical = torch.where(
            node_q_available, node_q_physical, torch.zeros_like(node_q_physical)
        )

        return (
            node_context,
            node_obs_available.unsqueeze(-1),
            node_q_norm.unsqueeze(-1),
            node_q_physical.unsqueeze(-1),
            node_q_available.unsqueeze(-1),
        )

    @staticmethod
    def _correct_edge_storage_mass_aware(
        edge_storage: torch.Tensor,
        edge_index: torch.Tensor,
        node_log_factor: torch.Tensor,
        node_q_residual_m3s: torch.Tensor,
        assimilation_hours: torch.Tensor,
        node_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_storage.ndim != 2:
            raise ValueError("v9 edge storage必须为[B,E]")
        if edge_storage.shape[1] == 0:
            empty = edge_storage.clone()
            return edge_storage, empty, empty
        source = edge_index[0]
        gate = node_available[:, source, 0]
        edge_log_factor = node_log_factor[:, source, 0] * gate.to(edge_storage.dtype)
        additive_volume = (
            node_q_residual_m3s[:, source, 0]
            * assimilation_hours[:, source]
            * 3600.0
            * gate.to(edge_storage.dtype)
        )
        candidate = edge_storage * torch.exp(edge_log_factor) + additive_volume
        corrected = torch.where(gate, candidate.clamp_min(0.0), edge_storage)
        return corrected, edge_log_factor, additive_volume

    def forward(self, batch: HydrologicGraphBatch) -> dict[str, Any]:
        _validate_v9_batch(
            batch,
            history_steps=self.history,
            forecast_internal_steps=self.forecast_internal_steps,
            target_steps=self.horizon,
            node_static_dim=self.node_static_dim,
            edge_static_dim=self.edge_static_dim,
        )
        station_index = batch.obs_station_index.long()
        obs_node_index = batch.obs_node_index.long()
        nodes = int(batch.history_rain.shape[2])
        node_static_norm = (
            batch.node_static
            - self.node_static_mean.to(
                device=batch.node_static.device, dtype=batch.node_static.dtype
            )
        ) / self.node_static_scale.to(
            device=batch.node_static.device, dtype=batch.node_static.dtype
        )
        history_rain_norm = (
            batch.history_rain - self.rain_mean.to(batch.history_rain.dtype)
        ) / self.rain_scale.to(batch.history_rain.dtype)
        future_rain_norm = (
            batch.future_rain - self.rain_mean.to(batch.future_rain.dtype)
        ) / self.rain_scale.to(batch.future_rain.dtype)

        batch_size = int(batch.history_rain.shape[0])
        initial_state = self.static_initializer(node_static_norm, batch_size)

        q_lat_history, runoff_history_diag, runoff_t0_state_raw = self._run_runoff(
            batch.history_rain,
            history_rain_norm,
            node_static_norm,
            batch.incremental_area_km2,
            initial_state,
        )
        q_nodes_history, routing_history_diag = self.routing(
            q_lat_history,
            batch.node_static,
            batch.edge_index,
            batch.edge_static,
        )
        if self.routing_mode == "kinematic_wave_gnn":
            edge_storage_t0_raw: torch.Tensor | None = routing_history_diag[
                "edge_storage"
            ]
            channel_available = torch.ones(
                (), dtype=torch.bool, device=batch.history_rain.device
            )
        else:
            edge_storage_t0_raw = None
            channel_available = torch.zeros(
                (), dtype=torch.bool, device=batch.history_rain.device
            )

        q_hist_mean = self._station_values(
            "q_history_mean", station_index, batch.q_history
        )
        q_hist_scale = self._station_values(
            "q_history_scale", station_index, batch.q_history
        )
        z_hist_mean = self._station_values(
            "z_history_mean", station_index, batch.z_history
        )
        z_hist_scale = self._station_values(
            "z_history_scale", station_index, batch.z_history
        )
        q_history_norm = torch.where(
            batch.q_mask,
            (batch.q_history - q_hist_mean) / q_hist_scale,
            torch.zeros_like(batch.q_history),
        )
        z_history_norm = torch.where(
            batch.z_mask,
            (batch.z_history - z_hist_mean) / z_hist_scale,
            torch.zeros_like(batch.z_history),
        )
        observation_context, observation_available = self.observation_encoder(
            q_history_norm,
            z_history_norm,
            batch.q_mask,
            batch.z_mask,
            station_index,
        )

        q0_warmup_obs = q_nodes_history[:, -1].index_select(1, obs_node_index)
        q_target_scale_for_correction = self._station_values(
            "q_target_scale", station_index, q0_warmup_obs.unsqueeze(1)
        ).squeeze(1)
        q0_observed_available = batch.q_mask[:, -1].bool()
        q0_residual_m3s = torch.where(
            q0_observed_available,
            batch.q_history[:, -1] - q0_warmup_obs,
            torch.zeros_like(q0_warmup_obs),
        )
        q0_residual_norm = q0_residual_m3s / q_target_scale_for_correction

        (
            node_obs_context,
            node_obs_available,
            node_q0_residual_norm,
            node_q0_residual_m3s,
            node_q0_available,
        ) = self._upstream_analysis_fields(
            observation_context=observation_context,
            observation_available=observation_available,
            q0_residual_norm=q0_residual_norm,
            q0_residual_m3s=q0_residual_m3s,
            q0_available=q0_observed_available,
            obs_node_index=obs_node_index,
            edge_index=batch.edge_index,
            incremental_area_km2=batch.incremental_area_km2,
            nodes=nodes,
        )
        runoff_t0_state, correction_diag = self.state_corrector(
            state=runoff_t0_state_raw,
            node_observation_context=node_obs_context,
            node_observation_available=node_obs_available,
            node_q0_residual_norm=node_q0_residual_norm,
            node_q0_residual_m3s=node_q0_residual_m3s,
            node_q0_residual_available=node_q0_available,
            node_static_norm=node_static_norm,
            incremental_area_km2=batch.incremental_area_km2,
        )

        if edge_storage_t0_raw is not None:
            (
                edge_storage_t0,
                edge_storage_log_factor,
                edge_storage_additive_m3,
            ) = self._correct_edge_storage_mass_aware(
                edge_storage_t0_raw,
                batch.edge_index,
                correction_diag["edge_node_log_factor"],
                node_q0_residual_m3s,
                correction_diag["assimilation_hours"],
                node_obs_available,
            )
            channel0_nodes = self._node_storage_from_edges(
                edge_storage_t0, batch.edge_index, nodes
            )
        else:
            edge_storage_t0 = None
            edge_storage_log_factor = q_nodes_history.new_zeros(
                (batch_size, int(batch.edge_index.shape[1]))
            )
            edge_storage_additive_m3 = edge_storage_log_factor.clone()
            channel0_nodes = q_nodes_history.new_zeros((batch_size, nodes))

        q_lat_future, runoff_future_diag, _ = self._run_runoff(
            batch.future_rain,
            future_rain_norm,
            node_static_norm,
            batch.incremental_area_km2,
            runoff_t0_state,
        )
        if self.routing_mode == "kinematic_wave_gnn":
            q_nodes_internal, routing_future_diag = self.routing(
                q_lat_future,
                batch.node_static,
                batch.edge_index,
                batch.edge_static,
                initial_edge_storage=edge_storage_t0,
            )
            channel_internal = routing_future_diag["node_channel_storage"]
            q0_edges = routing_future_diag["initial_edge_discharge_m3s"]
            q0_nodes_corrected = q_lat_history[:, -1].clone()
            if q0_edges.shape[1]:
                q0_nodes_corrected.index_add_(1, batch.edge_index[1], q0_edges)
        else:
            q_nodes_internal, routing_future_diag = self.routing(
                q_lat_future,
                batch.node_static,
                batch.edge_index,
                batch.edge_static,
            )
            channel_internal = torch.zeros_like(q_nodes_internal)
            q0_nodes_corrected = q_nodes_history[:, -1]

        target_indices = self.target_indices.to(q_nodes_internal.device)
        q_nodes = q_nodes_internal.index_select(1, target_indices)
        channel_nodes = channel_internal.index_select(1, target_indices)
        q_obs = q_nodes.index_select(2, obs_node_index)
        channel_obs = channel_nodes.index_select(2, obs_node_index)
        q0_model_corrected = q0_nodes_corrected.index_select(1, obs_node_index)
        # Forecast-origin analysis honours a valid observed Q0 exactly. This is
        # the origin used by the Z head and by future-Q change features.
        q0_analysis = torch.where(
            q0_observed_available,
            batch.q_history[:, -1],
            q0_model_corrected,
        )
        channel0 = channel0_nodes.index_select(1, obs_node_index)

        static_batch = node_static_norm.unsqueeze(0).expand(batch_size, -1, -1)
        node_context = self.node_context_projection(
            torch.cat([runoff_t0_state["h"], static_batch], dim=-1)
        )

        q_target_mean = self._station_values(
            "q_target_mean", station_index, q_obs
        )
        q_target_scale = self._station_values(
            "q_target_scale", station_index, q_obs
        )
        q_future_norm = (q_obs - q_target_mean) / q_target_scale
        q0_analysis_norm = (
            q0_analysis - q_target_mean.squeeze(1)
        ) / q_target_scale.squeeze(1)
        q_delta_norm = (q_obs - q0_analysis.unsqueeze(1)) / q_target_scale
        q0_observed_norm = torch.where(
            q0_observed_available,
            (batch.q_history[:, -1] - q_hist_mean.squeeze(1))
            / q_hist_scale.squeeze(1),
            torch.zeros_like(batch.q_history[:, -1]),
        )

        channel0_log = torch.log1p(channel0.clamp_min(0))
        channel_future_log = torch.log1p(channel_obs.clamp_min(0))
        channel_delta_log = channel_future_log - channel0_log.unsqueeze(1)
        z_state_features = self._z_state_features(batch, station_index)
        dz_scale = self._station_values("dz_target_scale", station_index, q_obs)
        z_delta, z_increment = self.z_head(
            node_context=node_context,
            observation_context=observation_context,
            obs_node_index=obs_node_index,
            obs_station_index=station_index,
            z_state_features=z_state_features,
            q0_model_norm=q0_analysis_norm,
            q0_observed_norm=q0_observed_norm,
            q0_observed_available=q0_observed_available,
            q_future_norm=q_future_norm,
            q_delta_norm=q_delta_norm,
            channel0_log=channel0_log,
            channel_future_log=channel_future_log,
            channel_delta_log=channel_delta_log,
            channel_available=channel_available,
            dz_scale=dz_scale,
        )

        diagnostics: dict[str, torch.Tensor] = {
            "history_observation_context": observation_context,
            "history_observation_available": observation_available,
            "history_node_context": node_context,
            "warmup_network_q_m3s": q_nodes_history,
            "network_q_m3s": q_nodes_internal,
            "local_runoff_q_m3s": q_lat_future,
            "q_origin_warmup_m3s": q0_warmup_obs,
            "q_origin_model_corrected_m3s": q0_model_corrected,
            "q_origin_analysis_m3s": q0_analysis,
            "z_increment_m": z_increment,
            "channel_state_available": channel_available,
            "state_correction_node_available": node_obs_available,
            "state_correction_q0_residual_norm": node_q0_residual_norm,
            "state_correction_q0_residual_m3s": node_q0_residual_m3s,
            "state_correction_edge_log_factor": edge_storage_log_factor,
            "state_correction_edge_additive_storage_m3": edge_storage_additive_m3,
        }
        for key, value in correction_diag.items():
            diagnostics[f"state_correction_{key}"] = value
        for key, value in runoff_history_diag.items():
            diagnostics[f"warmup_runoff_{key}"] = value
        for key, value in runoff_future_diag.items():
            diagnostics[f"forecast_runoff_{key}"] = value
        for key, value in routing_history_diag.items():
            diagnostics[f"warmup_routing_{key}"] = value
        for key, value in routing_future_diag.items():
            diagnostics[f"forecast_routing_{key}"] = value
        warmup_cfl = routing_history_diag.get("explicit_equivalent_substeps")
        forecast_cfl = routing_future_diag.get("explicit_equivalent_substeps")
        if isinstance(warmup_cfl, torch.Tensor) and isinstance(
            forecast_cfl, torch.Tensor
        ):
            diagnostics["explicit_equivalent_substeps"] = torch.cat(
                [warmup_cfl.reshape(-1), forecast_cfl.reshape(-1)]
            )
        elif isinstance(forecast_cfl, torch.Tensor):
            diagnostics["explicit_equivalent_substeps"] = forecast_cfl
        if edge_storage_t0 is not None:
            diagnostics["warmup_edge_storage_t0_raw_m3"] = edge_storage_t0_raw
            diagnostics["warmup_edge_storage_t0_m3"] = edge_storage_t0
            diagnostics["forecast_initial_edge_storage_m3"] = routing_future_diag[
                "initial_edge_storage_m3"
            ]
            diagnostics["future_node_channel_storage_m3"] = channel_internal

        return {
            "q": q_obs,
            "z": z_delta,
            "q_lat": q_lat_future,
            "q_nodes": q_nodes,
            "diagnostics": diagnostics,
        }
