"""v9 hydrologic graph model with physical warm-up and observation state correction."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from data.v8_schema import HydrologicGraphBatch
from models.routing import KinematicWaveGNN, PureDirectedGNN
from models.runoff.water_balance_v9 import ContinuousTimeWaterBalanceLSTM


def _as_buffer(values: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype)
    if not torch.isfinite(tensor).all():
        raise ValueError("v9 normalization buffer包含NaN/Inf")
    return tensor


def _validate_v9_batch(
    batch: HydrologicGraphBatch,
    *,
    history_steps: int,
    forecast_internal_steps: int,
    target_steps: int,
    node_static_dim: int,
    edge_static_dim: int,
) -> None:
    """Validate v9 shapes while allowing forcing and target time steps to differ."""
    if batch.history_rain.ndim != 4:
        raise ValueError("v9 history_rain应为[B,H,Nnode,1]")
    b, h, n, rain_dim = batch.history_rain.shape
    if h != history_steps or rain_dim != 1 or b <= 0 or n <= 0:
        raise ValueError("v9 history_rain时空维与temporal contract不一致")
    if tuple(batch.future_rain.shape) != (b, forecast_internal_steps, n, 1):
        raise ValueError("v9 future_rain内部步数与temporal contract不一致")
    if tuple(batch.node_static.shape) != (n, node_static_dim):
        raise ValueError("v9 node_static形状错误")
    if tuple(batch.incremental_area_km2.shape) != (n,):
        raise ValueError("v9 incremental_area_km2形状错误")
    if (
        batch.edge_index.dtype != torch.long
        or batch.edge_index.ndim != 2
        or batch.edge_index.shape[0] != 2
    ):
        raise ValueError("v9 edge_index必须为LongTensor [2,E]")
    edges = int(batch.edge_index.shape[1])
    if tuple(batch.edge_static.shape) != (edges, edge_static_dim):
        raise ValueError("v9 edge_static形状错误")
    if batch.obs_node_index.dtype != torch.long or batch.obs_node_index.ndim != 1:
        raise ValueError("v9 obs_node_index必须为LongTensor [Nobs]")
    obs = int(batch.obs_node_index.numel())
    if obs <= 0 or tuple(batch.obs_station_index.shape) != (obs,):
        raise ValueError("v9 observation catalogue非法")
    if (batch.obs_node_index < 0).any() or (batch.obs_node_index >= n).any():
        raise ValueError("v9 obs_node_index越界")
    for name in ("q_history", "z_history", "q_mask", "z_mask"):
        if tuple(getattr(batch, name).shape) != (b, history_steps, obs):
            raise ValueError(f"v9 {name}形状错误")
    for name in ("q_target", "z_target", "q_target_mask", "z_target_mask"):
        if tuple(getattr(batch, name).shape) != (b, target_steps, obs):
            raise ValueError(f"v9 {name}形状错误")
    for name in ("q_mask", "z_mask", "q_target_mask", "z_target_mask"):
        if getattr(batch, name).dtype != torch.bool:
            raise ValueError(f"v9 {name}必须为BoolTensor")
    for name in (
        "history_rain",
        "future_rain",
        "node_static",
        "incremental_area_km2",
        "edge_static",
        "q_history",
        "z_history",
        "q_target",
        "z_target",
    ):
        if not torch.isfinite(getattr(batch, name)).all():
            raise ValueError(f"v9 {name}含NaN/Inf；loader必须先按mask安全占位")
    if (batch.history_rain < 0).any() or (batch.future_rain < 0).any():
        raise ValueError("v9 rainfall forcing必须非负")
    if (batch.incremental_area_km2 <= 0).any():
        raise ValueError("v9 incremental area必须>0")


class StaticWarmupInitializer(nn.Module):
    """Static-conditioned prior at the beginning of the history warm-up."""

    def __init__(self, node_static_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_static_dim = int(node_static_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.node_static_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.h_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.storage_head = nn.Sequential(nn.Linear(self.hidden_dim, 2), nn.Softplus())

    def forward(
        self, node_static_norm: torch.Tensor, batch_size: int
    ) -> dict[str, torch.Tensor]:
        if (
            node_static_norm.ndim != 2
            or node_static_norm.shape[1] != self.node_static_dim
        ):
            raise ValueError("node_static_norm必须为[N,node_static_dim]")
        base = self.encoder(node_static_norm).unsqueeze(0).expand(batch_size, -1, -1)
        storage = self.storage_head(base)
        return {
            "h": torch.tanh(self.h_head(base)),
            "c": torch.tanh(self.c_head(base)),
            "storage_fast_mm": storage[..., 0],
            "storage_slow_mm": storage[..., 1],
        }


class ObservationHistoryEncoderV9(nn.Module):
    """V8-style sparse Q/Z history encoder; no extra recurrent module is added."""

    def __init__(self, hidden_dim: int, stations: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.LSTM(4, self.hidden_dim, batch_first=True)
        self.station_embedding = nn.Embedding(int(stations), self.hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        q_history_norm: torch.Tensor,
        z_history_norm: torch.Tensor,
        q_mask: torch.Tensor,
        z_mask: torch.Tensor,
        station_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_history_norm.shape != z_history_norm.shape:
            raise ValueError("Q/Z history normalized shape不一致")
        if q_mask.shape != q_history_norm.shape or z_mask.shape != q_history_norm.shape:
            raise ValueError("Q/Z history mask shape不一致")
        batch, history, obs = q_history_norm.shape
        sequence = torch.stack(
            (
                q_history_norm,
                q_mask.to(q_history_norm.dtype),
                z_history_norm,
                z_mask.to(z_history_norm.dtype),
            ),
            dim=-1,
        )
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch * obs, history, 4)
        _, (encoded, _) = self.encoder(sequence)
        encoded = encoded[-1].reshape(batch, obs, self.hidden_dim)
        station = self.station_embedding(station_index).unsqueeze(0).expand(
            batch, -1, -1
        )
        context = self.projection(torch.cat([encoded, station], dim=-1))
        available = (q_mask | z_mask).any(dim=1)
        context = torch.where(
            available.unsqueeze(-1), context, torch.zeros_like(context)
        )
        return context, available


class ObservationStateCorrectorV9(nn.Module):
    """Assimilate Q/Z history as a bounded residual correction to warm-up states.

    Physics/routing first evolve the 24 h history.  The observation encoder then
    corrects the forecast-origin state rather than replacing it.  With no Q/Z
    history mapped to a node the correction is exactly zero.
    """

    def __init__(
        self,
        node_static_dim: int,
        hidden_dim: int,
        *,
        hidden_residual_scale: float,
        storage_log_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.hidden_residual_scale = float(hidden_residual_scale)
        self.storage_log_scale = float(storage_log_scale)
        input_dim = 3 * self.hidden_dim + int(node_static_dim) + 3
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.h_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.storage_head = nn.Linear(self.hidden_dim, 2)
        self.edge_storage_head = nn.Linear(self.hidden_dim, 1)
        for head in (self.h_head, self.c_head, self.storage_head, self.edge_storage_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        *,
        state: dict[str, torch.Tensor],
        node_observation_context: torch.Tensor,
        node_observation_available: torch.Tensor,
        node_q0_residual_norm: torch.Tensor,
        node_q0_residual_available: torch.Tensor,
        node_static_norm: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch, nodes, hidden = state["h"].shape
        if hidden != self.hidden_dim or state["c"].shape != state["h"].shape:
            raise ValueError("v9 state corrector h/c形状错误")
        expected_context = (batch, nodes, self.hidden_dim)
        if node_observation_context.shape != expected_context:
            raise ValueError("v9 node observation context形状错误")
        gate = node_observation_available.to(state["h"].dtype)
        q_gate = node_q0_residual_available.to(state["h"].dtype)
        if gate.shape != (batch, nodes, 1) or q_gate.shape != gate.shape:
            raise ValueError("v9 state correction availability形状错误")
        if node_q0_residual_norm.shape != (batch, nodes, 1):
            raise ValueError("v9 q0 residual形状错误")
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
        delta_h = (
            self.hidden_residual_scale * torch.tanh(self.h_head(context)) * gate
        )
        delta_c = (
            self.hidden_residual_scale * torch.tanh(self.c_head(context)) * gate
        )
        storage_log = (
            self.storage_log_scale
            * torch.tanh(self.storage_head(context))
            * gate
        )
        storage_fast = (
            (state["storage_fast_mm"] + 1.0e-6) * torch.exp(storage_log[..., 0])
            - 1.0e-6
        ).clamp_min(0.0)
        storage_slow = (
            (state["storage_slow_mm"] + 1.0e-6) * torch.exp(storage_log[..., 1])
            - 1.0e-6
        ).clamp_min(0.0)
        corrected = {
            "h": state["h"] + delta_h,
            "c": state["c"] + delta_c,
            "storage_fast_mm": storage_fast,
            "storage_slow_mm": storage_slow,
        }
        edge_node_log_factor = (
            self.storage_log_scale
            * torch.tanh(self.edge_storage_head(context))
            * gate
        )
        return corrected, {
            "context": context,
            "delta_h": delta_h,
            "delta_c": delta_c,
            "storage_log_factor": storage_log,
            "edge_node_log_factor": edge_node_log_factor,
            "node_observation_available": node_observation_available,
            "node_q0_residual_norm": node_q0_residual_norm,
            "node_q0_residual_available": node_q0_residual_available,
        }


class PureLSTMRunoffV9(nn.Module):
    """Pure-AI runoff ablation with state continuation across warm-up/forecast."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.lstm = nn.LSTM(int(input_dim), self.hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Softplus())

    def forward(
        self,
        features: torch.Tensor,
        *,
        initial_h: torch.Tensor,
        initial_c: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if features.ndim != 4:
            raise ValueError("pure LSTM features必须为[B,T,N,D]")
        batch, steps, nodes, dim = features.shape
        expected = (batch, nodes, self.hidden_dim)
        if initial_h.shape != expected or initial_c.shape != expected:
            raise ValueError("pure LSTM initial h/c形状错误")
        sequence = features.permute(0, 2, 1, 3).reshape(batch * nodes, steps, dim)
        h0 = initial_h.reshape(batch * nodes, self.hidden_dim).unsqueeze(0)
        c0 = initial_c.reshape(batch * nodes, self.hidden_dim).unsqueeze(0)
        output, (h, c) = self.lstm(sequence, (h0, c0))
        q = self.head(output).squeeze(-1)
        q = q.reshape(batch, nodes, steps).permute(0, 2, 1).contiguous()
        return q, {
            "final_h": h[-1].reshape(batch, nodes, self.hidden_dim),
            "final_c": c[-1].reshape(batch, nodes, self.hidden_dim),
            "runoff_water_balance_residual": torch.full_like(q, float("nan")),
        }


class ExplicitStateDeltaZV9Head(nn.Module):
    """MLP predicts target-step water-level increments then cumulatively forms Delta-Z."""

    def __init__(self, hidden_dim: int, horizon: int, stations: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.station_embedding = nn.Embedding(int(stations), self.hidden_dim)
        input_dim = 3 * self.hidden_dim + 13 + 4 * self.horizon
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon),
        )
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.02)
        final = self.network[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
            nn.init.zeros_(final.bias)

    def forward(
        self,
        *,
        node_context: torch.Tensor,
        observation_context: torch.Tensor,
        obs_node_index: torch.Tensor,
        obs_station_index: torch.Tensor,
        z_state_features: torch.Tensor,
        q0_model_norm: torch.Tensor,
        q0_observed_norm: torch.Tensor,
        q0_observed_available: torch.Tensor,
        q_future_norm: torch.Tensor,
        q_delta_norm: torch.Tensor,
        channel0_log: torch.Tensor,
        channel_future_log: torch.Tensor,
        channel_delta_log: torch.Tensor,
        channel_available: torch.Tensor,
        dz_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, obs = q_future_norm.shape
        if horizon != self.horizon:
            raise ValueError("v9 Z head horizon不一致")
        mapped = node_context[:, obs_node_index]
        station = self.station_embedding(obs_station_index).unsqueeze(0).expand(
            batch, -1, -1
        )
        if z_state_features.shape != (batch, obs, 8):
            raise ValueError("z_state_features必须为[B,Nobs,8]")
        hydraulic = torch.cat(
            [
                q0_model_norm.detach().unsqueeze(-1),
                q0_observed_norm.detach().unsqueeze(-1),
                q0_observed_available.to(q_future_norm.dtype).unsqueeze(-1),
                q_future_norm.detach().permute(0, 2, 1),
                q_delta_norm.detach().permute(0, 2, 1),
                channel0_log.detach().unsqueeze(-1),
                channel_future_log.detach().permute(0, 2, 1),
                channel_delta_log.detach().permute(0, 2, 1),
                channel_available.to(q_future_norm.dtype)
                .view(1, 1, 1)
                .expand(batch, obs, 1),
            ],
            dim=-1,
        )
        features = torch.cat(
            [mapped, observation_context, station, z_state_features, hydraulic], dim=-1
        )
        increment_norm = self.network(features).permute(0, 2, 1).contiguous()
        increment_m = increment_norm * dz_scale
        delta_z = torch.cumsum(increment_m, dim=1)
        return delta_z, increment_m


class HydrologicGraphV9Model(nn.Module):
    """Shared E1--E4 v9 model with warm-up then V8-style state assimilation."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.history = int(cfg["history_length"])
        self.horizon = int(cfg["forecast_horizon"])
        self.node_static_dim = int(cfg["node_static_dim"])
        self.edge_static_dim = int(cfg["edge_static_dim"])
        self.hidden_dim = int(cfg["hidden_dim"])
        temporal = cfg["temporal"]
        self.forcing_step_seconds = float(temporal["forcing_step_seconds"])
        self.target_step_seconds = float(temporal["target_step_seconds"])
        self.forecast_internal_steps = int(
            round(
                float(temporal["forecast_duration_seconds"])
                / self.forcing_step_seconds
            )
        )
        self.target_stride = int(
            round(self.target_step_seconds / self.forcing_step_seconds)
        )
        target_indices = torch.arange(
            self.target_stride - 1,
            self.forecast_internal_steps,
            self.target_stride,
            dtype=torch.long,
        )
        if target_indices.numel() != self.horizon:
            raise ValueError("v9 temporal contract与forecast_horizon不一致")
        self.register_buffer("target_indices", target_indices)

        runtime = cfg.get("_runtime", {})
        normal = runtime.get("v8_normalization")
        if not isinstance(normal, Mapping):
            raise ValueError("v9缺少TRAIN-only normalization")
        stations = int(runtime.get("v8_station_count", 0))
        if stations <= 0:
            raise ValueError("v9缺少全局station catalogue")

        self.register_buffer("rain_mean", _as_buffer(normal["rain_mean"]).reshape(()))
        self.register_buffer("rain_scale", _as_buffer(normal["rain_scale"]).reshape(()))
        self.register_buffer(
            "node_static_mean", _as_buffer(normal["node_static_mean"]).reshape(1, -1)
        )
        self.register_buffer(
            "node_static_scale",
            _as_buffer(normal["node_static_scale"]).reshape(1, -1),
        )
        for name in (
            "q_history_mean",
            "q_history_scale",
            "z_history_mean",
            "z_history_scale",
            "q_target_mean",
            "q_target_scale",
            "dz_target_scale",
        ):
            self.register_buffer(name, _as_buffer(normal[name]).reshape(-1))

        self.static_initializer = StaticWarmupInitializer(
            self.node_static_dim, self.hidden_dim
        )
        self.observation_encoder = ObservationHistoryEncoderV9(
            self.hidden_dim, stations
        )
        correction_cfg = cfg.get("state_correction", {})
        self.state_corrector = ObservationStateCorrectorV9(
            self.node_static_dim,
            self.hidden_dim,
            hidden_residual_scale=float(
                correction_cfg.get("hidden_residual_scale", 0.25)
            ),
            storage_log_scale=float(correction_cfg.get("storage_log_scale", 0.35)),
        )
        self.node_context_projection = nn.Sequential(
            nn.Linear(self.hidden_dim + self.node_static_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.runoff_mode = str(cfg["runoff_mode"])
        if self.runoff_mode == "pure_lstm":
            self.runoff = PureLSTMRunoffV9(
                1 + self.node_static_dim, self.hidden_dim
            )
        elif self.runoff_mode == "water_balance_lstm":
            # Rainfall is an explicit controller input as well as the conserved
            # water mass input.  This fixes the old static-only gate dynamics.
            self.runoff = ContinuousTimeWaterBalanceLSTM(
                1 + self.node_static_dim, self.hidden_dim
            )
        else:
            raise ValueError(f"未知runoff_mode={self.runoff_mode!r}")

        self.routing_mode = str(cfg["routing_mode"])
        if self.routing_mode == "pure_gnn":
            self.routing = PureDirectedGNN(
                self.node_static_dim, self.edge_static_dim, self.hidden_dim
            )
        elif self.routing_mode == "kinematic_wave_gnn":
            self.routing = KinematicWaveGNN(
                self.node_static_dim,
                self.edge_static_dim,
                self.hidden_dim,
                cfg["physical_bounds"],
                cfg["solver"],
            )
        else:
            raise ValueError(f"未知routing_mode={self.routing_mode!r}")

        self.z_head = ExplicitStateDeltaZV9Head(
            self.hidden_dim, self.horizon, stations
        )
        self.trend_windows_seconds = tuple(
            int(value) for value in cfg["z_head"]["trend_windows_seconds"]
        )
        if len(self.trend_windows_seconds) != 3:
            raise ValueError("v9 Z head固定需要三个trend window")

    def _station_values(
        self, name: str, station_index: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        values = getattr(self, name).to(
            device=reference.device, dtype=reference.dtype
        )
        return values[station_index].view(1, 1, -1)

    def _run_runoff(
        self,
        rain_raw: torch.Tensor,
        rain_norm: torch.Tensor,
        node_static_norm: torch.Tensor,
        area: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch, steps, _, _ = rain_raw.shape
        static = node_static_norm.unsqueeze(0).unsqueeze(0).expand(
            batch, steps, -1, -1
        )
        features = torch.cat([rain_norm, static], dim=-1)
        if self.runoff_mode == "pure_lstm":
            q, diagnostics = self.runoff(
                features, initial_h=state["h"], initial_c=state["c"]
            )
            next_state = {
                "h": diagnostics["final_h"],
                "c": diagnostics["final_c"],
                "storage_fast_mm": state["storage_fast_mm"],
                "storage_slow_mm": state["storage_slow_mm"],
            }
        else:
            q, diagnostics = self.runoff(
                features,
                rain_raw,
                area,
                seconds=self.forcing_step_seconds,
                initial_state=(
                    state["h"],
                    state["c"],
                    state["storage_fast_mm"],
                    state["storage_slow_mm"],
                ),
            )
            next_state = {
                "h": diagnostics["final_h"],
                "c": diagnostics["final_c"],
                "storage_fast_mm": diagnostics["final_storage_fast_mm"],
                "storage_slow_mm": diagnostics["final_storage_slow_mm"],
            }
        return q, diagnostics, next_state

    @staticmethod
    def _aggregate_observations_to_nodes(
        values: torch.Tensor,
        available: torch.Tensor,
        obs_node_index: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or available.shape != values.shape[:2]:
            raise ValueError("v9 observation aggregation形状错误")
        batch, obs, features = values.shape
        node_values = values.new_zeros((batch, nodes, features))
        node_count = values.new_zeros((batch, nodes, 1))
        if obs:
            node_values.index_add_(1, obs_node_index, values)
            node_count.index_add_(
                1, obs_node_index, available.to(values.dtype).unsqueeze(-1)
            )
        node_values = node_values / node_count.clamp_min(1.0)
        return node_values, node_count.gt(0)

    @staticmethod
    def _aggregate_scalar_to_nodes(
        values: torch.Tensor,
        available: torch.Tensor,
        obs_node_index: torch.Tensor,
        nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or available.shape != values.shape:
            raise ValueError("v9 scalar observation aggregation形状错误")
        batch, obs = values.shape
        node_sum = values.new_zeros((batch, nodes, 1))
        node_count = values.new_zeros((batch, nodes, 1))
        if obs:
            node_sum.index_add_(1, obs_node_index, (values * available).unsqueeze(-1))
            node_count.index_add_(
                1, obs_node_index, available.to(values.dtype).unsqueeze(-1)
            )
        return node_sum / node_count.clamp_min(1.0), node_count.gt(0)

    @staticmethod
    def _correct_edge_storage(
        edge_storage: torch.Tensor,
        edge_index: torch.Tensor,
        node_log_factor: torch.Tensor,
        node_available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_storage.ndim != 2:
            raise ValueError("v9 edge storage必须为[B,E]")
        if edge_storage.shape[1] == 0:
            return edge_storage, edge_storage.clone()
        source, destination = edge_index
        src_factor = node_log_factor[:, source, 0]
        dst_factor = node_log_factor[:, destination, 0]
        src_gate = node_available[:, source, 0]
        dst_gate = node_available[:, destination, 0]
        edge_gate = (src_gate | dst_gate).to(edge_storage.dtype)
        edge_log_factor = 0.5 * (src_factor + dst_factor) * edge_gate
        corrected = edge_storage * torch.exp(edge_log_factor)
        return corrected.clamp_min(0.0), edge_log_factor

    @staticmethod
    def _node_storage_from_edges(
        edge_storage: torch.Tensor, edge_index: torch.Tensor, nodes: int
    ) -> torch.Tensor:
        result = edge_storage.new_zeros((edge_storage.shape[0], nodes))
        if edge_storage.shape[1]:
            result.index_add_(1, edge_index[1], edge_storage)
        return result

    @staticmethod
    def _masked_recent_slope(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int,
        dt_hours: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = min(int(steps), int(values.shape[1]))
        segment = values[:, -steps:]
        valid = mask[:, -steps:].bool()
        times = torch.arange(
            -(steps - 1), 1, device=values.device, dtype=values.dtype
        ) * float(dt_hours)
        weights = valid.to(values.dtype)
        count = weights.sum(dim=1)
        time = times.view(1, steps, 1)
        time_mean = (weights * time).sum(dim=1) / count.clamp_min(1.0)
        value_mean = (weights * segment).sum(dim=1) / count.clamp_min(1.0)
        centered_time = time - time_mean.unsqueeze(1)
        centered_value = segment - value_mean.unsqueeze(1)
        denominator = (weights * centered_time.square()).sum(dim=1)
        numerator = (weights * centered_time * centered_value).sum(dim=1)
        available = (count >= 2) & (denominator > 0)
        slope = torch.where(
            available,
            numerator / denominator.clamp_min(1.0e-12),
            torch.zeros_like(numerator),
        )
        return slope, available

    def _z_state_features(
        self,
        batch: HydrologicGraphBatch,
        station_index: torch.Tensor,
    ) -> torch.Tensor:
        z_hist_mean = self._station_values(
            "z_history_mean", station_index, batch.z_history
        )
        z_hist_scale = self._station_values(
            "z_history_scale", station_index, batch.z_history
        )
        z0_available = batch.z_mask[:, -1].bool()
        z0_norm = torch.where(
            z0_available,
            (batch.z_history[:, -1] - z_hist_mean.squeeze(1))
            / z_hist_scale.squeeze(1),
            torch.zeros_like(batch.z_history[:, -1]),
        )
        parts = [z0_norm, z0_available.to(z0_norm.dtype)]
        dt_hours = self.forcing_step_seconds / 3600.0
        for seconds in self.trend_windows_seconds:
            steps = max(2, int(round(seconds / self.forcing_step_seconds)) + 1)
            slope, available = self._masked_recent_slope(
                batch.z_history, batch.z_mask, steps=steps, dt_hours=dt_hours
            )
            slope_norm = slope / z_hist_scale.squeeze(1)
            parts.extend([slope_norm, available.to(slope_norm.dtype)])
        return torch.stack(parts, dim=-1)

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

        # 1) Physical/AI warm-up over the complete 24 h forcing history.
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

        # 2) V8-style Q/Z history encoder now acts as a residual state corrector.
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
        node_obs_context, node_obs_available = self._aggregate_observations_to_nodes(
            observation_context,
            observation_available,
            obs_node_index,
            nodes,
        )

        q0_warmup_obs = q_nodes_history[:, -1].index_select(1, obs_node_index)
        q_target_scale_for_correction = self._station_values(
            "q_target_scale", station_index, q0_warmup_obs.unsqueeze(1)
        ).squeeze(1)
        q0_observed_available = batch.q_mask[:, -1].bool()
        q0_residual_norm = torch.where(
            q0_observed_available,
            (batch.q_history[:, -1] - q0_warmup_obs)
            / q_target_scale_for_correction,
            torch.zeros_like(q0_warmup_obs),
        )
        node_q0_residual, node_q0_available = self._aggregate_scalar_to_nodes(
            q0_residual_norm,
            q0_observed_available,
            obs_node_index,
            nodes,
        )
        runoff_t0_state, correction_diag = self.state_corrector(
            state=runoff_t0_state_raw,
            node_observation_context=node_obs_context,
            node_observation_available=node_obs_available,
            node_q0_residual_norm=node_q0_residual,
            node_q0_residual_available=node_q0_available,
            node_static_norm=node_static_norm,
        )

        if edge_storage_t0_raw is not None:
            edge_storage_t0, edge_storage_log_factor = self._correct_edge_storage(
                edge_storage_t0_raw,
                batch.edge_index,
                correction_diag["edge_node_log_factor"],
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
            channel0_nodes = q_nodes_history.new_zeros((batch_size, nodes))

        # 3) Forecast continues corrected runoff state and corrected physical
        # channel storage.  No state is reset at the forecast origin.
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
            q0_nodes = q_lat_history[:, -1].clone()
            if q0_edges.shape[1]:
                q0_nodes.index_add_(1, batch.edge_index[1], q0_edges)
        else:
            q_nodes_internal, routing_future_diag = self.routing(
                q_lat_future,
                batch.node_static,
                batch.edge_index,
                batch.edge_static,
            )
            channel_internal = torch.zeros_like(q_nodes_internal)
            q0_nodes = q_nodes_history[:, -1]

        target_indices = self.target_indices.to(q_nodes_internal.device)
        q_nodes = q_nodes_internal.index_select(1, target_indices)
        channel_nodes = channel_internal.index_select(1, target_indices)
        q_obs = q_nodes.index_select(2, obs_node_index)
        channel_obs = channel_nodes.index_select(2, obs_node_index)
        q0_model = q0_nodes.index_select(1, obs_node_index)
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
        q0_model_norm = (
            q0_model - q_target_mean.squeeze(1)
        ) / q_target_scale.squeeze(1)
        q_delta_norm = (q_obs - q0_model.unsqueeze(1)) / q_target_scale
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
            q0_model_norm=q0_model_norm,
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
            "q_origin_model_m3s": q0_model,
            "z_increment_m": z_increment,
            "channel_state_available": channel_available,
            "state_correction_node_available": node_obs_available,
            "state_correction_q0_residual_norm": node_q0_residual,
            "state_correction_edge_log_factor": edge_storage_log_factor,
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
