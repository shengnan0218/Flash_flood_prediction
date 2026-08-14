"""Shared E1--E4 model for v8 hydrologic graphs and sparse observation initialization."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from data.v8_schema import HydrologicGraphBatch, validate_v8_batch
from models.observation import MonotonicQZObservation
from models.routing import KinematicWaveGNN, PureDirectedGNN
from models.runoff import WaterBalanceLSTM


def _as_buffer(values: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=dtype)
    if not torch.isfinite(tensor).all():
        raise ValueError("v8 normalization buffer包含NaN/Inf")
    return tensor


class HistoryInformedInitializer(nn.Module):
    """Fuse rainfall/static history with sparse Q/Z observations at mapped nodes."""

    def __init__(
        self,
        *,
        node_static_dim: int,
        hidden_dim: int,
        stations: int,
        history_hours: int,
    ) -> None:
        super().__init__()
        self.node_static_dim = int(node_static_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_hours = int(history_hours)
        self.physics_encoder = nn.LSTM(
            1 + self.node_static_dim, self.hidden_dim, batch_first=True
        )
        self.observation_encoder = nn.LSTM(4, self.hidden_dim, batch_first=True)
        self.station_embedding = nn.Embedding(int(stations), self.hidden_dim)
        self.observation_projection = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.hidden_dim + 1 + self.node_static_dim,
                self.hidden_dim,
            ),
            nn.SiLU(),
        )
        self.h_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.c_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.storage_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),
            nn.Softplus(),
        )
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        history_rain_norm: torch.Tensor,
        node_static_norm: torch.Tensor,
        q_history_norm: torch.Tensor,
        z_history_norm: torch.Tensor,
        q_mask: torch.Tensor,
        z_mask: torch.Tensor,
        obs_node_index: torch.Tensor,
        obs_station_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_rain_norm.ndim != 4:
            raise ValueError("history_rain_norm必须为[B,H,N,1]")
        batch, history, nodes, one = history_rain_norm.shape
        if history != self.history_hours or one != 1:
            raise ValueError("history rainfall长度/特征维与initializer不一致")
        if node_static_norm.shape != (nodes, self.node_static_dim):
            raise ValueError("node_static_norm必须为[N,node_static_dim]")
        obs = int(obs_node_index.numel())
        expected_obs = (batch, history, obs)
        for name, value in (
            ("q_history_norm", q_history_norm),
            ("z_history_norm", z_history_norm),
            ("q_mask", q_mask),
            ("z_mask", z_mask),
        ):
            if tuple(value.shape) != expected_obs:
                raise ValueError(f"{name}必须为[B,H,Nobs]")

        static = node_static_norm.unsqueeze(0).unsqueeze(0).expand(
            batch, history, -1, -1
        )
        physics_sequence = torch.cat([history_rain_norm, static], dim=-1)
        physics_sequence = (
            physics_sequence.permute(0, 2, 1, 3)
            .reshape(batch * nodes, history, -1)
        )
        _, (physics_h, physics_c) = self.physics_encoder(physics_sequence)
        physics_h = physics_h[-1].reshape(batch, nodes, self.hidden_dim)
        physics_c = physics_c[-1].reshape(batch, nodes, self.hidden_dim)

        obs_sequence = torch.stack(
            (
                q_history_norm,
                q_mask.to(q_history_norm.dtype),
                z_history_norm,
                z_mask.to(z_history_norm.dtype),
            ),
            dim=-1,
        )
        obs_sequence = (
            obs_sequence.permute(0, 2, 1, 3)
            .reshape(batch * obs, history, 4)
        )
        _, (obs_h, _) = self.observation_encoder(obs_sequence)
        obs_h = obs_h[-1].reshape(batch, obs, self.hidden_dim)
        station = self.station_embedding(obs_station_index).unsqueeze(0).expand(
            batch, -1, -1
        )
        obs_context = self.observation_projection(torch.cat([obs_h, station], dim=-1))
        obs_available = (q_mask | z_mask).any(dim=1)
        obs_context = torch.where(
            obs_available.unsqueeze(-1), obs_context, torch.zeros_like(obs_context)
        )

        node_obs = physics_h.new_zeros((batch, nodes, self.hidden_dim))
        node_obs_count = physics_h.new_zeros((batch, nodes, 1))
        if obs:
            node_obs.index_add_(1, obs_node_index, obs_context)
            availability_float = obs_available.to(physics_h.dtype).unsqueeze(-1)
            node_obs_count.index_add_(1, obs_node_index, availability_float)
        node_obs = node_obs / node_obs_count.clamp_min(1.0)
        node_has_obs = node_obs_count.gt(0).to(physics_h.dtype)

        static_batch = node_static_norm.unsqueeze(0).expand(batch, -1, -1)
        context = self.node_fusion(
            torch.cat([physics_h, node_obs, node_has_obs, static_batch], dim=-1)
        )
        h0 = torch.tanh(self.h_head(context))
        c0 = torch.tanh(self.c_head(context + physics_c))
        storage = self.storage_head(context)
        return {
            "h0": h0,
            "c0": c0,
            "storage_fast_mm": storage[..., 0],
            "storage_slow_mm": storage[..., 1],
            "node_context": context,
            "observation_context": obs_context,
            "observation_available": obs_available,
        }


class PureLSTMRunoffV8(nn.Module):
    """Pure-AI local runoff ablation with the same history-informed h/c state."""

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
            raise ValueError("pure LSTM features必须为[B,F,N,D]")
        batch, steps, nodes, dim = features.shape
        if initial_h.shape != (batch, nodes, self.hidden_dim):
            raise ValueError("pure LSTM initial_h形状错误")
        if initial_c.shape != initial_h.shape:
            raise ValueError("pure LSTM initial_c形状错误")
        sequence = features.permute(0, 2, 1, 3).reshape(
            batch * nodes, steps, dim
        )
        h0 = initial_h.reshape(batch * nodes, self.hidden_dim).unsqueeze(0)
        c0 = initial_c.reshape(batch * nodes, self.hidden_dim).unsqueeze(0)
        output, _ = self.lstm(sequence, (h0, c0))
        q = self.head(output).squeeze(-1)
        q = q.reshape(batch, nodes, steps).permute(0, 2, 1).contiguous()
        return q, {
            "runoff_water_balance_residual": torch.full_like(q, float("nan"))
        }


class IndependentDeltaZV8Head(nn.Module):
    """Independent station head; future routed Q is explanatory and detached."""

    def __init__(self, hidden_dim: int, horizon: int, stations: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.station_embedding = nn.Embedding(int(stations), self.hidden_dim)
        input_dim = 3 * self.hidden_dim + self.horizon
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
        q_future_norm: torch.Tensor,
    ) -> torch.Tensor:
        if q_future_norm.ndim != 3:
            raise ValueError("q_future_norm必须为[B,F,Nobs]")
        batch, horizon, obs = q_future_norm.shape
        if horizon != self.horizon:
            raise ValueError("Z head forecast horizon不一致")
        mapped_node_context = node_context[:, obs_node_index]
        if observation_context.shape != (batch, obs, self.hidden_dim):
            raise ValueError("observation_context形状错误")
        station = self.station_embedding(obs_station_index).unsqueeze(0).expand(
            batch, -1, -1
        )
        q_features = q_future_norm.detach().permute(0, 2, 1)
        features = torch.cat(
            [mapped_node_context, observation_context, station, q_features], dim=-1
        )
        z_norm = self.network(features)
        return z_norm.permute(0, 2, 1).contiguous()


class HydrologicGraphV8Model(nn.Module):
    """E1--E4 shared model; only runoff/routing physics switches differ."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.history = int(cfg["history_length"])
        self.horizon = int(cfg["forecast_horizon"])
        self.node_static_dim = int(cfg["node_static_dim"])
        self.edge_static_dim = int(cfg["edge_static_dim"])
        self.hidden_dim = int(cfg["hidden_dim"])
        runtime = cfg.get("_runtime", {})
        normal = runtime.get("v8_normalization")
        if not isinstance(normal, Mapping):
            raise ValueError("v8 model缺少_runtime.v8_normalization")
        stations = int(runtime.get("v8_station_count", 0))
        if stations <= 0:
            raise ValueError("v8 model缺少全局station catalogue")

        self.register_buffer("rain_mean", _as_buffer(normal["rain_mean"]).reshape(()))
        self.register_buffer("rain_scale", _as_buffer(normal["rain_scale"]).reshape(()))
        self.register_buffer(
            "node_static_mean", _as_buffer(normal["node_static_mean"]).reshape(1, -1)
        )
        self.register_buffer(
            "node_static_scale", _as_buffer(normal["node_static_scale"]).reshape(1, -1)
        )
        for name in (
            "q_history_mean",
            "q_history_scale",
            "z_history_mean",
            "z_history_scale",
            "q_target_mean",
            "q_target_scale",
            "dz_target_mean",
            "dz_target_scale",
        ):
            self.register_buffer(name, _as_buffer(normal[name]).reshape(-1))
        if self.q_history_mean.numel() != stations:
            raise ValueError("v8 station normalization数量与station catalogue不一致")

        self.initializer = HistoryInformedInitializer(
            node_static_dim=self.node_static_dim,
            hidden_dim=self.hidden_dim,
            stations=stations,
            history_hours=self.history,
        )
        runoff_mode = str(cfg["runoff_mode"])
        self.runoff_mode = runoff_mode
        if runoff_mode == "pure_lstm":
            self.runoff = PureLSTMRunoffV8(
                1 + self.node_static_dim, self.hidden_dim
            )
        elif runoff_mode == "water_balance_lstm":
            self.runoff = WaterBalanceLSTM(self.node_static_dim, self.hidden_dim)
        else:
            raise ValueError(f"未知runoff_mode={runoff_mode!r}")

        routing_mode = str(cfg["routing_mode"])
        self.routing_mode = routing_mode
        if routing_mode == "pure_gnn":
            self.routing = PureDirectedGNN(
                self.node_static_dim, self.edge_static_dim, self.hidden_dim
            )
        elif routing_mode == "kinematic_wave_gnn":
            self.routing = KinematicWaveGNN(
                self.node_static_dim,
                self.edge_static_dim,
                self.hidden_dim,
                cfg["physical_bounds"],
                cfg["solver"],
            )
        else:
            raise ValueError(f"未知routing_mode={routing_mode!r}")

        self.z_head = IndependentDeltaZV8Head(
            self.hidden_dim, self.horizon, stations
        )
        self.qz_consistency = MonotonicQZObservation(
            stations, embedding_dim=8, hidden_dim=16
        )

    def _station_values(
        self, name: str, station_index: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        values = getattr(self, name).to(device=reference.device, dtype=reference.dtype)
        return values[station_index].view(1, 1, -1)

    def forward(self, batch: HydrologicGraphBatch) -> dict[str, Any]:
        validate_v8_batch(
            batch,
            history_hours=self.history,
            forecast_hours=self.horizon,
            node_static_dim=self.node_static_dim,
            edge_static_dim=self.edge_static_dim,
        )
        station_index = batch.obs_station_index.long()
        rain_history_norm = (
            batch.history_rain - self.rain_mean.to(batch.history_rain.dtype)
        ) / self.rain_scale.to(batch.history_rain.dtype)
        rain_future_norm = (
            batch.future_rain - self.rain_mean.to(batch.future_rain.dtype)
        ) / self.rain_scale.to(batch.future_rain.dtype)
        node_static_norm = (
            batch.node_static
            - self.node_static_mean.to(
                device=batch.node_static.device, dtype=batch.node_static.dtype
            )
        ) / self.node_static_scale.to(
            device=batch.node_static.device, dtype=batch.node_static.dtype
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

        state = self.initializer(
            history_rain_norm=rain_history_norm,
            node_static_norm=node_static_norm,
            q_history_norm=q_history_norm,
            z_history_norm=z_history_norm,
            q_mask=batch.q_mask,
            z_mask=batch.z_mask,
            obs_node_index=batch.obs_node_index.long(),
            obs_station_index=station_index,
        )

        batch_size = int(batch.future_rain.shape[0])
        static_future = node_static_norm.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.horizon, -1, -1
        )
        if self.runoff_mode == "pure_lstm":
            runoff_features = torch.cat([rain_future_norm, static_future], dim=-1)
            q_lateral, runoff_diagnostics = self.runoff(
                runoff_features,
                initial_h=state["h0"],
                initial_c=state["c0"],
            )
        else:
            q_lateral, runoff_diagnostics = self.runoff(
                static_future,
                batch.future_rain,
                batch.incremental_area_km2,
                seconds=float(self.cfg["solver"]["seconds_per_step"]),
                initial_state=(
                    state["h0"],
                    state["c0"],
                    state["storage_fast_mm"],
                    state["storage_slow_mm"],
                ),
            )

        q_nodes, routing_diagnostics = self.routing(
            q_lateral,
            batch.node_static,
            batch.edge_index,
            batch.edge_static,
        )
        q_obs = q_nodes.index_select(2, batch.obs_node_index.long())

        q_target_mean = self._station_values(
            "q_target_mean", station_index, q_obs
        )
        q_target_scale = self._station_values(
            "q_target_scale", station_index, q_obs
        )
        q_future_norm = (q_obs - q_target_mean) / q_target_scale
        z_norm = self.z_head(
            node_context=state["node_context"],
            observation_context=state["observation_context"],
            obs_node_index=batch.obs_node_index.long(),
            obs_station_index=station_index,
            q_future_norm=q_future_norm,
        )
        dz_mean = self._station_values("dz_target_mean", station_index, z_norm)
        dz_scale = self._station_values("dz_target_scale", station_index, z_norm)
        z_delta = z_norm * dz_scale + dz_mean

        q0_raw = batch.q_history[:, -1]
        q0_available = batch.q_mask[:, -1].bool()
        q0 = torch.where(q0_available, q0_raw, torch.zeros_like(q0_raw)).clamp_min(0)
        q_level_future = self.qz_consistency(q_obs, None, station_index)
        q_level_origin = self.qz_consistency(q0.unsqueeze(1), None, station_index)
        consistency_delta = q_level_future - q_level_origin

        diagnostics: dict[str, torch.Tensor] = {
            **runoff_diagnostics,
            **routing_diagnostics,
            "history_node_context": state["node_context"],
            "history_observation_context": state["observation_context"],
            "history_observation_available": state["observation_available"],
            "q_origin_m3s": q0,
            "q_origin_available": q0_available,
            "qz_consistency_delta_z_m": consistency_delta,
            "qz_consistency_available": q0_available,
            "network_q_m3s": q_nodes,
            "local_runoff_q_m3s": q_lateral,
        }
        return {
            "q": q_obs,
            "z": z_delta,
            "q_lat": q_lateral,
            "q_nodes": q_nodes,
            "diagnostics": diagnostics,
        }
