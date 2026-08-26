"""Single production hydrologic model with four controlled physics ablations.

The architecture has one unambiguous flow:

    rainfall -> runoff LSTM -> directed river routing -> small residual MLP -> Q

The two configuration switches only select whether runoff and routing use their
physical forms.  No observation encoder, hidden-state correction, upstream
residual propagation, or channel-storage correction exists in this model.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from data.hydrologic_schema import HydrologicGraphBatch
from datasets.hydrologic_graph import validate_hydrologic_batch
from models.routing import KinematicWaveGNN, PureDirectedGNN
from models.runoff.water_balance_continuous import ContinuousTimeWaterBalanceLSTM


def _buffer(values: Any) -> torch.Tensor:
    value = torch.as_tensor(values, dtype=torch.float32)
    if not torch.isfinite(value).all():
        raise ValueError("normalization contains NaN/Inf")
    return value


class PureRunoffLSTM(nn.Module):
    """Unconstrained runoff ablation with the same inputs and hidden width."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, steps, nodes, width = features.shape
        sequence = features.permute(0, 2, 1, 3).reshape(batch * nodes, steps, width)
        encoded, _ = self.lstm(sequence)
        q = self.head(encoded).reshape(batch, nodes, steps).permute(0, 2, 1)
        return q, {"runoff_water_balance_residual": torch.full_like(q, float("nan"))}


class FixedStationRating(nn.Module):
    """Non-trainable TRAIN-only linear rating curves used only for reporting Z."""

    def __init__(self, statistics: Mapping[str, Any], station_ids: tuple[str, ...]) -> None:
        super().__init__()
        records = statistics.get("stations", {})
        slope, intercept, available = [], [], []
        for station in station_ids:
            record = records.get(station, {})
            ok = bool(record.get("available", False))
            slope.append(float(record.get("slope_m_per_m3s", 0.0)) if ok else 0.0)
            intercept.append(float(record.get("intercept_m", 0.0)) if ok else 0.0)
            available.append(ok)
        self.register_buffer("slope", torch.tensor(slope, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(intercept, dtype=torch.float32))
        self.register_buffer("available", torch.tensor(available, dtype=torch.bool))

    def select(self, station_index: torch.Tensor, reference: torch.Tensor):
        index = station_index.to(self.slope.device)
        return (
            self.slope[index].to(reference),
            self.intercept[index].to(reference),
            self.available[index].to(reference.device),
        )


class ResidualOutputMLP(nn.Module):
    """Small bounded correction around Q0 + physically routed Delta-Q."""

    def __init__(self, static_dim: int, hidden_dim: int, max_scale_fraction: float) -> None:
        super().__init__()
        self.max_scale_fraction = float(max_scale_fraction)
        self.network = nn.Sequential(
            nn.Linear(static_dim + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        routed_q: torch.Tensor,
        routed_q0: torch.Tensor,
        observed_q0: torch.Tensor,
        q0_available: torch.Tensor,
        q_mean: torch.Tensor,
        q_scale: torch.Tensor,
        outlet_static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, horizon, stations = routed_q.shape
        scale = q_scale.view(1, 1, stations).clamp_min(1.0e-6)
        mean = q_mean.view(1, 1, stations)
        route_delta = routed_q - routed_q0.unsqueeze(1)
        q0 = observed_q0.unsqueeze(1)
        available = q0_available.unsqueeze(1)
        lead = torch.linspace(
            1.0 / horizon, 1.0, horizon,
            device=routed_q.device, dtype=routed_q.dtype,
        ).view(1, horizon, 1).expand(batch, -1, stations)
        static = outlet_static.view(1, 1, stations, -1).expand(batch, horizon, -1, -1)
        features = torch.cat(
            [
                ((routed_q - mean) / scale).unsqueeze(-1),
                (route_delta / scale).unsqueeze(-1),
                ((q0 - mean) / scale).expand(-1, horizon, -1).unsqueeze(-1),
                available.expand(-1, horizon, -1).to(routed_q.dtype).unsqueeze(-1),
                lead.unsqueeze(-1),
                static,
            ],
            dim=-1,
        )
        correction = (
            torch.tanh(self.network(features).squeeze(-1))
            * scale
            * self.max_scale_fraction
        )
        anchored_base = observed_q0.unsqueeze(1) + route_delta
        base = torch.where(available, anchored_base, routed_q)
        prediction = torch.relu(base + correction)
        return prediction, base, correction


class HydrologicModel(nn.Module):
    """Shared model for E1--E4; only the two physics switches may differ."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = int(cfg["forecast_horizon"])
        self.node_static_dim = int(cfg["node_static_dim"])
        self.edge_static_dim = int(cfg["edge_static_dim"])
        self.hidden_dim = int(cfg["hidden_dim"])
        runtime = cfg.get("_runtime", {})
        normal = runtime.get("normalization")
        station_ids = tuple(runtime.get("station_ids", ()))
        if not isinstance(normal, Mapping) or not station_ids:
            raise ValueError("model runtime normalization/station catalogue missing")

        self.register_buffer("rain_mean", _buffer(normal["rain_mean"]).reshape(()))
        self.register_buffer("rain_scale", _buffer(normal["rain_scale"]).reshape(()))
        self.register_buffer("node_mean", _buffer(normal["node_static_mean"]).reshape(1, -1))
        self.register_buffer("node_scale", _buffer(normal["node_static_scale"]).reshape(1, -1))
        self.register_buffer("q_mean", _buffer(normal["q_target_mean"]).reshape(-1))
        self.register_buffer("q_scale", _buffer(normal["q_target_scale"]).reshape(-1))

        runoff_input = self.node_static_dim
        self.runoff_mode = str(cfg["runoff_mode"])
        if self.runoff_mode == "water_balance_lstm":
            self.runoff = ContinuousTimeWaterBalanceLSTM(runoff_input, self.hidden_dim)
        elif self.runoff_mode == "pure_lstm":
            self.runoff = PureRunoffLSTM(runoff_input + 1, self.hidden_dim)
        else:
            raise ValueError(f"unknown runoff_mode={self.runoff_mode!r}")

        self.routing_mode = str(cfg["routing_mode"])
        if self.routing_mode == "kinematic_wave_gnn":
            self.routing = KinematicWaveGNN(
                self.node_static_dim, self.edge_static_dim, self.hidden_dim,
                cfg["physical_bounds"], cfg["solver"],
            )
        elif self.routing_mode == "pure_gnn":
            self.routing = PureDirectedGNN(
                self.node_static_dim, self.edge_static_dim, self.hidden_dim
            )
        else:
            raise ValueError(f"unknown routing_mode={self.routing_mode!r}")

        output_cfg = cfg["output_head"]
        self.output_head = ResidualOutputMLP(
            self.node_static_dim,
            int(output_cfg["hidden_dim"]),
            float(output_cfg["max_correction_scale_fraction"]),
        )
        self.rating = FixedStationRating(runtime["rating_curves"], station_ids)

    def _runoff(self, rain: torch.Tensor, rain_norm: torch.Tensor, static_norm: torch.Tensor, area: torch.Tensor):
        batch, steps, nodes, _ = rain.shape
        static = static_norm.view(1, 1, nodes, -1).expand(batch, steps, -1, -1)
        if self.runoff_mode == "water_balance_lstm":
            return self.runoff(
                static, rain, area,
                seconds=float(self.cfg["solver"]["seconds_per_step"]),
            )
        return self.runoff(torch.cat([rain_norm, static], dim=-1))

    def forward(self, batch: HydrologicGraphBatch) -> dict[str, Any]:
        validate_hydrologic_batch(batch)
        history = batch.history_rain
        future = batch.future_rain
        rain = torch.cat([history, future], dim=1)
        rain_norm = (rain - self.rain_mean.to(rain)) / self.rain_scale.to(rain)
        static_norm = (batch.node_static - self.node_mean.to(batch.node_static)) / self.node_scale.to(batch.node_static)
        q_lat_all, runoff_diag = self._runoff(
            rain, rain_norm, static_norm, batch.incremental_area_km2
        )

        q_nodes_all, routing_diag = self.routing(
            q_lat_all, batch.node_static, batch.edge_index, batch.edge_static
        )
        origin_index = history.shape[1] - 1
        q_nodes = q_nodes_all[:, -self.horizon:]
        obs_node = batch.obs_node_index.long()
        routed_q = q_nodes.index_select(2, obs_node)
        routed_q0 = q_nodes_all[:, origin_index].index_select(1, obs_node)

        station_index = batch.obs_station_index.long()
        q_mean = self.q_mean[station_index].to(routed_q)
        q_scale = self.q_scale[station_index].to(routed_q)
        q0_available = batch.q_mask[:, -1].bool()
        observed_q0 = torch.where(q0_available, batch.q_history[:, -1], routed_q0)
        outlet_static = static_norm.index_select(0, obs_node)
        q, q_base, q_correction = self.output_head(
            routed_q, routed_q0, observed_q0, q0_available,
            q_mean, q_scale, outlet_static,
        )

        slope, intercept, rating_available = self.rating.select(station_index, q)
        z0_available = batch.z_mask[:, -1].bool()
        z_available = z0_available & rating_available.view(1, -1)
        z_delta_candidate = slope.view(1, 1, -1) * (q - observed_q0.unsqueeze(1))
        z_delta = torch.where(z_available.unsqueeze(1), z_delta_candidate, torch.zeros_like(z_delta_candidate))
        z_abs = torch.where(
            z_available.unsqueeze(1),
            batch.z_history[:, -1].unsqueeze(1) + z_delta,
            torch.zeros_like(z_delta),
        )
        z_raw = slope.view(1, 1, -1) * q + intercept.view(1, 1, -1)

        return {
            "q": q,
            "z": z_delta,
            "z_delta": z_delta,
            "z_abs": z_abs,
            "z_available_mask": z_available.unsqueeze(1).expand_as(z_delta),
            "z_rating_raw_abs": z_raw,
            "z_rating_raw_available_mask": rating_available.view(1, 1, -1).expand_as(z_raw),
            "q0_analysis": observed_q0,
            "q_lat": q_lat_all[:, -self.horizon:],
            "q_nodes": q_nodes,
            "diagnostics": {
                **{f"runoff_{k}": v for k, v in runoff_diag.items()},
                **{f"routing_{k}": v for k, v in routing_diag.items()},
                "routed_q_m3s": routed_q,
                "routed_q0_m3s": routed_q0,
                "q_residual_base_m3s": q_base,
                "q_output_correction_m3s": q_correction,
                "q_origin_observed_available": q0_available,
            },
        }
