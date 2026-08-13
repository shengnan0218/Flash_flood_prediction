"""Hybrid rainfall-runoff, directed-routing, and observation model."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from data.schema import validate_batch
from models.observation import IndependentDeltaZHead, MonotonicQZObservation
from models.routing import KinematicWaveGNN, PureDirectedGNN
from models.runoff import PureLSTMRunoff, WaterBalanceLSTM
from models.state_initialization import HydrologicalStateInitializer


class HybridFloodModel(nn.Module):
    """Run one of the E1--E4 ablations under a shared tensor contract."""

    def __init__(self, cfg: dict[str, Any], stations: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.horizon = int(cfg["forecast_horizon"])
        self.history = int(cfg["history_length"])
        self.seconds_per_step = float(cfg["solver"]["seconds_per_step"])
        self.use_observation_masks = bool(
            cfg.get("data", {}).get("use_observation_masks", False)
        )
        self.z_target_mode = str(cfg.get("loss", {}).get("z_target_mode", "absolute"))
        state_cfg = cfg.get("state_initialization", {})
        self.use_state_initialization = bool(state_cfg.get("enabled", False))
        self.state_initialization_mode = str(
            state_cfg.get("mode", "forecast_origin")
        )
        if self.use_state_initialization and self.state_initialization_mode != "forecast_origin":
            raise ValueError("当前state_initialization仅支持mode='forecast_origin'")
        self.expected = {
            key: int(cfg[key])
            for key in (
                "history_length",
                "forecast_horizon",
                "dynamic_dim",
                "node_static_dim",
                "edge_static_dim",
            )
        }

        runoff_input_dim = int(cfg["dynamic_dim"]) + 1
        if self.use_observation_masks:
            runoff_input_dim += 3
        hidden = int(cfg["hidden_dim"])
        self.hidden_dim = hidden
        runoff_mode = cfg["runoff_mode"]
        if runoff_mode == "water_balance_lstm":
            self.runoff = WaterBalanceLSTM(runoff_input_dim, hidden)
        elif runoff_mode == "pure_lstm":
            self.runoff = PureLSTMRunoff(runoff_input_dim, hidden)
        else:
            raise ValueError(f"未知runoff_mode={runoff_mode!r}")

        routing_mode = cfg["routing_mode"]
        if routing_mode == "kinematic_wave_gnn":
            self.routing = KinematicWaveGNN(
                int(cfg["node_static_dim"]),
                int(cfg["edge_static_dim"]),
                hidden,
                cfg["physical_bounds"],
                cfg["solver"],
            )
        elif routing_mode == "pure_gnn":
            self.routing = PureDirectedGNN(
                int(cfg["node_static_dim"]), int(cfg["edge_static_dim"]), hidden
            )
        else:
            raise ValueError(f"未知routing_mode={routing_mode!r}")

        if self.use_state_initialization:
            if runoff_mode != "water_balance_lstm":
                raise ValueError(
                    "P3 state initialization当前要求runoff_mode=water_balance_lstm"
                )
            if routing_mode != "kinematic_wave_gnn":
                raise ValueError(
                    "P3 state initialization当前要求routing_mode=kinematic_wave_gnn"
                )
            self.state_initializer = HydrologicalStateInitializer(
                runoff_input_dim,
                int(cfg["node_static_dim"]),
                int(cfg["edge_static_dim"]),
                hidden,
            )
        else:
            self.state_initializer = None

        # Monotonic Q-Z is retained as a hydraulic consistency relation.  It no
        # longer generates the primary P3 delta-Z prediction.
        self.observation = MonotonicQZObservation(stations)
        if self.use_state_initialization and self.z_target_mode == "delta_from_t0":
            self.independent_z_head: nn.Module | None = IndependentDeltaZHead(
                hidden,
                hidden,
                self.horizon,
                stations,
            )
        else:
            self.independent_z_head = None

    def _temporal_inputs(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = batch.dynamic_node_features.shape[1]
        total = history + self.horizon
        if batch.rainfall.shape[1] < total:
            raise ValueError(f"rainfall需覆盖history+forecast={total}小时")
        past = batch.dynamic_node_features
        future = past[:, -1:].expand(-1, self.horizon, -1, -1)
        dynamic = torch.cat([past, future], dim=1)
        rainfall = batch.rainfall[:, :total]
        parts = [dynamic, rainfall]
        if self.use_observation_masks:
            batch_size, _, nodes, _ = dynamic.shape
            unavailable = torch.zeros(
                batch_size,
                self.horizon,
                nodes,
                1,
                dtype=dynamic.dtype,
                device=dynamic.device,
            )
            q_availability = torch.cat(
                [batch.q_mask.unsqueeze(-1).to(dynamic.dtype), unavailable], dim=1
            )
            z_availability = torch.cat(
                [batch.z_mask.unsqueeze(-1).to(dynamic.dtype), unavailable], dim=1
            )
            if batch.rainfall_mask is None:
                rain_availability = torch.ones_like(rainfall)
            else:
                rain_availability = batch.rainfall_mask[:, :total].to(dynamic.dtype)
            parts.extend([q_availability, z_availability, rain_availability])
        return torch.cat(parts, dim=-1), rainfall

    @staticmethod
    def _physical_area(batch: Any) -> torch.Tensor:
        area = getattr(batch, "node_area_km2", None)
        if area is None:
            area = batch.node_static[:, 0]
        if area.ndim != 1 or area.shape[0] != batch.node_static.shape[0]:
            raise ValueError("node_area_km2必须为[N]并与当前河网节点对应")
        if not torch.isfinite(area).all() or (area <= 0).any():
            raise ValueError("物理增量汇水面积必须为有限正数，单位km²")
        return area

    @staticmethod
    def _latest_observed_history(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the latest valid history value per batch/node and availability."""
        if values.ndim != 3 or mask.shape != values.shape:
            raise ValueError("history values/mask必须同为[B,H,N]")
        valid = mask.bool() & torch.isfinite(values)
        batch, history, nodes = values.shape
        positions = torch.arange(history, device=values.device).view(1, history, 1)
        positions = positions.expand(batch, -1, nodes)
        last = positions.masked_fill(~valid, -1).amax(dim=1)
        available = last >= 0
        gather_index = last.clamp_min(0).unsqueeze(1)
        latest = values.gather(1, gather_index).squeeze(1)
        latest = torch.where(available, latest, torch.zeros_like(latest))
        return latest, available

    @staticmethod
    def _recent_observed_trend(
        values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return last-valid minus previous-valid observation per batch/node."""
        if values.ndim != 3 or mask.shape != values.shape:
            raise ValueError("trend values/mask必须同为[B,H,N]")
        valid = mask.bool() & torch.isfinite(values)
        batch, history, nodes = values.shape
        positions = torch.arange(history, device=values.device).view(1, history, 1)
        positions = positions.expand(batch, -1, nodes)
        last = positions.masked_fill(~valid, -1).amax(dim=1)
        has_last = last >= 0
        previous_candidates = valid & (positions < last.unsqueeze(1))
        previous = positions.masked_fill(~previous_candidates, -1).amax(dim=1)
        has_pair = has_last & (previous >= 0)
        last_value = values.gather(1, last.clamp_min(0).unsqueeze(1)).squeeze(1)
        previous_value = values.gather(
            1, previous.clamp_min(0).unsqueeze(1)
        ).squeeze(1)
        trend = torch.where(
            has_pair, last_value - previous_value, torch.zeros_like(last_value)
        )
        return trend, has_pair

    @staticmethod
    def _initial_node_channel_storage(
        batch: Any,
        routing_diagnostics: dict[str, torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Map initialized edge volumes to the same node-storage convention as routing."""
        batch_size, _, nodes = reference.shape
        node_storage = torch.zeros(
            batch_size, nodes, device=reference.device, dtype=reference.dtype
        )
        edge_storage = routing_diagnostics.get("initial_edge_storage_m3")
        if edge_storage is None:
            return node_storage
        edge_storage = edge_storage.to(device=reference.device, dtype=reference.dtype)
        edges = int(batch.edge_index.shape[1])
        if edge_storage.shape != (batch_size, edges):
            raise ValueError("initial_edge_storage_m3必须为[B,E]")
        if edges:
            destination = batch.edge_index[1].long().to(reference.device)
            node_storage.index_add_(1, destination, edge_storage)
        return node_storage

    def _independent_forecast_origin_z(
        self,
        batch: Any,
        q_future: torch.Tensor,
        channel_future: torch.Tensor | None,
        routing_diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict delta-Z independently and expose monotone Q-Z consistency target."""
        if self.independent_z_head is None:
            raise RuntimeError("P3 delta-Z独立预测头未构建")
        q_origin, q_origin_available = self._latest_observed_history(
            batch.q_history, batch.q_mask
        )
        initial_channel = self._initial_node_channel_storage(
            batch, routing_diagnostics, q_future
        )

        # This monotone relation is now only a weak physical consistency target.
        future_level_response = self.observation(
            q_future,
            channel_future,
            getattr(batch, "station_index", None),
        )
        origin_level_response = self.observation(
            q_origin.unsqueeze(1),
            initial_channel.unsqueeze(1),
            getattr(batch, "station_index", None),
        )
        consistency_delta = future_level_response - origin_level_response

        z_reference = getattr(batch, "z_reference", None)
        z_reference_mask = getattr(batch, "z_reference_mask", None)
        if z_reference is None or z_reference_mask is None:
            raise RuntimeError("P3 independent delta-Z要求forecast-origin z_reference")
        if z_reference.ndim != 2 or z_reference.shape != q_origin.shape:
            raise ValueError("z_reference必须为[B,N]并与forecast-origin Q一致")
        if z_reference_mask.shape != z_reference.shape:
            raise ValueError("z_reference_mask必须与z_reference同形状")
        reference_valid = z_reference_mask.bool() & torch.isfinite(z_reference)
        reference = torch.where(reference_valid, z_reference, torch.zeros_like(z_reference))
        recent_trend, recent_trend_valid = self._recent_observed_trend(
            batch.z_history, batch.z_mask
        )
        history_context = routing_diagnostics.get("history_context")
        if history_context is None:
            raise RuntimeError("P3 independent delta-Z缺少stage history_context")

        z_delta = self.independent_z_head(
            history_context,
            q_future,
            q_origin,
            q_origin_available,
            channel_future,
            initial_channel,
            reference,
            reference_valid,
            recent_trend,
            recent_trend_valid,
            getattr(batch, "station_index", None),
        )
        absolute_future = reference.unsqueeze(1) + z_delta
        diagnostics = {
            "forecast_origin_q_m3s": q_origin,
            "forecast_origin_q_available": q_origin_available,
            "initial_node_channel_storage_m3": initial_channel,
            "forecast_origin_level_response": origin_level_response.squeeze(1),
            "forecast_origin_z_m": reference,
            "forecast_origin_z_available": reference_valid,
            "recent_z_trend_m_per_step": recent_trend,
            "recent_z_trend_available": recent_trend_valid,
            "independent_delta_z_m": z_delta,
            "qz_consistency_delta_z_m": consistency_delta,
            "absolute_z_forecast_m": absolute_future,
        }
        return z_delta, diagnostics

    def _state_initialized_forward(
        self,
        batch: Any,
        features: torch.Tensor,
        rainfall: torch.Tensor,
        area_km2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.state_initializer is None:
            raise RuntimeError("state_initializer未构建")
        history_features = features[:, : self.history]
        future_features = features[:, self.history : self.history + self.horizon]
        future_rainfall = rainfall[:, self.history : self.history + self.horizon]
        state = self.state_initializer(
            history_features,
            batch.node_static,
            batch.edge_index,
            batch.edge_static,
        )
        q_lateral, runoff_diagnostics = self.runoff(
            future_features,
            future_rainfall,
            area_km2,
            seconds=self.seconds_per_step,
            initial_state=(
                state["h0"],
                state["c0"],
                state["storage_fast_mm"],
                state["storage_slow_mm"],
            ),
        )
        q_all, routing_diagnostics = self.routing(
            q_lateral,
            batch.node_static,
            batch.edge_index,
            batch.edge_static,
            initial_edge_discharge=state["edge_discharge_m3s"],
        )
        diagnostics: dict[str, torch.Tensor] = {
            **runoff_diagnostics,
            **routing_diagnostics,
            "initial_storage_fast_mm": state["storage_fast_mm"],
            "initial_storage_slow_mm": state["storage_slow_mm"],
            "initial_runoff_hidden_h": state["h0"],
            "initial_runoff_hidden_c": state["c0"],
            "history_context": state["history_context"],
        }
        return q_lateral, q_all, diagnostics

    def forward(
        self, batch: Any
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        validate_batch(batch, self.expected)
        features, rainfall = self._temporal_inputs(batch)
        area_km2 = self._physical_area(batch)

        if self.use_state_initialization:
            q_lateral, q_all, diagnostics = self._state_initialized_forward(
                batch, features, rainfall, area_km2
            )
            channel_all = diagnostics.get("node_channel_storage")
            if self.z_target_mode == "delta_from_t0":
                z_all, z_diagnostics = self._independent_forecast_origin_z(
                    batch, q_all, channel_all, diagnostics
                )
                diagnostics.update(z_diagnostics)
            else:
                z_all = self.observation(
                    q_all,
                    channel_all,
                    getattr(batch, "station_index", None),
                )
            return {
                "q": q_all,
                "z": z_all,
                "q_lat": q_lateral,
                "diagnostics": diagnostics,
            }

        q_lateral, runoff_diagnostics = self.runoff(
            features,
            rainfall,
            area_km2,
            seconds=self.seconds_per_step,
        )
        q_all, routing_diagnostics = self.routing(
            q_lateral,
            batch.node_static,
            batch.edge_index,
            batch.edge_static,
        )
        channel_all = routing_diagnostics.get("node_channel_storage")
        z_all = self.observation(
            q_all,
            channel_all,
            getattr(batch, "station_index", None),
        )
        return {
            "q": q_all[:, -self.horizon :],
            "z": z_all[:, -self.horizon :],
            "q_lat": q_lateral,
            "diagnostics": {**runoff_diagnostics, **routing_diagnostics},
        }


def set_finetune_strategy(model: HybridFloodModel, strategy: str) -> None:
    """Select the explicit Zhejiang transfer-learning parameter subset."""
    supported = {
        "observation_only",
        "observation_and_edge_parameters",
        "full_finetune",
    }
    if strategy not in supported:
        raise ValueError(f"未知finetune strategy={strategy!r}，支持{sorted(supported)}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.observation.parameters():
        parameter.requires_grad = True
    if model.independent_z_head is not None:
        for parameter in model.independent_z_head.parameters():
            parameter.requires_grad = True
    if strategy in {"observation_and_edge_parameters", "full_finetune"} and hasattr(
        model.routing, "edge_net"
    ):
        for parameter in model.routing.edge_net.parameters():
            parameter.requires_grad = True
    if strategy == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
