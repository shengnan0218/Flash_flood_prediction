"""Hybrid rainfall-runoff, directed-routing, and observation model."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from data.schema import validate_batch
from models.observation import MonotonicQZObservation
from models.routing import KinematicWaveGNN, PureDirectedGNN
from models.runoff import PureLSTMRunoff, WaterBalanceLSTM


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

        # Rainfall is always an explicit forcing.  In formal mode the three
        # availability masks are additional inputs, rather than treating an
        # imputed zero as indistinguishable from an observed zero.
        runoff_input_dim = int(cfg["dynamic_dim"]) + 1
        if self.use_observation_masks:
            runoff_input_dim += 3
        hidden = int(cfg["hidden_dim"])
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
        self.observation = MonotonicQZObservation(stations)

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
            # Backwards compatibility for the synthetic debug fixture only.
            area = batch.node_static[:, 0]
        if area.ndim != 1 or area.shape[0] != batch.node_static.shape[0]:
            raise ValueError("node_area_km2必须为[N]并与当前河网节点对应")
        if not torch.isfinite(area).all() or (area <= 0).any():
            raise ValueError("物理增量汇水面积必须为有限正数，单位km²")
        return area

    def forward(
        self, batch: Any
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        validate_batch(batch, self.expected)
        features, rainfall = self._temporal_inputs(batch)
        area_km2 = self._physical_area(batch)
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
    if strategy in {"observation_and_edge_parameters", "full_finetune"} and hasattr(
        model.routing, "edge_net"
    ):
        for parameter in model.routing.edge_net.parameters():
            parameter.requires_grad = True
    if strategy == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
