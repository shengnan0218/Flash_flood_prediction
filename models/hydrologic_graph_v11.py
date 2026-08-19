"""Formal V11 model: 72 h rainfall warm-up, 24 h Q/Z assimilation, Q-only forecast.

The trainable architecture is deliberately unchanged from V10. V11 changes the
physical state exposure: runoff/routing warm-up receives 72 h of rainfall while
the sparse observation encoder still receives only the most recent 24 h of Q/Z.
This prevents longer antecedent rainfall from becoming a longer station-history
shortcut.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from data.v8_schema import HydrologicGraphBatch
from datasets.hydrologic_graph_v11 import (
    OBS_HISTORY_HOURS,
    RAIN_HISTORY_HOURS,
    validate_v11_batch,
)
from models.hydrologic_graph_v10 import HydrologicGraphV10Model


class HydrologicGraphV11Model(HydrologicGraphV10Model):
    """V10 physical core with 72 h rainfall state warm-up and 24 h Q/Z analysis."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        if str(cfg.get("model_version", "")).lower() != "v11":
            raise ValueError("HydrologicGraphV11Model只接受model_version=v11")
        # V10's constructor is the preserved architecture implementation. Give
        # it a private 24 h compatibility view because its constructor contract
        # predates the split rain-history/observation-history semantics. No
        # formal V11 config is mutated or checkpointed as V10.
        base_cfg = deepcopy(cfg)
        base_cfg["model_version"] = "v10"
        base_cfg["history_length"] = OBS_HISTORY_HOURS
        base_cfg.setdefault("temporal", {})["history_duration_seconds"] = (
            OBS_HISTORY_HOURS * int(base_cfg["temporal"]["forcing_step_seconds"])
        )
        super().__init__(base_cfg)
        self.cfg = cfg
        self.rain_history_hours = RAIN_HISTORY_HOURS
        self.observation_history_hours = OBS_HISTORY_HOURS

    @staticmethod
    def _require_physical_q(q: torch.Tensor, label: str) -> None:
        if not torch.isfinite(q).all():
            raise FloatingPointError(f"v11 {label}含NaN/Inf")
        if (q < 0).any():
            raise ValueError(f"v11 {label}含负流量；不得静默截断")

    def forward(self, batch: HydrologicGraphBatch) -> dict[str, Any]:
        validate_v11_batch(batch)
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
        else:
            edge_storage_t0_raw = None

        # Q/Z remain a 24 h assimilation-only context even though rainfall warm-up
        # is 72 h. This is the central anti-shortcut design constraint of V11.
        if batch.q_history.shape[1] != self.observation_history_hours:
            raise ValueError("v11 observation history必须严格保持24 h")
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
        else:
            edge_storage_t0 = None
            edge_storage_log_factor = q_nodes_history.new_zeros(
                (batch_size, int(batch.edge_index.shape[1]))
            )
            edge_storage_additive_m3 = edge_storage_log_factor.clone()

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
            q0_nodes_corrected = q_nodes_history[:, -1]

        target_indices = self.target_indices.to(q_nodes_internal.device)
        q_nodes = q_nodes_internal.index_select(1, target_indices)
        q_obs = q_nodes.index_select(2, obs_node_index)
        q0_model_corrected = q0_nodes_corrected.index_select(1, obs_node_index)
        q0_analysis = torch.where(
            q0_observed_available,
            batch.q_history[:, -1],
            q0_model_corrected,
        )
        self._require_physical_q(q_obs, "forecast Q")
        self._require_physical_q(q0_analysis, "final-history-bin Q0_analysis")

        # Stage is unchanged from V10: detached Q -> TRAIN-only rating -> Z0 anchor.
        q_stage = q_obs.detach()
        q0_stage = q0_analysis.detach()
        slope, intercept, rating_available_station = self.rating.station_parameters(
            station_index, q_stage
        )
        slope_3d = slope.view(1, 1, -1)
        intercept_3d = intercept.view(1, 1, -1)
        z_rating_raw_abs = slope_3d * q_stage + intercept_3d
        z_rating_origin_abs = (
            slope.view(1, -1) * q0_stage + intercept.view(1, -1)
        )
        z0_observed_available = batch.z_mask[:, -1].bool()
        stage_origin_available = (
            z0_observed_available & rating_available_station.view(1, -1)
        )
        stage_available_mask = stage_origin_available.unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        z_delta_candidate = slope_3d * (q_stage - q0_stage.unsqueeze(1))
        z_abs_candidate = batch.z_history[:, -1].unsqueeze(1) + z_delta_candidate
        z_delta = torch.where(
            stage_available_mask, z_delta_candidate, torch.zeros_like(z_delta_candidate)
        )
        z_abs = torch.where(
            stage_available_mask, z_abs_candidate, torch.zeros_like(z_abs_candidate)
        )
        raw_available_mask = rating_available_station.view(1, 1, -1).expand(
            batch_size, self.horizon, -1
        )
        z_rating_raw_abs = torch.where(
            raw_available_mask,
            z_rating_raw_abs,
            torch.zeros_like(z_rating_raw_abs),
        )
        origin_residual = torch.where(
            stage_origin_available,
            batch.z_history[:, -1] - z_rating_origin_abs,
            torch.zeros_like(z_rating_origin_abs),
        )

        diagnostics: dict[str, torch.Tensor] = {
            "history_observation_context": observation_context,
            "history_observation_available": observation_available,
            "warmup_network_q_m3s": q_nodes_history,
            "network_q_m3s": q_nodes_internal,
            "local_runoff_q_m3s": q_lat_future,
            "q_origin_warmup_m3s": q0_warmup_obs,
            "q_origin_model_corrected_m3s": q0_model_corrected,
            "q_origin_analysis_m3s": q0_analysis,
            "q_origin_observed_available": q0_observed_available,
            "stage_z0_observed_available": z0_observed_available,
            "stage_rating_available": rating_available_station,
            "stage_rating_slope_m_per_m3s": slope,
            "stage_rating_intercept_m": intercept,
            "stage_rating_origin_abs_m": z_rating_origin_abs,
            "stage_origin_residual_m": origin_residual,
            "state_correction_node_available": node_obs_available,
            "state_correction_q0_residual_norm": node_q0_residual_norm,
            "state_correction_q0_residual_m3s": node_q0_residual_m3s,
            "state_correction_edge_log_factor": edge_storage_log_factor,
            "state_correction_edge_additive_storage_m3": edge_storage_additive_m3,
            "rain_history_hours": q_obs.new_tensor(float(self.rain_history_hours)),
            "observation_history_hours": q_obs.new_tensor(
                float(self.observation_history_hours)
            ),
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

        return {
            "q": q_obs,
            "z": z_delta,
            "z_delta": z_delta,
            "z_abs": z_abs,
            "z_available_mask": stage_available_mask,
            "z_rating_raw_abs": z_rating_raw_abs,
            "z_rating_raw_available_mask": raw_available_mask,
            "q0_analysis": q0_analysis,
            "q_lat": q_lat_future,
            "q_nodes": q_nodes,
            "diagnostics": diagnostics,
        }
