"""Formal v10: Q-only hydrologic forecast with fixed rating-derived stage.

V10 preserves the active v9 runoff, optimized routing, 24 h warm-up and
mass-aware forecast-origin Q/Z state assimilation.  The independent neural Z
head is removed completely.  Stage is a non-trainable station transformation:

    raw Z(t+h) = f_s(Q_hat(t+h))
    corrected Delta-Z(t+h) = f_s(Q_hat(t+h)) - f_s(Q0_analysis)
    corrected Z(t+h) = Z0_obs + corrected Delta-Z(t+h)

Q0_analysis honours observed Q0 exactly when available and otherwise uses the
v9 corrected physical/model origin discharge.  Corrected stage is available
only when the station has a TRAIN-only rating curve and exact forecast-origin
Z0 is observed.  No future Z observation is used.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from data.v8_schema import HydrologicGraphBatch
from models.hydrologic_graph_v9 import _validate_v9_batch
from models.hydrologic_graph_v9_assimilated import HydrologicGraphV9Model


class FixedStationLinearRatingV10(nn.Module):
    """Non-trainable TRAIN-only station linear Q-Z relationships."""

    def __init__(self, statistics: Mapping[str, Any], station_ids: tuple[str, ...]) -> None:
        super().__init__()
        station_stats = statistics.get("stations")
        if not isinstance(station_stats, Mapping):
            raise ValueError("v10 rating statistics缺少stations")
        count = len(station_ids)
        if count <= 0:
            raise ValueError("v10 rating station catalogue为空")
        slope = torch.zeros(count, dtype=torch.float32)
        intercept = torch.zeros(count, dtype=torch.float32)
        available = torch.zeros(count, dtype=torch.bool)
        pair_count = torch.zeros(count, dtype=torch.int64)
        q_min = torch.zeros(count, dtype=torch.float32)
        q_max = torch.zeros(count, dtype=torch.float32)
        for index, station in enumerate(station_ids):
            values = station_stats.get(station, {})
            if not isinstance(values, Mapping):
                raise ValueError(f"v10 rating station={station}统计非法")
            pair_count[index] = int(values.get("unique_train_pair_count", 0))
            if not bool(values.get("available", False)):
                continue
            a = float(values["slope_m_per_m3s"])
            b = float(values["intercept_m"])
            lower = float(values["q_min_m3s"])
            upper = float(values["q_max_m3s"])
            values_tensor = torch.tensor((a, b, lower, upper), dtype=torch.float64)
            if not torch.isfinite(values_tensor).all():
                raise ValueError(f"v10 rating station={station}参数含NaN/Inf")
            if a <= 0 or upper < lower:
                raise ValueError(f"v10 rating station={station}参数不满足物理/范围约束")
            slope[index] = a
            intercept[index] = b
            q_min[index] = lower
            q_max[index] = upper
            available[index] = True
        self.register_buffer("slope", slope)
        self.register_buffer("intercept", intercept)
        self.register_buffer("available", available)
        self.register_buffer("pair_count", pair_count)
        self.register_buffer("q_min_m3s", q_min)
        self.register_buffer("q_max_m3s", q_max)

    def station_parameters(
        self, station_index: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if station_index.dtype != torch.long or station_index.ndim != 1:
            raise ValueError("v10 rating station_index必须为[Nobs] LongTensor")
        if station_index.numel() and (
            station_index.min() < 0 or station_index.max() >= self.slope.numel()
        ):
            raise ValueError("v10 rating station_index越界")
        index = station_index.to(self.slope.device)
        slope = self.slope[index].to(device=reference.device, dtype=reference.dtype)
        intercept = self.intercept[index].to(device=reference.device, dtype=reference.dtype)
        available = self.available[index].to(device=reference.device)
        return slope, intercept, available


class HydrologicGraphV10Model(HydrologicGraphV9Model):
    """V9 physical/assimilation core with Q as the only learned forecast task."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        if str(cfg.get("model_version", "")).lower() != "v10":
            raise ValueError("HydrologicGraphV10Model只接受model_version=v10")

        # The preserved v9 base constructor historically creates its neural Z
        # head unconditionally.  Feed it a private constructor-only compatibility
        # view, never mutate/persist the formal v10 config, then remove every
        # Z-only module/buffer immediately.  This keeps v9 byte-for-byte intact
        # while ensuring the v10 state_dict/optimizer contain no neural Z path.
        base_cfg = dict(cfg)
        base_cfg["z_head"] = {"trend_windows_seconds": [3600, 10800, 21600]}
        super().__init__(base_cfg)
        self.cfg = cfg
        del self.z_head
        del self.node_context_projection
        del self.dz_target_scale
        delattr(self, "trend_windows_seconds")

        runtime = cfg.get("_runtime", {})
        station_ids = tuple(str(value) for value in runtime.get("v8_station_ids", ()))
        ratings = runtime.get("v10_rating_curves")
        if not station_ids or not isinstance(ratings, Mapping):
            raise ValueError("v10缺少全局station catalogue或TRAIN-only rating curves")
        self.rating = FixedStationLinearRatingV10(ratings, station_ids)
        stage = cfg.get("stage_output", {})
        if stage.get("method") != "train_only_station_linear_rating":
            raise ValueError("v10正式stage method必须为train_only_station_linear_rating")
        if stage.get("q0_source") != "observed_if_available_else_assimilated_model":
            raise ValueError("v10 q0_source与正式设计不一致")
        if stage.get("z0_source") != "exact_forecast_origin_observation_only":
            raise ValueError("v10 z0_source与正式设计不一致")
        if bool(stage.get("allow_backward_z_search", False)):
            raise ValueError("v10禁止向前搜索历史Z替代forecast-origin Z0")

    @staticmethod
    def _require_physical_q(q: torch.Tensor, label: str) -> None:
        if not torch.isfinite(q).all():
            raise FloatingPointError(f"v10 {label}含NaN/Inf")
        if (q < 0).any():
            raise ValueError(f"v10 {label}含负流量；不得在rating转换时静默截断")

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
            edge_storage_t0_raw: torch.Tensor | None = routing_history_diag["edge_storage"]
        else:
            edge_storage_t0_raw = None

        # Historical Q and Z remain assimilation inputs only.  They are never
        # supervised through a future Z loss in v10.
        q_hist_mean = self._station_values("q_history_mean", station_index, batch.q_history)
        q_hist_scale = self._station_values("q_history_scale", station_index, batch.q_history)
        z_hist_mean = self._station_values("z_history_mean", station_index, batch.z_history)
        z_hist_scale = self._station_values("z_history_scale", station_index, batch.z_history)
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
        # Missing observed Q0 has already selected the nonnegative physical/model
        # origin.  A present but negative observed Q0 is a data-contract error.
        self._require_physical_q(q0_analysis, "forecast-origin Q0_analysis")

        # Non-trainable station rating and exact Z0 residual anchoring.
        slope, intercept, rating_available_station = self.rating.station_parameters(
            station_index, q_obs
        )
        slope_3d = slope.view(1, 1, -1)
        intercept_3d = intercept.view(1, 1, -1)
        z_rating_raw_abs = slope_3d * q_obs + intercept_3d
        z_rating_origin_abs = (
            slope.view(1, -1) * q0_analysis + intercept.view(1, -1)
        )
        z0_observed_available = batch.z_mask[:, -1].bool()
        stage_origin_available = (
            z0_observed_available
            & rating_available_station.view(1, -1)
        )
        stage_available_mask = stage_origin_available.unsqueeze(1).expand(
            -1, self.horizon, -1
        )
        # For linear f_s(Q)=aQ+b the intercept cancels exactly.  Keep the
        # difference form explicit in semantics/diagnostics, but calculate the
        # algebraically identical slope*(Q-Q0) for numerical economy.
        z_delta_candidate = slope_3d * (q_obs - q0_analysis.unsqueeze(1))
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
        if isinstance(warmup_cfl, torch.Tensor) and isinstance(forecast_cfl, torch.Tensor):
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
            # ``z`` is retained only as a compatibility alias for derived Delta-Z;
            # it has no neural head and never enters the v10 training objective.
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
