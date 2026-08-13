"""Sampling and P3 normalization adapters for continuous-format Hunan data.

The frozen Step16/event-domain tensor/storage contract is never rewritten here.
This adapter only changes runtime sampling and, when explicitly requested by the
P3 rating-aligned mode, how already-loaded physical history observations are
presented to the model.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .hunan import HunanContinuousDataset as _BaseHunanContinuousDataset


class HunanContinuousDataset(_BaseHunanContinuousDataset):
    """Continuous-format dataset with split-safe P3 runtime adapters."""

    _NORMALIZATION_MODES = {"global", "train_aligned"}

    def __init__(
        self,
        *args: Any,
        dynamic_normalization_mode: str = "global",
        aligned_input_statistics: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        mode = str(dynamic_normalization_mode).strip().lower()
        if mode not in self._NORMALIZATION_MODES:
            raise ValueError(
                "dynamic_normalization_mode只能是global/train_aligned，"
                f"实际={dynamic_normalization_mode!r}"
            )
        self.dynamic_normalization_mode = mode
        self._aligned_input_statistics: dict[str, Any] | None = (
            dict(aligned_input_statistics)
            if aligned_input_statistics is not None
            else None
        )
        self._rating_curve_statistics_cache: dict[str, Any] | None = None
        super().__init__(*args, **kwargs)
        if self.dynamic_normalization_mode == "train_aligned" and not self.normalize_dynamic:
            raise ValueError("train_aligned输入归一化要求normalize_dynamic=true")

    @property
    def train_sampling_mode(self) -> str:
        domain = str(self._continuous_schema.get("sampling_domain", "")).strip()
        return "event_full_pass" if domain == "hydrologic_events_v1" else "response_weighted"

    def _active_station_ids(self) -> set[str]:
        return {
            node.station_id
            for graph_id in self.graph_ids
            for node in self._graphs[graph_id].nodes
        }

    def hydrologic_sampling_weights(
        self,
        *,
        q_scales: Mapping[str, float],
        delta_z_scales: Mapping[str, float],
        response_strength: float,
        response_cap: float,
        minimum_weight: float,
        maximum_weight: float,
    ) -> torch.Tensor:
        """Retain weighted sampling only for historical full-record continuous data."""
        if self.train_sampling_mode == "event_full_pass":
            raise RuntimeError(
                "hydrologic_events_v1禁止weighted/replacement sampling；"
                "TRAIN必须每epoch完整遍历一次event-domain sample_index，仅shuffle顺序"
            )
        return super().hydrologic_sampling_weights(
            q_scales=q_scales,
            delta_z_scales=delta_z_scales,
            response_strength=response_strength,
            response_cap=response_cap,
            minimum_weight=minimum_weight,
            maximum_weight=maximum_weight,
        )

    def train_rating_curve_statistics(self) -> dict[str, Any]:
        """Fit one TRAIN-only linear Q->Z relation per supervised outlet."""
        if self.split != "TRAIN":
            raise ValueError("rating curve只能从TRAIN拟合")
        if self._rating_curve_statistics_cache is not None:
            return self._rating_curve_statistics_cache
        horizons = torch.arange(1, self.forecast_hours + 1, dtype=torch.long)
        by_station: dict[str, dict[str, Any]] = {}
        samples_by_graph: dict[str, list[Any]] = {}
        for sample in self._samples:
            samples_by_graph.setdefault(sample.graph_id, []).append(sample)
        for graph_id, samples in samples_by_graph.items():
            graph = self._graphs[graph_id]
            outlet = next(node.node_index for node in graph.nodes if node.is_outlet)
            station_id = graph.outlet_id
            dynamic = self._dynamic[graph_id]
            used = torch.zeros(len(dynamic.timestamps), dtype=torch.bool)
            for start in range(0, len(samples), 100_000):
                chunk = samples[start : start + 100_000]
                origins = torch.tensor(
                    [self._origin_index(sample) for sample in chunk], dtype=torch.long
                )
                future = origins.unsqueeze(1) + horizons.unsqueeze(0)
                simultaneous = (
                    dynamic.flow_mask[future, outlet]
                    & dynamic.water_level_mask[future, outlet]
                )
                used[future[simultaneous]] = True
            valid_indices = used.nonzero(as_tuple=False).flatten()
            q = dynamic.flow[valid_indices, outlet].to(torch.float64)
            z = dynamic.water_level[valid_indices, outlet].to(torch.float64)
            count = int(q.numel())
            if count < 2:
                by_station[station_id] = {
                    "graph_id": graph_id,
                    "status": "INSUFFICIENT_PAIRED_TRAIN_POINTS",
                    "paired_unique_point_count": count,
                    "usable_linear": False,
                }
                continue
            q_mean = q.mean()
            z_mean = z.mean()
            q_anomaly = q - q_mean
            denominator = q_anomaly.square().sum()
            if float(denominator.item()) <= 0.0:
                slope = float("nan")
                intercept = float("nan")
                prediction = torch.full_like(z, z_mean)
            else:
                slope_tensor = (q_anomaly * (z - z_mean)).sum() / denominator
                slope = float(slope_tensor.item())
                intercept = float((z_mean - slope_tensor * q_mean).item())
                prediction = slope_tensor * q + (z_mean - slope_tensor * q_mean)
            residual = prediction - z
            sse = float(residual.square().sum().item())
            z_sst = float((z - z_mean).square().sum().item())
            fit_nse = 1.0 - sse / z_sst if z_sst > 0 else float("nan")
            fit_rmse = math.sqrt(sse / count)
            usable = (
                math.isfinite(slope)
                and slope > 0.0
                and math.isfinite(intercept)
                and math.isfinite(fit_rmse)
            )
            by_station[station_id] = {
                "graph_id": graph_id,
                "status": "APPLIED" if usable else "NON_MONOTONE_OR_DEGENERATE_LINEAR_FIT",
                "paired_unique_point_count": count,
                "slope_m_per_m3s": slope,
                "intercept_m": intercept,
                "fit_rmse_m": fit_rmse,
                "fit_nse": fit_nse,
                "q_min_m3s": float(q.min().item()),
                "q_max_m3s": float(q.max().item()),
                "z_min_m": float(z.min().item()),
                "z_max_m": float(z.max().item()),
                "usable_linear": usable,
            }
        self._rating_curve_statistics_cache = {
            "mode": "train_unique_target_timestamp_linear_ols",
            "computed_from_split": "TRAIN",
            "stations": by_station,
        }
        return self._rating_curve_statistics_cache

    def fit_aligned_input_statistics(
        self,
        *,
        q_scale_floor_m3s: float,
        delta_z_scale_floor_m: float,
    ) -> dict[str, Any]:
        """Fit TRAIN-only station-aware FLOW and relative-Z input statistics."""
        if self.split != "TRAIN":
            raise ValueError("aligned input statistics只能从TRAIN计算")
        if q_scale_floor_m3s <= 0 or delta_z_scale_floor_m <= 0:
            raise ValueError("input normalization floors必须为正")
        target_stats = self.train_target_statistics()
        flow_by_station: dict[str, dict[str, Any]] = {}
        dz_by_station: dict[str, dict[str, Any]] = {}
        samples_by_graph: dict[str, list[Any]] = {}
        for sample in self._samples:
            samples_by_graph.setdefault(sample.graph_id, []).append(sample)

        for graph_id, samples in samples_by_graph.items():
            graph = self._graphs[graph_id]
            dynamic = self._dynamic[graph_id]
            used_history = torch.zeros(len(dynamic.timestamps), dtype=torch.bool)
            for sample in samples:
                origin = self._origin_index(sample)
                used_history[origin - sample.history_hours + 1 : origin + 1] = True
            for node in graph.nodes:
                station = node.station_id
                valid = used_history & dynamic.flow_mask[:, node.node_index]
                flow = dynamic.flow[valid, node.node_index].to(torch.float64)
                if flow.numel():
                    mean = float(flow.mean().item())
                    raw_std = (
                        float(flow.std(unbiased=False).item())
                        if flow.numel() > 1
                        else 0.0
                    )
                    scale = max(raw_std, float(q_scale_floor_m3s))
                else:
                    mean = 0.0
                    raw_std = float("nan")
                    scale = float(q_scale_floor_m3s)
                candidate = {
                    "source": "unique TRAIN history observations",
                    "valid_unique_point_count": int(flow.numel()),
                    "mean_m3s": mean,
                    "raw_std_m3s": raw_std,
                    "scale_m3s": scale,
                    "floor_applied": (
                        not math.isfinite(raw_std)
                    ) or raw_std < q_scale_floor_m3s,
                }
                previous = flow_by_station.get(station)
                if previous is None or candidate["valid_unique_point_count"] > previous["valid_unique_point_count"]:
                    flow_by_station[station] = candidate

        active_stations = self._active_station_ids()
        z_moments: dict[str, list[float]] = {
            station: [0.0, 0.0, 0.0] for station in active_stations
        }
        for graph_id, samples in samples_by_graph.items():
            graph = self._graphs[graph_id]
            dynamic = self._dynamic[graph_id]
            offsets = torch.arange(-self.history_hours + 1, 1, dtype=torch.long)
            for start in range(0, len(samples), 25_000):
                chunk = samples[start : start + 25_000]
                origins = torch.tensor(
                    [self._origin_index(sample) for sample in chunk], dtype=torch.long
                )
                history_indices = origins.unsqueeze(1) + offsets.unsqueeze(0)
                for node in graph.nodes:
                    local = node.node_index
                    origin_valid = dynamic.water_level_mask[origins, local]
                    values = dynamic.water_level[history_indices, local].to(torch.float64)
                    valid = (
                        dynamic.water_level_mask[history_indices, local]
                        & origin_valid.unsqueeze(1)
                    )
                    relative = values - dynamic.water_level[origins, local].to(torch.float64).unsqueeze(1)
                    selected = relative[valid]
                    if not selected.numel():
                        continue
                    moments = z_moments[node.station_id]
                    moments[0] += float(selected.numel())
                    moments[1] += float(selected.sum().item())
                    moments[2] += float(selected.square().sum().item())
        for station, (count_value, total, squared) in z_moments.items():
            count = int(count_value)
            if count > 1:
                mean = total / count
                variance = max(squared / count - mean * mean, 0.0)
                raw_std = math.sqrt(variance)
            else:
                raw_std = 0.0 if count == 1 else float("nan")
            dz_by_station[station] = {
                "source": "TRAIN history Z(t)-Z(t0)",
                "valid_point_count": count,
                "center_m": 0.0,
                "raw_std_m": raw_std,
                "scale_m": max(
                    raw_std if math.isfinite(raw_std) else 0.0,
                    float(delta_z_scale_floor_m),
                ),
                "floor_applied": (
                    not math.isfinite(raw_std)
                ) or raw_std < delta_z_scale_floor_m,
            }

        for graph_id, q_statistics in target_stats["q_by_graph"].items():
            station = self._graphs[graph_id].outlet_id
            raw_std = float(q_statistics["std_m3s"])
            flow_by_station[station] = {
                "source": "TRAIN outlet Q supervision; exactly aligned with Q loss",
                "valid_unique_point_count": int(q_statistics["valid_unique_point_count"]),
                "mean_m3s": float(q_statistics["mean_m3s"]),
                "raw_std_m3s": raw_std,
                "scale_m3s": max(raw_std, float(q_scale_floor_m3s)),
                "floor_applied": raw_std < q_scale_floor_m3s,
            }
        for station, z_statistics in target_stats["delta_z_by_station"].items():
            raw_std = float(z_statistics["std_m"])
            dz_by_station[station] = {
                "source": "TRAIN delta-Z supervision; exactly aligned with Z loss",
                "valid_point_count": int(z_statistics["valid_point_count"]),
                "center_m": 0.0,
                "raw_std_m": raw_std,
                "scale_m": max(raw_std, float(delta_z_scale_floor_m)),
                "floor_applied": raw_std < delta_z_scale_floor_m,
            }
        return {
            "mode": "train_aligned",
            "computed_from_split": "TRAIN",
            "flow_by_station": flow_by_station,
            "relative_z_by_station": dz_by_station,
        }

    def set_aligned_input_statistics(self, statistics: Mapping[str, Any]) -> None:
        if self.dynamic_normalization_mode != "train_aligned":
            raise ValueError("仅train_aligned dataset允许注入aligned statistics")
        flow = statistics.get("flow_by_station")
        level = statistics.get("relative_z_by_station")
        if not isinstance(flow, Mapping) or not isinstance(level, Mapping):
            raise ValueError("aligned input statistics缺少flow_by_station/relative_z_by_station")
        required = self._active_station_ids()
        missing_flow = sorted(required - set(flow))
        missing_level = sorted(required - set(level))
        if missing_flow or missing_level:
            raise ValueError(
                "TRAIN aligned statistics未覆盖当前active graph站点: "
                f"FLOW缺少={missing_flow}, Z缺少={missing_level}"
            )
        self._aligned_input_statistics = dict(statistics)

    def aligned_input_statistics(self) -> dict[str, Any] | None:
        return self._aligned_input_statistics

    def __getitem__(self, index: int):
        batch = super().__getitem__(index)
        if self.dynamic_normalization_mode != "train_aligned":
            return batch
        if self._aligned_input_statistics is None:
            raise RuntimeError(
                f"{self.split} train_aligned dataset尚未注入TRAIN input statistics"
            )
        features = batch.dynamic_node_features.clone()
        flow_statistics = self._aligned_input_statistics["flow_by_station"]
        z_statistics = self._aligned_input_statistics["relative_z_by_station"]
        station_ids = batch.station_ids
        if station_ids is None:
            raise RuntimeError("train_aligned正式样本缺少station_ids")
        flow_mean = torch.tensor(
            [float(flow_statistics[station]["mean_m3s"]) for station in station_ids],
            dtype=batch.q_history.dtype,
            device=batch.q_history.device,
        )
        flow_scale = torch.tensor(
            [float(flow_statistics[station]["scale_m3s"]) for station in station_ids],
            dtype=batch.q_history.dtype,
            device=batch.q_history.device,
        )
        z_scale = torch.tensor(
            [float(z_statistics[station]["scale_m"]) for station in station_ids],
            dtype=batch.z_history.dtype,
            device=batch.z_history.device,
        )
        for feature_index, feature in enumerate(self.dynamic_features):
            if feature == "FLOW":
                normalized = (
                    batch.q_history - flow_mean.unsqueeze(0)
                ) / flow_scale.unsqueeze(0)
                features[..., feature_index] = torch.where(
                    batch.q_mask, normalized, torch.zeros_like(normalized)
                )
            elif feature == "WATER_LEVEL":
                z0 = batch.z_history[-1]
                z0_valid = batch.z_mask[-1]
                relative = batch.z_history - z0.unsqueeze(0)
                effective = batch.z_mask & z0_valid.unsqueeze(0)
                normalized = relative / z_scale.unsqueeze(0)
                features[..., feature_index] = torch.where(
                    effective, normalized, torch.zeros_like(normalized)
                )
        batch.dynamic_node_features = features
        return batch
