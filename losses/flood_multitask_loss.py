"""Mask-safe legacy and flood-specific multi-task losses.

The multi-task discharge objective is balanced at sample level.  Each Q
window first produces one point/peak/volume value, after which the TRAIN-only
graph-event sample weight is applied.  Water-level level and first-difference
terms retain valid-element means.  Physical errors are divided by explicit
TRAIN-only scales supplied in ``cfg['_runtime']``; the strict Q-normalization
experiment selects a per-graph scale without changing physical model tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from metrics.flood_metrics import masked_huber_stats


COMPONENTS = ("q_point", "q_peak", "q_volume", "z_level", "z_slope")


@dataclass(frozen=True)
class LossTerm:
    """Differentiable numerator plus its explicit aggregation denominator."""

    numerator: torch.Tensor
    denominator: int


def _checked_mask(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError(
            "prediction、target、mask形状必须完全一致，实际为"
            f"{tuple(prediction.shape)}、{tuple(target.shape)}、{tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        if torch.is_floating_point(mask) and not torch.isfinite(mask).all():
            raise ValueError("mask包含NaN/Inf")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask只能包含布尔值或0/1")
    valid = mask.bool()
    if valid.any() and not torch.isfinite(target[valid]).all():
        raise ValueError("有效mask内的target包含NaN/Inf")
    if valid.any() and not torch.isfinite(prediction[valid]).all():
        raise FloatingPointError("有效mask内的模型预测包含NaN/Inf")
    return valid


def _positive_scale(scales: Mapping[str, Any], name: str) -> float:
    value = float(scales.get(name, 1.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"TRAIN loss scale {name}必须为有限正数，实际={value}")
    return value


def _single_graph_id(batch: Any, batch_size: int) -> str:
    value = getattr(batch, "graph_id", None)
    if isinstance(value, str):
        identifiers = (value,)
    elif isinstance(value, (tuple, list)):
        identifiers = tuple(value)
    else:
        raise ValueError(
            "per-graph Q loss要求batch.graph_id为字符串或逐样本字符串序列"
        )
    if not identifiers or any(
        not isinstance(identifier, str) or not identifier.strip()
        for identifier in identifiers
    ):
        raise ValueError("per-graph Q loss的GRAPH_ID必须全部为非空字符串")
    if len(identifiers) not in {1, batch_size}:
        raise ValueError(
            f"batch.graph_id数量必须为1或batch size={batch_size}，"
            f"实际={len(identifiers)}"
        )
    unique = set(identifiers)
    if len(unique) != 1:
        raise ValueError(
            f"per-graph Q loss禁止一个batch混合多个GRAPH_ID，实际={sorted(unique)}"
        )
    return identifiers[0]


def _single_target_station_id(batch: Any, batch_size: int) -> str:
    value = getattr(batch, "target_station_id", None)
    if isinstance(value, str):
        identifiers = (value,)
    elif isinstance(value, (tuple, list)):
        identifiers = tuple(value)
    else:
        raise ValueError("per-station ΔZ loss要求batch.target_station_id")
    if len(identifiers) not in {1, batch_size} or len(set(identifiers)) != 1:
        raise ValueError("per-station ΔZ loss要求一个batch只含一个有效目标站")
    station = identifiers[0]
    if not isinstance(station, str) or not station.strip():
        raise ValueError("target_station_id必须是非空字符串")
    return station


def _sample_weights(batch: Any, batch_size: int, reference: torch.Tensor) -> torch.Tensor:
    value = getattr(batch, "sample_weight", None)
    if value is None:
        return torch.ones(batch_size, dtype=reference.dtype, device=reference.device)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (batch_size,):
        raise ValueError(f"sample_weight必须为[B]={batch_size}，实际={getattr(value, 'shape', None)}")
    weights = value.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample_weight必须全部为有限正数")
    return weights


def _weighted_sample_term(
    per_sample: torch.Tensor,
    active: torch.Tensor,
    weights: torch.Tensor,
    reference: torch.Tensor,
) -> LossTerm:
    if per_sample.ndim != 1 or active.shape != per_sample.shape or weights.shape != per_sample.shape:
        raise ValueError("逐样本loss、active和weight必须都是相同形状[B]")
    count = int(active.sum().item())
    if not count:
        return LossTerm(reference.reshape(-1)[:0].sum(), 0)
    numerator = (per_sample[active] * weights[active]).sum()
    return LossTerm(numerator, count)


def water_level_first_differences(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    history: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build causal Z first differences and their strict mask.

    Forecast hour 1 subtracts the latest valid observed Z in the history.  For
    later hours both prediction and observation subtract their own preceding
    forecast-hour value; both adjacent target masks must be valid.  If no
    history Z exists, only the first-hour slope mask is disabled.
    """

    valid_target = _checked_mask(prediction, target, target_mask)
    if prediction.ndim != 3:
        raise ValueError("Z slope要求prediction/target形状为[B,F,N]")
    if history.ndim != 3 or history_mask.shape != history.shape:
        raise ValueError("Z slope要求history/history_mask形状为[B,H,N]")
    if history.shape[0] != prediction.shape[0] or history.shape[2] != prediction.shape[2]:
        raise ValueError("Z history与forecast的batch/node维必须一致")
    if history_mask.dtype != torch.bool:
        if not ((history_mask == 0) | (history_mask == 1)).all():
            raise ValueError("history_mask只能包含布尔值或0/1")
    history_valid = history_mask.bool()
    if history_valid.any() and not torch.isfinite(history[history_valid]).all():
        raise ValueError("有效history_mask内的Z history包含NaN/Inf")

    batch_size, history_hours, nodes = history.shape
    positions = torch.arange(history_hours, device=history.device).view(1, -1, 1)
    positions = positions.expand(batch_size, -1, nodes)
    last_position = positions.masked_fill(~history_valid, -1).amax(dim=1)
    has_baseline = last_position >= 0
    safe_position = last_position.clamp_min(0).unsqueeze(1)
    baseline = history.gather(1, safe_position).squeeze(1)
    baseline = torch.where(has_baseline, baseline, torch.zeros_like(baseline))

    first_prediction = prediction[:, 0] - baseline
    first_target = target[:, 0] - baseline
    first_mask = valid_target[:, 0] & has_baseline
    if prediction.shape[1] == 1:
        return (
            first_prediction.unsqueeze(1),
            first_target.unsqueeze(1),
            first_mask.unsqueeze(1),
        )
    later_prediction = prediction[:, 1:] - prediction[:, :-1]
    later_target = target[:, 1:] - target[:, :-1]
    later_mask = valid_target[:, 1:] & valid_target[:, :-1]
    return (
        torch.cat((first_prediction.unsqueeze(1), later_prediction), dim=1),
        torch.cat((first_target.unsqueeze(1), later_target), dim=1),
        torch.cat((first_mask.unsqueeze(1), later_mask), dim=1),
    )


def delta_z_first_differences(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """First differences for an already causal ΔZ(t+h) trajectory.

    The h1 predecessor is exactly zero (= ΔZ at t0), while later horizons use
    adjacent forecast values.  No absolute history value is subtracted again.
    """

    valid = _checked_mask(prediction, target, target_mask)
    zero_prediction = torch.zeros_like(prediction[:, :1])
    zero_target = torch.zeros_like(target[:, :1])
    predecessor_prediction = torch.cat((zero_prediction, prediction[:, :-1]), dim=1)
    predecessor_target = torch.cat((zero_target, target[:, :-1]), dim=1)
    predecessor_valid = torch.cat(
        (torch.ones_like(valid[:, :1]), valid[:, :-1]), dim=1
    )
    return (
        prediction - predecessor_prediction,
        target - predecessor_target,
        valid & predecessor_valid,
    )


class FloodMultitaskLoss:
    """Compute legacy or Q-point/peak/volume + Z-level/slope objectives."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        loss_cfg = cfg.get("loss", {})
        self.mode = str(loss_cfg.get("mode", "legacy"))
        self.q_scale_mode = str(loss_cfg.get("q_scale_mode", "global"))
        self.z_target_mode = str(loss_cfg.get("z_target_mode", "absolute"))
        self.delta_z_scale_mode = str(
            loss_cfg.get("delta_z_scale_mode", "global")
        )
        if self.mode not in {"legacy", "multitask"}:
            raise ValueError(f"loss.mode必须是legacy/multitask，实际={self.mode!r}")
        if self.q_scale_mode not in {"global", "per_graph"}:
            raise ValueError(
                "loss.q_scale_mode必须是global/per_graph，"
                f"实际={self.q_scale_mode!r}"
            )
        if self.z_target_mode not in {"absolute", "delta_from_t0"}:
            raise ValueError(f"未知z_target_mode={self.z_target_mode!r}")

    def scales(self) -> tuple[float, float]:
        """Return the unchanged global TRAIN scales used outside Q supervision."""

        runtime_scales = self.cfg.get("_runtime", {}).get("loss_scales", {})
        return (
            _positive_scale(runtime_scales, "discharge"),
            _positive_scale(runtime_scales, "water_level"),
        )

    def q_scale_for_batch(self, batch: Any, batch_size: int) -> float:
        global_q_scale, _ = self.scales()
        if self.q_scale_mode == "global":
            return global_q_scale
        graph_id = _single_graph_id(batch, batch_size)
        runtime_scales = self.cfg.get("_runtime", {}).get("loss_scales", {})
        graph_scales = runtime_scales.get("discharge_by_graph")
        if not isinstance(graph_scales, Mapping) or graph_id not in graph_scales:
            raise ValueError(
                f"GRAPH_ID={graph_id}: 缺少TRAIN-only per-graph Q loss scale；"
                "禁止回退global std"
            )
        value = float(graph_scales[graph_id])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"GRAPH_ID={graph_id}: per-graph Q loss scale必须为有限正数，"
                f"实际={value}"
            )
        return value

    def z_scale_for_batch(self, batch: Any, batch_size: int) -> float:
        _, global_z_scale = self.scales()
        if self.delta_z_scale_mode == "global":
            return global_z_scale
        station_id = _single_target_station_id(batch, batch_size)
        station_scales = (
            self.cfg.get("_runtime", {})
            .get("loss_scales", {})
            .get("delta_z_by_station")
        )
        if not isinstance(station_scales, Mapping) or station_id not in station_scales:
            raise ValueError(
                f"STATION_ID={station_id}: 缺少TRAIN-only ΔZ scale；禁止回退全局WATER_LEVEL std"
            )
        value = float(station_scales[station_id])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"STATION_ID={station_id}: ΔZ scale必须为有限正数")
        return value

    def coefficients(self) -> dict[str, float]:
        if self.mode == "legacy":
            weights = self.cfg["loss_weights"]
            result = {
                "q_point": float(weights["discharge"]),
                "q_peak": 0.0,
                "q_volume": 0.0,
                "z_level": float(weights["water_level"]),
                "z_slope": 0.0,
            }
        else:
            loss = self.cfg["loss"]
            result = {
                "q_point": float(loss["discharge_weight"]) * float(loss["q_point_weight"]),
                "q_peak": float(loss["discharge_weight"]) * float(loss["q_peak_weight"]),
                "q_volume": float(loss["discharge_weight"]) * float(loss["q_volume_weight"]),
                "z_level": float(loss["water_level_weight"]) * float(loss["z_level_weight"]),
                "z_slope": float(loss["water_level_weight"]) * float(loss["z_slope_weight"]),
            }
        if any(not math.isfinite(value) or value < 0 for value in result.values()):
            raise ValueError(f"loss coefficients必须为有限非负数，实际={result}")
        return result

    def denominators(self, batch: Any) -> dict[str, int]:
        q_mask = batch.q_target_mask.bool()
        z_mask = batch.z_target_mask.bool()
        if self.mode == "legacy":
            return {
                "q_point": int(q_mask.sum().item()),
                "q_peak": 0,
                "q_volume": 0,
                "z_level": int(z_mask.sum().item()),
                "z_slope": 0,
            }
        q_active = q_mask.reshape(q_mask.shape[0], -1).any(dim=1)
        if self.z_target_mode == "delta_from_t0":
            _, _, slope_mask = delta_z_first_differences(
                batch.z_target, batch.z_target, z_mask
            )
        else:
            _, _, slope_mask = water_level_first_differences(
                batch.z_target,
                batch.z_target,
                z_mask,
                batch.z_history,
                batch.z_mask,
            )
        return {
            "q_point": (
                int(q_mask.sum().item())
                if self.z_target_mode == "delta_from_t0"
                else int(q_active.sum().item())
            ),
            "q_peak": int(q_active.sum().item()),
            "q_volume": int(q_active.sum().item()),
            "z_level": int(z_mask.sum().item()),
            "z_slope": int(slope_mask.sum().item()),
        }

    def batch_statistics(self, output: Mapping[str, Any], batch: Any) -> dict[str, LossTerm]:
        q_prediction = output["q"]
        z_prediction = output["z"]
        q_target = batch.q_target
        z_target = batch.z_target
        q_mask = _checked_mask(q_prediction, q_target, batch.q_target_mask)
        z_mask = _checked_mask(z_prediction, z_target, batch.z_target_mask)
        global_q_scale, global_z_scale = self.scales()

        if self.mode == "legacy":
            q_sum, q_count = masked_huber_stats(
                q_prediction / global_q_scale,
                q_target / global_q_scale,
                q_mask,
            )
            z_sum, z_count = masked_huber_stats(
                z_prediction / global_z_scale,
                z_target / global_z_scale,
                z_mask,
            )
            zero = (q_prediction.reshape(-1)[:0].sum() + z_prediction.reshape(-1)[:0].sum())
            return {
                "q_point": LossTerm(q_sum, q_count),
                "q_peak": LossTerm(zero, 0),
                "q_volume": LossTerm(zero, 0),
                "z_level": LossTerm(z_sum, z_count),
                "z_slope": LossTerm(zero, 0),
            }

        batch_size = q_prediction.shape[0]
        q_scale = (
            self.q_scale_for_batch(batch, batch_size)
            if q_mask.any()
            else global_q_scale
        )
        z_scale = self.z_scale_for_batch(batch, batch_size) if z_mask.any() else global_z_scale
        weights = _sample_weights(batch, batch_size, q_prediction)
        q_active = q_mask.reshape(batch_size, -1).any(dim=1)
        valid_counts = q_mask.reshape(batch_size, -1).sum(dim=1)
        safe_q_prediction = torch.where(q_mask, q_prediction, torch.zeros_like(q_prediction))
        safe_q_target = torch.where(q_mask, q_target, torch.zeros_like(q_target))
        q_error = (safe_q_prediction - safe_q_target) / q_scale
        q_point_elements = F.huber_loss(
            q_error, torch.zeros_like(q_error), delta=1.0, reduction="none"
        )
        q_point_per_sample = (
            q_point_elements.reshape(batch_size, -1).sum(dim=1)
            / valid_counts.clamp_min(1).to(q_point_elements.dtype)
        )
        if self.z_target_mode == "delta_from_t0":
            q_point = LossTerm(q_point_elements[q_mask].sum(), int(q_mask.sum().item()))
        else:
            q_point = _weighted_sample_term(
                q_point_per_sample, q_active, weights, q_prediction
            )

        negative_infinity = torch.full_like(q_prediction, float("-inf"))
        q_peak_prediction = torch.where(q_mask, q_prediction, negative_infinity).reshape(batch_size, -1).amax(dim=1)
        q_peak_target = torch.where(q_mask, q_target, negative_infinity).reshape(batch_size, -1).amax(dim=1)
        q_peak_prediction = torch.where(
            q_active, q_peak_prediction, torch.zeros_like(q_peak_prediction)
        )
        q_peak_target = torch.where(
            q_active, q_peak_target, torch.zeros_like(q_peak_target)
        )
        q_peak_error = (q_peak_prediction - q_peak_target) / q_scale
        q_peak = _weighted_sample_term(
            q_peak_error.square(), q_active, weights, q_prediction
        )

        q_mean_prediction = safe_q_prediction.reshape(batch_size, -1).sum(dim=1) / valid_counts.clamp_min(1)
        q_mean_target = safe_q_target.reshape(batch_size, -1).sum(dim=1) / valid_counts.clamp_min(1)
        q_volume_error = torch.where(
            q_active,
            (q_mean_prediction - q_mean_target) / q_scale,
            torch.zeros_like(q_mean_prediction),
        )
        q_volume = _weighted_sample_term(
            q_volume_error.square(), q_active, weights, q_prediction
        )

        z_level_sum, z_level_count = masked_huber_stats(
            z_prediction / z_scale, z_target / z_scale, z_mask
        )
        if self.z_target_mode == "delta_from_t0":
            slope_prediction, slope_target, slope_mask = delta_z_first_differences(
                z_prediction, z_target, z_mask
            )
        else:
            slope_prediction, slope_target, slope_mask = water_level_first_differences(
                z_prediction,
                z_target,
                z_mask,
                batch.z_history,
                batch.z_mask,
            )
        z_slope_sum, z_slope_count = masked_huber_stats(
            slope_prediction / z_scale,
            slope_target / z_scale,
            slope_mask,
        )
        return {
            "q_point": q_point,
            "q_peak": q_peak,
            "q_volume": q_volume,
            "z_level": LossTerm(z_level_sum, z_level_count),
            "z_slope": LossTerm(z_slope_sum, z_slope_count),
        }

    def combine(
        self,
        statistics: Mapping[str, LossTerm],
        denominators: Mapping[str, int] | None = None,
    ) -> torch.Tensor:
        coefficients = self.coefficients()
        reference = next(iter(statistics.values())).numerator
        total = reference * 0.0
        used = False
        for name in COMPONENTS:
            denominator = int(
                statistics[name].denominator
                if denominators is None
                else denominators[name]
            )
            if denominator:
                total = total + coefficients[name] * statistics[name].numerator / denominator
                used = used or coefficients[name] > 0
        if not used:
            raise ValueError("当前batch/group没有任何启用的有效Q/Z监督目标")
        return total

    def report(
        self,
        totals: Mapping[str, tuple[float, int]],
        *,
        q_valid_count: int,
        z_valid_count: int,
    ) -> dict[str, float | int]:
        means = {
            name: (value / count if count else float("nan"))
            for name, (value, count) in totals.items()
        }
        if self.mode == "legacy":
            q_weight = float(self.cfg["loss_weights"]["discharge"])
            z_weight = float(self.cfg["loss_weights"]["water_level"])
            total = 0.0
            if totals["q_point"][1]:
                total += q_weight * means["q_point"]
            if totals["z_level"][1]:
                total += z_weight * means["z_level"]
            return {
                "loss": total,
                "q_loss": means["q_point"],
                "z_loss": means["z_level"],
                "q_valid_count": q_valid_count,
                "z_valid_count": z_valid_count,
            }

        loss = self.cfg["loss"]
        q_total = 0.0
        for name, key in (
            ("q_point", "q_point_weight"),
            ("q_peak", "q_peak_weight"),
            ("q_volume", "q_volume_weight"),
        ):
            if totals[name][1]:
                q_total += float(loss[key]) * means[name]
        z_total = 0.0
        for name, key in (("z_level", "z_level_weight"), ("z_slope", "z_slope_weight")):
            if totals[name][1]:
                z_total += float(loss[key]) * means[name]
        total = 0.0
        if any(totals[name][1] for name in ("q_point", "q_peak", "q_volume")):
            total += float(loss["discharge_weight"]) * q_total
        if any(totals[name][1] for name in ("z_level", "z_slope")):
            total += float(loss["water_level_weight"]) * z_total
        return {
            "loss": total,
            "total_loss": total,
            "q_loss": q_total,
            "z_loss": z_total,
            "q_total_loss": q_total,
            "q_point_loss": means["q_point"],
            "q_peak_loss": means["q_peak"],
            "q_volume_loss": means["q_volume"],
            "z_total_loss": z_total,
            "z_level_loss": means["z_level"],
            "z_slope_loss": means["z_slope"],
            "q_valid_count": q_valid_count,
            "z_valid_count": z_valid_count,
            "z_slope_valid_count": totals["z_slope"][1],
        }
