"""Q-only objective with TRAIN-only high-flow emphasis.

The objective excludes sliding-window maximum loss.  High-flow
emphasis acts on physical target points whose observed discharge exceeds the
station-specific TRAIN P80 threshold.  The multiplier ramps smoothly from 1 at
P80 to 3 at P99 and is capped thereafter.  Thresholds are fitted from unique
TRAIN physical target hours, never from VALIDATION/TEST.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from losses.flood_multitask_loss import LossTerm
from metrics.flood_metrics import masked_huber_stats

_COMPONENTS = ("q_point", "q_high_flow", "q_volume")


class HydrologicLoss:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        loss = cfg.get("loss", {})
        if str(loss.get("mode", "")).lower() != "q_only_high_flow":
            raise ValueError("Q-only loss.mode必须为q_only_high_flow")
        expected = {
            "q_point_weight": 1.0,
            "q_high_flow_weight": 0.25,
            "q_volume_weight": 0.25,
            "high_flow_lower_quantile": 0.80,
            "high_flow_upper_quantile": 0.99,
            "high_flow_max_multiplier": 3.0,
        }
        changed = {
            key: loss.get(key)
            for key, value in expected.items()
            if not math.isclose(float(loss.get(key, float("nan"))), value)
        }
        if changed:
            raise ValueError(f"Q loss设计不一致: {changed}")
        if "q_peak_weight" in loss:
            raise ValueError("禁止保留sliding-window q_peak_weight")

        runtime = cfg.get("_runtime", {})
        normal = runtime.get("normalization")
        if not isinstance(normal, Mapping):
            raise ValueError("loss缺少TRAIN-only Q normalization")
        self.q_scales = torch.as_tensor(
            normal["q_target_scale"], dtype=torch.float32
        )
        if (
            self.q_scales.ndim != 1
            or not torch.isfinite(self.q_scales).all()
            or (self.q_scales <= 0).any()
        ):
            raise ValueError("per-station Q scale非法")

        station_ids = tuple(str(value) for value in runtime.get("station_ids", ()))
        quantiles = runtime.get("high_flow_quantiles")
        if not station_ids or not isinstance(quantiles, Mapping):
            raise ValueError("loss缺少station catalogue或TRAIN-only high-flow quantiles")
        station_stats = quantiles.get("stations")
        if not isinstance(station_stats, Mapping):
            raise ValueError("high-flow quantiles缺少stations")
        lower = torch.zeros(len(station_ids), dtype=torch.float32)
        upper = torch.zeros(len(station_ids), dtype=torch.float32)
        available = torch.zeros(len(station_ids), dtype=torch.bool)
        for index, station in enumerate(station_ids):
            record = station_stats.get(station, {})
            if not isinstance(record, Mapping) or not bool(record.get("available", False)):
                continue
            q80 = float(record["q80_m3s"])
            q99 = float(record["q99_m3s"])
            if not math.isfinite(q80) or not math.isfinite(q99) or q99 <= q80:
                raise ValueError(f"high-flow threshold非法: {station}: {q80}, {q99}")
            lower[index] = q80
            upper[index] = q99
            available[index] = True
        if len(station_ids) != int(self.q_scales.numel()):
            raise ValueError("Q scale与station catalogue长度不一致")
        self.high_flow_lower = lower
        self.high_flow_upper = upper
        self.high_flow_available = available
        self.high_flow_max_multiplier = float(loss["high_flow_max_multiplier"])

    def coefficients(self) -> dict[str, float]:
        loss = self.cfg["loss"]
        return {
            "q_point": float(loss["q_point_weight"]),
            "q_high_flow": float(loss["q_high_flow_weight"]),
            "q_volume": float(loss["q_volume_weight"]),
        }

    @staticmethod
    def _checked(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("Q prediction/target/mask shape必须一致")
        valid = mask.bool()
        if valid.any():
            if not torch.isfinite(prediction[valid]).all():
                raise FloatingPointError("有效Q预测含NaN/Inf")
            if not torch.isfinite(target[valid]).all():
                raise ValueError("有效Q target含NaN/Inf")
        return valid

    def _station_values(
        self, batch: Any, values: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        index = batch.obs_station_index.long().to(reference.device)
        return values.to(device=reference.device, dtype=reference.dtype)[index]

    def _scale(self, batch: Any, reference: torch.Tensor) -> torch.Tensor:
        return self._station_values(batch, self.q_scales, reference).view(1, 1, -1)

    def _high_flow_mask_and_weights(
        self,
        batch: Any,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lower = self._station_values(batch, self.high_flow_lower, target).view(1, 1, -1)
        upper = self._station_values(batch, self.high_flow_upper, target).view(1, 1, -1)
        available = self._station_values(
            batch, self.high_flow_available.to(torch.float32), target
        ).view(1, 1, -1).bool()
        high_flow = valid & available & (target >= lower)
        span = (upper - lower).clamp_min(torch.finfo(target.dtype).eps)
        ramp = ((target - lower) / span).clamp(0.0, 1.0)
        multiplier = 1.0 + (self.high_flow_max_multiplier - 1.0) * ramp
        return high_flow, multiplier

    def denominators(self, batch: Any) -> dict[str, int]:
        mask = batch.q_target_mask.bool()
        # Thresholding uses observed TRAIN/VAL targets only for loss accounting;
        # thresholds themselves are frozen TRAIN-only statistics.
        high_flow, _ = self._high_flow_mask_and_weights(
            batch, batch.q_target, mask
        )
        active = mask.any(dim=1)
        return {
            "q_point": int(mask.sum().item()),
            "q_high_flow": int(high_flow.sum().item()),
            "q_volume": int(active.sum().item()),
        }

    def batch_statistics(
        self, output: Mapping[str, Any], batch: Any
    ) -> dict[str, LossTerm]:
        prediction = output["q"]
        target = batch.q_target
        mask = self._checked(prediction, target, batch.q_target_mask)
        scale = self._scale(batch, prediction)
        error = (prediction - target) / scale

        point_sum, point_count = masked_huber_stats(
            error, torch.zeros_like(error), mask
        )

        high_flow, multiplier = self._high_flow_mask_and_weights(
            batch, target, mask
        )
        high_flow_count = int(high_flow.sum().item())
        if high_flow_count:
            element = F.huber_loss(
                error[high_flow],
                torch.zeros_like(error[high_flow]),
                delta=1.0,
                reduction="none",
            )
            high_flow_sum = (element * multiplier[high_flow]).sum()
        else:
            high_flow_sum = error[high_flow].sum()

        batch_size, _, obs = prediction.shape
        active = mask.any(dim=1)
        scale_bo = scale.squeeze(1).expand(batch_size, obs)
        valid_count = mask.sum(dim=1).clamp_min(1)
        safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        mean_prediction = safe_prediction.sum(dim=1) / valid_count
        mean_target = safe_target.sum(dim=1) / valid_count
        volume_error = (mean_prediction - mean_target) / scale_bo
        volume_sum = volume_error[active].square().sum()
        volume_count = int(active.sum().item())

        return {
            "q_point": LossTerm(point_sum, point_count),
            "q_high_flow": LossTerm(high_flow_sum, high_flow_count),
            "q_volume": LossTerm(volume_sum, volume_count),
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
        for name in _COMPONENTS:
            denominator = int(
                statistics[name].denominator
                if denominators is None
                else denominators[name]
            )
            if denominator:
                total = (
                    total
                    + coefficients[name]
                    * statistics[name].numerator
                    / denominator
                )
                used = True
        if not used:
            raise ValueError("当前batch/group没有有效Q监督")
        return total

    def report(
        self,
        totals: Mapping[str, tuple[float, int]],
        *,
        q_valid_count: int,
        z_valid_count: int,
    ) -> dict[str, float | int]:
        del z_valid_count
        means = {
            name: (float(value) / int(count) if int(count) else float("nan"))
            for name, (value, count) in totals.items()
        }
        coefficients = self.coefficients()
        total = sum(
            coefficients[name] * means[name]
            for name in _COMPONENTS
            if totals[name][1]
        )
        return {
            "loss": total,
            "total_loss": total,
            "q_loss": total,
            "q_total_loss": total,
            "q_point_loss": means["q_point"],
            "q_high_flow_loss": means["q_high_flow"],
            "q_volume_loss": means["q_volume"],
            "q_high_flow_valid_count": int(totals["q_high_flow"][1]),
            "q_valid_count": int(q_valid_count),
        }
