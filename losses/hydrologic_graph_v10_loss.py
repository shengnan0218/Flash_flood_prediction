"""Formal v10 Q-only objective.

Stage is derived diagnostically from Q and never contributes gradient or model
selection.  Q point/peak/volume definitions are intentionally identical to v9.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from losses.flood_multitask_loss import LossTerm
from metrics.flood_metrics import masked_huber_stats


_COMPONENTS = ("q_point", "q_peak", "q_volume")


class HydrologicGraphV10Loss:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        loss = cfg.get("loss", {})
        if str(loss.get("mode", "")).lower() != "q_only":
            raise ValueError("v10正式loss.mode必须为q_only")
        required = {
            "q_point_weight": 1.0,
            "q_peak_weight": 0.25,
            "q_volume_weight": 0.25,
        }
        changed = {
            key: loss.get(key)
            for key, expected in required.items()
            if not math.isclose(float(loss.get(key, float("nan"))), expected)
        }
        if changed:
            raise ValueError(f"v10 Q loss权重与正式设计不一致: {changed}")
        normal = cfg.get("_runtime", {}).get("v8_normalization")
        if not isinstance(normal, Mapping):
            raise ValueError("v10 loss缺少TRAIN-only normalization")
        self.q_scales = torch.as_tensor(normal["q_target_scale"], dtype=torch.float32)
        if (
            self.q_scales.ndim != 1
            or not torch.isfinite(self.q_scales).all()
            or (self.q_scales <= 0).any()
        ):
            raise ValueError("v10 per-station Q scale非法")

    def coefficients(self) -> dict[str, float]:
        loss = self.cfg["loss"]
        return {
            "q_point": float(loss["q_point_weight"]),
            "q_peak": float(loss["q_peak_weight"]),
            "q_volume": float(loss["q_volume_weight"]),
        }

    @staticmethod
    def _checked(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError("v10 Q prediction/target/mask shape必须一致")
        valid = mask.bool()
        if valid.any():
            if not torch.isfinite(prediction[valid]).all():
                raise FloatingPointError("v10有效Q预测含NaN/Inf")
            if not torch.isfinite(target[valid]).all():
                raise ValueError("v10有效Q target含NaN/Inf")
        return valid

    def _scale(self, batch: Any, reference: torch.Tensor) -> torch.Tensor:
        index = batch.obs_station_index.long().to(reference.device)
        scale = self.q_scales.to(device=reference.device, dtype=reference.dtype)[index]
        return scale.view(1, 1, -1)

    def denominators(self, batch: Any) -> dict[str, int]:
        mask = batch.q_target_mask.bool()
        active = mask.any(dim=1)
        return {
            "q_point": int(mask.sum().item()),
            "q_peak": int(active.sum().item()),
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
        q_point_sum, q_point_count = masked_huber_stats(
            error, torch.zeros_like(error), mask
        )

        batch_size, _, obs = prediction.shape
        active = mask.any(dim=1)
        negative_inf = torch.full_like(prediction, float("-inf"))
        peak_prediction = torch.where(mask, prediction, negative_inf).amax(dim=1)
        peak_target = torch.where(mask, target, negative_inf).amax(dim=1)
        peak_prediction = torch.where(active, peak_prediction, torch.zeros_like(peak_prediction))
        peak_target = torch.where(active, peak_target, torch.zeros_like(peak_target))
        scale_bo = scale.squeeze(1).expand(batch_size, obs)
        peak_error = (peak_prediction - peak_target) / scale_bo
        q_peak_sum = peak_error[active].square().sum()
        q_peak_count = int(active.sum().item())

        # Fixed hourly cadence: mean-Q bias is proportional to duration-normalized
        # forecast-volume bias, preserving the frozen v9 definition exactly.
        valid_count = mask.sum(dim=1).clamp_min(1)
        safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        mean_prediction = safe_prediction.sum(dim=1) / valid_count
        mean_target = safe_target.sum(dim=1) / valid_count
        volume_error = (mean_prediction - mean_target) / scale_bo
        q_volume_sum = volume_error[active].square().sum()
        return {
            "q_point": LossTerm(q_point_sum, q_point_count),
            "q_peak": LossTerm(q_peak_sum, q_peak_count),
            "q_volume": LossTerm(q_volume_sum, q_peak_count),
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
                statistics[name].denominator if denominators is None else denominators[name]
            )
            if denominator:
                total = total + coefficients[name] * statistics[name].numerator / denominator
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
            "q_peak_loss": means["q_peak"],
            "q_volume_loss": means["q_volume"],
            "q_valid_count": int(q_valid_count),
        }
