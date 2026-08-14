"""Layered Q/Delta-Z objective for v8 sparse observation stations."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from losses.flood_multitask_loss import LossTerm
from metrics.flood_metrics import masked_huber_stats


_COMPONENTS = (
    "q_point",
    "q_peak",
    "q_volume",
    "z_level",
    "z_slope",
    "qz_consistency",
)


class HydrologicGraphV8Loss:
    """Q point/peak/volume + Delta-Z level/slope + weak independent-head consistency."""

    QZ_CONSISTENCY_WEIGHT = 0.1

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        loss = cfg.get("loss", {})
        required = {
            "discharge_weight": 1.0,
            "water_level_weight": 1.0,
            "q_point_weight": 1.0,
            "q_peak_weight": 0.25,
            "q_volume_weight": 0.25,
            "z_level_weight": 1.0,
            "z_slope_weight": 0.25,
        }
        changed = {
            key: loss.get(key)
            for key, value in required.items()
            if not math.isclose(float(loss.get(key, float("nan"))), value)
        }
        if changed:
            raise ValueError(f"v8正式loss权重与冻结设计不一致: {changed}")
        runtime = cfg.get("_runtime", {})
        normal = runtime.get("v8_normalization")
        if not isinstance(normal, Mapping):
            raise ValueError("v8 loss缺少_runtime.v8_normalization")
        self.q_scales = torch.as_tensor(normal["q_target_scale"], dtype=torch.float32)
        self.z_scales = torch.as_tensor(normal["dz_target_scale"], dtype=torch.float32)
        if (
            self.q_scales.ndim != 1
            or self.z_scales.shape != self.q_scales.shape
            or not torch.isfinite(self.q_scales).all()
            or not torch.isfinite(self.z_scales).all()
            or (self.q_scales <= 0).any()
            or (self.z_scales <= 0).any()
        ):
            raise ValueError("v8 per-station Q/Delta-Z scale非法")

    def coefficients(self) -> dict[str, float]:
        loss = self.cfg["loss"]
        return {
            "q_point": float(loss["discharge_weight"]) * float(loss["q_point_weight"]),
            "q_peak": float(loss["discharge_weight"]) * float(loss["q_peak_weight"]),
            "q_volume": float(loss["discharge_weight"]) * float(loss["q_volume_weight"]),
            "z_level": float(loss["water_level_weight"]) * float(loss["z_level_weight"]),
            "z_slope": float(loss["water_level_weight"]) * float(loss["z_slope_weight"]),
            "qz_consistency": self.QZ_CONSISTENCY_WEIGHT,
        }

    @staticmethod
    def _checked(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, label: str
    ) -> torch.Tensor:
        if prediction.shape != target.shape or mask.shape != target.shape:
            raise ValueError(f"{label}: prediction/target/mask shape必须一致")
        valid = mask.bool()
        if valid.any():
            if not torch.isfinite(prediction[valid]).all():
                raise FloatingPointError(f"{label}: 有效预测含NaN/Inf")
            if not torch.isfinite(target[valid]).all():
                raise ValueError(f"{label}: 有效target含NaN/Inf")
        return valid

    def _scales(
        self, batch: Any, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        station_index = batch.obs_station_index.long().to(reference.device)
        q = self.q_scales.to(device=reference.device, dtype=reference.dtype)[station_index]
        z = self.z_scales.to(device=reference.device, dtype=reference.dtype)[station_index]
        return q.view(1, 1, -1), z.view(1, 1, -1)

    @staticmethod
    def _slope(
        prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_prediction = torch.zeros_like(prediction[:, :1])
        zero_target = torch.zeros_like(target[:, :1])
        predecessor_prediction = torch.cat(
            [zero_prediction, prediction[:, :-1]], dim=1
        )
        predecessor_target = torch.cat([zero_target, target[:, :-1]], dim=1)
        predecessor_mask = torch.cat(
            [torch.ones_like(mask[:, :1], dtype=torch.bool), mask[:, :-1].bool()],
            dim=1,
        )
        return (
            prediction - predecessor_prediction,
            target - predecessor_target,
            mask.bool() & predecessor_mask,
        )

    def denominators(self, batch: Any) -> dict[str, int]:
        q_mask = batch.q_target_mask.bool()
        z_mask = batch.z_target_mask.bool()
        q_active = q_mask.any(dim=1)
        _, _, slope_mask = self._slope(batch.z_target, batch.z_target, z_mask)
        q0_available = batch.q_mask[:, -1].bool()
        consistency_mask = z_mask & q0_available.unsqueeze(1)
        return {
            "q_point": int(q_mask.sum().item()),
            "q_peak": int(q_active.sum().item()),
            "q_volume": int(q_active.sum().item()),
            "z_level": int(z_mask.sum().item()),
            "z_slope": int(slope_mask.sum().item()),
            "qz_consistency": int(consistency_mask.sum().item()),
        }

    def batch_statistics(
        self, output: Mapping[str, Any], batch: Any
    ) -> dict[str, LossTerm]:
        q_prediction = output["q"]
        z_prediction = output["z"]
        q_target = batch.q_target
        z_target = batch.z_target
        q_mask = self._checked(q_prediction, q_target, batch.q_target_mask, "Q")
        z_mask = self._checked(z_prediction, z_target, batch.z_target_mask, "Delta-Z")
        q_scale, z_scale = self._scales(batch, q_prediction)

        q_error = (q_prediction - q_target) / q_scale
        q_point_sum, q_point_count = masked_huber_stats(
            q_error, torch.zeros_like(q_error), q_mask
        )

        batch_size, _, obs = q_prediction.shape
        q_active = q_mask.any(dim=1)
        negative_inf = torch.full_like(q_prediction, float("-inf"))
        peak_prediction = torch.where(q_mask, q_prediction, negative_inf).amax(dim=1)
        peak_target = torch.where(q_mask, q_target, negative_inf).amax(dim=1)
        peak_prediction = torch.where(
            q_active, peak_prediction, torch.zeros_like(peak_prediction)
        )
        peak_target = torch.where(q_active, peak_target, torch.zeros_like(peak_target))
        q_scale_bo = q_scale.squeeze(1).expand(batch_size, obs)
        peak_error = (peak_prediction - peak_target) / q_scale_bo
        q_peak_sum = peak_error[q_active].square().sum()
        q_peak_count = int(q_active.sum().item())

        valid_count = q_mask.sum(dim=1).clamp_min(1)
        safe_prediction = torch.where(q_mask, q_prediction, torch.zeros_like(q_prediction))
        safe_target = torch.where(q_mask, q_target, torch.zeros_like(q_target))
        mean_prediction = safe_prediction.sum(dim=1) / valid_count
        mean_target = safe_target.sum(dim=1) / valid_count
        volume_error = (mean_prediction - mean_target) / q_scale_bo
        q_volume_sum = volume_error[q_active].square().sum()
        q_volume_count = q_peak_count

        z_error = (z_prediction - z_target) / z_scale
        z_level_sum, z_level_count = masked_huber_stats(
            z_error, torch.zeros_like(z_error), z_mask
        )
        slope_prediction, slope_target, slope_mask = self._slope(
            z_prediction, z_target, z_mask
        )
        slope_error = (slope_prediction - slope_target) / z_scale
        z_slope_sum, z_slope_count = masked_huber_stats(
            slope_error, torch.zeros_like(slope_error), slope_mask
        )

        diagnostics = output.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("v8 Q-Z consistency要求model diagnostics")
        consistency_target = diagnostics.get("qz_consistency_delta_z_m")
        consistency_available = diagnostics.get("qz_consistency_available")
        if not isinstance(consistency_target, torch.Tensor):
            raise ValueError("缺少qz_consistency_delta_z_m")
        if consistency_target.shape != z_prediction.shape:
            raise ValueError("Q-Z consistency target必须与Z预测同形状")
        if not isinstance(consistency_available, torch.Tensor):
            raise ValueError("缺少qz_consistency_available")
        if consistency_available.shape != (batch_size, obs):
            raise ValueError("qz_consistency_available必须为[B,Nobs]")
        consistency_mask = z_mask & consistency_available.bool().unsqueeze(1)
        consistency_error = (z_prediction - consistency_target) / z_scale
        qz_sum, qz_count = masked_huber_stats(
            consistency_error, torch.zeros_like(consistency_error), consistency_mask
        )

        return {
            "q_point": LossTerm(q_point_sum, q_point_count),
            "q_peak": LossTerm(q_peak_sum, q_peak_count),
            "q_volume": LossTerm(q_volume_sum, q_volume_count),
            "z_level": LossTerm(z_level_sum, z_level_count),
            "z_slope": LossTerm(z_slope_sum, z_slope_count),
            "qz_consistency": LossTerm(qz_sum, qz_count),
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
                total = total + coefficients[name] * statistics[name].numerator / denominator
                used = used or coefficients[name] > 0
        if not used:
            raise ValueError("当前batch/group没有有效Q/Z监督")
        return total

    def report(
        self,
        totals: Mapping[str, tuple[float, int]],
        *,
        q_valid_count: int,
        z_valid_count: int,
    ) -> dict[str, float | int]:
        means = {
            name: (float(value) / int(count) if int(count) else float("nan"))
            for name, (value, count) in totals.items()
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
        for name, key in (
            ("z_level", "z_level_weight"),
            ("z_slope", "z_slope_weight"),
        ):
            if totals[name][1]:
                z_total += float(loss[key]) * means[name]
        qz = (
            self.QZ_CONSISTENCY_WEIGHT * means["qz_consistency"]
            if totals["qz_consistency"][1]
            else 0.0
        )
        total = (
            float(loss["discharge_weight"]) * q_total
            + float(loss["water_level_weight"]) * z_total
            + qz
        )
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
            "qz_consistency_loss": means["qz_consistency"],
            "qz_consistency_weight": self.QZ_CONSISTENCY_WEIGHT,
            "q_valid_count": int(q_valid_count),
            "z_valid_count": int(z_valid_count),
            "z_slope_valid_count": int(totals["z_slope"][1]),
            "qz_consistency_valid_count": int(totals["qz_consistency"][1]),
        }
