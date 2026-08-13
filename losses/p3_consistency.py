"""P3 independent-Z loss adapter with a weak Q-Z consistency constraint."""
from __future__ import annotations

from typing import Any, Mapping

import torch

from metrics.flood_metrics import masked_huber_stats
from .flood_multitask_loss import FloodMultitaskLoss as _BaseFloodMultitaskLoss
from .flood_multitask_loss import LossTerm


class FloodMultitaskLoss(_BaseFloodMultitaskLoss):
    """Keep legacy P3 consistency while allowing rating-backed P3 to replace it.

    Historical P3 independent-Z experiments use a fixed 0.1 Huber consistency
    penalty between the independent delta-Z trajectory and the learned monotone
    hydraulic relation.  The revised ``p3_rating_aligned`` runtime deliberately
    disables that legacy term because Q->Z is now the primary differentiable
    prediction backbone itself; retaining an additional weak consistency target
    would double-count a superseded relation and change the agreed objective.
    """

    P3_QZ_CONSISTENCY_WEIGHT = 0.1
    _BASE_COMPONENTS = ("q_point", "q_peak", "q_volume", "z_level", "z_slope")

    def _p3_consistency_enabled(self) -> bool:
        state = self.cfg.get("state_initialization", {})
        runtime = self.cfg.get("_runtime", {})
        rating_aligned = bool(
            isinstance(runtime, Mapping) and runtime.get("p3_rating_aligned", False)
        )
        return bool(
            self.mode == "multitask"
            and self.z_target_mode == "delta_from_t0"
            and isinstance(state, Mapping)
            and state.get("enabled", False)
            and not rating_aligned
        )

    def coefficients(self) -> dict[str, float]:
        result = super().coefficients()
        if self._p3_consistency_enabled():
            result["qz_consistency"] = self.P3_QZ_CONSISTENCY_WEIGHT
        return result

    def denominators(self, batch: Any) -> dict[str, int]:
        result = super().denominators(batch)
        if self._p3_consistency_enabled():
            result["qz_consistency"] = int(batch.z_target_mask.bool().sum().item())
        return result

    def batch_statistics(
        self, output: Mapping[str, Any], batch: Any
    ) -> dict[str, LossTerm]:
        result = super().batch_statistics(output, batch)
        if not self._p3_consistency_enabled():
            return result
        diagnostics = output.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("P3 Q-Z consistency要求model diagnostics")
        physical_delta = diagnostics.get("qz_consistency_delta_z_m")
        if not isinstance(physical_delta, torch.Tensor):
            raise ValueError(
                "P3 Q-Z consistency缺少qz_consistency_delta_z_m"
            )
        z_prediction = output["z"]
        if physical_delta.shape != z_prediction.shape:
            raise ValueError("Q-Z consistency target必须与独立Z预测同形状")
        z_mask = batch.z_target_mask.bool()
        batch_size = z_prediction.shape[0]
        z_scale = self.z_scale_for_batch(batch, batch_size)
        numerator, count = masked_huber_stats(
            z_prediction / z_scale,
            physical_delta / z_scale,
            z_mask,
        )
        result["qz_consistency"] = LossTerm(numerator, count)
        return result

    def combine(
        self,
        statistics: Mapping[str, LossTerm],
        denominators: Mapping[str, int] | None = None,
    ) -> torch.Tensor:
        base_statistics = {name: statistics[name] for name in self._BASE_COMPONENTS}
        base_denominators = (
            None
            if denominators is None
            else {name: denominators[name] for name in self._BASE_COMPONENTS}
        )
        total = super().combine(base_statistics, base_denominators)
        if not self._p3_consistency_enabled():
            return total
        term = statistics["qz_consistency"]
        denominator = int(
            term.denominator
            if denominators is None
            else denominators["qz_consistency"]
        )
        if denominator:
            total = total + self.P3_QZ_CONSISTENCY_WEIGHT * term.numerator / denominator
        return total

    def report(
        self,
        totals: Mapping[str, tuple[float, int]],
        *,
        q_valid_count: int,
        z_valid_count: int,
    ) -> dict[str, float | int]:
        base_totals = {name: totals[name] for name in self._BASE_COMPONENTS}
        report = super().report(
            base_totals,
            q_valid_count=q_valid_count,
            z_valid_count=z_valid_count,
        )
        if not self._p3_consistency_enabled():
            return report
        numerator, count = totals["qz_consistency"]
        consistency = float(numerator) / int(count) if count else float("nan")
        contribution = (
            self.P3_QZ_CONSISTENCY_WEIGHT * consistency if count else 0.0
        )
        report["qz_consistency_loss"] = consistency
        report["qz_consistency_weight"] = self.P3_QZ_CONSISTENCY_WEIGHT
        report["qz_consistency_valid_count"] = int(count)
        report["loss"] = float(report["loss"]) + contribution
        if "total_loss" in report:
            report["total_loss"] = float(report["total_loss"]) + contribution
        return report
