"""Trainer adapter for formal v10 Q-only learning."""
from __future__ import annotations

from typing import Any, Iterable

import torch

from losses.hydrologic_graph_v10_loss import HydrologicGraphV10Loss
from metrics.flood_metrics import masked_regression_sums, regression_metrics, valid_target_count
from trainers.v8_trainer import V8Trainer


class V10Trainer(V8Trainer):
    """Reuse mature fit/checkpoint machinery while evaluating Q only per epoch."""

    def __init__(self, model: torch.nn.Module, cfg: dict, device: torch.device) -> None:
        super().__init__(model, cfg, device)
        self.loss_engine = HydrologicGraphV10Loss(cfg)

    @staticmethod
    def _merge(target: dict[str, float | int], source: dict[str, float | int]) -> None:
        for key in target:
            target[key] = target[key] + source[key]

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Any],
        *,
        include_group_metrics: bool = False,
        include_group_details: bool = False,
        include_validation_diagnostics: bool = False,
        include_diagnostic_details: bool = False,
    ) -> dict[str, Any]:
        # V10 checkpoint selection is deliberately Q-only val_loss.  Detailed
        # Q/stage grouping is reserved for the final v10 evaluator.
        del (
            include_group_metrics,
            include_group_details,
            include_validation_diagnostics,
            include_diagnostic_details,
        )
        self.model.eval()
        loss_totals = {name: [0.0, 0] for name in self.loss_engine.coefficients()}
        q_valid_total = 0
        regression = {
            "count": 0,
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "error": 0.0,
            "prediction": 0.0,
            "target": 0.0,
            "prediction_squared": 0.0,
            "target_squared": 0.0,
            "cross": 0.0,
        }
        batch_count = 0
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            statistics = self.loss_engine.batch_statistics(output, batch)
            for name, term in statistics.items():
                loss_totals[name][0] += float(term.numerator.detach().item())
                loss_totals[name][1] += int(term.denominator)
            q_valid_total += valid_target_count(batch.q_target, batch.q_target_mask)
            self._merge(
                regression,
                masked_regression_sums(output["q"], batch.q_target, batch.q_target_mask),
            )
            batch_count += 1
        if batch_count == 0:
            raise ValueError("v10 evaluation DataLoader为空")
        result = self.loss_engine.report(
            {name: (float(value), int(count)) for name, (value, count) in loss_totals.items()},
            q_valid_count=q_valid_total,
            z_valid_count=0,
        )
        q_metrics = regression_metrics(regression)
        result.update(
            {
                "q_mae": float(q_metrics["mae"]),
                "q_rmse": float(q_metrics["rmse"]),
                "q_nse": float(q_metrics["nse"]),
                "q_kge": float(q_metrics["kge"]),
                "q_valid_count": int(q_metrics["valid_count"]),
            }
        )
        return result
