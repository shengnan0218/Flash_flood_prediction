"""Trainer adapter for formal V10 Q-only learning."""
from __future__ import annotations

from typing import Any, Iterable

import torch

from losses.hydrologic_graph_v10_loss import HydrologicGraphV10Loss
from metrics.flood_metrics import (
    masked_regression_sums,
    regression_metrics,
    valid_target_count,
)
from trainers.v8_trainer import V8Trainer


class V10Trainer(V8Trainer):
    """Reuse mature V8 fit/checkpoint methods without constructing its Z loss."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        # Do NOT call V8Trainer.__init__.  That constructor intentionally
        # instantiates the frozen V8 Q+Z objective and therefore rejects the
        # formal V10 Q-only config before V10 can replace the loss.  Reproduce
        # only the generic optimiser/checkpoint state here; inherited train/fit/
        # checkpoint methods remain reused unchanged.
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        opt = cfg["optimizer"]
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("v10模型没有可训练参数")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(opt["lr"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.loss_engine = HydrologicGraphV10Loss(cfg)
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.selection_mode = str(
            cfg.get("validation_selection", {}).get("mode", "val_loss")
        )
        if self.selection_mode != "val_loss":
            raise ValueError("v10固定使用Q-only val_loss checkpoint selection")
        self.best = float("inf")
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None

    @staticmethod
    def _merge(
        target: dict[str, float | int], source: dict[str, float | int]
    ) -> None:
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
        # Q/stage grouping is reserved for the final V10 evaluator.
        del (
            include_group_metrics,
            include_group_details,
            include_validation_diagnostics,
            include_diagnostic_details,
        )
        self.model.eval()
        loss_totals = {
            name: [0.0, 0] for name in self.loss_engine.coefficients()
        }
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
            q_valid_total += valid_target_count(
                batch.q_target, batch.q_target_mask
            )
            self._merge(
                regression,
                masked_regression_sums(
                    output["q"], batch.q_target, batch.q_target_mask
                ),
            )
            batch_count += 1
        if batch_count == 0:
            raise ValueError("v10 evaluation DataLoader为空")
        result = self.loss_engine.report(
            {
                name: (float(value), int(count))
                for name, (value, count) in loss_totals.items()
            },
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
