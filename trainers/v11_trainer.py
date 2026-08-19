"""Trainer adapter for formal V11 event-balanced Q-only learning."""
from __future__ import annotations

from typing import Any, Iterable

import torch

from losses.hydrologic_graph_v11_loss import HydrologicGraphV11Loss
from metrics.flood_metrics import (
    masked_regression_sums,
    regression_metrics,
    valid_target_count,
)
from trainers.v10_trainer import V10Trainer


def _empty_regression() -> dict[str, float | int]:
    return {
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


def _merge(
    target: dict[str, float | int], source: dict[str, float | int]
) -> None:
    for key in target:
        target[key] = target[key] + source[key]


class V11Trainer(V10Trainer):
    """V10 fit/checkpoint mechanics with V11 objective and generalization logging."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        # Do not call V10Trainer.__init__: its formal loss constructor requires
        # the retired V10 window-peak component. Generic optimizer/checkpoint
        # state is identical and is initialized explicitly here.
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        opt = cfg["optimizer"]
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("v11模型没有可训练参数")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(opt["lr"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.loss_engine = HydrologicGraphV11Loss(cfg)
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.selection_mode = str(
            cfg.get("validation_selection", {}).get("mode", "val_loss")
        )
        if self.selection_mode != "val_loss":
            raise ValueError("v11固定使用Q-only val_loss checkpoint selection")
        self.best = float("inf")
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None

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
        """Evaluate loss/Q plus persistence and Delta-Q in the same forward pass.

        Extra metrics are diagnostics only. Checkpoint selection remains V11
        validation loss; persistence/Delta-Q never contribute gradient.
        """
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
        all_q = _empty_regression()
        model_q0_subset = _empty_regression()
        persistence = _empty_regression()
        delta_q = _empty_regression()
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
            _merge(
                all_q,
                masked_regression_sums(
                    output["q"], batch.q_target, batch.q_target_mask
                ),
            )

            q0_available = batch.q_mask[:, -1].bool()
            q0_mask = batch.q_target_mask.bool() & q0_available.unsqueeze(1)
            q0 = batch.q_history[:, -1].unsqueeze(1).expand_as(batch.q_target)
            _merge(
                model_q0_subset,
                masked_regression_sums(output["q"], batch.q_target, q0_mask),
            )
            _merge(
                persistence,
                masked_regression_sums(q0, batch.q_target, q0_mask),
            )
            _merge(
                delta_q,
                masked_regression_sums(
                    output["q"] - q0,
                    batch.q_target - q0,
                    q0_mask,
                ),
            )
            batch_count += 1

        if batch_count == 0:
            raise ValueError("v11 evaluation DataLoader为空")
        result = self.loss_engine.report(
            {
                name: (float(value), int(count))
                for name, (value, count) in loss_totals.items()
            },
            q_valid_count=q_valid_total,
            z_valid_count=0,
        )
        q_metrics = regression_metrics(all_q)
        model_q0_metrics = regression_metrics(model_q0_subset)
        persistence_metrics = regression_metrics(persistence)
        delta_metrics = regression_metrics(delta_q)
        model_sse = float(model_q0_subset["squared_error"])
        persistence_sse = float(persistence["squared_error"])
        skill = (
            1.0 - model_sse / persistence_sse
            if int(persistence["count"]) > 0 and persistence_sse > 0.0
            else float("nan")
        )
        result.update(
            {
                "q_mae": float(q_metrics["mae"]),
                "q_rmse": float(q_metrics["rmse"]),
                "q_nse": float(q_metrics["nse"]),
                "q_kge": float(q_metrics["kge"]),
                "q_valid_count": int(q_metrics["valid_count"]),
                "q0_observed_valid_count": int(model_q0_metrics["valid_count"]),
                "q0_subset_model_nse": float(model_q0_metrics["nse"]),
                "q0_persistence_nse": float(persistence_metrics["nse"]),
                "q_skill_over_persistence": float(skill),
                "delta_q_rmse": float(delta_metrics["rmse"]),
                "delta_q_nse": float(delta_metrics["nse"]),
            }
        )
        return result
