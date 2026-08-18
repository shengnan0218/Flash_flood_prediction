"""Trainer adapter for the v9 warm-up and explicit-state Delta-Z objective."""
from __future__ import annotations

from typing import Any, Iterable

import torch

from losses.hydrologic_graph_v9_loss import HydrologicGraphV9Loss
from trainers.v8_trainer import V8Trainer


class V9Trainer(V8Trainer):
    """Reuse v8 fit/checkpoint machinery with the v9 loss engine.

    The legacy event-validation diagnostics expect the old station-as-node batch
    metadata.  v9 uses Nnode/Nobs separation and has a dedicated station-aware
    evaluator, so per-epoch validation intentionally keeps only global physical
    metrics and val_loss.
    """

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        super().__init__(model, cfg, device)
        self.loss_engine = HydrologicGraphV9Loss(cfg)

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
        del include_validation_diagnostics, include_diagnostic_details
        return super().evaluate(
            loader,
            include_group_metrics=include_group_metrics,
            include_group_details=include_group_details,
            include_validation_diagnostics=False,
            include_diagnostic_details=False,
        )
