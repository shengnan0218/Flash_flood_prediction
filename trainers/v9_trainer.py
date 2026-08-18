"""Trainer adapter for the v9 warm-up and explicit-state Delta-Z objective."""
from __future__ import annotations

import torch

from losses.hydrologic_graph_v9_loss import HydrologicGraphV9Loss
from trainers.v8_trainer import V8Trainer


class V9Trainer(V8Trainer):
    """Reuse v8 fit/evaluation/checkpoint machinery with the v9 loss engine."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        super().__init__(model, cfg, device)
        self.loss_engine = HydrologicGraphV9Loss(cfg)
