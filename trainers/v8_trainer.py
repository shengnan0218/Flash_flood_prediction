"""Trainer adapter for the v8 sparse-observation objective."""
from __future__ import annotations

import torch

from losses.hydrologic_graph_v8_loss import HydrologicGraphV8Loss
from trainers.trainer import Trainer


class V8Trainer(Trainer):
    """Reuse checkpoint/evaluation/fit machinery with the v8 loss engine."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        # Deliberately do not call Trainer.__init__: the legacy loss constructor
        # does not understand v8 per-station Q normalization.
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        opt = cfg["optimizer"]
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("模型没有可训练参数")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=opt["lr"], weight_decay=opt["weight_decay"]
        )
        self.loss_engine = HydrologicGraphV8Loss(cfg)
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.selection_mode = str(
            cfg.get("validation_selection", {}).get("mode", "val_loss")
        )
        if self.selection_mode != "val_loss":
            raise ValueError(
                "v8当前正式训练固定使用val_loss checkpoint selection；"
                "最终论文指标由独立TEST评价给出"
            )
        self.best = float("inf")
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None
