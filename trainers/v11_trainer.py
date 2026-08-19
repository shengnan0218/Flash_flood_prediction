"""Trainer adapter for formal V11 event-balanced Q-only learning."""
from __future__ import annotations

import torch

from losses.hydrologic_graph_v11_loss import HydrologicGraphV11Loss
from trainers.v10_trainer import V10Trainer


class V11Trainer(V10Trainer):
    """Reuse V10 fit/evaluation/checkpoint machinery with the V11 objective."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        # Do not call V10Trainer.__init__: its formal loss constructor requires
        # the retired V10 window-peak component. The generic optimizer/checkpoint
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
