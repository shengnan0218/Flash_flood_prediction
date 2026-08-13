"""Diagnose P3 delta-Z collapse without modifying training state."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable

# Allow direct execution from the repository root via
# ``python scripts/diagnose_p3_z.py ...``.  When a file inside ``scripts/`` is
# executed directly, Python otherwise places only that directory at sys.path[0]
# and cannot resolve top-level project packages such as ``scripts``/``trainers``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.common import setup_training
from trainers import Trainer


def _masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    return value[valid].detach().float().cpu()


def _summary(value: torch.Tensor) -> dict[str, float | int]:
    value = value.float().reshape(-1)
    if not value.numel():
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(a.square().sum() * b.square().sum())
    if not torch.isfinite(denom) or float(denom) <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def _concat(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts) if parts else torch.empty(0)


def _collect_validation(
    trainer: Trainer,
    loader: Iterable,
    max_batches: int,
) -> dict[str, object]:
    model = trainer.model
    model.eval()
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    physical: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []
    q_predictions: list[torch.Tensor] = []
    reference_total = 0
    reference_valid = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            if batches >= max_batches:
                break
            batch = batch.to(trainer.device)
            out = model(batch)
            diagnostics = out.get("diagnostics", {})
            if "physical_delta_z_m" not in diagnostics:
                raise RuntimeError("checkpoint/model没有P3 physical_delta_z_m diagnostics")
            if "stage_memory_residual_m" not in diagnostics:
                raise RuntimeError("checkpoint/model没有P3 stage_memory_residual_m diagnostics")
            mask = batch.z_target_mask.bool()
            targets.append(_masked_values(batch.z_target, mask))
            predictions.append(_masked_values(out["z"], mask))
            physical.append(_masked_values(diagnostics["physical_delta_z_m"], mask))
            residuals.append(_masked_values(diagnostics["stage_memory_residual_m"], mask))
            q_predictions.append(_masked_values(out["q"], batch.q_target_mask.bool()))
            if batch.z_reference_mask is not None:
                reference_total += int(batch.z_reference_mask.numel())
                reference_valid += int(batch.z_reference_mask.sum().item())
            batches += 1
    target = _concat(targets)
    prediction = _concat(predictions)
    physical_delta = _concat(physical)
    residual = _concat(residuals)
    q_prediction = _concat(q_predictions)
    if target.numel() != prediction.numel() or target.numel() != physical_delta.numel() or target.numel() != residual.numel():
        raise RuntimeError("Z诊断张量有效元素数量不一致")
    error = prediction - target
    return {
        "batches": batches,
        "z_reference_valid_fraction": (
            reference_valid / reference_total if reference_total else float("nan")
        ),
        "z_target": _summary(target),
        "z_prediction": _summary(prediction),
        "physical_delta": _summary(physical_delta),
        "stage_residual": _summary(residual),
        "z_error": _summary(error),
        "z_mae_m": float(error.abs().mean()) if error.numel() else float("nan"),
        "corr_prediction_target": _corr(prediction, target),
        "corr_physical_target": _corr(physical_delta, target),
        "corr_residual_target": _corr(residual, target),
        "corr_residual_target_minus_physical": _corr(residual, target - physical_delta),
        "q_prediction": _summary(q_prediction),
    }


def _gradient_diagnostic(trainer: Trainer, loader: Iterable) -> dict[str, object]:
    model = trainer.model
    model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    batch = next(iter(loader)).to(trainer.device)
    out = model(batch)
    loss, parts = trainer._loss(out, batch)
    loss.backward()

    def group_stats(prefix: str) -> dict[str, float | int]:
        grad_sq = 0.0
        param_sq = 0.0
        grad_params = 0
        params = 0
        for name, parameter in model.named_parameters():
            if not name.startswith(prefix):
                continue
            params += 1
            param_sq += float(parameter.detach().float().square().sum().cpu())
            if parameter.grad is not None:
                grad = parameter.grad.detach().float()
                grad_sq += float(grad.square().sum().cpu())
                if float(grad.abs().max().cpu()) > 0:
                    grad_params += 1
        return {
            "parameter_tensors": params,
            "nonzero_gradient_tensors": grad_params,
            "parameter_l2": math.sqrt(param_sq),
            "gradient_l2": math.sqrt(grad_sq),
        }

    result = {
        "loss": float(loss.detach().cpu()),
        "parts": parts,
        "stage_residual_head": group_stats("stage_residual_head"),
        "observation": group_stats("observation"),
        "state_initializer": group_stats("state_initializer"),
        "runoff": group_stats("runoff"),
    }
    trainer.optimizer.zero_grad(set_to_none=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="只读诊断P3 delta-Z塌缩")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-batches", type=int, default=64)
    args = parser.parse_args()

    cfg, model, train_loader, validation_loader, device = setup_training(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    trainer = Trainer(model, cfg, device)
    checkpoint_path = Path(args.checkpoint or cfg["training"]["checkpoint"])
    checkpoint = trainer.load_weights(checkpoint_path, strict=True)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "validation": _collect_validation(trainer, validation_loader, args.max_batches),
        "gradient": _gradient_diagnostic(trainer, train_loader),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
