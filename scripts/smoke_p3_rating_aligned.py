"""One-batch smoke test for the revised rating-aligned P3.

No optimizer step, checkpoint write, or training log write is performed.  The
backward pass uses only Z level/slope terms so nonzero runoff/routing gradients
specifically verify the new Z -> rating(Q) -> Q gradient bridge.
"""
from __future__ import annotations

import argparse
import json

import torch

from losses import FloodMultitaskLoss
from scripts.p3_rating_aligned_runtime import setup_training_rating_aligned


def _grad_norm(module: torch.nn.Module) -> tuple[float, int]:
    squared = 0.0
    nonzero_tensors = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        if bool((value != 0).any()):
            nonzero_tensors += 1
        squared += float(value.square().sum().item())
    return squared ** 0.5, nonzero_tensors


def main() -> None:
    parser = argparse.ArgumentParser(description="rating-aligned P3 one-batch smoke test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default=None)
    args = parser.parse_args()

    cfg, model, train_loader, _validation_loader, device = setup_training_rating_aligned(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    dataset = train_loader.dataset
    if getattr(dataset, "dynamic_normalization_mode", None) != "train_aligned":
        raise RuntimeError("TRAIN dataset未启用train_aligned normalization")
    stats = dataset.aligned_input_statistics()
    if not isinstance(stats, dict) or stats.get("computed_from_split") != "TRAIN":
        raise RuntimeError("aligned input statistics不是TRAIN-only")
    if model.rating_curve is None or not bool(model.rating_curve.available.any()):
        raise RuntimeError("没有配置任何可用TRAIN rating curve")

    batch = next(iter(train_loader)).to(device)
    model = model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)
    out = model(batch)
    diagnostics = out.get("diagnostics", {})
    if "rating_delta_z_m" not in diagnostics or "z_residual_delta_m" not in diagnostics:
        raise RuntimeError("forward未进入rating-backed Z路径")

    loss_engine = FloodMultitaskLoss(cfg)
    statistics = loss_engine.batch_statistics(out, batch)
    coefficients = loss_engine.coefficients()
    reference = out["z"].reshape(-1)[:0].sum()
    z_only_loss = reference * 0.0
    z_terms_used = 0
    for name in ("z_level", "z_slope"):
        term = statistics[name]
        if term.denominator and coefficients[name] > 0:
            z_only_loss = (
                z_only_loss
                + coefficients[name] * term.numerator / int(term.denominator)
            )
            z_terms_used += 1
    if z_terms_used == 0:
        raise RuntimeError("抽到的TRAIN batch没有有效Z监督，请重新运行smoke")
    if not torch.isfinite(z_only_loss):
        raise FloatingPointError("smoke Z-only loss出现NaN/Inf")
    z_only_loss.backward()

    runoff_norm, runoff_tensors = _grad_norm(model.runoff)
    routing_norm, routing_tensors = _grad_norm(model.routing)
    initializer_norm, initializer_tensors = _grad_norm(model.state_initializer)
    residual_norm, residual_tensors = _grad_norm(model.independent_z_head)
    if runoff_norm <= 0.0:
        raise RuntimeError(
            "Z-only backward没有到达runoff参数；rating(Q)梯度桥未生效"
        )

    rating_available = diagnostics["rating_available"].detach().bool()
    anchor_source = diagnostics["q_origin_anchor_source"].detach().cpu()
    report = {
        "status": "PASS",
        "device": str(device),
        "graph_id": cfg["data"].get("graph_id"),
        "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
        "p3_rating_aligned": bool(cfg["_runtime"].get("p3_rating_aligned")),
        "input_normalization_mode": dataset.dynamic_normalization_mode,
        "rating_available_fraction_in_batch": float(rating_available.float().mean().item()),
        "q_anchor_source_counts": {
            "learned_only": int((anchor_source == 0).sum().item()),
            "exact_q_t0": int((anchor_source == 1).sum().item()),
            "inverse_rating_from_z_t0": int((anchor_source == 2).sum().item()),
        },
        "z_only_loss": float(z_only_loss.detach().item()),
        "z_only_gradient_norms": {
            "runoff": {"l2": runoff_norm, "nonzero_tensors": runoff_tensors},
            "routing": {"l2": routing_norm, "nonzero_tensors": routing_tensors},
            "state_initializer": {
                "l2": initializer_norm,
                "nonzero_tensors": initializer_tensors,
            },
            "z_residual_head": {
                "l2": residual_norm,
                "nonzero_tensors": residual_tensors,
            },
        },
        "rating_fit": cfg["_runtime"]["rating_curves"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
