"""Formal single-graph P3 training with read-only state diagnostics.

The optimization path is production code: setup_training() + Trainer.fit().
The callback only performs eval/no-grad observation on a deterministic
VALIDATION prefix. No diagnostic quantity enters the loss or forward dynamics.
After training, the best VALIDATION checkpoint is evaluated once on TEST.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import setup_evaluation, setup_training, validate_checkpoint_config  # noqa: E402
from trainers import Trainer  # noqa: E402


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1).cpu()
    y = y.detach().float().reshape(-1).cpu()
    if x.numel() != y.numel():
        raise ValueError("correlation两侧样本数不一致")
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(x.square().sum() * y.square().sum())
    if float(denom) <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def _stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().reshape(-1).cpu()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {key: float("nan") for key in ("mean", "std", "min", "max")}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _outlet_index(batch: Any) -> int:
    supervision = batch.q_target_mask.any(dim=(0, 1)) | batch.z_target_mask.any(
        dim=(0, 1)
    )
    indices = torch.nonzero(supervision, as_tuple=False).flatten()
    if indices.numel() != 1:
        raise ValueError(
            f"单graph state audit要求唯一监督出口节点，实际={indices.tolist()}"
        )
    return int(indices.item())


@torch.no_grad()
def state_audit(
    trainer: Trainer,
    loader: Any,
    *,
    epoch: int,
    max_batches: int,
) -> dict[str, Any]:
    """Observe Q(t0) -> initialized storage -> H1 release on VALIDATION."""
    model = trainer.model
    was_training = model.training
    model.eval()

    q_t0_values: list[torch.Tensor] = []
    sf0_values: list[torch.Tensor] = []
    ss0_values: list[torch.Tensor] = []
    st0_values: list[torch.Tensor] = []
    h0_norm_values: list[torch.Tensor] = []
    c0_norm_values: list[torch.Tensor] = []
    kf_h1_values: list[torch.Tensor] = []
    ks_h1_values: list[torch.Tensor] = []
    qlat_h1_values: list[torch.Tensor] = []
    qpred_t0mask_values: list[torch.Tensor] = []
    qobs_h1_values: list[torch.Tensor] = []
    qpred_h1_values: list[torch.Tensor] = []

    batches = 0
    for cpu_batch in loader:
        if batches >= max_batches:
            break
        batch = cpu_batch.to(trainer.device)
        output = model(batch)
        diagnostics = output["diagnostics"]
        outlet = _outlet_index(batch)

        required = (
            "initial_storage_fast_mm",
            "initial_storage_slow_mm",
            "initial_runoff_hidden_h",
            "initial_runoff_hidden_c",
            "k_fast",
            "k_slow",
        )
        missing = [name for name in required if name not in diagnostics]
        if missing:
            raise KeyError(f"state audit缺少diagnostics字段: {missing}")

        sf0 = diagnostics["initial_storage_fast_mm"][:, outlet]
        ss0 = diagnostics["initial_storage_slow_mm"][:, outlet]
        h0 = diagnostics["initial_runoff_hidden_h"][:, outlet]
        c0 = diagnostics["initial_runoff_hidden_c"][:, outlet]
        kf_h1 = diagnostics["k_fast"][:, 0, outlet]
        ks_h1 = diagnostics["k_slow"][:, 0, outlet]
        qlat_h1 = output["q_lat"][:, 0, outlet]
        qpred_h1 = output["q"][:, 0, outlet]

        q_t0_mask = batch.q_mask[:, -1, outlet].bool()
        if q_t0_mask.any():
            q_t0_values.append(batch.q_history[:, -1, outlet][q_t0_mask].detach().cpu())
            sf0_values.append(sf0[q_t0_mask].detach().cpu())
            ss0_values.append(ss0[q_t0_mask].detach().cpu())
            st0_values.append((sf0 + ss0)[q_t0_mask].detach().cpu())
            h0_norm_values.append(h0.norm(dim=-1)[q_t0_mask].detach().cpu())
            c0_norm_values.append(c0.norm(dim=-1)[q_t0_mask].detach().cpu())
            kf_h1_values.append(kf_h1[q_t0_mask].detach().cpu())
            ks_h1_values.append(ks_h1[q_t0_mask].detach().cpu())
            qlat_h1_values.append(qlat_h1[q_t0_mask].detach().cpu())
            qpred_t0mask_values.append(qpred_h1[q_t0_mask].detach().cpu())

        h1_mask = batch.q_target_mask[:, 0, outlet].bool()
        if h1_mask.any():
            qobs_h1_values.append(
                batch.q_target[:, 0, outlet][h1_mask].detach().cpu()
            )
            qpred_h1_values.append(qpred_h1[h1_mask].detach().cpu())
        batches += 1

    if was_training:
        model.train()
    if not q_t0_values:
        raise ValueError("state audit固定VALIDATION前缀没有有效Q(t0)")

    q_t0 = torch.cat(q_t0_values)
    sf0 = torch.cat(sf0_values)
    ss0 = torch.cat(ss0_values)
    st0 = torch.cat(st0_values)
    h0_norm = torch.cat(h0_norm_values)
    c0_norm = torch.cat(c0_norm_values)
    kf_h1 = torch.cat(kf_h1_values)
    ks_h1 = torch.cat(ks_h1_values)
    qlat_h1 = torch.cat(qlat_h1_values)
    qpred_t0mask = torch.cat(qpred_t0mask_values)

    result: dict[str, Any] = {
        "epoch": epoch,
        "audit_batches": batches,
        "q_t0_count": int(q_t0.numel()),
        "q_t0_mean_m3s": _stats(q_t0)["mean"],
        "initial_fast_storage_mean_mm": _stats(sf0)["mean"],
        "initial_slow_storage_mean_mm": _stats(ss0)["mean"],
        "initial_total_storage_mean_mm": _stats(st0)["mean"],
        "initial_total_storage_std_mm": _stats(st0)["std"],
        "h0_norm_mean": _stats(h0_norm)["mean"],
        "c0_norm_mean": _stats(c0_norm)["mean"],
        "k_fast_h1_mean": _stats(kf_h1)["mean"],
        "k_slow_h1_mean": _stats(ks_h1)["mean"],
        "q_lateral_h1_mean_m3s": _stats(qlat_h1)["mean"],
        "q_pred_h1_mean_m3s": _stats(qpred_t0mask)["mean"],
        "corr_q_t0_initial_total_storage": _corr(q_t0, st0),
        "corr_q_t0_initial_fast_storage": _corr(q_t0, sf0),
        "corr_q_t0_initial_slow_storage": _corr(q_t0, ss0),
        "corr_q_t0_q_lateral_h1": _corr(q_t0, qlat_h1),
        "corr_q_t0_q_pred_h1": _corr(q_t0, qpred_t0mask),
    }

    if qobs_h1_values:
        qobs_h1 = torch.cat(qobs_h1_values)
        qpred_h1 = torch.cat(qpred_h1_values)
        qerr_h1 = qpred_h1 - qobs_h1
        result.update(
            {
                "q_obs_h1_count": int(qobs_h1.numel()),
                "q_obs_h1_mean_m3s": _stats(qobs_h1)["mean"],
                "q_pred_valid_h1_mean_m3s": _stats(qpred_h1)["mean"],
                "q_h1_bias_m3s": _stats(qerr_h1)["mean"],
                "q_h1_mae_m3s": float(qerr_h1.abs().mean()),
                "q_h1_rmse_m3s": float(torch.sqrt(qerr_h1.square().mean())),
                "corr_q_obs_h1_q_pred_h1": _corr(qobs_h1, qpred_h1),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="正式单graph P3连续Q+Z训练，并旁路审计forecast-origin state"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "hunan_p3_state_init_single_q611e0340.yaml"),
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default="Q_611E0340")
    parser.add_argument("--state-audit-every", type=int, default=10)
    parser.add_argument("--state-audit-batches", type=int, default=8)
    parser.add_argument(
        "--state-audit-csv",
        default=str(PROJECT_ROOT / "outputs" / "hunan_p3_state_init_single_q611e0340_state_audit.csv"),
    )
    parser.add_argument(
        "--test-json",
        default=str(PROJECT_ROOT / "outputs" / "hunan_p3_state_init_single_q611e0340_test_metrics.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.state_audit_every <= 0 or args.state_audit_batches <= 0:
        parser.error("state audit cadence/batches必须>0")

    audit_path = Path(args.state_audit_csv).expanduser().resolve()
    test_path = Path(args.test_json).expanduser().resolve()
    if not args.overwrite:
        existing = [path for path in (audit_path, test_path) if path.exists()]
        if existing:
            raise FileExistsError(
                f"state/test输出已存在，请换路径或加--overwrite: {existing}"
            )
    else:
        audit_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)

    cfg, model, train_loader, validation_loader, device = setup_training(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    if not bool(cfg.get("state_initialization", {}).get("enabled", False)):
        raise ValueError("该实验要求P3 state_initialization.enabled=true")

    print(
        json.dumps(
            {
                "experiment": "formal_single_graph_p3_state_probe",
                "graph_id": args.graph_id,
                "dataset_root": cfg["data"]["dataset_root"],
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "target_variable": cfg["data"]["target_variable"],
                "train_weighted_sampling": bool(cfg["train_sampling"]["enabled"]),
                "epochs": int(cfg["training"]["epochs"]),
                "batch_size": int(cfg["batch_size"]),
                "state_audit_every": args.state_audit_every,
                "state_audit_batches": args.state_audit_batches,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    trainer = Trainer(model, cfg, device)

    def callback(epoch: int, row: dict[str, float | int]) -> None:
        if (
            epoch % args.state_audit_every != 0
            and epoch != int(cfg["training"]["epochs"]) - 1
        ):
            return
        audit = state_audit(
            trainer,
            validation_loader,
            epoch=epoch,
            max_batches=args.state_audit_batches,
        )
        print("STATE_AUDIT", json.dumps(audit, ensure_ascii=False, allow_nan=True))
        _append_csv(audit_path, audit)

    trainer.fit(
        train_loader,
        validation_loader,
        overwrite=args.overwrite,
        epoch_callback=callback,
    )

    best_path = Path(cfg["training"]["checkpoint"])
    if not best_path.is_file():
        raise FileNotFoundError(f"训练结束后缺少best checkpoint: {best_path}")

    test_cfg, test_model, test_loader, test_device = setup_evaluation(
        args.config,
        split="TEST",
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    test_trainer = Trainer(test_model, test_cfg, test_device)
    checkpoint = test_trainer.load_weights(best_path)
    validate_checkpoint_config(checkpoint, test_cfg, resume=False)
    test_metrics = test_trainer.evaluate(test_loader)
    payload = {
        "graph_id": args.graph_id,
        "checkpoint": str(best_path.resolve()),
        "test_samples": len(test_loader.dataset),
        "metrics": {
            key: (_finite(float(value)) if isinstance(value, (int, float)) else value)
            for key, value in test_metrics.items()
            if not key.startswith("_")
        },
    }
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("TEST_RESULT", json.dumps(payload, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
