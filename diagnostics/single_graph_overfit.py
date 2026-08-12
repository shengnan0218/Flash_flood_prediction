"""Single-graph Q-only overfit diagnostic for the Hunan P2 continuous dataset.

This is a memorization test, not a formal experiment. It keeps the current model
forward path and Q loss definition, but:
- uses exactly one GRAPH_ID;
- selects a fixed, response-rich subset of TRAIN windows;
- disables weighted sampling;
- removes all Z supervision by zeroing z_target_mask;
- evaluates on the same fixed TRAIN subset after every epoch.

A successful run should be able to drive TRAIN Q NSE very close to 1. Failure to
memorize this tiny subset is evidence to inspect the Q forward/loss/routing path
before further tuning the province-scale training formulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from data.device import resolve_device, seed_everything  # noqa: E402
from datasets import HunanContinuousDataset, collate_hunan_graph_events  # noqa: E402
from models import HybridFloodModel  # noqa: E402
from scripts.common import (  # noqa: E402
    _dataset_nodes,
    _runtime_config_from_mapping,
    _runtime_metadata,
)
from trainers import Trainer  # noqa: E402


class QOnlySubset(Dataset):
    """Expose fixed base-dataset indices while disabling every Z target."""

    def __init__(self, base: HunanContinuousDataset, indices: list[int]) -> None:
        if not indices:
            raise ValueError("overfit subset不能为空")
        self.base = base
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        batch = self.base[self.indices[index]]
        # Targets never enter model.forward; zeroing the mask is sufficient to
        # remove Z from loss/metrics. Zero the target too to make the diagnostic
        # contract explicit and robust against accidental downstream use.
        batch.z_target = torch.zeros_like(batch.z_target)
        batch.z_target_mask = torch.zeros_like(batch.z_target_mask, dtype=torch.bool)
        # Formal P2 sampling weights are not part of this memorization test.
        batch.sample_weight = torch.tensor(1.0, dtype=torch.float32)
        return batch


def _meta_item(value: Any, index: int) -> str:
    if isinstance(value, (tuple, list)):
        return str(value[index])
    return "" if value is None else str(value)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_or_none(item) for item in value]
    return value


def _build_base_dataset(cfg: dict[str, Any]) -> HunanContinuousDataset:
    data_cfg = cfg["data"]
    return HunanContinuousDataset(
        data_cfg["dataset_root"],
        "TRAIN",
        history_hours=cfg["history_length"],
        forecast_hours=cfg["forecast_horizon"],
        graph_id=data_cfg["graph_id"],
        normalize_dynamic=data_cfg["normalize_dynamic"],
        future_rainfall_mode=data_cfg["future_rainfall_mode"],
        use_observation_masks=data_cfg["use_observation_masks"],
        strict=data_cfg["strict_validation"],
    )


def _response_candidates(
    dataset: HunanContinuousDataset, min_valid_horizons: int
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        target_nodes = torch.nonzero(
            sample.q_target_mask.any(dim=0), as_tuple=False
        ).flatten()
        if target_nodes.numel() != 1:
            continue
        outlet = int(target_nodes.item())
        valid = sample.q_target_mask[:, outlet]
        valid_count = int(valid.sum().item())
        if valid_count < min_valid_horizons:
            continue
        if not bool(sample.q_mask[-1, outlet]):
            continue

        q_t0 = float(sample.q_history[-1, outlet].item())
        target = sample.q_target[:, outlet]
        valid_target = target[valid]
        q_min = float(valid_target.min().item())
        q_max = float(valid_target.max().item())
        max_delta = float((valid_target - q_t0).abs().max().item())
        target_range = q_max - q_min
        response_score = max(max_delta, target_range)

        candidates.append(
            {
                "dataset_index": dataset_index,
                "sample_id": _meta_item(sample.sample_id, 0),
                "forecast_time": _meta_item(sample.forecast_time, 0),
                "target_station_id": _meta_item(sample.target_station_id, 0),
                "q_t0_m3s": q_t0,
                "q_target_min_m3s": q_min,
                "q_target_max_m3s": q_max,
                "q_target_range_m3s": target_range,
                "q_max_abs_delta_from_t0_m3s": max_delta,
                "response_score": response_score,
                "valid_horizons": valid_count,
            }
        )
    candidates.sort(
        key=lambda row: (-float(row["response_score"]), int(row["dataset_index"]))
    )
    return candidates


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的记录: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_hunan_graph_events,
        generator=generator,
    )


@torch.no_grad()
def _export_q_predictions(
    trainer: Trainer, loader: DataLoader, path: Path
) -> None:
    trainer.model.eval()
    rows: list[dict[str, Any]] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(trainer.device)
        output = trainer.model(batch)
        prediction = output["q"].detach().cpu()
        target = cpu_batch.q_target
        mask = cpu_batch.q_target_mask
        batch_size = int(prediction.shape[0])
        for sample_index in range(batch_size):
            valid_positions = torch.nonzero(
                mask[sample_index], as_tuple=False
            )
            for horizon_index, node_index in valid_positions.tolist():
                rows.append(
                    {
                        "sample_id": _meta_item(cpu_batch.sample_id, sample_index),
                        "graph_id": _meta_item(cpu_batch.graph_id, sample_index),
                        "target_station_id": _meta_item(
                            cpu_batch.target_station_id, sample_index
                        ),
                        "forecast_time": _meta_item(
                            cpu_batch.forecast_time, sample_index
                        ),
                        "lead_hour": int(horizon_index) + 1,
                        "node_index": int(node_index),
                        "q_observed_m3s": float(
                            target[sample_index, horizon_index, node_index].item()
                        ),
                        "q_predicted_m3s": float(
                            prediction[sample_index, horizon_index, node_index].item()
                        ),
                    }
                )
    if rows:
        _write_rows(path, rows)


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"诊断输出目录已存在且非空: {path}. "
            "请换一个--output-dir，或确认后加--overwrite。"
        )
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="单graph固定TRAIN窗口Q-only过拟合诊断；不修改正式P2训练输出"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "hunan_p2_continuous_multitask.yaml"),
    )
    parser.add_argument(
        "--dataset-root",
        default=str(PROJECT_ROOT / "_model_dataset_v6_continuous_multitask"),
    )
    parser.add_argument("--graph-id", required=True)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--min-valid-horizons",
        type=int,
        default=6,
        help="入选窗口至少具有多少个有效Q forecast hours；默认要求6/6完整",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="默认沿用正式配置batch_size",
    )
    parser.add_argument(
        "--success-nse",
        type=float,
        default=0.98,
        help="仅用于最终PASS/FAIL标签，不触发提前停止",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples必须>0")
    if args.epochs <= 0:
        parser.error("--epochs必须>0")
    if args.min_valid_horizons <= 0:
        parser.error("--min-valid-horizons必须>0")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size必须>0")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (PROJECT_ROOT / "outputs" / f"diag_single_graph_{args.graph_id}").resolve()
    )
    _prepare_output_dir(output_dir, args.overwrite)

    source = load_config(args.config)
    if source["data"].get("dataset_type") != "continuous":
        raise ValueError("single-graph overfit诊断要求continuous P2配置")

    # Keep the formal architecture, optimizer, Q objective and physical solver.
    # Only diagnostic-only training controls/output paths are changed.
    source["train_sampling"]["enabled"] = False
    source["training"]["epochs"] = int(args.epochs)
    source["training"]["patience"] = int(args.epochs)
    source["training"]["early_stopping"] = False
    source["training"]["checkpoint"] = str(output_dir / "best_q_nse.pt")
    source["training"]["final_checkpoint"] = str(output_dir / "final.pt")
    source["training"]["log_csv"] = str(output_dir / "train_history.csv")
    source["hyperparameter_optimization"]["enabled"] = False

    cfg = _runtime_config_from_mapping(
        source,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    seed_everything(int(cfg["seed"]))

    dataset = _build_base_dataset(cfg)
    if len(dataset) == 0:
        raise ValueError(f"GRAPH_ID={args.graph_id}没有TRAIN样本")

    proxy_loader = SimpleNamespace(dataset=dataset)
    cfg["_runtime"] = _runtime_metadata(proxy_loader, cfg)
    nodes = _dataset_nodes(proxy_loader)
    model = HybridFloodModel(cfg, nodes)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    trainer = Trainer(model, cfg, device)

    candidates = _response_candidates(dataset, args.min_valid_horizons)
    if len(candidates) < args.num_samples:
        raise ValueError(
            f"GRAPH_ID={args.graph_id}仅找到{len(candidates)}个满足"
            f"Q(t0)有效且forecast Q至少{args.min_valid_horizons}小时有效的TRAIN窗口，"
            f"不足--num-samples={args.num_samples}"
        )

    selected = candidates[: args.num_samples]
    selected_indices = [int(row["dataset_index"]) for row in selected]
    _write_rows(output_dir / "selected_samples.csv", selected)

    subset = QOnlySubset(dataset, selected_indices)
    batch_size = int(args.batch_size or cfg["batch_size"])
    train_loader = _build_loader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        seed=int(cfg["seed"]),
        pin_memory=bool(cfg["pin_memory"]),
    )
    eval_loader = _build_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        seed=int(cfg["seed"]),
        pin_memory=bool(cfg["pin_memory"]),
    )

    selection_scores = [float(row["response_score"]) for row in selected]
    setup_report = {
        "diagnostic": "single_graph_q_only_overfit",
        "graph_id": args.graph_id,
        "dataset_root": cfg["data"]["dataset_root"],
        "num_samples": len(subset),
        "epochs": args.epochs,
        "batch_size": batch_size,
        "weighted_sampling": False,
        "z_supervision": False,
        "q_objective": "unchanged formal multitask Q point+peak+volume objective",
        "success_nse": args.success_nse,
        "response_score_min": min(selection_scores),
        "response_score_median": sorted(selection_scores)[len(selection_scores) // 2],
        "response_score_max": max(selection_scores),
        "device": str(device),
    }
    print(json.dumps(setup_report, ensure_ascii=False, indent=2))

    history_rows: list[dict[str, Any]] = []
    best_nse = float("-inf")
    best_epoch = -1
    best_path = output_dir / "best_q_nse.pt"

    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        fit_metrics = trainer.evaluate(eval_loader)

        q_nse = float(fit_metrics.get("q_nse", float("nan")))
        row: dict[str, Any] = {"epoch": epoch}
        row.update(
            {
                f"train_{key}": value
                for key, value in train_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        row.update(
            {
                f"fit_{key}": value
                for key, value in fit_metrics.items()
                if isinstance(value, (int, float))
            }
        )
        history_rows.append(row)

        print(
            {
                "epoch": epoch,
                "train_q_loss": train_metrics.get("q_loss"),
                "fit_q_nse": fit_metrics.get("q_nse"),
                "fit_q_kge": fit_metrics.get("q_kge"),
                "fit_q_rmse": fit_metrics.get("q_rmse"),
                "fit_q_mae": fit_metrics.get("q_mae"),
            }
        )

        if math.isfinite(q_nse) and q_nse > best_nse:
            best_nse = q_nse
            best_epoch = epoch
            trainer.save_checkpoint(
                best_path,
                epoch,
                row,
                kind="diagnostic_best_q_nse",
            )

    _write_rows(output_dir / "train_history.csv", history_rows)

    final_metrics = trainer.evaluate(eval_loader)
    trainer.save_checkpoint(
        output_dir / "final.pt",
        args.epochs - 1,
        {"epoch": args.epochs - 1, **final_metrics},
        kind="diagnostic_final",
    )
    _export_q_predictions(trainer, eval_loader, output_dir / "final_q_predictions.csv")

    best_metrics: dict[str, Any] | None = None
    if best_epoch >= 0 and best_path.is_file():
        trainer.load_weights(best_path)
        best_metrics = trainer.evaluate(eval_loader)
        _export_q_predictions(
            trainer, eval_loader, output_dir / "best_q_predictions.csv"
        )

    passed = math.isfinite(best_nse) and best_nse >= float(args.success_nse)
    summary = {
        **setup_report,
        "best_epoch": best_epoch,
        "best_q_nse": best_nse if math.isfinite(best_nse) else None,
        "status": "PASS_MEMORIZATION" if passed else "FAIL_MEMORIZATION",
        "interpretation": (
            "当前Q路径可以记住该固定小样本集；下一步优先检查多流域/continuous训练 formulation。"
            if passed
            else "当前Q路径未能记住该固定小样本集；优先检查Q forward/loss/routing实现或模型结构。"
        ),
        "best_metrics": _finite_or_none(best_metrics),
        "final_metrics": _finite_or_none(final_metrics),
        "files": {
            "selected_samples": str((output_dir / "selected_samples.csv").resolve()),
            "history": str((output_dir / "train_history.csv").resolve()),
            "best_checkpoint": str(best_path.resolve()),
            "final_checkpoint": str((output_dir / "final.pt").resolve()),
            "best_predictions": str((output_dir / "best_q_predictions.csv").resolve()),
            "final_predictions": str((output_dir / "final_q_predictions.csv").resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_finite_or_none(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_finite_or_none(summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
