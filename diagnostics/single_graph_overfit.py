"""Single-graph Q-only overfit diagnostic for the Hunan P2 continuous dataset.

This is a memorization test, not a formal experiment. It keeps the current model
forward path and Q loss definition, but:
- uses exactly one GRAPH_ID;
- selects a fixed, response-rich subset of TRAIN windows;
- disables weighted sampling;
- removes all Z supervision by zeroing z_target_mask;
- evaluates on the same fixed TRAIN subset after every epoch.

Use ``--preflight-only`` first. The preflight inspects rainfall forcing and
initial predictions, performs exactly one backward pass and one optimizer step,
and reports whether gradients, parameters and Q predictions actually change.
It does not enter the multi-epoch memorization loop.
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
        batch.z_target = torch.zeros_like(batch.z_target)
        batch.z_target_mask = torch.zeros_like(batch.z_target_mask, dtype=torch.bool)
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


def _tensor_report(value: torch.Tensor) -> dict[str, float | int]:
    flat = value.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "zero_fraction": float("nan"),
            "near_zero_fraction": float("nan"),
        }
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "zero_fraction": float((flat == 0).float().mean().item()),
        "near_zero_fraction": float((flat.abs() <= 1.0e-6).float().mean().item()),
    }


def _area_weighted_rain(sample: Any, history_hours: int) -> tuple[float, float]:
    rain = sample.rainfall.detach().float().squeeze(-1)
    if rain.ndim != 2:
        raise ValueError(f"rainfall预期[T,N]（去掉末维后），实际={tuple(rain.shape)}")
    area = sample.node_area_km2
    if area is None:
        raise ValueError("preflight要求node_area_km2以计算面积加权流域平均降雨")
    area = area.detach().float()
    if area.ndim != 1 or area.shape[0] != rain.shape[1]:
        raise ValueError("node_area_km2与rainfall节点维不一致")
    weights = area / area.sum()
    basin_hourly = (rain * weights.unsqueeze(0)).sum(dim=1)
    history_total = float(basin_hourly[:history_hours].sum().item())
    forecast_total = float(basin_hourly[history_hours:].sum().item())
    return history_total, forecast_total


def _annotate_preflight_samples(
    dataset: HunanContinuousDataset,
    selected: list[dict[str, Any]],
    history_hours: int,
    dry_epsilon_mm: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selected_row in selected:
        sample = dataset[int(selected_row["dataset_index"])]
        history_rain, forecast_rain = _area_weighted_rain(sample, history_hours)
        target_nodes = torch.nonzero(
            sample.q_target_mask.any(dim=0), as_tuple=False
        ).flatten()
        outlet = int(target_nodes.item())
        valid = sample.q_target_mask[:, outlet]
        target = sample.q_target[:, outlet][valid].float()
        row = dict(selected_row)
        row.update(
            {
                "history_model_forcing_basin_rain_mm": history_rain,
                "forecast_model_forcing_basin_rain_mm": forecast_rain,
                "total_model_forcing_basin_rain_mm": history_rain + forecast_rain,
                "history_dry": history_rain <= dry_epsilon_mm,
                "forecast_dry": forecast_rain <= dry_epsilon_mm,
                "history_and_forecast_dry": (
                    history_rain + forecast_rain <= dry_epsilon_mm
                ),
                "q_target_mean_m3s": float(target.mean().item()),
                "q_target_std_m3s": float(target.std(unbiased=False).item()),
            }
        )
        rows.append(row)
    return rows


def _rain_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    history = [float(row["history_model_forcing_basin_rain_mm"]) for row in rows]
    forecast = [float(row["forecast_model_forcing_basin_rain_mm"]) for row in rows]
    total = [float(row["total_model_forcing_basin_rain_mm"]) for row in rows]
    return {
        "sample_count": count,
        "history_dry_fraction": sum(bool(row["history_dry"]) for row in rows) / count,
        "forecast_dry_fraction": sum(bool(row["forecast_dry"]) for row in rows) / count,
        "history_and_forecast_dry_fraction": sum(
            bool(row["history_and_forecast_dry"]) for row in rows
        )
        / count,
        "history_rain_mean_mm": sum(history) / count,
        "history_rain_max_mm": max(history),
        "forecast_rain_mean_mm": sum(forecast) / count,
        "forecast_rain_max_mm": max(forecast),
        "total_rain_mean_mm": sum(total) / count,
        "total_rain_max_mm": max(total),
    }


@torch.no_grad()
def _prediction_preflight_report(
    trainer: Trainer, loader: DataLoader
) -> dict[str, dict[str, float | int]]:
    trainer.model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    lateral: list[torch.Tensor] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(trainer.device)
        output = trainer.model(batch)
        q_prediction = output["q"].detach().cpu()
        valid = cpu_batch.q_target_mask.bool()
        predictions.append(q_prediction[valid])
        targets.append(cpu_batch.q_target[valid].detach().cpu())
        lateral.append(output["q_lat"].detach().cpu().reshape(-1))
    return {
        "observed_q": _tensor_report(torch.cat(targets)),
        "predicted_q": _tensor_report(torch.cat(predictions)),
        "q_lateral_all_nodes_times": _tensor_report(torch.cat(lateral)),
    }


def _gradient_group_report(
    model: torch.nn.Module, prefix: str | None = None
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    parameter_tensors = 0
    gradient_tensors = 0
    nonzero_gradient_tensors = 0
    parameter_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or (prefix is not None and not name.startswith(prefix)):
            continue
        parameter_tensors += 1
        parameter_elements += parameter.numel()
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_tensors += 1
        detached = gradient.detach().float()
        if bool((detached != 0).any()):
            nonzero_gradient_tensors += 1
        squared += float(detached.square().sum().item())
        if detached.numel():
            max_abs = max(max_abs, float(detached.abs().max().item()))
    return {
        "l2_norm": math.sqrt(squared),
        "max_abs": max_abs,
        "parameter_tensors": parameter_tensors,
        "parameter_elements": parameter_elements,
        "gradient_tensors": gradient_tensors,
        "nonzero_gradient_tensors": nonzero_gradient_tensors,
    }


def _parameter_delta_report(
    before: dict[str, torch.Tensor], model: torch.nn.Module, prefix: str | None = None
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed_tensors = 0
    parameter_tensors = 0
    changed_elements = 0
    for name, parameter in model.named_parameters():
        if name not in before or (prefix is not None and not name.startswith(prefix)):
            continue
        parameter_tensors += 1
        delta = parameter.detach() - before[name]
        delta_float = delta.float()
        if bool((delta_float != 0).any()):
            changed_tensors += 1
            changed_elements += int((delta_float != 0).sum().item())
        squared += float(delta_float.square().sum().item())
        if delta_float.numel():
            max_abs = max(max_abs, float(delta_float.abs().max().item()))
    return {
        "l2_norm": math.sqrt(squared),
        "max_abs": max_abs,
        "parameter_tensors": parameter_tensors,
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
    }


def _single_optimizer_step_preflight(
    trainer: Trainer, loader: DataLoader
) -> dict[str, Any]:
    cpu_batch = next(iter(loader))
    batch = cpu_batch.to(trainer.device)
    trainer.model.eval()
    trainer.optimizer.zero_grad(set_to_none=True)
    before_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }

    with torch.autocast(
        device_type=trainer.device.type,
        dtype=torch.float16,
        enabled=trainer.amp,
    ):
        output_before = trainer.model(batch)
        statistics = trainer.loss_engine.batch_statistics(output_before, batch)
        loss = trainer.loss_engine.combine(statistics)
    q_before = output_before["q"].detach().clone()
    q_lat_before = output_before["q_lat"].detach().clone()

    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError("preflight单batch Q loss出现NaN/Inf")
    trainer.scaler.scale(loss).backward()
    trainer.scaler.unscale_(trainer.optimizer)

    gradients_before_clip = {
        "runoff": _gradient_group_report(trainer.model, "runoff."),
        "routing": _gradient_group_report(trainer.model, "routing."),
        "observation": _gradient_group_report(trainer.model, "observation."),
        "total": _gradient_group_report(trainer.model, None),
    }
    clip_return = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(),
        float(trainer.cfg["training"]["gradient_clip"]),
        error_if_nonfinite=not trainer.amp,
    )
    gradients_after_clip = {
        "runoff": _gradient_group_report(trainer.model, "runoff."),
        "routing": _gradient_group_report(trainer.model, "routing."),
        "observation": _gradient_group_report(trainer.model, "observation."),
        "total": _gradient_group_report(trainer.model, None),
    }

    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad(set_to_none=True)

    parameter_delta = {
        "runoff": _parameter_delta_report(before_parameters, trainer.model, "runoff."),
        "routing": _parameter_delta_report(before_parameters, trainer.model, "routing."),
        "observation": _parameter_delta_report(
            before_parameters, trainer.model, "observation."
        ),
        "total": _parameter_delta_report(before_parameters, trainer.model, None),
    }

    with torch.no_grad():
        output_after = trainer.model(batch)
    q_after = output_after["q"].detach()
    q_lat_after = output_after["q_lat"].detach()
    q_change = q_after - q_before
    q_lat_change = q_lat_after - q_lat_before
    valid = batch.q_target_mask.bool()

    return {
        "batch_size": int(q_before.shape[0]),
        "loss_before_step": float(loss.detach().item()),
        "gradient_clip_threshold": float(trainer.cfg["training"]["gradient_clip"]),
        "gradient_norm_returned_by_clip": float(clip_return.detach().cpu().item()),
        "gradients_before_clip": gradients_before_clip,
        "gradients_after_clip": gradients_after_clip,
        "parameter_delta_after_step": parameter_delta,
        "q_prediction_change_valid_targets": _tensor_report(q_change[valid]),
        "q_prediction_change_all_nodes": _tensor_report(q_change),
        "q_lateral_change_all_nodes_times": _tensor_report(q_lat_change),
    }


def _run_preflight(
    trainer: Trainer,
    eval_loader: DataLoader,
    dataset: HunanContinuousDataset,
    selected: list[dict[str, Any]],
    output_dir: Path,
    cfg: dict[str, Any],
    dry_epsilon_mm: float,
) -> dict[str, Any]:
    rows = _annotate_preflight_samples(
        dataset,
        selected,
        int(cfg["history_length"]),
        dry_epsilon_mm,
    )
    _write_rows(output_dir / "preflight_samples.csv", rows)
    rain = _rain_summary(rows)
    initial_prediction = _prediction_preflight_report(trainer, eval_loader)
    update_path = _single_optimizer_step_preflight(trainer, eval_loader)

    gradients_path = output_dir / "preflight_gradients.json"
    gradients_path.write_text(
        json.dumps(_finite_or_none(update_path), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    total_grad_norm = float(
        update_path["gradients_before_clip"]["total"]["l2_norm"]
    )
    runoff_grad_norm = float(
        update_path["gradients_before_clip"]["runoff"]["l2_norm"]
    )
    routing_grad_norm = float(
        update_path["gradients_before_clip"]["routing"]["l2_norm"]
    )
    total_parameter_delta = float(
        update_path["parameter_delta_after_step"]["total"]["l2_norm"]
    )
    prediction_change_max = float(
        update_path["q_prediction_change_valid_targets"]["max"]
    )
    prediction_change_abs = max(
        abs(float(update_path["q_prediction_change_valid_targets"]["min"])),
        abs(prediction_change_max),
    )
    predicted_std = float(initial_prediction["predicted_q"]["std"])

    gradient_present = math.isfinite(total_grad_norm) and total_grad_norm > 1.0e-12
    parameter_changed = (
        math.isfinite(total_parameter_delta) and total_parameter_delta > 0.0
    )
    prediction_changed = (
        math.isfinite(prediction_change_abs) and prediction_change_abs > 1.0e-9
    )
    prediction_collapsed = math.isfinite(predicted_std) and predicted_std <= 1.0e-8

    if not gradient_present:
        status = "FAIL_NO_EFFECTIVE_GRADIENT"
        interpretation = (
            "Q loss没有向可训练参数产生有效梯度；优先检查Q forward、零初始水文/河道状态及物理约束造成的梯度阻断。"
        )
    elif not parameter_changed:
        status = "FAIL_OPTIMIZER_NO_PARAMETER_UPDATE"
        interpretation = (
            "存在Q梯度，但一次optimizer step后参数未变化；优先检查AMP/scaler、optimizer参数组和梯度裁剪。"
        )
    elif not prediction_changed:
        status = "FAIL_PARAMETER_UPDATE_NO_Q_CHANGE"
        interpretation = (
            "参数发生变化，但同一batch的Q预测没有可检测变化；优先检查更新参数是否真正位于Q输出通路。"
        )
    else:
        status = "PASS_UPDATE_PATH"
        interpretation = (
            "Q梯度、参数更新和预测变化均存在；若多epoch指标仍完全不变，应继续检查更新量尺度、训练循环和物理状态表达。"
        )

    summary = {
        "diagnostic": "single_graph_q_only_preflight",
        "graph_id": cfg["data"]["graph_id"],
        "sample_count": len(rows),
        "dry_epsilon_mm": dry_epsilon_mm,
        "rain_forcing": rain,
        "initial_prediction": initial_prediction,
        "key_gradient_norms_before_clip": {
            "runoff": runoff_grad_norm,
            "routing": routing_grad_norm,
            "total": total_grad_norm,
        },
        "total_parameter_delta_l2": total_parameter_delta,
        "max_abs_q_prediction_change_after_one_step": prediction_change_abs,
        "initial_q_prediction_collapsed": prediction_collapsed,
        "status": status,
        "interpretation": interpretation,
        "files": {
            "samples": str((output_dir / "preflight_samples.csv").resolve()),
            "gradients": str(gradients_path.resolve()),
            "summary": str((output_dir / "preflight_summary.json").resolve()),
        },
    }
    summary_path = output_dir / "preflight_summary.json"
    summary_path.write_text(
        json.dumps(_finite_or_none(summary), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


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
            valid_positions = torch.nonzero(mask[sample_index], as_tuple=False)
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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只做降雨/初始预测/一次backward/一次optimizer step检查，不进入epoch训练",
    )
    parser.add_argument(
        "--dry-epsilon-mm",
        type=float,
        default=1.0e-8,
        help="面积加权流域累计降雨<=该阈值时判为dry",
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
    if not math.isfinite(args.dry_epsilon_mm) or args.dry_epsilon_mm < 0:
        parser.error("--dry-epsilon-mm必须是有限非负数")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (PROJECT_ROOT / "outputs" / f"diag_single_graph_{args.graph_id}").resolve()
    )
    _prepare_output_dir(output_dir, args.overwrite)

    source = load_config(args.config)
    if source["data"].get("dataset_type") != "continuous":
        raise ValueError("single-graph overfit诊断要求continuous P2配置")

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

    if args.preflight_only:
        summary = _run_preflight(
            trainer,
            eval_loader,
            dataset,
            selected,
            output_dir,
            cfg,
            float(args.dry_epsilon_mm),
        )
        print(
            json.dumps(
                _finite_or_none(summary),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return

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
