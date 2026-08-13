"""Read-only audit of Q-prediction -> station rating curve -> delta-Z.

The audit answers one narrow question: if the already-trained model's discharge
forecast is passed through a TRAIN-only station Q-Z relation, does stage skill
improve relative to the model's current direct delta-Z head?

Three paths are compared on the same VALIDATION samples:
  A) current model direct delta-Z output;
  B) model Q prediction -> TRAIN-only linear rating curve -> delta-Z;
  C) observed future Q -> the same TRAIN-only rating curve -> delta-Z (oracle).

The rating curve is fit only from unique TRAIN target timestamps where Q and Z
are simultaneously valid.  Both raw OLS and a clearly reported gross-outlier
robust refit are evaluated.  No model weights, dataset files, checkpoints, or
training outputs are modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from data.device import resolve_device
from models import HybridFloodModel
from scripts.common import (
    _dataset_nodes,
    _ensure_matching_graph,
    _make_loader,
    _runtime_config,
    _runtime_metadata,
    validate_checkpoint_config,
)


def _as_list(value: Any, batch_size: int, name: str) -> list[str]:
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, (tuple, list)):
        if len(value) != batch_size:
            raise ValueError(f"{name}长度应为batch size={batch_size}，实际={len(value)}")
        return [str(item) for item in value]
    raise ValueError(f"{name}必须为字符串或逐样本字符串序列")


def _parse_time(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("forecast_time不能为空")
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        )
    except ValueError as exc:
        raise ValueError(f"无效forecast_time={value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _target_node(batch: Any, station_id: str) -> int:
    station_ids = getattr(batch, "station_ids", None)
    if station_ids is None:
        if batch.q_target.shape[2] == 1:
            return 0
        raise ValueError("多节点batch缺少station_ids，无法定位目标站")
    matches = [index for index, value in enumerate(station_ids) if str(value) == station_id]
    if len(matches) != 1:
        raise ValueError(
            f"target station {station_id!r}在station_ids中应唯一出现，实际={matches}"
        )
    return matches[0]


def _fit_ols(q: torch.Tensor, z: torch.Tensor) -> dict[str, float]:
    q = q.double().reshape(-1)
    z = z.double().reshape(-1)
    if q.numel() != z.numel() or q.numel() < 2:
        raise ValueError("rating curve至少需要2个成对Q/Z观测")
    q_mean = q.mean()
    z_mean = z.mean()
    denominator = ((q - q_mean) ** 2).sum()
    if float(denominator.item()) <= 0.0:
        raise ValueError("TRAIN Q无方差，无法拟合rating curve")
    slope = ((q - q_mean) * (z - z_mean)).sum() / denominator
    intercept = z_mean - slope * q_mean
    prediction = slope * q + intercept
    residual = prediction - z
    rmse = torch.sqrt((residual**2).mean())
    z_anomaly = z - z_mean
    nse_denominator = (z_anomaly**2).sum()
    nse = (
        1.0 - float((residual**2).sum().item()) / float(nse_denominator.item())
        if float(nse_denominator.item()) > 0.0
        else float("nan")
    )
    return {
        "slope_m_per_m3s": float(slope.item()),
        "intercept_m": float(intercept.item()),
        "fit_rmse_m": float(rmse.item()),
        "fit_nse": nse,
    }


def _median(values: torch.Tensor) -> float:
    values = values.double().reshape(-1)
    return float(values.median().item())


def _rating_fits(q: torch.Tensor, z: torch.Tensor) -> dict[str, Any]:
    raw = _fit_ols(q, z)
    slope = raw["slope_m_per_m3s"]
    intercept = raw["intercept_m"]
    residual = slope * q.double() + intercept - z.double()
    residual_median = _median(residual)
    mad = _median((residual - residual_median).abs())
    robust_sigma = 1.4826 * mad
    # This is deliberately only a gross-outlier filter.  It is not tuned to
    # improve validation performance.  The 0.20 m floor prevents ordinary
    # rating scatter from being silently removed; all exclusion counts and the
    # threshold are emitted in the report.
    threshold = max(0.20, 8.0 * robust_sigma)
    inlier = (residual - residual_median).abs() <= threshold
    inlier_count = int(inlier.sum().item())
    if inlier_count < 2:
        raise ValueError("gross-outlier过滤后不足2个TRAIN Q/Z点")
    robust = _fit_ols(q[inlier], z[inlier])
    return {
        "paired_train_points": int(q.numel()),
        "raw_ols": raw,
        "gross_outlier_refit": {
            **robust,
            "residual_median_m_from_raw_fit": residual_median,
            "residual_mad_m_from_raw_fit": mad,
            "robust_sigma_m_from_raw_fit": robust_sigma,
            "exclusion_threshold_m": threshold,
            "included_count": inlier_count,
            "excluded_count": int(q.numel()) - inlier_count,
        },
    }


def _collect_unique_train_pairs(loader: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    paired: dict[tuple[str, datetime], tuple[float, float]] = {}
    conflict_count = 0
    conflict_preview: list[dict[str, Any]] = []
    candidate_count = 0

    for batch in loader:
        q_target = batch.q_target.detach().cpu().float()
        z_target = batch.z_target.detach().cpu().float()
        q_mask = batch.q_target_mask.detach().cpu().bool()
        z_mask = batch.z_target_mask.detach().cpu().bool()
        z_reference = None if batch.z_reference is None else batch.z_reference.detach().cpu().float()
        z_reference_mask = (
            None
            if batch.z_reference_mask is None
            else batch.z_reference_mask.detach().cpu().bool()
        )
        if z_reference is None or z_reference_mask is None:
            raise ValueError("rating audit要求delta-from-t0 z_reference")
        batch_size, horizon, _ = q_target.shape
        forecast_times = _as_list(batch.forecast_time, batch_size, "forecast_time")
        station_ids = _as_list(batch.target_station_id, batch_size, "target_station_id")

        for sample_index in range(batch_size):
            station_id = station_ids[sample_index]
            node = _target_node(batch, station_id)
            if not bool(z_reference_mask[sample_index, node]):
                continue
            z0 = float(z_reference[sample_index, node].item())
            t0 = _parse_time(forecast_times[sample_index])
            for h in range(horizon):
                if not (
                    bool(q_mask[sample_index, h, node])
                    and bool(z_mask[sample_index, h, node])
                ):
                    continue
                candidate_count += 1
                timestamp = t0 + timedelta(hours=h + 1)
                q = float(q_target[sample_index, h, node].item())
                z = z0 + float(z_target[sample_index, h, node].item())
                key = (station_id, timestamp)
                if key in paired:
                    previous_q, previous_z = paired[key]
                    if abs(previous_q - q) > 1.0e-6 or abs(previous_z - z) > 1.0e-6:
                        conflict_count += 1
                        if len(conflict_preview) < 20:
                            conflict_preview.append(
                                {
                                    "station_id": station_id,
                                    "timestamp": timestamp.isoformat(sep=" "),
                                    "existing_q_m3s": previous_q,
                                    "new_q_m3s": q,
                                    "existing_z_m": previous_z,
                                    "new_z_m": z,
                                }
                            )
                    continue
                paired[key] = (q, z)

    if len(paired) < 2:
        raise ValueError("TRAIN中没有足够的唯一Q/Z同时有效target时间点")
    ordered = [paired[key] for key in sorted(paired, key=lambda item: (item[0], item[1]))]
    q = torch.tensor([value[0] for value in ordered], dtype=torch.float64)
    z = torch.tensor([value[1] for value in ordered], dtype=torch.float64)
    audit = {
        "candidate_paired_target_occurrences": candidate_count,
        "unique_paired_train_timestamps": len(paired),
        "duplicate_value_conflict_count": conflict_count,
        "duplicate_value_conflicts_preview": conflict_preview,
        "q_min_m3s": float(q.min().item()),
        "q_max_m3s": float(q.max().item()),
        "z_min_m": float(z.min().item()),
        "z_max_m": float(z.max().item()),
    }
    return q, z, audit


def _metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    count = int(valid.sum().item())
    if count == 0:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "nse": float("nan"),
            "corr": float("nan"),
        }
    p = prediction[valid].double()
    y = target[valid].double()
    error = p - y
    mae = float(error.abs().mean().item())
    rmse = float(torch.sqrt(error.square().mean()).item())
    bias = float(error.mean().item())
    y_anomaly = y - y.mean()
    denominator = float(y_anomaly.square().sum().item())
    nse = (
        1.0 - float(error.square().sum().item()) / denominator
        if denominator > 0.0
        else float("nan")
    )
    p_anomaly = p - p.mean()
    corr_denominator = torch.sqrt(p_anomaly.square().sum() * y_anomaly.square().sum())
    corr = (
        float((p_anomaly * y_anomaly).sum().item() / corr_denominator.item())
        if torch.isfinite(corr_denominator) and float(corr_denominator.item()) > 0.0
        else float("nan")
    )
    return {
        "count": count,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "nse": nse,
        "corr": corr,
    }


def _report_path(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    unit_suffix: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "overall": _metrics(prediction, target, mask),
        "by_horizon": {},
    }
    for key in ("mae", "rmse", "bias"):
        report["overall"][f"{key}_{unit_suffix}"] = report["overall"].pop(key)
    for h in range(target.shape[1]):
        values = _metrics(prediction[:, h], target[:, h], mask[:, h])
        for key in ("mae", "rmse", "bias"):
            values[f"{key}_{unit_suffix}"] = values.pop(key)
        report["by_horizon"][f"h{h + 1}"] = values
    return report


def _curve_delta(
    q: torch.Tensor,
    z_reference: torch.Tensor,
    slope: float,
    intercept: float,
) -> torch.Tensor:
    absolute_z = q * float(slope) + float(intercept)
    return absolute_z - z_reference.unsqueeze(1)


def _evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    fits: dict[str, Any],
) -> dict[str, Any]:
    q_predictions: list[torch.Tensor] = []
    q_targets: list[torch.Tensor] = []
    q_masks: list[torch.Tensor] = []
    z_direct: list[torch.Tensor] = []
    z_targets: list[torch.Tensor] = []
    z_masks: list[torch.Tensor] = []
    z_references: list[torch.Tensor] = []
    z_reference_masks: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.to(device))
            q_predictions.append(output["q"].detach().cpu().float())
            z_direct.append(output["z"].detach().cpu().float())
            q_targets.append(batch.q_target.detach().cpu().float())
            q_masks.append(batch.q_target_mask.detach().cpu().bool())
            z_targets.append(batch.z_target.detach().cpu().float())
            z_masks.append(batch.z_target_mask.detach().cpu().bool())
            if batch.z_reference is None or batch.z_reference_mask is None:
                raise ValueError("rating audit要求VALIDATION提供z_reference")
            z_references.append(batch.z_reference.detach().cpu().float())
            z_reference_masks.append(batch.z_reference_mask.detach().cpu().bool())

    q_pred = torch.cat(q_predictions, dim=0)
    q_true = torch.cat(q_targets, dim=0)
    q_mask = torch.cat(q_masks, dim=0)
    dz_direct = torch.cat(z_direct, dim=0)
    dz_true = torch.cat(z_targets, dim=0)
    z_mask = torch.cat(z_masks, dim=0)
    z0 = torch.cat(z_references, dim=0)
    z0_mask = torch.cat(z_reference_masks, dim=0)
    valid_z = z_mask & z0_mask.unsqueeze(1)
    common_oracle = valid_z & q_mask

    result: dict[str, Any] = {
        "q_prediction": _report_path(q_pred, q_true, q_mask, unit_suffix="m3s"),
        "z_direct_current_model": _report_path(
            dz_direct, dz_true, valid_z, unit_suffix="m"
        ),
        "common_oracle_valid_count": int(common_oracle.sum().item()),
        "rating_curves": {},
    }

    curve_specs = {
        "raw_ols": fits["raw_ols"],
        "gross_outlier_refit": fits["gross_outlier_refit"],
    }
    for name, curve in curve_specs.items():
        slope = float(curve["slope_m_per_m3s"])
        intercept = float(curve["intercept_m"])
        predicted_q_delta_z = _curve_delta(q_pred, z0, slope, intercept)
        oracle_q_delta_z = _curve_delta(q_true, z0, slope, intercept)
        result["rating_curves"][name] = {
            "parameters": {
                "slope_m_per_m3s": slope,
                "intercept_m": intercept,
            },
            "predicted_q_to_delta_z_all_z_valid": _report_path(
                predicted_q_delta_z, dz_true, valid_z, unit_suffix="m"
            ),
            "oracle_observed_q_to_delta_z": _report_path(
                oracle_q_delta_z, dz_true, common_oracle, unit_suffix="m"
            ),
            "same_common_mask_comparison": {
                "current_direct_z": _report_path(
                    dz_direct, dz_true, common_oracle, unit_suffix="m"
                ),
                "predicted_q_rating_z": _report_path(
                    predicted_q_delta_z, dz_true, common_oracle, unit_suffix="m"
                ),
                "oracle_q_rating_z": _report_path(
                    oracle_q_delta_z, dz_true, common_oracle, unit_suffix="m"
                ),
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读审计：当前Q预测经TRAIN-only rating curve后能否改善ΔZ"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--split", default="VALIDATION", choices=("VALIDATION", "TEST")
    )
    args = parser.parse_args()

    cfg = _runtime_config(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    if str(cfg.get("loss", {}).get("z_target_mode")) != "delta_from_t0":
        raise ValueError("rating audit要求loss.z_target_mode=delta_from_t0")

    dynamic_cache: dict = {}
    train_loader = _make_loader(
        cfg,
        cfg["data"]["train_split"],
        shuffle=False,
        dynamic_cache=dynamic_cache,
    )
    evaluation_loader = _make_loader(
        cfg,
        args.split,
        shuffle=False,
        dynamic_cache=dynamic_cache,
    )
    _ensure_matching_graph(train_loader, evaluation_loader)
    cfg["_runtime"] = _runtime_metadata(train_loader, cfg)

    train_q, train_z, pair_audit = _collect_unique_train_pairs(train_loader)
    fits = _rating_fits(train_q, train_z)
    if fits["raw_ols"]["slope_m_per_m3s"] <= 0:
        raise ValueError("TRAIN raw rating curve slope非正，Q-Z关系不符合单调增假设")
    if fits["gross_outlier_refit"]["slope_m_per_m3s"] <= 0:
        raise ValueError("TRAIN robust rating curve slope非正，Q-Z关系不符合单调增假设")

    model = HybridFloodModel(cfg, _dataset_nodes(train_loader))
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    checkpoint_path = Path(
        args.checkpoint if args.checkpoint is not None else cfg["training"]["checkpoint"]
    ).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint必须包含model state_dict")
    validate_checkpoint_config(checkpoint, cfg, resume=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    report = {
        "split": args.split,
        "graph_id": cfg["data"].get("graph_id"),
        "dataset_root": str(Path(cfg["data"]["dataset_root"]).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_kind": checkpoint.get("checkpoint_kind"),
        "rating_curve_training_domain": {
            "split": "TRAIN",
            "source": "unique target timestamps with simultaneous Q and Z supervision",
            **pair_audit,
        },
        "rating_curve_fits": fits,
        "evaluation": _evaluate(model, evaluation_loader, device, fits),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
