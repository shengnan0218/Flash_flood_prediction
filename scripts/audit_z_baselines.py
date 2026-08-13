"""Read-only baseline audit for forecast-origin delta-Z predictability.

The script evaluates three deterministic stage baselines on one existing split:

B0: persistence,          delta Z(h) = 0
B1: latest 1 h slope,    delta Z(h) = h * [Z(t0)-Z(t0-1)]
B3: latest 3 h slope,    delta Z(h) = h * [Z(t0)-Z(t0-3)] / 3

No model is constructed, no checkpoint is loaded, and no training/output state is
modified. Metrics are computed directly in physical metres against the dataset's
existing delta-from-t0 Z target.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.common import _make_loader, _runtime_config


def _metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    count = int(valid.sum().item())
    if count == 0:
        return {
            "count": 0,
            "mae_m": float("nan"),
            "rmse_m": float("nan"),
            "nse": float("nan"),
            "corr": float("nan"),
            "bias_m": float("nan"),
        }
    p = prediction[valid].double()
    y = target[valid].double()
    error = p - y
    mae = float(error.abs().mean().item())
    rmse = float(torch.sqrt(error.square().mean()).item())
    bias = float(error.mean().item())

    y_anomaly = y - y.mean()
    denominator = float(y_anomaly.square().sum().item())
    if denominator > 0.0:
        nse = 1.0 - float(error.square().sum().item()) / denominator
    else:
        nse = float("nan")

    p_anomaly = p - p.mean()
    corr_denominator = torch.sqrt(p_anomaly.square().sum() * y_anomaly.square().sum())
    if torch.isfinite(corr_denominator) and float(corr_denominator.item()) > 0.0:
        corr = float((p_anomaly * y_anomaly).sum().item() / corr_denominator.item())
    else:
        corr = float("nan")
    return {
        "count": count,
        "mae_m": mae,
        "rmse_m": rmse,
        "nse": nse,
        "corr": corr,
        "bias_m": bias,
    }


def _empty_parts(horizon: int) -> dict[str, list[torch.Tensor]]:
    return {
        "target": [],
        "target_mask": [],
        "b0": [],
        "b0_mask": [],
        "b1": [],
        "b1_mask": [],
        "b3": [],
        "b3_mask": [],
        "common_mask": [],
    }


def _cat(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        return torch.empty(0)
    return torch.cat(parts, dim=0)


def _evaluate(loader: Any, expected_horizon: int) -> dict[str, Any]:
    parts = _empty_parts(expected_horizon)
    sample_count = 0

    for batch in loader:
        z_history = batch.z_history.detach().cpu().float()
        z_history_mask = batch.z_mask.detach().cpu().bool()
        z_target = batch.z_target.detach().cpu().float()
        z_target_mask = batch.z_target_mask.detach().cpu().bool()

        if z_history.ndim != 3 or z_target.ndim != 3:
            raise ValueError("Z history/target必须为[B,H,N]/[B,F,N]")
        if z_history.shape[1] < 4:
            raise ValueError("3 h slope baseline要求至少4个history时刻（含t0）")
        if z_target.shape[1] != expected_horizon:
            raise ValueError(
                f"forecast horizon应为{expected_horizon}，实际={z_target.shape[1]}"
            )
        if z_history_mask.shape != z_history.shape or z_target_mask.shape != z_target.shape:
            raise ValueError("Z mask形状与对应张量不一致")

        # Dataset history contract ends at forecast origin t0.
        z0 = z_history[:, -1]
        z1 = z_history[:, -2]
        z3 = z_history[:, -4]
        m0 = z_history_mask[:, -1] & torch.isfinite(z0)
        m1 = z_history_mask[:, -2] & torch.isfinite(z1)
        m3 = z_history_mask[:, -4] & torch.isfinite(z3)

        batch_size, horizon, nodes = z_target.shape
        lead = torch.arange(1, horizon + 1, dtype=z_target.dtype).view(1, horizon, 1)

        b0 = torch.zeros_like(z_target)
        slope1 = z0 - z1
        slope3 = (z0 - z3) / 3.0
        b1 = lead * slope1.unsqueeze(1)
        b3 = lead * slope3.unsqueeze(1)

        b0_mask = z_target_mask & m0.unsqueeze(1)
        b1_mask = z_target_mask & (m0 & m1).unsqueeze(1)
        b3_mask = z_target_mask & (m0 & m3).unsqueeze(1)
        common_mask = b0_mask & b1_mask & b3_mask

        parts["target"].append(z_target)
        parts["target_mask"].append(z_target_mask)
        parts["b0"].append(b0)
        parts["b0_mask"].append(b0_mask)
        parts["b1"].append(b1)
        parts["b1_mask"].append(b1_mask)
        parts["b3"].append(b3)
        parts["b3_mask"].append(b3_mask)
        parts["common_mask"].append(common_mask)
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("loader没有样本")

    tensors = {name: _cat(values) for name, values in parts.items()}
    target = tensors["target"]
    target_mask = tensors["target_mask"]
    total_target = int(target_mask.sum().item())

    result: dict[str, Any] = {
        "sample_count": sample_count,
        "target_valid_count": total_target,
        "target_summary": _metrics(torch.zeros_like(target), target, target_mask),
        "baselines": {},
    }

    for name, description in (
        ("b0_persistence", "delta_Z(h)=0"),
        ("b1_latest_1h_slope", "delta_Z(h)=h*[Z(t0)-Z(t0-1)]"),
        ("b3_latest_3h_slope", "delta_Z(h)=h*[Z(t0)-Z(t0-3)]/3"),
    ):
        short = name.split("_")[0]
        prediction = tensors[short]
        mask = tensors[f"{short}_mask"]
        valid_count = int(mask.sum().item())
        baseline: dict[str, Any] = {
            "definition": description,
            "valid_count": valid_count,
            "coverage_of_target": valid_count / total_target if total_target else float("nan"),
            "overall": _metrics(prediction, target, mask),
            "by_horizon": {},
        }
        for horizon_index in range(expected_horizon):
            baseline["by_horizon"][f"h{horizon_index + 1}"] = _metrics(
                prediction[:, horizon_index],
                target[:, horizon_index],
                mask[:, horizon_index],
            )
        result["baselines"][name] = baseline

    common = tensors["common_mask"]
    result["common_valid_comparison"] = {
        "valid_count": int(common.sum().item()),
        "coverage_of_target": (
            int(common.sum().item()) / total_target if total_target else float("nan")
        ),
        "b0_persistence": _metrics(tensors["b0"], target, common),
        "b1_latest_1h_slope": _metrics(tensors["b1"], target, common),
        "b3_latest_3h_slope": _metrics(tensors["b3"], target, common),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计ΔZ简单基线可预报性")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--split", default="VALIDATION", choices=("TRAIN", "VALIDATION", "TEST"))
    args = parser.parse_args()

    cfg = _runtime_config(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    if str(cfg.get("loss", {}).get("z_target_mode")) != "delta_from_t0":
        raise ValueError("baseline audit要求loss.z_target_mode=delta_from_t0")

    split = str(args.split).upper()
    loader = _make_loader(cfg, split, shuffle=False)
    report = {
        "split": split,
        "graph_id": cfg["data"].get("graph_id"),
        "dataset_root": str(Path(cfg["data"]["dataset_root"]).resolve()),
        "history_hours": int(cfg["history_length"]),
        "forecast_hours": int(cfg["forecast_horizon"]),
        "result": _evaluate(loader, int(cfg["forecast_horizon"])),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
