"""TRAIN-only deterministic calibration of station Q->Z rating functions.

The paired Q/Z observations are used only to calibrate the observation
function itself.  They do not train a neural Z residual head.  The calibrated
curve is a monotone piecewise-linear function obtained by pool-adjacent-
violators (PAV) regression on unique TRAIN target timestamps.  Its knots are
frozen data-derived facts during model training.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _paired_train_outlet_observations(dataset: Any, graph_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(dataset, "split", None) != "TRAIN":
        raise ValueError("rating calibration只能从TRAIN dataset拟合")
    graph = dataset._graphs[graph_id]
    outlet = next(node.node_index for node in graph.nodes if node.is_outlet)
    dynamic = dataset._dynamic[graph_id]
    samples = [sample for sample in dataset._samples if sample.graph_id == graph_id]
    horizons = torch.arange(1, dataset.forecast_hours + 1, dtype=torch.long)
    used = torch.zeros(len(dynamic.timestamps), dtype=torch.bool)
    for start in range(0, len(samples), 100_000):
        chunk = samples[start : start + 100_000]
        origins = torch.tensor(
            [dataset._origin_index(sample) for sample in chunk], dtype=torch.long
        )
        future = origins.unsqueeze(1) + horizons.unsqueeze(0)
        simultaneous = (
            dynamic.flow_mask[future, outlet]
            & dynamic.water_level_mask[future, outlet]
        )
        used[future[simultaneous]] = True
    indices = used.nonzero(as_tuple=False).flatten()
    return (
        dynamic.flow[indices, outlet].to(torch.float64),
        dynamic.water_level[indices, outlet].to(torch.float64),
    )


def _pav_monotone_knots(q: torch.Tensor, z: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return strictly increasing PAV block-centroid knots.

    Equal Q values are first pooled. Adjacent blocks are merged whenever their
    fitted Z means fail to increase. Equality is merged too, so consecutive
    returned Z knots are strictly increasing and the piecewise-linear mapping
    has positive slope wherever it is defined.
    """
    if q.ndim != 1 or z.shape != q.shape or q.numel() < 2:
        raise ValueError("PAV rating calibration要求至少2个一维Q/Z配对点")
    order = torch.argsort(q)
    q_sorted = q[order].tolist()
    z_sorted = z[order].tolist()

    grouped: list[dict[str, float]] = []
    for q_value, z_value in zip(q_sorted, z_sorted):
        if grouped and q_value == grouped[-1]["q"]:
            grouped[-1]["weight"] += 1.0
            grouped[-1]["sum_z"] += z_value
        else:
            grouped.append(
                {"q": float(q_value), "weight": 1.0, "sum_z": float(z_value)}
            )
    if len(grouped) < 2:
        raise ValueError("TRAIN配对Q只有一个唯一值，无法标定单调rating curve")

    blocks: list[dict[str, float]] = []
    for group in grouped:
        block = {
            "weight": group["weight"],
            "sum_q": group["q"] * group["weight"],
            "sum_z": group["sum_z"],
        }
        blocks.append(block)
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["sum_z"] / left["weight"]
            right_mean = right["sum_z"] / right["weight"]
            if left_mean < right_mean:
                break
            merged = {
                "weight": left["weight"] + right["weight"],
                "sum_q": left["sum_q"] + right["sum_q"],
                "sum_z": left["sum_z"] + right["sum_z"],
            }
            blocks[-2:] = [merged]

    q_knots = [block["sum_q"] / block["weight"] for block in blocks]
    z_knots = [block["sum_z"] / block["weight"] for block in blocks]
    if len(q_knots) < 2:
        raise ValueError("PAV后只剩一个单调block")
    if any(b <= a for a, b in zip(q_knots, q_knots[1:])):
        raise ValueError("PAV q knots未严格递增")
    if any(b <= a for a, b in zip(z_knots, z_knots[1:])):
        raise ValueError("PAV z knots未严格递增")
    return q_knots, z_knots


def _piecewise_linear_predict(
    q: torch.Tensor,
    q_knots: list[float],
    z_knots: list[float],
) -> torch.Tensor:
    x = torch.tensor(q_knots, dtype=torch.float64)
    y = torch.tensor(z_knots, dtype=torch.float64)
    right = torch.searchsorted(x, q).clamp(1, x.numel() - 1)
    left = right - 1
    x0 = x[left]
    x1 = x[right]
    y0 = y[left]
    y1 = y[right]
    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (q - x0)


def fit_train_monotone_rating_statistics(dataset: Any) -> dict[str, Any]:
    """Augment the existing TRAIN linear audit with frozen monotone calibration."""
    if getattr(dataset, "split", None) != "TRAIN":
        raise ValueError("rating calibration只能从TRAIN dataset拟合")
    linear = dataset.train_rating_curve_statistics()
    stations = linear.get("stations", {})
    calibrated: dict[str, dict[str, Any]] = {}

    for graph_id in dataset.graph_ids:
        graph = dataset._graphs[graph_id]
        station_id = graph.outlet_id
        base = dict(stations.get(station_id, {}))
        q, z = _paired_train_outlet_observations(dataset, graph_id)
        count = int(q.numel())
        if count < 2:
            base.update(
                {
                    "calibration_status": "INSUFFICIENT_PAIRED_TRAIN_POINTS",
                    "usable_calibrated": False,
                }
            )
            calibrated[station_id] = base
            continue
        try:
            q_knots, z_knots = _pav_monotone_knots(q, z)
        except ValueError as exc:
            base.update(
                {
                    "calibration_status": f"FAILED: {exc}",
                    "usable_calibrated": False,
                }
            )
            calibrated[station_id] = base
            continue

        prediction = _piecewise_linear_predict(q, q_knots, z_knots)
        residual = prediction - z
        sse = float(residual.square().sum().item())
        rmse = math.sqrt(sse / count)
        z_mean = z.mean()
        sst = float((z - z_mean).square().sum().item())
        nse = 1.0 - sse / sst if sst > 0 else float("nan")
        base.update(
            {
                "calibration_status": "APPLIED",
                "calibration_method": "TRAIN-only PAV monotone regression + piecewise-linear interpolation",
                "calibration_knot_count": len(q_knots),
                "calibrated_q_knots_m3s": q_knots,
                "calibrated_z_knots_m": z_knots,
                "calibrated_fit_rmse_m": rmse,
                "calibrated_fit_nse": nse,
                "usable_calibrated": True,
            }
        )
        calibrated[station_id] = base

    return {
        "mode": "train_unique_target_timestamp_monotone_pwl_pav",
        "computed_from_split": "TRAIN",
        "paired_data_role": "calibrate frozen Q-to-Z observation function only; never a neural Z residual target",
        "stations": calibrated,
    }
