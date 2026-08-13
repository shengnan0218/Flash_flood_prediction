"""TRAIN-only deterministic calibration of station Q->Z rating functions.

Paired Q/Z observations calibrate only the frozen observation function. They
never train a neural Z residual.  To keep the derivative stable in all-domain
training, raw pairs are first aggregated into a data-sized number of Q-quantile
bins (Sturges rule), then a weighted pool-adjacent-violators regression enforces
monotonicity.  The resulting low-degree-of-freedom knot sequence is interpolated
inside the observed TRAIN range.  Outside that range, extrapolation uses the
station's global TRAIN OLS Q-Z slope rather than a noisy edge-local segment.
"""
from __future__ import annotations

import math
from typing import Any

import torch


def _paired_train_outlet_observations(
    dataset: Any, graph_id: str
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _global_ols(q: torch.Tensor, z: torch.Tensor) -> tuple[float, float]:
    if q.numel() < 2:
        raise ValueError("OLS至少需要2个配对点")
    q_mean = q.mean()
    z_mean = z.mean()
    anomaly = q - q_mean
    denominator = anomaly.square().sum()
    if float(denominator.item()) <= 0.0:
        raise ValueError("TRAIN配对Q无变异，无法拟合rating slope")
    slope_tensor = (anomaly * (z - z_mean)).sum() / denominator
    slope = float(slope_tensor.item())
    intercept = float((z_mean - slope_tensor * q_mean).item())
    if not math.isfinite(slope) or slope <= 0.0 or not math.isfinite(intercept):
        raise ValueError("TRAIN全局Q-Z OLS不是有限正斜率关系")
    return slope, intercept


def _sturges_bin_count(sample_count: int, unique_q_count: int) -> int:
    """Data-derived low-DOF bin count; no station-specific tuning constant."""
    if sample_count < 2 or unique_q_count < 2:
        raise ValueError("rating calibration需要至少2个样本和2个唯一Q值")
    proposed = int(math.ceil(math.log2(sample_count) + 1.0))
    return max(2, min(proposed, unique_q_count))


def _quantile_bin_centroids(
    q: torch.Tensor, z: torch.Tensor, bin_count: int
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate sorted paired observations into approximately equal-count bins."""
    order = torch.argsort(q)
    q_sorted = q[order]
    z_sorted = z[order]
    n = int(q.numel())
    boundaries = [int(round(index * n / bin_count)) for index in range(bin_count + 1)]
    boundaries[0] = 0
    boundaries[-1] = n
    q_bins: list[float] = []
    z_bins: list[float] = []
    weights: list[float] = []
    for index in range(bin_count):
        start = boundaries[index]
        end = boundaries[index + 1]
        if end <= start:
            continue
        q_part = q_sorted[start:end]
        z_part = z_sorted[start:end]
        q_value = float(q_part.mean().item())
        z_value = float(z_part.mean().item())
        weight = float(end - start)
        if q_bins and math.isclose(q_value, q_bins[-1], rel_tol=0.0, abs_tol=1.0e-12):
            combined_weight = weights[-1] + weight
            z_bins[-1] = (
                z_bins[-1] * weights[-1] + z_value * weight
            ) / combined_weight
            q_bins[-1] = (
                q_bins[-1] * weights[-1] + q_value * weight
            ) / combined_weight
            weights[-1] = combined_weight
        else:
            q_bins.append(q_value)
            z_bins.append(z_value)
            weights.append(weight)
    if len(q_bins) < 2:
        raise ValueError("Q分位聚合后不足2个唯一bin")
    return q_bins, z_bins, weights


def _weighted_pav_knots(
    q_values: list[float],
    z_values: list[float],
    weights: list[float],
) -> tuple[list[float], list[float]]:
    """Weighted PAV over aggregated bins, returning strictly monotone centroids."""
    if not (len(q_values) == len(z_values) == len(weights)) or len(q_values) < 2:
        raise ValueError("weighted PAV输入长度无效")
    blocks: list[dict[str, float]] = []
    for q_value, z_value, weight in zip(q_values, z_values, weights):
        if weight <= 0:
            raise ValueError("PAV bin weight必须为正")
        blocks.append(
            {
                "weight": weight,
                "sum_q": q_value * weight,
                "sum_z": z_value * weight,
            }
        )
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["sum_z"] / left["weight"]
            right_mean = right["sum_z"] / right["weight"]
            if left_mean < right_mean:
                break
            blocks[-2:] = [
                {
                    "weight": left["weight"] + right["weight"],
                    "sum_q": left["sum_q"] + right["sum_q"],
                    "sum_z": left["sum_z"] + right["sum_z"],
                }
            ]
    q_knots = [block["sum_q"] / block["weight"] for block in blocks]
    z_knots = [block["sum_z"] / block["weight"] for block in blocks]
    if len(q_knots) < 2:
        raise ValueError("单调聚合后只剩一个block")
    if any(b <= a for a, b in zip(q_knots, q_knots[1:])):
        raise ValueError("calibrated Q knots未严格递增")
    if any(b <= a for a, b in zip(z_knots, z_knots[1:])):
        raise ValueError("calibrated Z knots未严格递增")
    return q_knots, z_knots


def _stable_piecewise_predict(
    q: torch.Tensor,
    q_knots: list[float],
    z_knots: list[float],
    extrapolation_slope: float,
) -> torch.Tensor:
    x = torch.tensor(q_knots, dtype=torch.float64)
    y = torch.tensor(z_knots, dtype=torch.float64)
    right = torch.searchsorted(x, q).clamp(1, x.numel() - 1)
    left = right - 1
    x0 = x[left]
    x1 = x[right]
    y0 = y[left]
    y1 = y[right]
    local_slope = (y1 - y0) / (x1 - x0)
    inside = y0 + local_slope * (q - x0)
    below = y[0] + extrapolation_slope * (q - x[0])
    above = y[-1] + extrapolation_slope * (q - x[-1])
    return torch.where(q < x[0], below, torch.where(q > x[-1], above, inside))


def _segment_slopes(q_knots: list[float], z_knots: list[float]) -> list[float]:
    return [
        (z1 - z0) / (q1 - q0)
        for q0, q1, z0, z1 in zip(
            q_knots, q_knots[1:], z_knots, z_knots[1:]
        )
    ]


def fit_train_monotone_rating_statistics(dataset: Any) -> dict[str, Any]:
    """Fit frozen low-DOF monotone station rating functions from TRAIN only."""
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
        unique_q = int(torch.unique(q).numel()) if count else 0
        if count < 2 or unique_q < 2:
            base.update(
                {
                    "calibration_status": "INSUFFICIENT_PAIRED_TRAIN_VARIATION",
                    "usable_calibrated": False,
                }
            )
            calibrated[station_id] = base
            continue
        try:
            global_slope, global_intercept = _global_ols(q, z)
            bin_count = _sturges_bin_count(count, unique_q)
            q_bins, z_bins, weights = _quantile_bin_centroids(q, z, bin_count)
            q_knots, z_knots = _weighted_pav_knots(q_bins, z_bins, weights)
        except ValueError as exc:
            base.update(
                {
                    "calibration_status": f"FAILED: {exc}",
                    "usable_calibrated": False,
                }
            )
            calibrated[station_id] = base
            continue

        prediction = _stable_piecewise_predict(q, q_knots, z_knots, global_slope)
        residual = prediction - z
        sse = float(residual.square().sum().item())
        rmse = math.sqrt(sse / count)
        z_mean = z.mean()
        sst = float((z - z_mean).square().sum().item())
        nse = 1.0 - sse / sst if sst > 0 else float("nan")
        slopes = _segment_slopes(q_knots, z_knots)
        base.update(
            {
                "calibration_status": "APPLIED",
                "calibration_method": (
                    "TRAIN-only Sturges Q-quantile aggregation + weighted PAV "
                    "+ piecewise-linear interpolation; global TRAIN OLS slope extrapolation"
                ),
                "calibration_raw_paired_count": count,
                "calibration_unique_q_count": unique_q,
                "calibration_bin_count": bin_count,
                "calibration_knot_count": len(q_knots),
                "calibrated_q_knots_m3s": q_knots,
                "calibrated_z_knots_m": z_knots,
                "calibrated_fit_rmse_m": rmse,
                "calibrated_fit_nse": nse,
                "interior_slope_min_m_per_m3s": min(slopes),
                "interior_slope_max_m_per_m3s": max(slopes),
                "extrapolation_slope_m_per_m3s": global_slope,
                "global_ols_intercept_m": global_intercept,
                "usable_calibrated": True,
            }
        )
        calibrated[station_id] = base

    return {
        "mode": "train_unique_pairs_low_dof_monotone_pwl_sturges_pav",
        "computed_from_split": "TRAIN",
        "paired_data_role": (
            "calibrate frozen Q-to-Z observation function only; never a neural Z residual target"
        ),
        "extrapolation": "station global TRAIN OLS slope anchored at calibrated edge knot",
        "stations": calibrated,
    }
