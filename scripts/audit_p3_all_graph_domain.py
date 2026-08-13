"""Read-only all-graph audit for P3 event-domain calibrated training.

This script does not train or modify the dataset.  It builds the exact current
P3 calibrated TRAIN/VALIDATION runtime, then reports per-graph/per-station loss
scales, supervision support, rating quality, and validation Q/Z target ranges.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.p3_rating_calibrated_runtime import setup_training_rating_calibrated


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _samples_by_graph(dataset: Any) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = defaultdict(list)
    for sample in dataset._samples:
        result[sample.graph_id].append(sample)
    return dict(result)


def _split_support(dataset: Any) -> dict[str, dict[str, Any]]:
    grouped = _samples_by_graph(dataset)
    rows: dict[str, dict[str, Any]] = {}
    horizons = torch.arange(1, dataset.forecast_hours + 1, dtype=torch.long)
    for graph_id in sorted(dataset.graph_ids):
        samples = grouped.get(graph_id, [])
        graph = dataset._graphs[graph_id]
        outlet = next(node.node_index for node in graph.nodes if node.is_outlet)
        dynamic = dataset._dynamic[graph_id]
        q_used = torch.zeros(len(dynamic.timestamps), dtype=torch.bool)
        z_used = torch.zeros(len(dynamic.timestamps), dtype=torch.bool)
        q_window_points = 0
        z_window_points = 0
        for start in range(0, len(samples), 100_000):
            chunk = samples[start : start + 100_000]
            if not chunk:
                continue
            origins = torch.tensor(
                [dataset._origin_index(sample) for sample in chunk], dtype=torch.long
            )
            future = origins.unsqueeze(1) + horizons.unsqueeze(0)
            q_mask = dynamic.flow_mask[future, outlet]
            z_mask = dynamic.water_level_mask[future, outlet]
            q_window_points += int(q_mask.sum().item())
            z_window_points += int(z_mask.sum().item())
            q_used[future[q_mask]] = True
            z_used[future[z_mask]] = True
        q_indices = q_used.nonzero(as_tuple=False).flatten()
        z_indices = z_used.nonzero(as_tuple=False).flatten()
        q = dynamic.flow[q_indices, outlet].to(torch.float64)
        z = dynamic.water_level[z_indices, outlet].to(torch.float64)
        rows[graph_id] = {
            "sample_count": len(samples),
            "event_count": len({sample.event_id for sample in samples if sample.event_id}),
            "q_window_valid_point_count": q_window_points,
            "z_window_valid_point_count": z_window_points,
            "q_unique_target_time_count": int(q.numel()),
            "z_unique_target_time_count": int(z.numel()),
            "q_mean_m3s": float(q.mean().item()) if q.numel() else None,
            "q_std_m3s": float(q.std(unbiased=False).item()) if q.numel() > 1 else None,
            "q_min_m3s": float(q.min().item()) if q.numel() else None,
            "q_max_m3s": float(q.max().item()) if q.numel() else None,
            "z_mean_m": float(z.mean().item()) if z.numel() else None,
            "z_std_m": float(z.std(unbiased=False).item()) if z.numel() > 1 else None,
            "z_min_m": float(z.min().item()) if z.numel() else None,
            "z_max_m": float(z.max().item()) if z.numel() else None,
        }
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计P3全域事件训练的scale/rating/support")
    parser.add_argument(
        "--config",
        default="configs/hunan_p3_state_init_all_graphs_event_domain_rating_calibrated_stable.yaml",
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/p3_all_graph_domain_audit",
    )
    args = parser.parse_args()

    cfg, model, train_loader, validation_loader, _ = setup_training_rating_calibrated(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=None,
    )
    train = train_loader.dataset
    validation = validation_loader.dataset
    if model.independent_z_head is not None:
        raise RuntimeError("审计入口检测到neural Z residual head，当前并非calibrated P3")

    target_stats = train.train_target_statistics()
    q_stats = target_stats.get("q_by_graph", {})
    dz_stats = target_stats.get("delta_z_by_station", {})
    rating = cfg["_runtime"].get("rating_curves", {}).get("stations", {})
    input_stats = cfg["_runtime"].get("input_normalization", {})
    flow_input = input_stats.get("flow_by_station", {})
    dz_input = input_stats.get("relative_z_by_station", {})
    train_support = _split_support(train)
    val_support = _split_support(validation)

    graph_rows: list[dict[str, Any]] = []
    for graph_id in sorted(train.graph_ids):
        graph = train._graphs[graph_id]
        station = graph.outlet_id
        qs = q_stats.get(graph_id, {})
        dzs = dz_stats.get(station, {})
        rf = rating.get(station, {})
        fi = flow_input.get(station, {})
        zi = dz_input.get(station, {})
        dz_scale = _finite(dzs.get("std_m"))
        if dz_scale is not None:
            dz_scale = max(dz_scale, float(cfg["loss"]["delta_z_scale_floor_m"]))
        rating_rmse = _finite(rf.get("calibrated_fit_rmse_m"))
        row: dict[str, Any] = {
            "graph_id": graph_id,
            "station_id": station,
            "train_sample_count": train_support[graph_id]["sample_count"],
            "train_event_count": train_support[graph_id]["event_count"],
            "validation_sample_count": val_support[graph_id]["sample_count"],
            "validation_event_count": val_support[graph_id]["event_count"],
            "train_q_unique_supervision_count": qs.get("valid_unique_point_count"),
            "train_q_mean_m3s": qs.get("mean_m3s"),
            "train_q_raw_std_m3s": qs.get("std_m3s"),
            "train_q_loss_scale_m3s": max(
                float(qs.get("std_m3s", 0.0)),
                float(cfg["loss"]["q_scale_floor_m3s"]),
            ) if qs else None,
            "train_delta_z_valid_point_count": dzs.get("valid_point_count"),
            "train_delta_z_raw_std_m": dzs.get("std_m"),
            "train_delta_z_loss_scale_m": dz_scale,
            "flow_input_scale_m3s": fi.get("scale_m3s"),
            "delta_z_input_scale_m": zi.get("scale_m"),
            "rating_status": rf.get("calibration_status"),
            "rating_paired_count": rf.get("calibration_raw_paired_count", rf.get("paired_unique_point_count")),
            "rating_unique_q_count": rf.get("calibration_unique_q_count"),
            "rating_knot_count": rf.get("calibration_knot_count"),
            "rating_linear_fit_rmse_m": rf.get("fit_rmse_m"),
            "rating_linear_fit_nse": rf.get("fit_nse"),
            "rating_calibrated_fit_rmse_m": rating_rmse,
            "rating_calibrated_fit_nse": rf.get("calibrated_fit_nse"),
            "rating_rmse_over_delta_z_scale": (
                rating_rmse / dz_scale
                if rating_rmse is not None and dz_scale is not None and dz_scale > 0
                else None
            ),
            "rating_interior_slope_min": rf.get("interior_slope_min_m_per_m3s"),
            "rating_interior_slope_max": rf.get("interior_slope_max_m_per_m3s"),
            "rating_extrapolation_slope": rf.get("extrapolation_slope_m_per_m3s"),
        }
        for prefix, support in (("train", train_support[graph_id]), ("validation", val_support[graph_id])):
            for key, value in support.items():
                if key in {"sample_count", "event_count"}:
                    continue
                row[f"{prefix}_{key}"] = value
        graph_rows.append(row)

    out = Path(args.output_dir)
    _write_csv(out / "graph_station_audit.csv", graph_rows)

    ratio_rows = sorted(
        [row for row in graph_rows if _finite(row.get("rating_rmse_over_delta_z_scale")) is not None],
        key=lambda row: float(row["rating_rmse_over_delta_z_scale"]),
        reverse=True,
    )
    q_range_rows = sorted(
        [row for row in graph_rows if _finite(row.get("validation_q_max_m3s")) is not None],
        key=lambda row: float(row["validation_q_max_m3s"]),
        reverse=True,
    )
    q_std_rows = sorted(
        [row for row in graph_rows if _finite(row.get("validation_q_std_m3s")) is not None],
        key=lambda row: float(row["validation_q_std_m3s"]),
        reverse=True,
    )
    fit_rows = sorted(
        [row for row in graph_rows if _finite(row.get("rating_calibrated_fit_nse")) is not None],
        key=lambda row: float(row["rating_calibrated_fit_nse"]),
    )

    summary = {
        "config": args.config,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "train_sampling_mode": getattr(train, "train_sampling_mode", None),
        "train_batch_sampler": type(train_loader.batch_sampler).__name__,
        "train_sample_weight_present": train[0].sample_weight is not None,
        "graph_count": len(graph_rows),
        "train_sample_count": len(train),
        "validation_sample_count": len(validation),
        "train_event_count": len({sample.event_id for sample in train._samples if sample.event_id}),
        "validation_event_count": len({sample.event_id for sample in validation._samples if sample.event_id}),
        "rating_unusable_stations": [
            row["station_id"] for row in graph_rows if row.get("rating_status") != "APPLIED"
        ],
        "worst_rating_rmse_over_delta_z_scale": ratio_rows[:15],
        "lowest_rating_calibrated_nse": fit_rows[:15],
        "largest_validation_q_max": q_range_rows[:15],
        "largest_validation_q_std": q_std_rows[:15],
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print(json.dumps({
        "output_dir": str(out.resolve()),
        "graph_count": len(graph_rows),
        "train_sampling_mode": summary["train_sampling_mode"],
        "train_sample_weight_present": summary["train_sample_weight_present"],
        "worst_rating_rmse_over_delta_z_scale": [
            {
                "graph_id": row["graph_id"],
                "station_id": row["station_id"],
                "ratio": row["rating_rmse_over_delta_z_scale"],
                "rating_rmse_m": row["rating_calibrated_fit_rmse_m"],
                "delta_z_scale_m": row["train_delta_z_loss_scale_m"],
                "rating_nse": row["rating_calibrated_fit_nse"],
            }
            for row in ratio_rows[:10]
        ],
        "largest_validation_q_std": [
            {
                "graph_id": row["graph_id"],
                "station_id": row["station_id"],
                "q_std_m3s": row["validation_q_std_m3s"],
                "q_max_m3s": row["validation_q_max_m3s"],
            }
            for row in q_std_rows[:10]
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
