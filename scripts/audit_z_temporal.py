"""Read-only temporal audit for one station's water-level predictability.

This script never constructs a model, never loads a checkpoint, never trains,
and never writes into the experiment output directory.  It uses the formal
loader's physical-unit ``z_history`` and delta-from-t0 ``z_target`` tensors to
check whether hourly stage dynamics are temporally coherent and predictive.

Reported diagnostics include:
- strict Z(t0) reference-vs-history alignment;
- unique-hour stage/increment statistics after de-duplicating overlapping windows;
- lag-1 autocorrelation of consecutive hourly stage increments;
- sign-persistence of consecutive increments;
- correlations between past 1 h / 3 h trends and future h1..h6 delta-Z;
- representative t0-6..t0+6 windows for visual inspection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.common import _make_loader, _runtime_config


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
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f"forecast_time必须整点，实际={value!r}")
    return parsed


def _finite_corr(x: list[float], y: list[float]) -> dict[str, float | int]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(x, y)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    count = len(pairs)
    if count < 2:
        return {"count": count, "corr": float("nan")}
    tx = torch.tensor([a for a, _ in pairs], dtype=torch.float64)
    ty = torch.tensor([b for _, b in pairs], dtype=torch.float64)
    ax = tx - tx.mean()
    ay = ty - ty.mean()
    denominator = torch.sqrt(ax.square().sum() * ay.square().sum())
    if not torch.isfinite(denominator) or float(denominator.item()) <= 0.0:
        corr = float("nan")
    else:
        corr = float((ax * ay).sum().item() / denominator.item())
    return {"count": count, "corr": corr}


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _distribution(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0}
    tensor = torch.tensor(finite, dtype=torch.float64)
    ordered = sorted(finite)
    absolute = [abs(value) for value in finite]
    return {
        "count": len(finite),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(ordered[0]),
        "p01": _quantile(ordered, 0.01),
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": float(ordered[-1]),
        "fraction_exact_zero": sum(value == 0.0 for value in finite) / len(finite),
        "fraction_abs_le_0_01m": sum(value <= 0.01 for value in absolute) / len(finite),
        "fraction_abs_le_0_05m": sum(value <= 0.05 for value in absolute) / len(finite),
        "fraction_abs_ge_0_20m": sum(value >= 0.20 for value in absolute) / len(finite),
        "fraction_abs_ge_0_50m": sum(value >= 0.50 for value in absolute) / len(finite),
    }


def _station_node_index(batch: Any, station_id: str) -> int:
    station_ids = getattr(batch, "station_ids", None)
    if station_ids is None:
        if batch.z_history.shape[2] == 1:
            return 0
        raise ValueError("多节点batch缺少station_ids，无法定位target station")
    candidates = [index for index, value in enumerate(station_ids) if str(value) == station_id]
    if len(candidates) != 1:
        raise ValueError(
            f"target station {station_id!r}在station_ids中应唯一出现一次，实际={candidates}"
        )
    return candidates[0]


def _record_observation(
    observations: dict[datetime, float],
    timestamp: datetime,
    value: float,
    *,
    tolerance: float,
    conflicts: list[dict[str, Any]],
) -> None:
    if timestamp not in observations:
        observations[timestamp] = float(value)
        return
    previous = observations[timestamp]
    difference = abs(previous - float(value))
    if difference > tolerance:
        conflicts.append(
            {
                "timestamp": timestamp.isoformat(sep=" "),
                "existing_z_m": previous,
                "new_z_m": float(value),
                "abs_difference_m": difference,
            }
        )


def _window_payload(row: dict[str, Any], history_back: int = 6) -> dict[str, Any]:
    t0: datetime = row["forecast_time"]
    history = row["history"]
    history_mask = row["history_mask"]
    target = row["target"]
    target_mask = row["target_mask"]
    z0 = row["z0"]
    points: list[dict[str, Any]] = []
    history_hours = len(history)
    start_index = max(0, history_hours - 1 - history_back)
    for index in range(start_index, history_hours):
        offset = index - (history_hours - 1)
        valid = bool(history_mask[index])
        value = float(history[index]) if valid else None
        points.append(
            {
                "offset_h": offset,
                "timestamp": (t0 + timedelta(hours=offset)).isoformat(sep=" "),
                "z_m": value,
                "delta_from_t0_m": (None if value is None else value - z0),
                "valid": valid,
                "source": "history",
            }
        )
    for horizon, (delta, valid) in enumerate(zip(target, target_mask), start=1):
        is_valid = bool(valid)
        value = z0 + float(delta) if is_valid else None
        points.append(
            {
                "offset_h": horizon,
                "timestamp": (t0 + timedelta(hours=horizon)).isoformat(sep=" "),
                "z_m": value,
                "delta_from_t0_m": (float(delta) if is_valid else None),
                "valid": is_valid,
                "source": "target",
            }
        )
    return {
        "sample_id": row["sample_id"],
        "event_id": row["event_id"],
        "target_station_id": row["station_id"],
        "forecast_time": t0.isoformat(sep=" "),
        "z0_m": z0,
        "max_abs_future_delta_z_m": row["max_abs_future_delta"],
        "series": points,
    }


def _evaluate(loader: Any, seed: int, representative_count: int) -> dict[str, Any]:
    rows_by_origin: dict[tuple[str, datetime], dict[str, Any]] = {}
    observations: dict[datetime, float] = {}
    observation_conflicts: list[dict[str, Any]] = []
    duplicate_origin_count = 0
    duplicate_origin_event_conflicts: list[dict[str, Any]] = []
    reference_differences: list[float] = []
    reference_mismatch_count = 0
    sample_rows_seen = 0

    for batch in loader:
        z_history = batch.z_history.detach().cpu().float()
        z_mask = batch.z_mask.detach().cpu().bool()
        z_target = batch.z_target.detach().cpu().float()
        z_target_mask = batch.z_target_mask.detach().cpu().bool()
        if z_history.ndim != 3 or z_target.ndim != 3:
            raise ValueError("z_history/z_target必须为[B,H,N]/[B,F,N]")
        batch_size = z_history.shape[0]
        forecast_times = _as_list(batch.forecast_time, batch_size, "forecast_time")
        event_ids = _as_list(batch.event_id, batch_size, "event_id")
        sample_ids = _as_list(batch.sample_id, batch_size, "sample_id")
        station_ids = _as_list(batch.target_station_id, batch_size, "target_station_id")
        references = None if batch.z_reference is None else batch.z_reference.detach().cpu().float()
        reference_masks = None if batch.z_reference_mask is None else batch.z_reference_mask.detach().cpu().bool()

        for sample_index in range(batch_size):
            sample_rows_seen += 1
            station_id = station_ids[sample_index]
            node = _station_node_index(batch, station_id)
            t0 = _parse_time(forecast_times[sample_index])
            history = z_history[sample_index, :, node]
            history_mask = z_mask[sample_index, :, node]
            target = z_target[sample_index, :, node]
            target_mask = z_target_mask[sample_index, :, node]
            if not bool(history_mask[-1]):
                raise ValueError(
                    f"sample={sample_ids[sample_index]}: t0 Z history无效；delta-Z target契约异常"
                )
            z0 = float(history[-1].item())

            if references is not None and reference_masks is not None:
                if bool(reference_masks[sample_index, node]):
                    reference = float(references[sample_index, node].item())
                    difference = abs(reference - z0)
                    reference_differences.append(difference)
                    if difference > 1.0e-6:
                        reference_mismatch_count += 1

            for history_index in range(history.shape[0]):
                if not bool(history_mask[history_index]):
                    continue
                offset = history_index - (history.shape[0] - 1)
                timestamp = t0 + timedelta(hours=int(offset))
                _record_observation(
                    observations,
                    timestamp,
                    float(history[history_index].item()),
                    tolerance=1.0e-6,
                    conflicts=observation_conflicts,
                )
            for horizon_index in range(target.shape[0]):
                if not bool(target_mask[horizon_index]):
                    continue
                timestamp = t0 + timedelta(hours=horizon_index + 1)
                _record_observation(
                    observations,
                    timestamp,
                    z0 + float(target[horizon_index].item()),
                    tolerance=1.0e-5,
                    conflicts=observation_conflicts,
                )

            key = (station_id, t0)
            future_valid = target[target_mask]
            max_abs_future = (
                float(future_valid.abs().max().item()) if future_valid.numel() else 0.0
            )
            row = {
                "station_id": station_id,
                "forecast_time": t0,
                "event_id": event_ids[sample_index],
                "sample_id": sample_ids[sample_index],
                "history": history.tolist(),
                "history_mask": history_mask.tolist(),
                "target": target.tolist(),
                "target_mask": target_mask.tolist(),
                "z0": z0,
                "max_abs_future_delta": max_abs_future,
            }
            if key in rows_by_origin:
                duplicate_origin_count += 1
                previous = rows_by_origin[key]
                if previous["event_id"] != row["event_id"]:
                    duplicate_origin_event_conflicts.append(
                        {
                            "forecast_time": t0.isoformat(sep=" "),
                            "station_id": station_id,
                            "event_id_existing": previous["event_id"],
                            "event_id_duplicate": row["event_id"],
                        }
                    )
            else:
                rows_by_origin[key] = row

    if not rows_by_origin:
        raise ValueError("没有可审计的forecast-origin样本")

    # Build unique consecutive-hour increments after de-duplicating overlapping windows.
    ordered_times = sorted(observations)
    increments_by_end: dict[datetime, float] = {}
    increments: list[float] = []
    for timestamp in ordered_times:
        previous = timestamp - timedelta(hours=1)
        if previous not in observations:
            continue
        increment = observations[timestamp] - observations[previous]
        increments_by_end[timestamp] = increment
        increments.append(increment)

    lag_previous: list[float] = []
    lag_next: list[float] = []
    same_sign_count = 0
    nonzero_pair_count = 0
    for timestamp, current_increment in increments_by_end.items():
        next_end = timestamp + timedelta(hours=1)
        if next_end not in increments_by_end:
            continue
        next_increment = increments_by_end[next_end]
        lag_previous.append(current_increment)
        lag_next.append(next_increment)
        if current_increment != 0.0 and next_increment != 0.0:
            nonzero_pair_count += 1
            if (current_increment > 0) == (next_increment > 0):
                same_sign_count += 1

    # Forecast-origin trend -> future delta-Z correlations, one row per unique t0.
    rows = list(rows_by_origin.values())
    trend_results: dict[str, Any] = {
        "past_1h_vs_future_delta_z": {},
        "past_3h_mean_slope_vs_future_delta_z": {},
    }
    for horizon_index in range(len(rows[0]["target"])):
        past1: list[float] = []
        past3: list[float] = []
        future1: list[float] = []
        future3: list[float] = []
        for row in rows:
            history = row["history"]
            history_mask = row["history_mask"]
            target = row["target"]
            target_mask = row["target_mask"]
            if bool(target_mask[horizon_index]):
                future = float(target[horizon_index])
                if len(history) >= 2 and bool(history_mask[-1]) and bool(history_mask[-2]):
                    past1.append(float(history[-1] - history[-2]))
                    future1.append(future)
                if len(history) >= 4 and bool(history_mask[-1]) and bool(history_mask[-4]):
                    past3.append(float((history[-1] - history[-4]) / 3.0))
                    future3.append(future)
        trend_results["past_1h_vs_future_delta_z"][f"h{horizon_index + 1}"] = _finite_corr(
            past1, future1
        )
        trend_results["past_3h_mean_slope_vs_future_delta_z"][f"h{horizon_index + 1}"] = _finite_corr(
            past3, future3
        )

    # Representative windows: half strongest future responses, half deterministic random.
    representative_count = max(2, int(representative_count))
    strongest_count = representative_count // 2
    random_count = representative_count - strongest_count
    strongest = sorted(
        rows,
        key=lambda row: row["max_abs_future_delta"],
        reverse=True,
    )[:strongest_count]
    strongest_keys = {(row["station_id"], row["forecast_time"]) for row in strongest}
    remaining = [
        row
        for row in rows
        if (row["station_id"], row["forecast_time"]) not in strongest_keys
    ]
    rng = random.Random(seed)
    random_rows = rng.sample(remaining, min(random_count, len(remaining)))

    max_reference_difference = max(reference_differences) if reference_differences else float("nan")
    result = {
        "sample_rows_seen": sample_rows_seen,
        "unique_forecast_origins": len(rows_by_origin),
        "duplicate_forecast_origin_rows": duplicate_origin_count,
        "duplicate_origin_event_conflict_count": len(duplicate_origin_event_conflicts),
        "duplicate_origin_event_conflicts_preview": duplicate_origin_event_conflicts[:20],
        "forecast_origin_alignment": {
            "reference_comparison_count": len(reference_differences),
            "reference_vs_history_t0_mismatch_count_gt_1e-6m": reference_mismatch_count,
            "max_abs_reference_vs_history_t0_difference_m": max_reference_difference,
        },
        "overlapping_window_consistency": {
            "unique_observed_timestamps": len(observations),
            "conflict_count": len(observation_conflicts),
            "conflicts_preview": observation_conflicts[:20],
        },
        "hourly_stage_increment_m": _distribution(increments),
        "increment_temporal_dependence": {
            "lag1_increment_correlation": _finite_corr(lag_previous, lag_next),
            "nonzero_consecutive_increment_pair_count": nonzero_pair_count,
            "same_sign_fraction_among_nonzero_pairs": (
                same_sign_count / nonzero_pair_count if nonzero_pair_count else float("nan")
            ),
        },
        "past_trend_vs_future_delta_z": trend_results,
        "representative_windows": {
            "strongest_future_response": [_window_payload(row) for row in strongest],
            "deterministic_random": [_window_payload(row) for row in random_rows],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计单站Z时间连续性和ΔZ可预报性")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--graph-id", default=None)
    parser.add_argument(
        "--split", default="VALIDATION", choices=("TRAIN", "VALIDATION", "TEST")
    )
    parser.add_argument("--representative-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    cfg = _runtime_config(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    if str(cfg.get("loss", {}).get("z_target_mode")) != "delta_from_t0":
        raise ValueError("temporal audit要求loss.z_target_mode=delta_from_t0")
    split = str(args.split).upper()
    loader = _make_loader(cfg, split, shuffle=False)
    report = {
        "split": split,
        "graph_id": cfg["data"].get("graph_id"),
        "dataset_root": str(Path(cfg["data"]["dataset_root"]).resolve()),
        "history_hours": int(cfg["history_length"]),
        "forecast_hours": int(cfg["forecast_horizon"]),
        "result": _evaluate(loader, args.seed, args.representative_count),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
