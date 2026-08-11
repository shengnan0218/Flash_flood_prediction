"""Physical-unit Q, absolute-Z and delta-Z evaluation for P2 TEST events."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch


def _strings(batch: Any, name: str, size: int) -> tuple[str, ...]:
    value = getattr(batch, name, None)
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"P2 evaluation要求{name}为逐样本字符串")
    return tuple(str(item) for item in value)


def _regression(rows: Iterable[dict], observed: str, predicted: str) -> dict[str, float | int]:
    pairs = [
        (float(row[observed]), float(row[predicted]))
        for row in rows
        if row.get(observed) not in {None, ""} and row.get(predicted) not in {None, ""}
    ]
    count = len(pairs)
    if not count:
        return {"valid_count": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan, "nse": math.nan}
    errors = [prediction - observation for observation, prediction in pairs]
    mean_observed = sum(observation for observation, _ in pairs) / count
    sse = sum(error * error for error in errors)
    denominator = sum((observation - mean_observed) ** 2 for observation, _ in pairs)
    return {
        "valid_count": count,
        "mae": sum(abs(error) for error in errors) / count,
        "rmse": math.sqrt(sse / count),
        "bias": sum(errors) / count,
        "nse": 1.0 - sse / denominator if denominator > 0 else math.nan,
    }


VARIABLES = {
    "Q": ("Q_OBS", "Q_PRED"),
    "DELTA_Z": ("DELTA_Z_OBS", "DELTA_Z_PRED"),
    "Z": ("Z_OBS", "Z_PRED"),
}


def _group_metrics(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(str(row[key]) for key in keys), []).append(row)
    output: list[dict] = []
    for group, values in sorted(grouped.items()):
        prefix = dict(zip(keys, group))
        for variable, (observed, predicted) in VARIABLES.items():
            output.append({**prefix, "VARIABLE": variable, **_regression(values, observed, predicted)})
    return output


def _peak_metrics(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, int], list[dict]] = {}
    for row in rows:
        if row["Q_OBS"] == "":
            continue
        key = (row["GRAPH_ID"], row["EVENT_ID"], row["STATION_ID"], int(row["HORIZON_HOURS"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict] = []
    for (graph, event, station, horizon), values in sorted(grouped.items()):
        # Repeated target hours at one lead are deterministic; lexical SAMPLE_ID
        # breaks the unlikely tie without consulting observations.
        by_time: dict[str, dict] = {}
        for row in sorted(values, key=lambda item: item["SAMPLE_ID"]):
            by_time.setdefault(row["TARGET_TIME"], row)
        series = list(by_time.values())
        observed_peak = max(series, key=lambda row: float(row["Q_OBS"]))
        predicted_peak = max(series, key=lambda row: float(row["Q_PRED"]))
        observed_time = datetime.fromisoformat(observed_peak["TARGET_TIME"])
        predicted_time = datetime.fromisoformat(predicted_peak["TARGET_TIME"])
        output.append(
            {
                "GRAPH_ID": graph,
                "EVENT_ID": event,
                "STATION_ID": station,
                "HORIZON_HOURS": horizon,
                "Q_PEAK_OBS": float(observed_peak["Q_OBS"]),
                "Q_PEAK_PRED": float(predicted_peak["Q_PRED"]),
                "Q_PEAK_MAGNITUDE_ERROR": float(predicted_peak["Q_PRED"]) - float(observed_peak["Q_OBS"]),
                "Q_PEAK_TIME_OBS": observed_peak["TARGET_TIME"],
                "Q_PEAK_TIME_PRED": predicted_peak["TARGET_TIME"],
                "Q_PEAK_TIMING_ERROR_HOURS": (predicted_time - observed_time).total_seconds() / 3600.0,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_p2_flood_events(
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        q_prediction = output["q"].detach().cpu()
        dz_prediction = output["z"].detach().cpu()
        q_target = batch.q_target.detach().cpu()
        dz_target = batch.z_target.detach().cpu()
        q_mask = batch.q_target_mask.detach().cpu()
        dz_mask = batch.z_target_mask.detach().cpu()
        reference = batch.z_reference.detach().cpu()
        reference_mask = batch.z_reference_mask.detach().cpu()
        size, horizons, nodes = q_prediction.shape
        metadata = {
            name: _strings(batch, name, size)
            for name in ("sample_id", "event_id", "graph_id", "target_station_id", "forecast_time")
        }
        station_ids = batch.station_ids
        if not isinstance(station_ids, tuple) or len(station_ids) != nodes:
            raise ValueError("P2 evaluation缺少station_ids")
        for sample_index in range(size):
            station = metadata["target_station_id"][sample_index]
            node = station_ids.index(station)
            origin = datetime.fromisoformat(metadata["forecast_time"][sample_index])
            for horizon in range(horizons):
                q_valid = bool(q_mask[sample_index, horizon, node])
                dz_valid = bool(dz_mask[sample_index, horizon, node])
                baseline_valid = bool(reference_mask[sample_index, node])
                if not q_valid and not dz_valid:
                    continue
                baseline = float(reference[sample_index, node]) if baseline_valid else math.nan
                dz_obs = float(dz_target[sample_index, horizon, node]) if dz_valid else math.nan
                dz_pred = float(dz_prediction[sample_index, horizon, node]) if dz_valid else math.nan
                rows.append(
                    {
                        "SAMPLE_ID": metadata["sample_id"][sample_index],
                        "EVENT_ID": metadata["event_id"][sample_index],
                        "GRAPH_ID": metadata["graph_id"][sample_index],
                        "STATION_ID": station,
                        "FORECAST_TIME": origin.strftime("%Y-%m-%d %H:%M:%S"),
                        "TARGET_TIME": (origin + timedelta(hours=horizon + 1)).strftime("%Y-%m-%d %H:%M:%S"),
                        "HORIZON_HOURS": horizon + 1,
                        "Q_OBS": float(q_target[sample_index, horizon, node]) if q_valid else "",
                        "Q_PRED": float(q_prediction[sample_index, horizon, node]) if q_valid else "",
                        "DELTA_Z_OBS": dz_obs if dz_valid else "",
                        "DELTA_Z_PRED": dz_pred if dz_valid else "",
                        "Z_T0_OBS": baseline if baseline_valid else "",
                        "Z_OBS": baseline + dz_obs if dz_valid and baseline_valid else "",
                        "Z_PRED": baseline + dz_pred if dz_valid and baseline_valid else "",
                        "Q_MASK": int(q_valid),
                        "DELTA_Z_MASK": int(dz_valid),
                        "QZ_JOINT_MASK": int(q_valid and dz_valid),
                    }
                )
    if not rows:
        raise ValueError("flood-event TEST index没有任何有效Q/ΔZ评价点")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "test_predictions_q_z_delta_z.csv", rows)
    reports = {
        "metrics_overall.csv": _group_metrics(rows, ()),
        "metrics_by_graph.csv": _group_metrics(rows, ("GRAPH_ID",)),
        "metrics_by_station.csv": _group_metrics(rows, ("STATION_ID",)),
        "metrics_by_event.csv": _group_metrics(rows, ("GRAPH_ID", "EVENT_ID")),
        "metrics_by_horizon.csv": _group_metrics(rows, ("HORIZON_HOURS",)),
    }
    for name, report in reports.items():
        _write_csv(output / name, report)
    peaks = _peak_metrics(rows)
    _write_csv(output / "q_peak_metrics_by_event_horizon.csv", peaks)
    q_events = [row for row in reports["metrics_by_event.csv"] if row["VARIABLE"] == "Q"]
    top_errors = sorted(
        q_events,
        key=lambda row: float(row["rmse"]) if math.isfinite(float(row["rmse"])) else -math.inf,
        reverse=True,
    )[:20]
    _write_csv(output / "test_q_top20_error_events.csv", top_errors)
    summary = {
        "sample_horizon_rows": len(rows),
        "events": len({row["EVENT_ID"] for row in rows}),
        "graphs": len({row["GRAPH_ID"] for row in rows}),
        "stations": len({row["STATION_ID"] for row in rows}),
        "q_valid_points": sum(int(row["Q_MASK"]) for row in rows),
        "delta_z_valid_points": sum(int(row["DELTA_Z_MASK"]) for row in rows),
        "qz_joint_valid_points": sum(int(row["QZ_JOINT_MASK"]) for row in rows),
        "normalization_rule": "checkpoint TRAIN-only per-graph Q and per-station delta-Z scales; no TEST fitting",
        "z_reconstruction": "Z_pred(t+h)=observed Z(t0)+predicted delta-Z(t+h)",
        "output_dir": str(output),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
