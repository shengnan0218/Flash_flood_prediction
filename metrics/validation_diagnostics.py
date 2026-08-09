"""Event-, graph-, and station-level validation diagnostics.

The training objective is intentionally not referenced here.  This module only
collects detached physical-unit predictions during evaluation.  Sliding-window
forecasts are converted to one operational rolling series by retaining the
shortest available lead for each real event/station/target timestamp.  Thus a
target hour is counted once instead of up to ``forecast_horizon`` times.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .flood_metrics import (
    masked_regression_sums,
    regression_metric_status,
    regression_metrics,
)


DEDUPLICATION_RULE = (
    "Within each EVENT_ID/station/target timestamp, retain the prediction with "
    "the shortest lead_hours (latest issue time); ties use lexical SAMPLE_ID."
)
DELTA_Z_BASELINE_RULE = (
    "For each retained forecast point, subtract the latest valid observed water "
    "level at the target station within that sample's history window, never later "
    "than FORECAST_TIME, from both observation and prediction."
)
RELATIVE_Q_MIN_M3_S = 1.0
HOURLY_VOLUME_SECONDS = 3600.0


@dataclass(frozen=True)
class ForecastPoint:
    variable: str
    graph_id: str
    event_id: str
    sample_id: str
    station_id: str
    forecast_time: datetime
    target_time: datetime
    lead_hours: int
    observed: float
    predicted: float
    event_rain_start: str
    event_rain_end: str
    event_hydro_start: str
    event_hydro_end: str
    event_peak_time: str
    event_sample_start: str
    event_sample_end: str
    baseline_value: float | None = None
    baseline_time: datetime | None = None
    candidate_count: int = 1


def _parse_time(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是有效ISO日期时间，实际为{value!r}") from exc
    if parsed.tzinfo is not None:
        raise ValueError(f"{name}必须使用数据集已规范化的无时区时间")
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f"{name}必须对齐整点，实际为{value!r}")
    return parsed


def _batch_strings(
    batch: Any,
    name: str,
    batch_size: int,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = getattr(batch, name, None)
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (tuple, list)):
        values = tuple(value)
    else:
        raise ValueError(f"validation diagnostics要求batch.{name}为逐样本字符串")
    if len(values) != batch_size:
        raise ValueError(
            f"batch.{name}数量必须等于batch size={batch_size}，实际={len(values)}"
        )
    if any(not isinstance(item, str) or (not allow_empty and not item.strip()) for item in values):
        qualifier = "字符串" if allow_empty else "非空字符串"
        raise ValueError(f"batch.{name}必须全部为{qualifier}")
    return values


def latest_history_baseline(
    history: torch.Tensor,
    mask: torch.Tensor,
    station_index: int,
    forecast_time: datetime,
) -> tuple[float, datetime] | None:
    """Return the latest causal observed Z baseline in one sample history."""

    if history.ndim != 2 or mask.shape != history.shape:
        raise ValueError("ΔZ基准要求history/mask形状均为[H,N]")
    if not 0 <= station_index < history.shape[1]:
        raise ValueError("ΔZ基准站点索引越界")
    valid = mask[:, station_index].bool().nonzero(as_tuple=False).flatten()
    if not valid.numel():
        return None
    position = int(valid[-1].item())
    value = float(history[position, station_index].item())
    if not math.isfinite(value):
        raise ValueError("ΔZ有效历史基准包含NaN/Inf")
    age_hours = int(history.shape[0]) - 1 - position
    baseline_time = forecast_time - timedelta(hours=age_hours)
    if baseline_time > forecast_time:  # defensive proof of the causal contract
        raise AssertionError("ΔZ基准时间晚于起报时刻，发生未来信息泄漏")
    return value, baseline_time


def deduplicate_shortest_lead(
    points: Iterable[ForecastPoint],
) -> list[ForecastPoint]:
    """Select one deterministic rolling forecast for every real target hour."""

    grouped: dict[tuple[str, str, str, str, datetime], list[ForecastPoint]] = {}
    for point in points:
        key = (
            point.variable,
            point.graph_id,
            point.event_id,
            point.station_id,
            point.target_time,
        )
        grouped.setdefault(key, []).append(point)
    selected: list[ForecastPoint] = []
    for candidates in grouped.values():
        choice = min(candidates, key=lambda item: (item.lead_hours, item.sample_id))
        selected.append(replace(choice, candidate_count=len(candidates)))
    return sorted(
        selected,
        key=lambda item: (
            item.graph_id,
            item.event_id,
            item.station_id,
            item.target_time,
            item.variable,
        ),
    )


def _regression_sums_from_values(
    predicted: Iterable[float], observed: Iterable[float]
) -> dict[str, float | int]:
    prediction = torch.tensor(list(predicted), dtype=torch.float64)
    target = torch.tensor(list(observed), dtype=torch.float64)
    mask = torch.ones_like(target, dtype=torch.bool)
    return masked_regression_sums(prediction, target, mask)


def _quantile(values: Iterable[float], probability: float) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("nan")
    position = (len(finite) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _metric_fields(
    sums: dict[str, float | int], prefix: str
) -> dict[str, float | int | str]:
    metrics = regression_metrics(sums)
    statuses = regression_metric_status(sums)
    return {
        f"{prefix}_nse": metrics["nse"],
        f"{prefix}_nse_status": statuses["nse"],
        f"{prefix}_kge": metrics["kge"],
        f"{prefix}_kge_status": statuses["kge"],
        f"{prefix}_mae": metrics["mae"],
        f"{prefix}_rmse": metrics["rmse"],
        f"{prefix}_bias": metrics["bias"],
    }


def _group_points(
    points: Iterable[ForecastPoint], key
) -> dict[Any, list[ForecastPoint]]:
    result: dict[Any, list[ForecastPoint]] = {}
    for point in points:
        result.setdefault(key(point), []).append(point)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class ValidationDiagnostics:
    summary_metrics: dict[str, float | int]
    summary: dict[str, Any]
    q_by_graph: list[dict[str, Any]]
    q_by_event: list[dict[str, Any]]
    q_top20_error_events: list[dict[str, Any]]
    q_top20_sse_events: list[dict[str, Any]]
    z_by_station: list[dict[str, Any]]
    delta_z_by_station: list[dict[str, Any]]

    def write(
        self,
        output_dir: str | Path,
        *,
        split: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Overwrite one bounded set of detailed diagnostics atomically."""

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        prefix = split.strip().lower()
        tables = {
            f"{prefix}_q_by_graph.csv": self.q_by_graph,
            f"{prefix}_q_by_event.csv": self.q_by_event,
            f"{prefix}_q_top20_error_events.csv": self.q_top20_error_events,
            f"{prefix}_q_top20_sse_events.csv": self.q_top20_sse_events,
            f"{prefix}_z_by_station.csv": self.z_by_station,
            f"{prefix}_delta_z_by_station.csv": self.delta_z_by_station,
        }
        written: dict[str, str] = {}
        for filename, rows in tables.items():
            path = output / filename
            fieldnames = (
                list(rows[0])
                if rows
                else _EMPTY_TABLE_FIELDS[filename[len(prefix) + 1 :]]
            )
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
            written[filename] = str(path)
        summary = dict(self.summary)
        if context:
            summary["run_context"] = dict(context)
        summary_path = output / f"{prefix}_diagnostics_summary.json"
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
        written[summary_path.name] = str(summary_path)
        return written


_EMPTY_TABLE_FIELDS = {
    "q_by_graph.csv": [
        "GRAPH_ID", "q_nse", "q_kge", "q_mae", "q_rmse", "q_bias",
        "valid_count", "event_count",
    ],
    "q_by_event.csv": [
        "GRAPH_ID", "EVENT_ID", "target_station_id", "valid_q_count", "q_sse",
    ],
    "q_top20_error_events.csv": [
        "rank", "GRAPH_ID", "EVENT_ID", "q_rmse",
    ],
    "q_top20_sse_events.csv": [
        "rank", "GRAPH_ID", "EVENT_ID", "q_sse",
        "sse_fraction_of_total", "cumulative_sse_fraction",
    ],
    "z_by_station.csv": [
        "station_id", "z_nse", "z_kge", "z_mae", "z_rmse", "z_bias",
        "valid_count", "event_count",
    ],
    "delta_z_by_station.csv": [
        "station_id", "delta_z_nse", "delta_z_mae", "delta_z_rmse",
        "delta_z_bias", "valid_count", "event_count",
    ],
}


class ValidationDiagnosticsAccumulator:
    """Collect detached formal-validation predictions with strict metadata."""

    def __init__(self) -> None:
        self.q_points: list[ForecastPoint] = []
        self.z_points: list[ForecastPoint] = []
        self.event_sample_ids: dict[tuple[str, str], set[str]] = {}

    def add_batch(self, batch: Any, output: Mapping[str, Any]) -> None:
        q_prediction = output["q"].detach().cpu()
        z_prediction = output["z"].detach().cpu()
        q_target = batch.q_target.detach().cpu()
        z_target = batch.z_target.detach().cpu()
        q_mask = batch.q_target_mask.detach().cpu()
        z_mask = batch.z_target_mask.detach().cpu()
        z_history = batch.z_history.detach().cpu()
        z_history_mask = batch.z_mask.detach().cpu()
        if q_prediction.ndim != 3 or z_prediction.shape != q_prediction.shape:
            raise ValueError("validation diagnostics要求Q/Z预测形状一致且为[B,F,N]")
        batch_size, forecast_hours, nodes = q_prediction.shape
        names = (
            "sample_id",
            "event_id",
            "graph_id",
            "target_station_id",
            "forecast_time",
            "event_rain_start",
            "event_rain_end",
            "event_hydro_start",
            "event_hydro_end",
            "event_peak_time",
            "event_sample_start",
            "event_sample_end",
        )
        values = {
            name: _batch_strings(
                batch,
                name,
                batch_size,
                allow_empty=name == "event_hydro_end",
            )
            for name in names
        }
        station_ids = getattr(batch, "station_ids", None)
        if not isinstance(station_ids, tuple) or len(station_ids) != nodes:
            raise ValueError("validation diagnostics要求batch.station_ids按节点顺序提供")

        for sample_index in range(batch_size):
            station_id = values["target_station_id"][sample_index]
            try:
                node_index = station_ids.index(station_id)
            except ValueError as exc:
                raise ValueError(
                    f"target_station_id={station_id!r}不在batch.station_ids中"
                ) from exc
            forecast_time = _parse_time(
                values["forecast_time"][sample_index], "FORECAST_TIME"
            )
            baseline = latest_history_baseline(
                z_history[sample_index],
                z_history_mask[sample_index],
                node_index,
                forecast_time,
            )
            metadata = {
                name: values[name][sample_index]
                for name in (
                    "event_rain_start",
                    "event_rain_end",
                    "event_hydro_start",
                    "event_hydro_end",
                    "event_peak_time",
                    "event_sample_start",
                    "event_sample_end",
                )
            }
            graph_id = values["graph_id"][sample_index]
            event_id = values["event_id"][sample_index]
            sample_id = values["sample_id"][sample_index]
            self.event_sample_ids.setdefault((graph_id, event_id), set()).add(sample_id)
            for horizon in range(forecast_hours):
                target_time = forecast_time + timedelta(hours=horizon + 1)
                common = {
                    "graph_id": graph_id,
                    "event_id": event_id,
                    "sample_id": sample_id,
                    "station_id": station_id,
                    "forecast_time": forecast_time,
                    "target_time": target_time,
                    "lead_hours": horizon + 1,
                    **metadata,
                }
                if bool(q_mask[sample_index, horizon, node_index]):
                    self.q_points.append(
                        ForecastPoint(
                            variable="Q",
                            observed=float(q_target[sample_index, horizon, node_index]),
                            predicted=float(q_prediction[sample_index, horizon, node_index]),
                            **common,
                        )
                    )
                if bool(z_mask[sample_index, horizon, node_index]):
                    baseline_value, baseline_time = (
                        baseline if baseline is not None else (None, None)
                    )
                    self.z_points.append(
                        ForecastPoint(
                            variable="Z",
                            observed=float(z_target[sample_index, horizon, node_index]),
                            predicted=float(z_prediction[sample_index, horizon, node_index]),
                            baseline_value=baseline_value,
                            baseline_time=baseline_time,
                            **common,
                        )
                    )

    def finalize(self) -> ValidationDiagnostics:
        q_unique = deduplicate_shortest_lead(self.q_points)
        z_unique = deduplicate_shortest_lead(self.z_points)
        q_by_event = self._q_event_rows(q_unique)
        q_by_graph = self._q_graph_rows(q_unique)
        z_by_station = self._z_station_rows(z_unique, delta=False)
        delta_by_station = self._z_station_rows(z_unique, delta=True)
        q_total_sse = sum(float(row["q_sse"]) for row in q_by_event)
        error_top = sorted(
            q_by_event,
            key=lambda row: (float(row["q_rmse"]), float(row["q_sse"])),
            reverse=True,
        )[:20]
        q_top20_error = [
            {"rank": rank, "ranking_metric": "q_rmse_desc", **row}
            for rank, row in enumerate(error_top, 1)
        ]
        cumulative = 0.0
        q_top20_sse: list[dict[str, Any]] = []
        for rank, row in enumerate(
            sorted(q_by_event, key=lambda item: float(item["q_sse"]), reverse=True)[:20],
            1,
        ):
            fraction = float(row["q_sse"]) / q_total_sse if q_total_sse > 0 else float("nan")
            cumulative += fraction if math.isfinite(fraction) else 0.0
            q_top20_sse.append(
                {
                    "rank": rank,
                    "ranking_metric": "q_sse_desc",
                    **row,
                    "sse_fraction_of_total": fraction,
                    "cumulative_sse_fraction": cumulative if q_total_sse > 0 else float("nan"),
                }
            )

        delta_points = [point for point in z_unique if point.baseline_value is not None]
        delta_sums = _regression_sums_from_values(
            (point.predicted - float(point.baseline_value) for point in delta_points),
            (point.observed - float(point.baseline_value) for point in delta_points),
        )
        delta_metrics = regression_metrics(delta_sums)
        summary_metrics: dict[str, float | int] = {
            "q_graph_nse_median": _quantile((row["q_nse"] for row in q_by_graph), 0.5),
            "q_graph_kge_median": _quantile((row["q_kge"] for row in q_by_graph), 0.5),
            "z_station_nse_median": _quantile((row["z_nse"] for row in z_by_station), 0.5),
            "z_station_kge_median": _quantile((row["z_kge"] for row in z_by_station), 0.5),
            "z_station_mae_median": _quantile((row["z_mae"] for row in z_by_station), 0.5),
            "z_station_nse_p25": _quantile((row["z_nse"] for row in z_by_station), 0.25),
            "z_station_nse_p75": _quantile((row["z_nse"] for row in z_by_station), 0.75),
            "z_station_kge_p25": _quantile((row["z_kge"] for row in z_by_station), 0.25),
            "z_station_kge_p75": _quantile((row["z_kge"] for row in z_by_station), 0.75),
            "z_station_mae_p25": _quantile((row["z_mae"] for row in z_by_station), 0.25),
            "z_station_mae_p75": _quantile((row["z_mae"] for row in z_by_station), 0.75),
            "z_station_valid_count": len(z_by_station),
            "z_station_nse_defined_count": sum(
                math.isfinite(float(row["z_nse"])) for row in z_by_station
            ),
            "z_station_kge_defined_count": sum(
                math.isfinite(float(row["z_kge"])) for row in z_by_station
            ),
            "delta_z_mae": delta_metrics["mae"],
            "delta_z_rmse": delta_metrics["rmse"],
            "delta_z_nse": delta_metrics["nse"],
            "delta_z_bias": delta_metrics["bias"],
            "delta_z_station_nse_median": _quantile(
                (row["delta_z_nse"] for row in delta_by_station), 0.5
            ),
            "delta_z_station_mae_median": _quantile(
                (row["delta_z_mae"] for row in delta_by_station), 0.5
            ),
            "delta_z_station_valid_count": len(delta_by_station),
            "delta_z_station_nse_defined_count": sum(
                math.isfinite(float(row["delta_z_nse"]))
                for row in delta_by_station
            ),
        }
        raw_q_sums = _regression_sums_from_values(
            (point.predicted for point in self.q_points),
            (point.observed for point in self.q_points),
        )
        summary = {
            "deduplication_rule": DEDUPLICATION_RULE,
            "delta_z_baseline_rule": DELTA_Z_BASELINE_RULE,
            "relative_q_denominator_rule": (
                f"Relative peak/volume error is defined only when observed peak "
                f"or mean valid discharge is >= {RELATIVE_Q_MIN_M3_S:g} m3/s."
            ),
            "total_q_sse": q_total_sse,
            "window_weighted_total_q_sse": float(raw_q_sums["squared_error"]),
            "q_valid_forecast_points_before_dedup": len(self.q_points),
            "q_unique_event_time_points": len(q_unique),
            "q_duplicate_weight_factor": (
                len(self.q_points) / len(q_unique) if q_unique else float("nan")
            ),
            "top_1_event_sse_fraction": self._top_sse_fraction(q_by_event, q_total_sse, 1),
            "top_5_event_sse_fraction": self._top_sse_fraction(q_by_event, q_total_sse, 5),
            "top_10_event_sse_fraction": self._top_sse_fraction(q_by_event, q_total_sse, 10),
            "top_20_event_sse_fraction": self._top_sse_fraction(q_by_event, q_total_sse, 20),
            "worst_graph_by_q_nse": self._worst_graph(q_by_graph),
            "median_graph_q_nse": summary_metrics["q_graph_nse_median"],
            "median_graph_q_kge": summary_metrics["q_graph_kge_median"],
            "median_station_z_nse": summary_metrics["z_station_nse_median"],
            "median_station_z_kge": summary_metrics["z_station_kge_median"],
            "delta_z_overall_metrics": {
                **{
                    name: value
                    for name, value in delta_metrics.items()
                    if name != "kge"
                },
                "nse_status": regression_metric_status(delta_sums)["nse"],
                "skipped_missing_baseline_count": len(z_unique) - len(delta_points),
            },
            "counts": {
                "q_events": len(q_by_event),
                "q_graphs": len(q_by_graph),
                "z_stations": len(z_by_station),
                "delta_z_stations": len(delta_by_station),
                "z_unique_event_time_points": len(z_unique),
                "delta_z_valid_points": len(delta_points),
            },
        }
        return ValidationDiagnostics(
            summary_metrics,
            summary,
            q_by_graph,
            q_by_event,
            q_top20_error,
            q_top20_sse,
            z_by_station,
            delta_by_station,
        )

    @staticmethod
    def _top_sse_fraction(rows: list[dict[str, Any]], total: float, count: int) -> float:
        if total <= 0:
            return float("nan")
        ordered = sorted((float(row["q_sse"]) for row in rows), reverse=True)
        return sum(ordered[:count]) / total

    @staticmethod
    def _worst_graph(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        defined = [row for row in rows if math.isfinite(float(row["q_nse"]))]
        if not defined:
            return None
        row = min(defined, key=lambda item: float(item["q_nse"]))
        return {"GRAPH_ID": row["GRAPH_ID"], "q_nse": row["q_nse"]}

    def _q_graph_rows(self, points: list[ForecastPoint]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for graph_id, group in sorted(_group_points(points, lambda point: point.graph_id).items()):
            sums = _regression_sums_from_values(
                (point.predicted for point in group),
                (point.observed for point in group),
            )
            rows.append(
                {
                    "GRAPH_ID": graph_id,
                    **_metric_fields(sums, "q"),
                    "q_sse": float(sums["squared_error"]),
                    "valid_count": int(sums["count"]),
                    "event_count": len({point.event_id for point in group}),
                    "raw_window_point_count": sum(point.candidate_count for point in group),
                    "aggregation_rule": "shortest_lead_unique_event_station_target_time",
                }
            )
        return rows

    def _q_event_rows(self, points: list[ForecastPoint]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grouped = _group_points(points, lambda point: (point.graph_id, point.event_id))
        for (graph_id, event_id), group in sorted(grouped.items()):
            ordered = sorted(group, key=lambda point: point.target_time)
            first = ordered[0]
            sums = _regression_sums_from_values(
                (point.predicted for point in ordered),
                (point.observed for point in ordered),
            )
            observed_peak = max(
                ordered,
                key=lambda point: (point.observed, -point.target_time.timestamp()),
            )
            predicted_peak = max(
                ordered,
                key=lambda point: (point.predicted, -point.target_time.timestamp()),
            )
            peak_error = predicted_peak.predicted - observed_peak.observed
            relative_peak = (
                peak_error / observed_peak.observed
                if observed_peak.observed >= RELATIVE_Q_MIN_M3_S
                else float("nan")
            )
            timing_error = (
                predicted_peak.target_time - observed_peak.target_time
            ).total_seconds() / HOURLY_VOLUME_SECONDS
            observed_volume = (
                sum(point.observed for point in ordered) * HOURLY_VOLUME_SECONDS
            )
            predicted_volume = (
                sum(point.predicted for point in ordered) * HOURLY_VOLUME_SECONDS
            )
            volume_error = predicted_volume - observed_volume
            mean_observed_q = observed_volume / (len(ordered) * HOURLY_VOLUME_SECONDS)
            relative_volume = (
                volume_error / observed_volume
                if observed_volume > 0 and mean_observed_q >= RELATIVE_Q_MIN_M3_S
                else float("nan")
            )
            rows.append(
                {
                    "GRAPH_ID": graph_id,
                    "EVENT_ID": event_id,
                    "target_station_id": first.station_id,
                    "rain_start": first.event_rain_start,
                    "rain_end": first.event_rain_end,
                    "hydro_start": first.event_hydro_start,
                    "hydro_end": first.event_hydro_end,
                    "official_event_peak_time": first.event_peak_time,
                    "event_sample_start": first.event_sample_start,
                    "event_sample_end": first.event_sample_end,
                    "evaluated_target_start": ordered[0].target_time.isoformat(sep=" "),
                    "evaluated_target_end": ordered[-1].target_time.isoformat(sep=" "),
                    "sample_count": len(self.event_sample_ids.get((graph_id, event_id), set())),
                    "raw_valid_q_forecast_point_count": sum(
                        point.candidate_count for point in ordered
                    ),
                    "valid_q_count": len(ordered),
                    "peak_obs": observed_peak.observed,
                    "peak_pred": predicted_peak.predicted,
                    "peak_error": peak_error,
                    "absolute_peak_error": abs(peak_error),
                    "relative_peak_error": relative_peak,
                    "relative_peak_error_status": (
                        "DEFINED"
                        if math.isfinite(relative_peak)
                        else "OBS_PEAK_BELOW_1_M3_S"
                    ),
                    "observed_peak_time": observed_peak.target_time.isoformat(sep=" "),
                    "predicted_peak_time": predicted_peak.target_time.isoformat(sep=" "),
                    "peak_timing_error_hours": timing_error,
                    "absolute_peak_timing_error_hours": abs(timing_error),
                    "observed_volume": observed_volume,
                    "predicted_volume": predicted_volume,
                    "volume_error": volume_error,
                    "relative_volume_error": relative_volume,
                    "relative_volume_error_status": (
                        "DEFINED"
                        if math.isfinite(relative_volume)
                        else "OBS_MEAN_Q_BELOW_1_M3_S"
                    ),
                    "volume_unit": "m3_over_valid_unique_hourly_points",
                    **_metric_fields(sums, "q"),
                    "q_sse": float(sums["squared_error"]),
                    "aggregation_rule": "shortest_lead_unique_event_station_target_time",
                }
            )
        return rows

    def _z_station_rows(
        self, points: list[ForecastPoint], *, delta: bool
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grouped = _group_points(points, lambda point: point.station_id)
        for station_id, station_points in sorted(grouped.items()):
            usable = (
                [point for point in station_points if point.baseline_value is not None]
                if delta
                else station_points
            )
            if not usable:
                continue
            predicted = [
                point.predicted - float(point.baseline_value)
                if delta
                else point.predicted
                for point in usable
            ]
            observed = [
                point.observed - float(point.baseline_value)
                if delta
                else point.observed
                for point in usable
            ]
            sums = _regression_sums_from_values(predicted, observed)
            prefix = "delta_z" if delta else "z"
            metric_fields = _metric_fields(sums, prefix)
            if delta:
                # Delta-Z can have a mean close to zero, which makes KGE's
                # bias ratio unstable and hard to interpret.  Do not emit it.
                metric_fields.pop("delta_z_kge")
                metric_fields.pop("delta_z_kge_status")
            row: dict[str, Any] = {
                "station_id": station_id,
                "graph_ids": ";".join(sorted({point.graph_id for point in usable})),
                **metric_fields,
                "valid_count": int(sums["count"]),
                "event_count": len({point.event_id for point in usable}),
                "raw_window_point_count": sum(point.candidate_count for point in usable),
                "aggregation_rule": "shortest_lead_unique_event_station_target_time",
            }
            if delta:
                row["skipped_missing_baseline_count"] = len(station_points) - len(usable)
                row["baseline_rule"] = "latest_valid_history_z_at_or_before_forecast_time"
            rows.append(row)
        return rows
