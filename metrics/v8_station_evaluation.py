"""Station/basin/event/lead-time evaluation for v8 sparse observations.

All reported Q metrics are in m3/s and all Z metrics use the frozen v8 target
semantics: Delta-Z(t+h)=Z(t+h)-Z(t0), in metres.  The evaluator never
broadcasts sparse observations to computational nodes and never refits any
normalization statistics.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import torch

from metrics.flood_metrics import (
    hydrograph_sample_sums,
    masked_regression_sums,
    regression_metric_status,
    regression_metrics,
)


_REGRESSION_SUM_KEYS = (
    "count",
    "absolute_error",
    "squared_error",
    "error",
    "prediction",
    "target",
    "prediction_squared",
    "target_squared",
    "cross",
)
_HYDRO_SUM_KEYS = (
    "peak_absolute_error",
    "peak_signed_error",
    "peak_relative_error",
    "peak_timing_absolute_error",
    "peak_timing_signed_error",
    "peak_count",
    "peak_relative_count",
    "relative_volume_error",
    "volume_count",
)


def _empty_regression() -> dict[str, float | int]:
    return {
        "count": 0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "error": 0.0,
        "prediction": 0.0,
        "target": 0.0,
        "prediction_squared": 0.0,
        "target_squared": 0.0,
        "cross": 0.0,
    }


def _empty_hydrograph() -> dict[str, float | int]:
    return {
        "peak_absolute_error": 0.0,
        "peak_signed_error": 0.0,
        "peak_relative_error": 0.0,
        "peak_timing_absolute_error": 0.0,
        "peak_timing_signed_error": 0.0,
        "peak_count": 0,
        "peak_relative_count": 0,
        "relative_volume_error": 0.0,
        "volume_count": 0,
    }


def _merge(target: dict[str, float | int], source: Mapping[str, float | int]) -> None:
    for key in target:
        target[key] = target[key] + source[key]


def _regression_fields(prefix: str, sums: Mapping[str, float | int]) -> dict[str, Any]:
    metrics = regression_metrics(dict(sums))
    status = regression_metric_status(dict(sums))
    return {
        f"{prefix}_valid_count": int(metrics["valid_count"]),
        f"{prefix}_mae": float(metrics["mae"]),
        f"{prefix}_rmse": float(metrics["rmse"]),
        f"{prefix}_bias": float(metrics["bias"]),
        f"{prefix}_nse": float(metrics["nse"]),
        f"{prefix}_kge": float(metrics["kge"]),
        f"{prefix}_nse_status": status["nse"],
        f"{prefix}_kge_status": status["kge"],
    }


def _hydrograph_fields(prefix: str, sums: Mapping[str, float | int]) -> dict[str, Any]:
    peak_count = int(sums["peak_count"])
    relative_count = int(sums["peak_relative_count"])
    volume_count = int(sums["volume_count"])

    def mean(name: str, count: int) -> float:
        return float(sums[name]) / count if count else float("nan")

    return {
        f"{prefix}_window_count": peak_count,
        f"{prefix}_window_peak_mae": mean("peak_absolute_error", peak_count),
        f"{prefix}_window_peak_bias": mean("peak_signed_error", peak_count),
        f"{prefix}_window_relative_peak_bias": mean(
            "peak_relative_error", relative_count
        ),
        f"{prefix}_window_peak_timing_mae_h": mean(
            "peak_timing_absolute_error", peak_count
        ),
        f"{prefix}_window_peak_timing_bias_h": mean(
            "peak_timing_signed_error", peak_count
        ),
        f"{prefix}_window_relative_volume_bias": mean(
            "relative_volume_error", volume_count
        ),
    }


def _finite_macro(rows: Iterable[Mapping[str, Any]], field: str) -> tuple[float, int]:
    values = []
    for row in rows:
        value = row.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return (sum(values) / len(values) if values else float("nan"), len(values))


def _macro_regression(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("mae", "rmse", "bias", "nse", "kge"):
        value, count = _finite_macro(rows, f"{prefix}_{metric}")
        result[metric] = value
        result[f"{metric}_defined_count"] = count
    result["station_count"] = len(rows)
    result["station_with_target_count"] = sum(
        int(row.get(f"{prefix}_valid_count", 0)) > 0 for row in rows
    )
    return result


def _macro_hydrograph(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, target in (
        ("window_peak_mae", "window_peak_mae"),
        ("window_peak_bias", "window_peak_bias"),
        ("window_relative_peak_bias", "window_relative_peak_bias"),
        ("window_peak_timing_mae_h", "window_peak_timing_mae_h"),
        ("window_peak_timing_bias_h", "window_peak_timing_bias_h"),
        ("window_relative_volume_bias", "window_relative_volume_bias"),
    ):
        value, count = _finite_macro(rows, f"{prefix}_{source}")
        result[target] = value
        result[f"{target}_defined_count"] = count
    return result


def _normalise_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool_value(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def _batch_strings(value: Any, batch_size: int, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) * batch_size
    if isinstance(value, (tuple, list)) and len(value) == batch_size:
        return tuple(str(item) for item in value)
    raise ValueError(f"v8 evaluation要求{label}包含每个sample的元数据")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _load_station_metadata(dataset: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    path = Path(dataset.root) / "graph/station_observation_mapping.csv"
    if not path.is_file():
        raise FileNotFoundError(f"v8 evaluation缺少station mapping: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    required = {"GRAPH_ID", "STATION_ID", "IS_OUTLET_STATION"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"station mapping缺少字段: {sorted(missing)}")
    frame["GRAPH_ID"] = frame["GRAPH_ID"].map(_normalise_id)
    frame["STATION_ID"] = frame["STATION_ID"].map(_normalise_id)
    if frame.duplicated(["GRAPH_ID", "STATION_ID"]).any():
        raise ValueError("station mapping含重复GRAPH_ID/STATION_ID")

    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    outlet_by_graph: dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        graph_id = row["GRAPH_ID"]
        station_id = row["STATION_ID"]
        is_outlet = _bool_value(row["IS_OUTLET_STATION"])
        role = row.get("STATION_ROLE") or ("OUTLET" if is_outlet else "INTERNAL")
        metadata[(graph_id, station_id)] = {
            "graph_id": graph_id,
            "station_id": station_id,
            "station_role": role,
            "is_outlet_station": is_outlet,
        }
        if is_outlet:
            if graph_id in outlet_by_graph:
                raise ValueError(f"{graph_id}: station mapping存在多个outlet station")
            outlet_by_graph[graph_id] = station_id
    expected_graphs = set(getattr(dataset, "graph_ids", ()))
    if expected_graphs - set(outlet_by_graph):
        raise ValueError(
            f"evaluation graph缺少outlet station: {sorted(expected_graphs-set(outlet_by_graph))}"
        )
    return metadata, outlet_by_graph


@torch.no_grad()
def evaluate_v8_station_aware(
    trainer: Any,
    loader: Iterable[Any],
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Evaluate v8 once and write five compact, paper-oriented output files."""

    model = trainer.model
    device = trainer.device
    loss_engine = trainer.loss_engine
    model.eval()
    dataset = loader.dataset
    station_meta, outlet_by_graph = _load_station_metadata(dataset)

    q_global = _empty_regression()
    z_global = _empty_regression()
    q_hydro_global = _empty_hydrograph()
    station_q: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        _empty_regression
    )
    station_z: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        _empty_regression
    )
    station_q_hydro: dict[tuple[str, str], dict[str, float | int]] = defaultdict(
        _empty_hydrograph
    )
    graph_q: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
    graph_z: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
    graph_outlet_q: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
    graph_outlet_z: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
    graph_outlet_hydro: dict[str, dict[str, float | int]] = defaultdict(
        _empty_hydrograph
    )
    event_q: dict[tuple[str, str, str], dict[str, float | int]] = defaultdict(
        _empty_regression
    )
    event_z: dict[tuple[str, str, str], dict[str, float | int]] = defaultdict(
        _empty_regression
    )
    event_q_hydro: dict[tuple[str, str, str], dict[str, float | int]] = defaultdict(
        _empty_hydrograph
    )
    event_sample_count: dict[tuple[str, str, str], int] = defaultdict(int)

    horizon = int(trainer.cfg["forecast_horizon"])
    lead_global = {
        task: [_empty_regression() for _ in range(horizon)] for task in ("q", "z")
    }
    lead_station = {
        task: defaultdict(lambda: [_empty_regression() for _ in range(horizon)])
        for task in ("q", "z")
    }

    coefficients = loss_engine.coefficients()
    loss_totals = {name: [0.0, 0] for name in coefficients}
    q_valid_total = 0
    z_valid_total = 0
    sample_total = 0
    observed_event_ids: set[str] = set()

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        q_prediction = output["q"]
        z_prediction = output["z"]
        batch_size = int(q_prediction.shape[0])
        if q_prediction.shape != batch.q_target.shape:
            raise ValueError("v8 evaluation Q prediction/target shape不一致")
        if z_prediction.shape != batch.z_target.shape:
            raise ValueError("v8 evaluation Delta-Z prediction/target shape不一致")
        if q_prediction.shape[1] != horizon:
            raise ValueError("v8 evaluation forecast horizon不一致")

        statistics = loss_engine.batch_statistics(output, batch)
        for name, term in statistics.items():
            loss_totals[name][0] += float(term.numerator.detach().item())
            loss_totals[name][1] += int(term.denominator)
        q_valid_total += int(batch.q_target_mask.sum().item())
        z_valid_total += int(batch.z_target_mask.sum().item())
        sample_total += batch_size

        graph_ids = _batch_strings(batch.graph_id, batch_size, "graph_id")
        event_ids = _batch_strings(batch.event_id, batch_size, "event_id")
        graph_id = _normalise_id(graph_ids[0])
        if any(_normalise_id(value) != graph_id for value in graph_ids):
            raise ValueError("v8 evaluation batch混入多个graph")
        observed_event_ids.update(event_ids)

        _merge(q_global, masked_regression_sums(q_prediction, batch.q_target, batch.q_target_mask))
        _merge(z_global, masked_regression_sums(z_prediction, batch.z_target, batch.z_target_mask))
        _merge(q_hydro_global, hydrograph_sample_sums(q_prediction, batch.q_target, batch.q_target_mask))
        _merge(graph_q[graph_id], masked_regression_sums(q_prediction, batch.q_target, batch.q_target_mask))
        _merge(graph_z[graph_id], masked_regression_sums(z_prediction, batch.z_target, batch.z_target_mask))

        outlet_station = outlet_by_graph[graph_id]
        try:
            outlet_position = batch.obs_station_ids.index(outlet_station)
        except ValueError as exc:
            raise ValueError(
                f"{graph_id}: batch obs_station_ids缺少正式outlet={outlet_station}"
            ) from exc
        outlet_slice = slice(outlet_position, outlet_position + 1)
        _merge(
            graph_outlet_q[graph_id],
            masked_regression_sums(
                q_prediction[:, :, outlet_slice],
                batch.q_target[:, :, outlet_slice],
                batch.q_target_mask[:, :, outlet_slice],
            ),
        )
        _merge(
            graph_outlet_z[graph_id],
            masked_regression_sums(
                z_prediction[:, :, outlet_slice],
                batch.z_target[:, :, outlet_slice],
                batch.z_target_mask[:, :, outlet_slice],
            ),
        )
        _merge(
            graph_outlet_hydro[graph_id],
            hydrograph_sample_sums(
                q_prediction[:, :, outlet_slice],
                batch.q_target[:, :, outlet_slice],
                batch.q_target_mask[:, :, outlet_slice],
            ),
        )

        for lead in range(horizon):
            _merge(
                lead_global["q"][lead],
                masked_regression_sums(
                    q_prediction[:, lead : lead + 1],
                    batch.q_target[:, lead : lead + 1],
                    batch.q_target_mask[:, lead : lead + 1],
                ),
            )
            _merge(
                lead_global["z"][lead],
                masked_regression_sums(
                    z_prediction[:, lead : lead + 1],
                    batch.z_target[:, lead : lead + 1],
                    batch.z_target_mask[:, lead : lead + 1],
                ),
            )

        for station_position, raw_station_id in enumerate(batch.obs_station_ids):
            station_id = _normalise_id(raw_station_id)
            key = (graph_id, station_id)
            if key not in station_meta:
                raise ValueError(f"batch station未出现在mapping: {key}")
            station_slice = slice(station_position, station_position + 1)
            q_station_sums = masked_regression_sums(
                q_prediction[:, :, station_slice],
                batch.q_target[:, :, station_slice],
                batch.q_target_mask[:, :, station_slice],
            )
            z_station_sums = masked_regression_sums(
                z_prediction[:, :, station_slice],
                batch.z_target[:, :, station_slice],
                batch.z_target_mask[:, :, station_slice],
            )
            _merge(station_q[key], q_station_sums)
            _merge(station_z[key], z_station_sums)
            _merge(
                station_q_hydro[key],
                hydrograph_sample_sums(
                    q_prediction[:, :, station_slice],
                    batch.q_target[:, :, station_slice],
                    batch.q_target_mask[:, :, station_slice],
                ),
            )

            for lead in range(horizon):
                _merge(
                    lead_station["q"][key][lead],
                    masked_regression_sums(
                        q_prediction[:, lead : lead + 1, station_slice],
                        batch.q_target[:, lead : lead + 1, station_slice],
                        batch.q_target_mask[:, lead : lead + 1, station_slice],
                    ),
                )
                _merge(
                    lead_station["z"][key][lead],
                    masked_regression_sums(
                        z_prediction[:, lead : lead + 1, station_slice],
                        batch.z_target[:, lead : lead + 1, station_slice],
                        batch.z_target_mask[:, lead : lead + 1, station_slice],
                    ),
                )

            for sample_index, event_id in enumerate(event_ids):
                event_key = (str(event_id), graph_id, station_id)
                event_sample_count[event_key] += 1
                sample_slice = slice(sample_index, sample_index + 1)
                _merge(
                    event_q[event_key],
                    masked_regression_sums(
                        q_prediction[sample_slice, :, station_slice],
                        batch.q_target[sample_slice, :, station_slice],
                        batch.q_target_mask[sample_slice, :, station_slice],
                    ),
                )
                _merge(
                    event_z[event_key],
                    masked_regression_sums(
                        z_prediction[sample_slice, :, station_slice],
                        batch.z_target[sample_slice, :, station_slice],
                        batch.z_target_mask[sample_slice, :, station_slice],
                    ),
                )
                _merge(
                    event_q_hydro[event_key],
                    hydrograph_sample_sums(
                        q_prediction[sample_slice, :, station_slice],
                        batch.q_target[sample_slice, :, station_slice],
                        batch.q_target_mask[sample_slice, :, station_slice],
                    ),
                )

    if sample_total == 0:
        raise ValueError("v8 evaluation loader为空")

    station_rows: list[dict[str, Any]] = []
    for key in sorted(station_meta):
        if key[0] not in set(getattr(dataset, "graph_ids", ())):
            continue
        meta = station_meta[key]
        row = {
            "GRAPH_ID": meta["graph_id"],
            "STATION_ID": meta["station_id"],
            "STATION_ROLE": meta["station_role"],
            "IS_OUTLET_STATION": int(meta["is_outlet_station"]),
        }
        row.update(_regression_fields("q", station_q[key]))
        row.update(_hydrograph_fields("q", station_q_hydro[key]))
        row.update(_regression_fields("delta_z", station_z[key]))
        station_rows.append(row)

    graph_rows: list[dict[str, Any]] = []
    for graph_id in sorted(getattr(dataset, "graph_ids", ())):
        graph_station_rows = [row for row in station_rows if row["GRAPH_ID"] == graph_id]
        outlet_station_id = outlet_by_graph[graph_id]
        row = {
            "GRAPH_ID": graph_id,
            "OBS_STATION_COUNT": len(graph_station_rows),
            "OUTLET_STATION_ID": outlet_station_id,
        }
        row.update(_regression_fields("allobs_q", graph_q[graph_id]))
        row.update(_regression_fields("allobs_delta_z", graph_z[graph_id]))
        row.update(_regression_fields("outlet_q", graph_outlet_q[graph_id]))
        row.update(_hydrograph_fields("outlet_q", graph_outlet_hydro[graph_id]))
        row.update(_regression_fields("outlet_delta_z", graph_outlet_z[graph_id]))
        graph_rows.append(row)

    event_rows: list[dict[str, Any]] = []
    for event_key in sorted(event_sample_count):
        event_id, graph_id, station_id = event_key
        meta = station_meta[(graph_id, station_id)]
        row = {
            "EVENT_ID": event_id,
            "GRAPH_ID": graph_id,
            "STATION_ID": station_id,
            "STATION_ROLE": meta["station_role"],
            "IS_OUTLET_STATION": int(meta["is_outlet_station"]),
            "FORECAST_WINDOW_COUNT": event_sample_count[event_key],
        }
        row.update(_regression_fields("q", event_q[event_key]))
        row.update(_hydrograph_fields("q", event_q_hydro[event_key]))
        row.update(_regression_fields("delta_z", event_z[event_key]))
        event_rows.append(row)

    lead_rows: list[dict[str, Any]] = []
    outlet_keys = {key for key, meta in station_meta.items() if meta["is_outlet_station"]}
    active_station_keys = [
        key for key in sorted(station_meta) if key[0] in set(getattr(dataset, "graph_ids", ()))
    ]
    for task, prefix in (("Q", "q"), ("DELTA_Z", "delta_z")):
        storage_key = "q" if task == "Q" else "z"
        for lead in range(horizon):
            pooled = _regression_fields(prefix, lead_global[storage_key][lead])
            lead_rows.append(
                {
                    "TASK": task,
                    "LEAD_HOUR": lead + 1,
                    "SCOPE": "POOLED_ALL_OBS",
                    **pooled,
                }
            )
            station_metric_rows = [
                _regression_fields(prefix, lead_station[storage_key][key][lead])
                for key in active_station_keys
            ]
            macro = _macro_regression(station_metric_rows, prefix)
            lead_rows.append(
                {
                    "TASK": task,
                    "LEAD_HOUR": lead + 1,
                    "SCOPE": "MACRO_STATION",
                    **macro,
                }
            )
            outlet_metric_rows = [
                _regression_fields(prefix, lead_station[storage_key][key][lead])
                for key in active_station_keys
                if key in outlet_keys
            ]
            outlet_macro = _macro_regression(outlet_metric_rows, prefix)
            lead_rows.append(
                {
                    "TASK": task,
                    "LEAD_HOUR": lead + 1,
                    "SCOPE": "MACRO_OUTLET",
                    **outlet_macro,
                }
            )

    loss_report = loss_engine.report(
        {
            name: (float(value), int(count))
            for name, (value, count) in loss_totals.items()
        },
        q_valid_count=q_valid_total,
        z_valid_count=z_valid_total,
    )
    outlet_station_rows = [row for row in station_rows if row["IS_OUTLET_STATION"] == 1]
    summary = {
        "split": str(split).upper(),
        "samples": sample_total,
        "events": len(observed_event_ids),
        "graphs": len(getattr(dataset, "graph_ids", ())),
        "observation_stations": len(station_rows),
        "outlet_stations": len(outlet_station_rows),
        "internal_stations": len(station_rows) - len(outlet_station_rows),
        "target_semantics": {
            "q": "physical discharge Q in m3/s",
            "z": "Delta-Z(t+h)=Z(t+h)-Z(t0) in m",
        },
        "loss": loss_report,
        "q": {
            "pooled_all_observations": _regression_fields("q", q_global),
            "macro_station": _macro_regression(station_rows, "q"),
            "macro_outlet": _macro_regression(outlet_station_rows, "q"),
            "window_hydrograph_pooled": _hydrograph_fields("q", q_hydro_global),
            "window_hydrograph_macro_station": _macro_hydrograph(station_rows, "q"),
            "window_hydrograph_macro_outlet": _macro_hydrograph(
                outlet_station_rows, "q"
            ),
        },
        "delta_z": {
            "pooled_all_observations": _regression_fields("delta_z", z_global),
            "macro_station": _macro_regression(station_rows, "delta_z"),
            "macro_outlet": _macro_regression(outlet_station_rows, "delta_z"),
        },
        "metric_interpretation": {
            "macro_station": "unweighted arithmetic mean across stations where the metric is defined",
            "macro_outlet": "unweighted arithmetic mean across the 33 basin outlet stations where the metric is defined",
            "event_station": "metrics aggregate all six-hour forecast windows belonging to one frozen hydrologic event and station; peak/timing/volume fields are window-level, not a reconstructed single event hydrograph",
        },
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    station_path = destination / "station_metrics.csv"
    graph_path = destination / "graph_metrics.csv"
    event_path = destination / "event_station_metrics.csv"
    lead_path = destination / "lead_time_metrics.csv"
    summary_path = destination / "evaluation_summary.json"
    pd.DataFrame(station_rows).to_csv(station_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(graph_rows).to_csv(graph_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(event_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(lead_rows).to_csv(lead_path, index=False, encoding="utf-8-sig")
    summary_payload = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "dataset_root": str(Path(dataset.root).resolve()),
        "data_contract": dataset.contract.get("contract"),
        "summary": summary,
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary_payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary": _json_safe(summary),
        "output_dir": str(destination),
        "files": {
            "summary": str(summary_path),
            "station_metrics": str(station_path),
            "graph_metrics": str(graph_path),
            "event_station_metrics": str(event_path),
            "lead_time_metrics": str(lead_path),
        },
    }
