"""Paper-oriented station/basin/event/lead-time evaluation for v8.

Q is reported in m3/s. Z follows the frozen v8 target semantics
Delta-Z(t+h)=Z(t+h)-Z(t0), in metres. Sparse observations remain on Nobs;
no station value is broadcast to computational nodes and no statistics are fit
at evaluation time.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from metrics.flood_metrics import regression_metric_status, regression_metrics


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


def _array_sums(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("evaluation prediction/target/mask shape必须一致")
    valid_prediction = prediction[mask]
    valid_target = target[mask]
    if not valid_target.size:
        return _empty_regression()
    if not np.isfinite(valid_prediction).all():
        raise FloatingPointError("evaluation有效预测含NaN/Inf")
    if not np.isfinite(valid_target).all():
        raise ValueError("evaluation有效target含NaN/Inf")
    error = valid_prediction - valid_target
    return {
        "count": int(valid_target.size),
        "absolute_error": float(np.abs(error).sum()),
        "squared_error": float(np.square(error).sum()),
        "error": float(error.sum()),
        "prediction": float(valid_prediction.sum()),
        "target": float(valid_target.sum()),
        "prediction_squared": float(np.square(valid_prediction).sum()),
        "target_squared": float(np.square(valid_target).sum()),
        "cross": float((valid_prediction * valid_target).sum()),
    }


def _array_hydrograph(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    """Window-level Q peak/timing/volume metrics for [B,H,Nobs]."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("hydrograph prediction/target/mask shape必须一致")
    if prediction.ndim != 3:
        raise ValueError("hydrograph evaluation要求[B,H,Nobs]")
    out = _empty_hydrograph()
    for sample_index in range(prediction.shape[0]):
        for station_index in range(prediction.shape[2]):
            valid = np.flatnonzero(mask[sample_index, :, station_index])
            if not valid.size:
                continue
            pred = prediction[sample_index, valid, station_index]
            obs = target[sample_index, valid, station_index]
            if not np.isfinite(pred).all() or not np.isfinite(obs).all():
                raise FloatingPointError("hydrograph有效预测/target含NaN/Inf")
            predicted_peak = float(pred.max())
            observed_peak = float(obs.max())
            peak_error = predicted_peak - observed_peak
            out["peak_absolute_error"] += abs(peak_error)
            out["peak_signed_error"] += peak_error
            if observed_peak != 0:
                out["peak_relative_error"] += peak_error / observed_peak
                out["peak_relative_count"] += 1
            predicted_time = int(valid[int(pred.argmax())])
            observed_time = int(valid[int(obs.argmax())])
            timing_error = predicted_time - observed_time
            out["peak_timing_absolute_error"] += abs(timing_error)
            out["peak_timing_signed_error"] += timing_error
            out["peak_count"] += 1
            observed_volume = float(obs.sum())
            if observed_volume != 0:
                out["relative_volume_error"] += float(pred.sum() - obs.sum()) / observed_volume
                out["volume_count"] += 1
    return out


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
        f"{prefix}_window_relative_peak_bias": mean("peak_relative_error", relative_count),
        f"{prefix}_window_peak_timing_mae_h": mean("peak_timing_absolute_error", peak_count),
        f"{prefix}_window_peak_timing_bias_h": mean("peak_timing_signed_error", peak_count),
        f"{prefix}_window_relative_volume_bias": mean("relative_volume_error", volume_count),
    }


def _finite_macro(rows: Iterable[Mapping[str, Any]], field: str) -> tuple[float, int]:
    values = [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float)) and math.isfinite(float(row[field]))
    ]
    return (sum(values) / len(values) if values else float("nan"), len(values))


def _macro_regression(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "station_count": len(rows),
        "station_with_target_count": sum(
            int(row.get(f"{prefix}_valid_count", 0)) > 0 for row in rows
        ),
    }
    for metric in ("mae", "rmse", "bias", "nse", "kge"):
        value, count = _finite_macro(rows, f"{prefix}_{metric}")
        result[metric] = value
        result[f"{metric}_defined_count"] = count
    return result


def _macro_hydrograph(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in (
        "window_peak_mae",
        "window_peak_bias",
        "window_relative_peak_bias",
        "window_peak_timing_mae_h",
        "window_peak_timing_bias_h",
        "window_relative_volume_bias",
    ):
        value, count = _finite_macro(rows, f"{prefix}_{metric}")
        result[metric] = value
        result[f"{metric}_defined_count"] = count
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


def _load_station_metadata(dataset: Any):
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
        graph_id, station_id = row["GRAPH_ID"], row["STATION_ID"]
        is_outlet = _bool_value(row["IS_OUTLET_STATION"])
        metadata[(graph_id, station_id)] = {
            "graph_id": graph_id,
            "station_id": station_id,
            "station_role": row.get("STATION_ROLE") or ("OUTLET" if is_outlet else "INTERNAL"),
            "is_outlet_station": is_outlet,
        }
        if is_outlet:
            if graph_id in outlet_by_graph:
                raise ValueError(f"{graph_id}: 存在多个outlet station")
            outlet_by_graph[graph_id] = station_id
    active_graphs = set(getattr(dataset, "graph_ids", ()))
    if active_graphs - set(outlet_by_graph):
        raise ValueError(f"evaluation graph缺少outlet station: {sorted(active_graphs-set(outlet_by_graph))}")
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
    """Evaluate the model once and write five compact final-evaluation files."""

    model, device, loss_engine = trainer.model, trainer.device, trainer.loss_engine
    model.eval()
    dataset = loader.dataset
    station_meta, outlet_by_graph = _load_station_metadata(dataset)
    active_graphs = set(getattr(dataset, "graph_ids", ()))

    q_global, z_global = _empty_regression(), _empty_regression()
    q_hydro_global = _empty_hydrograph()
    station_q = defaultdict(_empty_regression)
    station_z = defaultdict(_empty_regression)
    station_hydro = defaultdict(_empty_hydrograph)
    graph_q = defaultdict(_empty_regression)
    graph_z = defaultdict(_empty_regression)
    graph_outlet_q = defaultdict(_empty_regression)
    graph_outlet_z = defaultdict(_empty_regression)
    graph_outlet_hydro = defaultdict(_empty_hydrograph)
    event_q = defaultdict(_empty_regression)
    event_z = defaultdict(_empty_regression)
    event_hydro = defaultdict(_empty_hydrograph)
    event_sample_count: dict[tuple[str, str, str], int] = defaultdict(int)

    horizon = int(trainer.cfg["forecast_horizon"])
    lead_global = {
        task: [_empty_regression() for _ in range(horizon)] for task in ("q", "z")
    }
    lead_station = {
        task: defaultdict(lambda: [_empty_regression() for _ in range(horizon)])
        for task in ("q", "z")
    }
    loss_totals = {name: [0.0, 0] for name in loss_engine.coefficients()}
    q_valid_total = z_valid_total = sample_total = 0
    observed_event_ids: set[str] = set()

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        q_pred_t, z_pred_t = output["q"], output["z"]
        if q_pred_t.shape != batch.q_target.shape or z_pred_t.shape != batch.z_target.shape:
            raise ValueError("v8 evaluation prediction/target shape不一致")
        batch_size = int(q_pred_t.shape[0])
        if q_pred_t.shape[1] != horizon:
            raise ValueError("v8 evaluation forecast horizon不一致")

        statistics = loss_engine.batch_statistics(output, batch)
        for name, term in statistics.items():
            loss_totals[name][0] += float(term.numerator.detach().item())
            loss_totals[name][1] += int(term.denominator)
        q_valid_total += int(batch.q_target_mask.sum().item())
        z_valid_total += int(batch.z_target_mask.sum().item())
        sample_total += batch_size

        # One host transfer per tensor keeps detailed grouping from causing
        # thousands of GPU synchronisations on the 72k-sample TEST split.
        q_pred = q_pred_t.detach().float().cpu().numpy()
        z_pred = z_pred_t.detach().float().cpu().numpy()
        q_target = batch.q_target.detach().float().cpu().numpy()
        z_target = batch.z_target.detach().float().cpu().numpy()
        q_mask = batch.q_target_mask.detach().cpu().numpy().astype(bool)
        z_mask = batch.z_target_mask.detach().cpu().numpy().astype(bool)

        graph_ids = _batch_strings(batch.graph_id, batch_size, "graph_id")
        event_ids = _batch_strings(batch.event_id, batch_size, "event_id")
        graph_id = _normalise_id(graph_ids[0])
        if any(_normalise_id(value) != graph_id for value in graph_ids):
            raise ValueError("v8 evaluation batch混入多个graph")
        observed_event_ids.update(event_ids)

        _merge(q_global, _array_sums(q_pred, q_target, q_mask))
        _merge(z_global, _array_sums(z_pred, z_target, z_mask))
        _merge(q_hydro_global, _array_hydrograph(q_pred, q_target, q_mask))
        _merge(graph_q[graph_id], _array_sums(q_pred, q_target, q_mask))
        _merge(graph_z[graph_id], _array_sums(z_pred, z_target, z_mask))

        outlet_station = outlet_by_graph[graph_id]
        if outlet_station not in batch.obs_station_ids:
            raise ValueError(f"{graph_id}: batch缺少正式outlet={outlet_station}")
        outlet_position = batch.obs_station_ids.index(outlet_station)
        s = slice(outlet_position, outlet_position + 1)
        _merge(graph_outlet_q[graph_id], _array_sums(q_pred[:, :, s], q_target[:, :, s], q_mask[:, :, s]))
        _merge(graph_outlet_z[graph_id], _array_sums(z_pred[:, :, s], z_target[:, :, s], z_mask[:, :, s]))
        _merge(graph_outlet_hydro[graph_id], _array_hydrograph(q_pred[:, :, s], q_target[:, :, s], q_mask[:, :, s]))

        for lead in range(horizon):
            _merge(lead_global["q"][lead], _array_sums(q_pred[:, lead:lead+1], q_target[:, lead:lead+1], q_mask[:, lead:lead+1]))
            _merge(lead_global["z"][lead], _array_sums(z_pred[:, lead:lead+1], z_target[:, lead:lead+1], z_mask[:, lead:lead+1]))

        for station_position, raw_station_id in enumerate(batch.obs_station_ids):
            station_id = _normalise_id(raw_station_id)
            key = (graph_id, station_id)
            if key not in station_meta:
                raise ValueError(f"batch station未出现在mapping: {key}")
            s = slice(station_position, station_position + 1)
            _merge(station_q[key], _array_sums(q_pred[:, :, s], q_target[:, :, s], q_mask[:, :, s]))
            _merge(station_z[key], _array_sums(z_pred[:, :, s], z_target[:, :, s], z_mask[:, :, s]))
            _merge(station_hydro[key], _array_hydrograph(q_pred[:, :, s], q_target[:, :, s], q_mask[:, :, s]))
            for lead in range(horizon):
                _merge(lead_station["q"][key][lead], _array_sums(q_pred[:, lead:lead+1, s], q_target[:, lead:lead+1, s], q_mask[:, lead:lead+1, s]))
                _merge(lead_station["z"][key][lead], _array_sums(z_pred[:, lead:lead+1, s], z_target[:, lead:lead+1, s], z_mask[:, lead:lead+1, s]))
            for sample_index, event_id in enumerate(event_ids):
                event_key = (str(event_id), graph_id, station_id)
                event_sample_count[event_key] += 1
                b = slice(sample_index, sample_index + 1)
                _merge(event_q[event_key], _array_sums(q_pred[b, :, s], q_target[b, :, s], q_mask[b, :, s]))
                _merge(event_z[event_key], _array_sums(z_pred[b, :, s], z_target[b, :, s], z_mask[b, :, s]))
                _merge(event_hydro[event_key], _array_hydrograph(q_pred[b, :, s], q_target[b, :, s], q_mask[b, :, s]))

    if sample_total == 0:
        raise ValueError("v8 evaluation loader为空")

    station_rows: list[dict[str, Any]] = []
    for key in sorted(station_meta):
        if key[0] not in active_graphs:
            continue
        meta = station_meta[key]
        row = {
            "GRAPH_ID": meta["graph_id"],
            "STATION_ID": meta["station_id"],
            "STATION_ROLE": meta["station_role"],
            "IS_OUTLET_STATION": int(meta["is_outlet_station"]),
        }
        row.update(_regression_fields("q", station_q[key]))
        row.update(_hydrograph_fields("q", station_hydro[key]))
        row.update(_regression_fields("delta_z", station_z[key]))
        station_rows.append(row)

    graph_rows: list[dict[str, Any]] = []
    for graph_id in sorted(active_graphs):
        row = {
            "GRAPH_ID": graph_id,
            "OBS_STATION_COUNT": sum(r["GRAPH_ID"] == graph_id for r in station_rows),
            "OUTLET_STATION_ID": outlet_by_graph[graph_id],
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
        row.update(_hydrograph_fields("q", event_hydro[event_key]))
        row.update(_regression_fields("delta_z", event_z[event_key]))
        event_rows.append(row)

    active_station_keys = [key for key in sorted(station_meta) if key[0] in active_graphs]
    outlet_keys = {key for key in active_station_keys if station_meta[key]["is_outlet_station"]}
    lead_rows: list[dict[str, Any]] = []
    for task, storage_key, prefix in (("Q", "q", "q"), ("DELTA_Z", "z", "delta_z")):
        for lead in range(horizon):
            lead_rows.append({
                "TASK": task,
                "LEAD_HOUR": lead + 1,
                "SCOPE": "POOLED_ALL_OBS",
                **_regression_fields(prefix, lead_global[storage_key][lead]),
            })
            station_metrics = [_regression_fields(prefix, lead_station[storage_key][key][lead]) for key in active_station_keys]
            outlet_metrics = [_regression_fields(prefix, lead_station[storage_key][key][lead]) for key in active_station_keys if key in outlet_keys]
            lead_rows.append({"TASK": task, "LEAD_HOUR": lead + 1, "SCOPE": "MACRO_STATION", **_macro_regression(station_metrics, prefix)})
            lead_rows.append({"TASK": task, "LEAD_HOUR": lead + 1, "SCOPE": "MACRO_OUTLET", **_macro_regression(outlet_metrics, prefix)})

    loss_report = loss_engine.report(
        {name: (float(value), int(count)) for name, (value, count) in loss_totals.items()},
        q_valid_count=q_valid_total,
        z_valid_count=z_valid_total,
    )
    outlet_rows = [row for row in station_rows if row["IS_OUTLET_STATION"] == 1]
    summary = {
        "split": str(split).upper(),
        "samples": sample_total,
        "events": len(observed_event_ids),
        "graphs": len(active_graphs),
        "observation_stations": len(station_rows),
        "outlet_stations": len(outlet_rows),
        "internal_stations": len(station_rows) - len(outlet_rows),
        "target_semantics": {
            "q": "physical discharge Q in m3/s",
            "z": "Delta-Z(t+h)=Z(t+h)-Z(t0) in m",
        },
        "loss": loss_report,
        "q": {
            "pooled_all_observations": _regression_fields("q", q_global),
            "macro_station": _macro_regression(station_rows, "q"),
            "macro_outlet": _macro_regression(outlet_rows, "q"),
            "window_hydrograph_pooled": _hydrograph_fields("q", q_hydro_global),
            "window_hydrograph_macro_station": _macro_hydrograph(station_rows, "q"),
            "window_hydrograph_macro_outlet": _macro_hydrograph(outlet_rows, "q"),
        },
        "delta_z": {
            "pooled_all_observations": _regression_fields("delta_z", z_global),
            "macro_station": _macro_regression(station_rows, "delta_z"),
            "macro_outlet": _macro_regression(outlet_rows, "delta_z"),
        },
        "metric_interpretation": {
            "macro_station": "unweighted arithmetic mean across stations where each metric is defined",
            "macro_outlet": "unweighted arithmetic mean across basin outlet stations where each metric is defined",
            "event_station": "all frozen six-hour forecast windows belonging to one event/station are aggregated; peak/timing/volume fields remain window-level, not a reconstructed single event hydrograph",
        },
    }

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "summary": destination / "evaluation_summary.json",
        "station_metrics": destination / "station_metrics.csv",
        "graph_metrics": destination / "graph_metrics.csv",
        "event_station_metrics": destination / "event_station_metrics.csv",
        "lead_time_metrics": destination / "lead_time_metrics.csv",
    }
    pd.DataFrame(station_rows).to_csv(files["station_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(graph_rows).to_csv(files["graph_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(files["event_station_metrics"], index=False, encoding="utf-8-sig")
    pd.DataFrame(lead_rows).to_csv(files["lead_time_metrics"], index=False, encoding="utf-8-sig")
    payload = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "dataset_root": str(Path(dataset.root).resolve()),
        "data_contract": dataset.contract.get("contract"),
        "summary": summary,
    }
    files["summary"].write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": _json_safe(summary),
        "output_dir": str(destination),
        "files": {key: str(path) for key, path in files.items()},
    }
