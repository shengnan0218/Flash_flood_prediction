#!/usr/bin/env python3
"""Build formal V11 tensors: 72 h rainfall warm-up, 24 h Q/Z history.

The V8 dataset remains frozen and untouched. V11 preserves every V8 graph,
event, split and forecast origin, copies all Q/Z/static tensors exactly, and
rebuilds only rainfall exposure from the authoritative computational-unit
rainfall source so each origin receives 72 antecedent hours.

No zero padding is allowed outside the source rainfall valid period. If the
source does not cover the extra 48 antecedent hours needed by the earliest V8
origin, the build fails rather than inventing dry conditions.

The builder also writes TRAIN-only, unique-physical-hour high-flow quantiles and
a deterministic event phase label used only by the V11 TRAIN sampler.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

RAIN_HISTORY_HOURS = 72
OBS_HISTORY_HOURS = 24
FORECAST_HOURS = 6
CONTRACT_NAME = "hydrologic-computational-graph-72h-rain-24h-observation-v11"
EVENT_PHASES = ("LOW", "RISING", "PEAK", "RECESSION")
NPZ_KEYS = {
    "node_id",
    "node_static",
    "incremental_area_km2",
    "edge_index",
    "edge_static",
    "obs_station_id",
    "obs_node_index",
    "sample_id",
    "split_code",
    "forecast_time_unix_hour",
    "history_rain",
    "future_rain",
    "q_history",
    "q_history_mask",
    "z_history",
    "z_history_mask",
    "q_target",
    "q_target_mask",
    "z_target",
    "z_target_mask",
}


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if not array.size:
            return
        count = int(array.size)
        mean = float(array.mean())
        delta = mean - self.mean
        total = self.count + count
        self.m2 += (
            float(np.square(array - mean).sum())
            + delta * delta * self.count * count / total
        )
        self.mean += delta * count / total
        self.count = total
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self, *, scale_floor: float = 1.0e-6) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("V11 TRAIN rainfall normalization没有样本")
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        return {
            "available": True,
            "count": self.count,
            "mean": self.mean,
            "std": std,
            "min": self.minimum,
            "max": self.maximum,
            "scale": max(std, scale_floor),
            "floor_applied": std < scale_floor,
        }


def _norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def _require(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}不存在: {path}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v8-dataset",
        type=Path,
        default=root / "_model_dataset_v8_hydrologic_graph",
        help="冻结V8 dataset；只读。",
    )
    parser.add_argument(
        "--hydrologic-graph",
        type=Path,
        default=root / "_hydrologic_graph_v1",
        help="包含完整 computational-unit rainfall 源的hydrologic graph目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "_model_dataset_v11_72h_event_balanced",
    )
    return parser.parse_args()


def _load_rainfall_matrix(
    graph_id: str,
    node_ids: tuple[str, ...],
    graph_root: Path,
    *,
    required_start_hour: int,
    required_end_hour: int,
) -> np.ndarray:
    """Materialise dense hourly rainfall over the exact required V11 period."""
    if required_end_hour < required_start_hour:
        raise ValueError("rainfall required period非法")
    coverage_path = _require(
        graph_root / "rainfall/node_rainfall_coverage.csv",
        "computational-unit rainfall coverage",
    )
    coverage = _read_csv(coverage_path, dtype=str)
    required_columns = {"GRAPH_ID", "NODE_ID", "VALID_START", "VALID_END", "ZERO_SEMANTICS"}
    missing = required_columns - set(coverage.columns)
    if missing:
        raise ValueError(f"rainfall coverage缺字段: {sorted(missing)}")
    coverage["GRAPH_ID"] = coverage["GRAPH_ID"].map(_norm_id)
    coverage["NODE_ID"] = coverage["NODE_ID"].map(_norm_id)
    coverage = coverage[coverage["GRAPH_ID"].eq(graph_id)].copy()
    if len(coverage) != len(node_ids) or set(coverage["NODE_ID"]) != set(node_ids):
        raise ValueError(f"{graph_id}: rainfall coverage与V8 node catalogue不一致")
    if not coverage["ZERO_SEMANTICS"].eq(
        "ABSENT_SPARSE_ROW_WITHIN_VALID_PERIOD_IS_0_MM"
    ).all():
        raise ValueError(f"{graph_id}: rainfall zero semantics不允许把缺记录解释为0")

    required_start = pd.Timestamp(required_start_hour, unit="h")
    required_end = pd.Timestamp(required_end_hour, unit="h")
    valid_start = pd.to_datetime(coverage["VALID_START"], errors="raise")
    valid_end = pd.to_datetime(coverage["VALID_END"], errors="raise")
    if (valid_start > required_start).any() or (valid_end < required_end).any():
        offenders = coverage.loc[
            (valid_start > required_start) | (valid_end < required_end),
            ["NODE_ID", "VALID_START", "VALID_END"],
        ].head(20)
        raise ValueError(
            f"{graph_id}: 72h antecedent rainfall超出正式有效期；禁止zero-pad。"
            f" required=[{required_start},{required_end}], offenders="
            f"{offenders.to_dict('records')}"
        )

    count = required_end_hour - required_start_hour + 1
    matrix = np.zeros((count, len(node_ids)), dtype=np.float32)
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    sparse_path = _require(
        graph_root
        / f"rainfall/node_hourly_rain_sparse/graph_{graph_id}_hourly_sparse.csv",
        f"{graph_id} sparse rainfall",
    )
    frame = _read_csv(sparse_path, dtype=str)
    if frame.empty:
        return matrix
    required_sparse = {"GRAPH_ID", "NODE_ID", "START_TIME", "END_TIME", "RAIN_MM"}
    missing = required_sparse - set(frame.columns)
    if missing:
        raise ValueError(f"{graph_id}: sparse rainfall缺字段{sorted(missing)}")
    frame["GRAPH_ID"] = frame["GRAPH_ID"].map(_norm_id)
    frame["NODE_ID"] = frame["NODE_ID"].map(_norm_id)
    if set(frame["GRAPH_ID"]) != {graph_id} or set(frame["NODE_ID"]) - set(node_ids):
        raise ValueError(f"{graph_id}: sparse rainfall graph/node不一致")
    if frame.duplicated(["NODE_ID", "START_TIME"]).any():
        raise ValueError(f"{graph_id}: sparse rainfall存在重复node-hour")
    start = pd.to_datetime(frame["START_TIME"], errors="raise")
    end = pd.to_datetime(frame["END_TIME"], errors="raise")
    if not (end - start).eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"{graph_id}: sparse rainfall含非1h interval")
    unix_hour = (start.astype("int64") // 3_600_000_000_000).to_numpy(np.int64)
    selector = (unix_hour >= required_start_hour) & (unix_hour <= required_end_hour)
    if not selector.any():
        return matrix
    selected = frame.loc[selector].copy()
    selected_hours = unix_hour[selector]
    values = pd.to_numeric(selected["RAIN_MM"], errors="raise").to_numpy(np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{graph_id}: sparse positive rainfall必须为有限正数")
    row_index = selected_hours - required_start_hour
    column_index = selected["NODE_ID"].map(node_to_index).to_numpy(np.int64)
    matrix[row_index, column_index] = values.astype(np.float32)
    return matrix


def _consistent_pair(
    store: dict[tuple[str, int], float],
    key: tuple[str, int],
    value: float,
    *,
    label: str,
) -> None:
    previous = store.get(key)
    if previous is not None:
        if abs(previous - value) > 1.0e-5:
            raise ValueError(f"{label}重叠窗口物理时刻冲突: {key}: {previous} vs {value}")
        return
    store[key] = value


def _event_phase_labels(
    sample_rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    outlet_obs: int,
) -> list[str]:
    q = arrays["q_target"]
    mask = arrays["q_target_mask"].astype(bool)
    origins = arrays["forecast_time_unix_hour"].astype(np.int64)
    by_event: dict[str, dict[int, float]] = {}
    for row in sample_rows.itertuples(index=False):
        tensor_row = int(row.TENSOR_ROW)
        event = str(row.EVENT_ID)
        event_points = by_event.setdefault(event, {})
        for lead in range(FORECAST_HOURS):
            if not mask[tensor_row, outlet_obs, lead]:
                continue
            hour = int(origins[tensor_row]) + lead + 1
            value = float(q[tensor_row, outlet_obs, lead])
            previous = event_points.get(hour)
            if previous is not None and abs(previous - value) > 1.0e-5:
                raise ValueError(f"{event}: outlet Q在重叠窗口物理时刻冲突")
            event_points[hour] = value

    event_stats: dict[str, tuple[float, float, int]] = {}
    for event, points in by_event.items():
        if not points:
            continue
        hours = np.asarray(sorted(points), dtype=np.int64)
        values = np.asarray([points[int(hour)] for hour in hours], dtype=np.float64)
        peak = float(values.max())
        peak_hour = int(hours[np.flatnonzero(np.isclose(values, peak, atol=1.0e-5, rtol=0))[0]])
        event_stats[event] = (peak, float(np.median(values)), peak_hour)

    labels: list[str] = []
    for row in sample_rows.itertuples(index=False):
        tensor_row = int(row.TENSOR_ROW)
        event = str(row.EVENT_ID)
        q_supervised = int(row.Q_TARGET_VALID_COUNT) > 0
        stats = event_stats.get(event)
        if q_supervised and stats is None:
            raise ValueError(f"{event}: Q-supervised样本没有outlet event reference")
        if stats is None:
            labels.append("LOW")
            continue
        event_peak, event_median, peak_hour = stats
        local_valid = mask[tensor_row, outlet_obs]
        local_values = q[tensor_row, outlet_obs, local_valid].astype(np.float64)
        if q_supervised and local_values.size == 0:
            raise ValueError(
                f"{event}/{row.SAMPLE_ID}: Q-supervised样本没有outlet Q target；"
                "V11 phase sampler不能用内部站替代outlet flood phase"
            )
        if local_values.size:
            window_max = float(local_values.max())
            if event_peak > 0 and window_max >= 0.8 * event_peak:
                phase = "PEAK"
            elif window_max <= event_median:
                phase = "LOW"
            elif float(local_values[-1]) >= float(local_values[0]):
                phase = "RISING"
            else:
                phase = "RECESSION"
        else:
            phase = "RISING" if int(origins[tensor_row]) < peak_hour else "RECESSION"
        labels.append(phase)
    if len(labels) != len(sample_rows) or set(labels) - set(EVENT_PHASES):
        raise RuntimeError("V11 EVENT_PHASE生成失败")
    return labels


def _high_flow_payload(
    paired: dict[tuple[str, int], float],
    station_ids: tuple[str, ...],
    outlet_stations: set[str],
) -> dict[str, Any]:
    by_station: dict[str, list[float]] = {station: [] for station in station_ids}
    for (station, _hour), value in paired.items():
        by_station[station].append(value)
    stations: dict[str, dict[str, Any]] = {}
    missing_outlets: list[str] = []
    for station in station_ids:
        values = np.asarray(by_station[station], dtype=np.float64)
        record: dict[str, Any] = {
            "available": False,
            "unique_train_physical_hour_count": int(values.size),
        }
        if values.size >= 20:
            q80 = float(np.quantile(values, 0.80))
            q99 = float(np.quantile(values, 0.99))
            if math.isfinite(q80) and math.isfinite(q99) and q99 > q80:
                record.update(
                    {
                        "available": True,
                        "q80_m3s": q80,
                        "q99_m3s": q99,
                        "q_max_m3s": float(values.max()),
                    }
                )
        if station in outlet_stations and not record["available"]:
            missing_outlets.append(station)
        stations[station] = record
    if missing_outlets:
        raise ValueError(
            "V11 high-flow objective要求每个outlet有TRAIN-only P80/P99，缺失="
            f"{missing_outlets}"
        )
    payload: dict[str, Any] = {
        "method": "TRAIN_ONLY_UNIQUE_PHYSICAL_TARGET_HOUR_QUANTILES",
        "fit_split": "TRAIN",
        "deduplication_key": "STATION_ID+PHYSICAL_TARGET_UNIX_HOUR",
        "lower_quantile": 0.80,
        "upper_quantile": 0.99,
        "station_count": len(station_ids),
        "available_station_count": sum(bool(v["available"]) for v in stations.values()),
        "outlet_station_count": len(outlet_stations),
        "outlet_missing_threshold": missing_outlets,
        "unique_pair_count": len(paired),
        "duplicate_value_conflict_count": 0,
        "stations": stations,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main() -> None:
    args = parse_args()
    v8_root = args.v8_dataset.expanduser().resolve()
    graph_root = args.hydrologic_graph.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing V11 dataset: {output}")
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)

    v8_report = _require(v8_root / "BUILD_AND_QC.md", "V8 QC report")
    if "FINAL QC STATUS: PASS" not in v8_report.read_text(encoding="utf-8"):
        raise ValueError("V8 source dataset不是QC PASS")
    v8_contract_path = _require(
        v8_root / "metadata/dataset_contract.json", "V8 dataset contract"
    )
    v8_contract = json.loads(v8_contract_path.read_text(encoding="utf-8-sig"))
    if v8_contract.get("contract") != "hydrologic-computational-graph-sparse-observation-v1":
        raise ValueError("V11 builder只接受正式V8 hydrologic graph source")
    if int(v8_contract.get("history_hours", -1)) != OBS_HISTORY_HOURS:
        raise ValueError("V8 source Q/Z history不是24 h")
    if int(v8_contract.get("forecast_hours", -1)) != FORECAST_HOURS:
        raise ValueError("V8 source forecast不是6 h")

    sample_path = _require(v8_root / "samples/sample_index.csv", "V8 sample index")
    event_path = _require(v8_root / "events/hydrologic_events.csv", "V8 event table")
    mapping_path = _require(
        v8_root / "graph/station_observation_mapping.csv", "V8 station mapping"
    )
    samples = _read_csv(sample_path, dtype=str)
    events = _read_csv(event_path, dtype=str)
    mapping = _read_csv(mapping_path, dtype=str)
    for frame in (samples, events, mapping):
        frame["GRAPH_ID"] = frame["GRAPH_ID"].map(_norm_id)
    mapping["STATION_ID"] = mapping["STATION_ID"].map(_norm_id)
    samples["SPLIT"] = samples["SPLIT"].str.upper()
    samples["TENSOR_ROW"] = pd.to_numeric(samples["TENSOR_ROW"], errors="raise").astype(np.int64)
    samples["Q_TARGET_VALID_COUNT"] = pd.to_numeric(
        samples["Q_TARGET_VALID_COUNT"], errors="raise"
    ).astype(np.int64)
    if len(samples) != 279_574 or len(events) != 2_807:
        raise ValueError("V8 frozen sample/event cardinality发生变化")
    if samples["SAMPLE_ID"].duplicated().any() or events["EVENT_ID"].duplicated().any():
        raise ValueError("V8 source SAMPLE_ID/EVENT_ID不唯一")
    if set(samples["SPLIT"]) != {"TRAIN", "VALIDATION", "TEST"}:
        raise ValueError("V8 source split集合非法")

    graph_ids = tuple(sorted(samples["GRAPH_ID"].unique().tolist()))
    if len(graph_ids) != 33:
        raise ValueError(f"V11必须保持33 graphs，实际={len(graph_ids)}")
    station_ids = tuple(sorted(mapping["STATION_ID"].unique().tolist()))
    if len(station_ids) != 39:
        raise ValueError(f"V11必须保持39 observation stations，实际={len(station_ids)}")
    outlet_stations = set(
        mapping.loc[mapping["IS_OUTLET_STATION"].map(_bool), "STATION_ID"].tolist()
    )
    if len(outlet_stations) != 33:
        raise ValueError("V11必须保持33 outlet stations")

    stage.mkdir(parents=True)
    try:
        (stage / "graph").mkdir()
        (stage / "events").mkdir()
        (stage / "samples/tensors").mkdir(parents=True)
        (stage / "metadata").mkdir()
        for name in (
            "node_catalog.csv",
            "edge_topology.csv",
            "node_static_attributes.csv",
            "edge_static_attributes.csv",
            "station_observation_mapping.csv",
        ):
            shutil.copy2(_require(v8_root / "graph" / name, name), stage / "graph" / name)
        shutil.copy2(event_path, stage / "events/hydrologic_events.csv")

        rain_stats = RunningStats()
        high_flow_pairs: dict[tuple[str, int], float] = {}
        output_indices: list[pd.DataFrame] = []
        tensor_paths: list[Path] = []
        phase_counts = {split: {phase: 0 for phase in EVENT_PHASES} for split in ("TRAIN", "VALIDATION", "TEST")}

        for number, graph_id in enumerate(graph_ids, start=1):
            graph_samples = samples[samples["GRAPH_ID"].eq(graph_id)].copy()
            tensor_names = graph_samples["TENSOR_FILE"].unique().tolist()
            if len(tensor_names) != 1:
                raise ValueError(f"{graph_id}: V8 graph应唯一对应一个tensor file")
            relative = str(tensor_names[0]).replace("\\", "/")
            source_tensor = (v8_root / relative).resolve()
            if v8_root not in source_tensor.parents:
                raise ValueError(f"{graph_id}: TENSOR_FILE越出V8 root")
            with np.load(_require(source_tensor, f"{graph_id} V8 tensor"), allow_pickle=False) as archive:
                if set(archive.files) != NPZ_KEYS:
                    raise ValueError(f"{graph_id}: V8 tensor key集合发生变化")
                arrays = {key: archive[key].copy() for key in archive.files}

            sample_count = len(arrays["sample_id"])
            if sample_count != len(graph_samples):
                raise ValueError(f"{graph_id}: sample index/tensor cardinality不一致")
            graph_samples = graph_samples.sort_values("TENSOR_ROW").copy()
            if graph_samples["TENSOR_ROW"].tolist() != list(range(sample_count)):
                raise ValueError(f"{graph_id}: TENSOR_ROW不是完整0-based序列")
            if arrays["sample_id"].astype(str).tolist() != graph_samples["SAMPLE_ID"].astype(str).tolist():
                raise ValueError(f"{graph_id}: sample_id与tensor row不一致")

            node_ids = tuple(_norm_id(value) for value in arrays["node_id"].tolist())
            origins = arrays["forecast_time_unix_hour"].astype(np.int64)
            required_start = int(origins.min()) - (RAIN_HISTORY_HOURS - 1)
            required_end = int(origins.max()) + FORECAST_HOURS
            rain = _load_rainfall_matrix(
                graph_id,
                node_ids,
                graph_root,
                required_start_hour=required_start,
                required_end_hour=required_end,
            )
            history_index = (
                origins[:, None]
                + np.arange(-(RAIN_HISTORY_HOURS - 1), 1, dtype=np.int64)[None, :]
                - required_start
            )
            future_index = (
                origins[:, None]
                + np.arange(1, FORECAST_HOURS + 1, dtype=np.int64)[None, :]
                - required_start
            )
            history_rain = np.transpose(rain[history_index], (0, 2, 1)).astype(np.float32)
            future_rain = np.transpose(rain[future_index], (0, 2, 1)).astype(np.float32)
            if arrays["future_rain"].shape != future_rain.shape or not np.allclose(
                arrays["future_rain"], future_rain, rtol=0.0, atol=1.0e-6
            ):
                difference = float(
                    np.max(np.abs(arrays["future_rain"].astype(np.float64) - future_rain.astype(np.float64)))
                )
                raise ValueError(
                    f"{graph_id}: raw rainfall source与冻结V8 future_rain不一致，max_diff={difference}"
                )
            arrays["history_rain"] = history_rain
            arrays["future_rain"] = future_rain

            local_stations = tuple(
                _norm_id(value) for value in arrays["obs_station_id"].tolist()
            )
            graph_mapping = mapping[mapping["GRAPH_ID"].eq(graph_id)]
            graph_outlets = graph_mapping.loc[
                graph_mapping["IS_OUTLET_STATION"].map(_bool), "STATION_ID"
            ].tolist()
            if len(graph_outlets) != 1 or graph_outlets[0] not in local_stations:
                raise ValueError(f"{graph_id}: 无法唯一定位outlet observation")
            outlet_obs = local_stations.index(graph_outlets[0])

            phase_labels = _event_phase_labels(graph_samples, arrays, outlet_obs)
            graph_samples["EVENT_PHASE"] = phase_labels
            graph_samples["RAIN_HISTORY_HOURS"] = RAIN_HISTORY_HOURS
            graph_samples["OBS_HISTORY_HOURS"] = OBS_HISTORY_HOURS
            for split, phase in zip(graph_samples["SPLIT"], phase_labels):
                phase_counts[str(split)][phase] += 1

            train_rows = graph_samples[graph_samples["SPLIT"].eq("TRAIN")]
            train_indices = train_rows["TENSOR_ROW"].to_numpy(np.int64)
            if not len(train_indices):
                raise ValueError(f"{graph_id}: 没有TRAIN样本")
            rain_stats.update(history_rain[train_indices])
            rain_stats.update(future_rain[train_indices])

            q = arrays["q_target"]
            q_mask = arrays["q_target_mask"].astype(bool)
            for row in train_rows.itertuples(index=False):
                tensor_row = int(row.TENSOR_ROW)
                origin = int(origins[tensor_row])
                for obs, station in enumerate(local_stations):
                    for lead in range(FORECAST_HOURS):
                        if not q_mask[tensor_row, obs, lead]:
                            continue
                        value = float(q[tensor_row, obs, lead])
                        if not math.isfinite(value):
                            raise ValueError("TRAIN有效Q target含NaN/Inf")
                        _consistent_pair(
                            high_flow_pairs,
                            (station, origin + lead + 1),
                            value,
                            label="V11 high-flow quantile",
                        )

            relative_out = f"samples/tensors/graph_{graph_id}.npz"
            tensor_path = stage / relative_out
            np.savez_compressed(tensor_path, **arrays)
            tensor_paths.append(tensor_path)
            graph_samples["TENSOR_FILE"] = relative_out
            output_indices.append(graph_samples)
            print(
                f"[v11 tensor {number:02d}/33] {graph_id}: samples={sample_count}, "
                f"rain_history=72h, obs_history=24h",
                flush=True,
            )

        sample_output = pd.concat(output_indices, ignore_index=True)
        # Restore the exact frozen V8 sample order before publication.
        order = {sample_id: index for index, sample_id in enumerate(samples["SAMPLE_ID"].tolist())}
        sample_output["_V8_ORDER"] = sample_output["SAMPLE_ID"].map(order)
        if sample_output["_V8_ORDER"].isna().any():
            raise ValueError("V11产生未知SAMPLE_ID")
        sample_output = sample_output.sort_values("_V8_ORDER").drop(columns="_V8_ORDER")
        if sample_output["SAMPLE_ID"].tolist() != samples["SAMPLE_ID"].tolist():
            raise ValueError("V11 SAMPLE_ID/order未严格继承V8")
        for field in ("EVENT_ID", "GRAPH_ID", "FORECAST_TIME", "SPLIT"):
            if sample_output[field].astype(str).tolist() != samples[field].astype(str).tolist():
                raise ValueError(f"V11 {field}未严格继承V8")
        if len(sample_output) != len(samples):
            raise ValueError("V11 sample cardinality改变")

        high_flow = _high_flow_payload(
            high_flow_pairs, station_ids, outlet_stations
        )
        normalization = deepcopy(v8_contract["normalization"])
        normalization["rain_mm"] = rain_stats.result()
        normalization["computed_from_split"] = "TRAIN"
        normalization["fit_scope"] = "TRAIN_SAMPLE_EXPOSURE_ONLY_V11_72H_RAIN"
        normalization["rain_history_hours"] = RAIN_HISTORY_HOURS
        normalization["qz_observation_history_hours"] = OBS_HISTORY_HOURS

        # Keep all original V8 index columns, then append V11-only metadata.
        output_columns = list(samples.columns)
        for field in ("RAIN_HISTORY_HOURS", "OBS_HISTORY_HOURS", "EVENT_PHASE"):
            if field not in output_columns:
                output_columns.append(field)
        sample_output[output_columns].to_csv(
            stage / "samples/sample_index.csv", index=False, encoding="utf-8-sig"
        )

        earliest_origin = pd.to_datetime(samples["FORECAST_TIME"], errors="raise").min()
        latest_origin = pd.to_datetime(samples["FORECAST_TIME"], errors="raise").max()
        contract = {
            "contract": CONTRACT_NAME,
            "format_version": 11,
            "graph_count": int(v8_contract["graph_count"]),
            "computational_node_count": int(v8_contract["computational_node_count"]),
            "edge_count": int(v8_contract["edge_count"]),
            "observation_station_count": int(v8_contract["observation_station_count"]),
            "rain_history_hours": RAIN_HISTORY_HOURS,
            "observation_history_hours": OBS_HISTORY_HOURS,
            "forecast_hours": FORECAST_HOURS,
            "time_zone": v8_contract.get("time_zone", "Asia/Shanghai"),
            "global_time_start": v8_contract.get("global_time_start"),
            "global_time_end": v8_contract.get("global_time_end"),
            "time_splits": v8_contract.get("time_splits"),
            "timestamp_semantics": v8_contract.get("timestamp_semantics"),
            "temporal_domain": {
                "events": "events/hydrologic_events.csv",
                "samples": "samples/sample_index.csv",
                "inheritance": (
                    "exact V8 graph/event/split/SAMPLE_ID/FORECAST_TIME domain; "
                    "only antecedent rainfall tensor exposure is extended"
                ),
            },
            "antecedent_rainfall": {
                "source": str(graph_root),
                "hours": RAIN_HISTORY_HOURS,
                "extra_hours_before_v8_history": RAIN_HISTORY_HOURS - OBS_HISTORY_HOURS,
                "zero_padding_outside_valid_period": False,
                "coverage_check": "PER_NODE_FAIL_IF_REQUIRED_HOUR_OUTSIDE_VALID_PERIOD",
                "earliest_required_hour": (
                    earliest_origin - pd.Timedelta(hours=RAIN_HISTORY_HOURS - 1)
                ).isoformat(),
                "latest_required_hour": (
                    latest_origin + pd.Timedelta(hours=FORECAST_HOURS)
                ).isoformat(),
                "future_rain_crosscheck": "bitwise-domain numeric equality to V8 future_rain within 1e-6",
            },
            "observation_history": {
                "hours": OBS_HISTORY_HOURS,
                "source": "copied byte-value/mask arrays from frozen V8 tensors",
                "purpose": "forecast-origin Q/Z assimilation only",
                "extended_to_72h": False,
            },
            "event_phase": {
                "labels": list(EVENT_PHASES),
                "TRAIN_usage": "event-balanced phase-stratified sampler only",
                "definition": {
                    "PEAK": "6h outlet target window max >= 0.8 * event outlet peak",
                    "LOW": "6h outlet target window max <= event outlet median",
                    "RISING": "otherwise last valid outlet target >= first valid outlet target",
                    "RECESSION": "otherwise",
                },
                "phase_counts": phase_counts,
            },
            "high_flow_quantiles": high_flow,
            "normalization": normalization,
            "source_v8": {
                "root": str(v8_root),
                "dataset_contract_sha256": _sha256(v8_contract_path),
                "sample_index_sha256": _sha256(sample_path),
                "event_table_sha256": _sha256(event_path),
            },
            "tensor_files": "samples/tensors/graph_{GRAPH_ID}.npz",
            "tensor_schema": {
                "history_rain": "float32 [S,Nnode,72], mm, dense physics warm-up forcing",
                "future_rain": "float32 [S,Nnode,6], mm, dense observed-hindcast forcing",
                "q_history": "float32 [S,Nobs,24], copied V8 values, NaN iff mask=false",
                "z_history": "float32 [S,Nobs,24], copied V8 values, NaN iff mask=false",
                "q_target": "float32 [S,Nobs,6], copied V8 raw Q m3/s",
                "z_target": "float32 [S,Nobs,6], copied V8 Delta-Z",
                "static_and_topology": "byte-value equivalent to frozen V8 source",
            },
            "node_static_features": v8_contract["node_static_features"],
            "edge_static_features": v8_contract["edge_static_features"],
            "rain_zero_semantics": v8_contract["rain_zero_semantics"],
            "observation_semantics": v8_contract["observation_semantics"],
        }
        contract_path = stage / "metadata/dataset_contract.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        # Independent on-disk checks before atomic publication.
        if len(tensor_paths) != 33:
            raise ValueError("V11必须生成33个graph tensor files")
        for tensor_path in tensor_paths:
            with np.load(tensor_path, allow_pickle=False) as tensor:
                if set(tensor.files) != NPZ_KEYS:
                    raise ValueError(f"{tensor_path.name}: V11 tensor key集合非法")
                if tensor["history_rain"].shape[2] != RAIN_HISTORY_HOURS:
                    raise ValueError(f"{tensor_path.name}: history_rain不是72h")
                if tensor["q_history"].shape[2] != OBS_HISTORY_HOURS:
                    raise ValueError(f"{tensor_path.name}: q_history不是24h")
                if tensor["z_history"].shape[2] != OBS_HISTORY_HOURS:
                    raise ValueError(f"{tensor_path.name}: z_history不是24h")
                if tensor["q_target"].shape[2] != FORECAST_HOURS:
                    raise ValueError(f"{tensor_path.name}: q_target不是6h")
                if not np.isfinite(tensor["history_rain"]).all() or (
                    tensor["history_rain"] < 0
                ).any():
                    raise ValueError(f"{tensor_path.name}: 72h rainfall非法")
                for value_key, mask_key in (
                    ("q_history", "q_history_mask"),
                    ("z_history", "z_history_mask"),
                    ("q_target", "q_target_mask"),
                    ("z_target", "z_target_mask"),
                ):
                    values = tensor[value_key]
                    masks = tensor[mask_key].astype(bool)
                    if (
                        values.shape != masks.shape
                        or not np.isfinite(values[masks]).all()
                        or np.isfinite(values[~masks]).any()
                    ):
                        raise ValueError(f"{tensor_path.name}: {value_key} mask/value QC失败")

        train_q = sample_output[
            sample_output["SPLIT"].eq("TRAIN")
            & pd.to_numeric(sample_output["Q_TARGET_VALID_COUNT"], errors="raise").gt(0)
        ]
        train_event_count = int(train_q["EVENT_ID"].nunique())
        if train_event_count <= 0:
            raise ValueError("V11 TRAIN没有Q-supervised event")
        report = f"""# Model Dataset V11 — 72 h Antecedent Rainfall / 24 h Observation Assimilation

- Source V8 samples/events/splits/origins: preserved exactly ({len(sample_output)} samples; {len(events)} events).
- Graphs: {len(graph_ids)}; stations: {len(station_ids)}; outlets: {len(outlet_stations)}.
- Rainfall physical warm-up: 72 h, source-valid-period checked per node, no out-of-period zero padding.
- Q/Z observation assimilation history: 24 h, copied from frozen V8.
- Forecast: 6 h.
- Q-supervised TRAIN events available to event-balanced sampler: {train_event_count}.
- Event phases: LOW/RISING/PEAK/RECESSION; labels are used only for TRAIN sampling.
- High-flow thresholds: TRAIN-only unique physical target hours, station-specific P80/P99; all outlets covered.
- TRAIN rainfall normalization recomputed from V11 72 h + 6 h exposure; Q/Z/static normalization inherited from frozen V8.
- Future rainfall recomputation cross-checked against every frozen V8 future-rain tensor to 1e-6.

**FINAL QC STATUS: PASS**
"""
        (stage / "BUILD_AND_QC.md").write_text(report, encoding="utf-8")
        if "FINAL QC STATUS: PASS" not in (stage / "BUILD_AND_QC.md").read_text(encoding="utf-8"):
            raise RuntimeError("V11 QC report未记录PASS")
        stage.rename(output)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output": str(output),
                    "samples": len(sample_output),
                    "events": len(events),
                    "train_q_events": train_event_count,
                    "rain_history_hours": RAIN_HISTORY_HOURS,
                    "observation_history_hours": OBS_HISTORY_HOURS,
                    "high_flow_available_stations": high_flow["available_station_count"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
