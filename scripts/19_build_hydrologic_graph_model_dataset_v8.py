#!/usr/bin/env python3
"""Build v8 tensors for the 10-km2 hydrologic computational graph.

The builder preserves the frozen v7 events, split boundaries and sample
origins.  It changes only the representation:

* rainfall is dense physics forcing on every computational unit;
* Q/Z remain sparse observations at mapped stations only;
* graph topology/static facts come from ``_hydrologic_graph_v1``;
* all fitted statistics use TRAIN sample exposure only.

The output is built in a sibling staging directory, fully validated, then
atomically published.  Existing formal directories are never overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterable

import numpy as np
import pandas as pd


HISTORY_HOURS = 24
FORECAST_HOURS = 6
EXCLUDED_GRAPH = "Q_61512000"
EXCLUDED_STATION = "61512000"
SPLIT_CODE = {"TRAIN": 0, "VALIDATION": 1, "TEST": 2}
NODE_STATIC_FEATURES = (
    "log_incremental_area",
    "log_upstream_area",
    "mean_hillslope_flow_distance_m",
    "mean_slope_deg",
    "elevation_std_m",
    "drainage_density_km_per_km2",
    "soil_log_ksat_0_30cm",
    "soil_profile_depth_cm",
    "forest_fraction",
    "impervious_fraction",
)
EDGE_STATIC_FEATURES = ("reach_length_m", "reach_slope_m_per_m")


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if not array.size:
            return
        count = int(array.size)
        mean = float(array.mean())
        delta = mean - self.mean
        total = self.count + count
        self.m2 += float(np.square(array - mean).sum()) + delta * delta * self.count * count / total
        self.mean += delta * count / total
        self.count = total
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))

    def result(self, scale_floor: float | None = None) -> dict:
        if self.count == 0:
            return {
                "available": False, "count": 0, "mean": None, "std": None,
                "min": None, "max": None, "scale": None, "floor_applied": False,
            }
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        floor_applied = bool(scale_floor is not None and std < scale_floor)
        return {
            "available": True, "count": self.count, "mean": self.mean,
            "std": std, "min": self.minimum, "max": self.maximum,
            "scale": max(std, scale_floor) if scale_floor is not None else std,
            "floor_applied": floor_applied,
        }


def parse_args() -> argparse.Namespace:
    workflow = Path(__file__).resolve().parent
    root = workflow.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hydrologic-graph", type=Path,
        default=root / "project/_hydrologic_graph_v1",
    )
    parser.add_argument(
        "--temporal-dataset", type=Path,
        default=root / "project/_model_dataset_v7_event_multitask",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "project/_model_dataset_v8_hydrologic_graph",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} missing: {result}")
    return result


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def numeric_matrix(frame: pd.DataFrame, features: Iterable[str], label: str) -> np.ndarray:
    fields = list(features)
    missing = [field for field in fields if field not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing fields: {missing}")
    matrix = frame[fields].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains missing/nonfinite values")
    return matrix


def ensure_hourly_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    if start != start.floor("h") or end != end.floor("h") or start >= end:
        raise ValueError("invalid frozen global hourly boundaries")
    return pd.date_range(start, end, freq="h")


def validate_graph_facts(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_static: pd.DataFrame,
    edge_static: pd.DataFrame,
    mapping: pd.DataFrame,
) -> set[str]:
    required_nodes = {"GRAPH_ID", "NODE_ID", "NODE_INDEX", "IS_OUTLET", "incremental_area_km2"}
    required_edges = {"GRAPH_ID", "EDGE_ID", "FROM_NODE_ID", "TO_NODE_ID", "FROM_NODE_INDEX", "TO_NODE_INDEX"}
    required_mapping = {
        "GRAPH_ID", "STATION_ID", "STATION_ROLE", "IS_OUTLET_STATION",
        "MAPPED_NODE_ID", "MAPPED_NODE_INDEX", "SNAP_DISTANCE_M",
    }
    for actual, required, label in (
        (nodes, required_nodes, "node catalog"),
        (edges, required_edges, "edge topology"),
        (mapping, required_mapping, "station mapping"),
    ):
        if not required.issubset(actual.columns):
            raise ValueError(f"{label} missing fields: {sorted(required-set(actual.columns))}")
    graphs = set(nodes.GRAPH_ID.map(norm_id))
    if len(graphs) != 33 or EXCLUDED_GRAPH in graphs:
        raise ValueError(f"hydrologic graph must contain exactly 33 non-excluded basins; actual={len(graphs)}")
    if set(edges.GRAPH_ID.map(norm_id)) - graphs:
        raise ValueError("edge topology contains an unknown graph")
    if set(node_static.GRAPH_ID.map(norm_id)) != graphs or set(edge_static.GRAPH_ID.map(norm_id)) - graphs:
        raise ValueError("static graph set differs from node catalog")
    if set(mapping.GRAPH_ID.map(norm_id)) != graphs or mapping.STATION_ID.map(norm_id).eq(EXCLUDED_STATION).any():
        raise ValueError("observation mapping graph/station set is invalid")
    if mapping.duplicated(["GRAPH_ID", "STATION_ID"]).any():
        raise ValueError("station mapping contains duplicate graph-station rows")

    for graph_id in sorted(graphs):
        graph_nodes = nodes[nodes.GRAPH_ID.eq(graph_id)].copy()
        graph_edges = edges[edges.GRAPH_ID.eq(graph_id)].copy()
        indices = pd.to_numeric(graph_nodes.NODE_INDEX, errors="raise").astype(int)
        if sorted(indices) != list(range(len(graph_nodes))):
            raise ValueError(f"{graph_id}: NODE_INDEX is not contiguous zero-based")
        node_id_to_index = dict(zip(graph_nodes.NODE_ID, indices))
        if int(pd.to_numeric(graph_nodes.IS_OUTLET, errors="raise").sum()) != 1:
            raise ValueError(f"{graph_id}: expected one computational outlet")
        if len(graph_edges) != len(graph_nodes) - 1:
            raise ValueError(f"{graph_id}: expected a directed drainage tree with E=N-1")
        indegree = {node_id: 0 for node_id in node_id_to_index}
        outgoing = {node_id: [] for node_id in node_id_to_index}
        for edge in graph_edges.itertuples(index=False):
            if edge.FROM_NODE_ID not in node_id_to_index or edge.TO_NODE_ID not in node_id_to_index:
                raise ValueError(f"{graph_id}: edge endpoint absent from node catalog")
            if int(edge.FROM_NODE_INDEX) != node_id_to_index[edge.FROM_NODE_ID] or int(edge.TO_NODE_INDEX) != node_id_to_index[edge.TO_NODE_ID]:
                raise ValueError(f"{graph_id}: edge ID/index endpoint mismatch")
            indegree[edge.TO_NODE_ID] += 1
            outgoing[edge.FROM_NODE_ID].append(edge.TO_NODE_ID)
        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            node_id = queue.pop()
            visited += 1
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        outlet_id = graph_nodes.loc[pd.to_numeric(graph_nodes.IS_OUTLET).eq(1), "NODE_ID"].iloc[0]
        terminals = {node_id for node_id, targets in outgoing.items() if not targets}
        if visited != len(graph_nodes) or terminals != {outlet_id}:
            raise ValueError(f"{graph_id}: graph is not a connected DAG with one outlet")
        graph_mapping = mapping[mapping.GRAPH_ID.eq(graph_id)]
        for row in graph_mapping.itertuples(index=False):
            if row.MAPPED_NODE_ID not in node_id_to_index or int(row.MAPPED_NODE_INDEX) != node_id_to_index[row.MAPPED_NODE_ID]:
                raise ValueError(f"{graph_id}/{row.STATION_ID}: mapped node ID/index mismatch")
        outlet_mapping = graph_mapping[pd.to_numeric(graph_mapping.IS_OUTLET_STATION).eq(1)]
        if len(outlet_mapping) != 1 or outlet_mapping.MAPPED_NODE_ID.iloc[0] != outlet_id:
            raise ValueError(f"{graph_id}: outlet observation is not mapped to computational outlet")

    if len(nodes) != 237 or len(edges) != 204 or len(mapping) != 39:
        raise ValueError("frozen hydrologic graph cardinality changed from QC-PASS v1")
    if len(node_static) != len(nodes) or len(edge_static) != len(edges):
        raise ValueError("static/topology cardinality mismatch")
    return graphs


def validate_temporal_contract(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    schema: dict,
    graphs: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    if int(schema.get("history_hours", -1)) != HISTORY_HOURS or int(schema.get("forecast_hours", -1)) != FORECAST_HOURS:
        raise ValueError("frozen temporal dataset is not history=24/forecast=6")
    start = pd.Timestamp(schema["global_time_start"])
    end = pd.Timestamp(schema["global_time_end"])
    timeline = ensure_hourly_index(start, end)
    required_samples = {
        "SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "INPUT_START", "FORECAST_TIME",
        "TARGET_START", "TARGET_END", "HISTORY_HOURS", "FORECAST_HOURS", "SPLIT",
    }
    required_events = {"EVENT_ID", "GRAPH_ID", "SPLIT", "EVENT_START", "EVENT_END"}
    if not required_samples.issubset(samples.columns) or not required_events.issubset(events.columns):
        raise ValueError("frozen sample/event table misses required fields")
    samples = samples.copy()
    events = events.copy()
    samples["GRAPH_ID"] = samples.GRAPH_ID.map(norm_id)
    events["GRAPH_ID"] = events.GRAPH_ID.map(norm_id)
    samples = samples[samples.GRAPH_ID.isin(graphs)].copy()
    events = events[events.GRAPH_ID.isin(graphs)].copy()
    if set(samples.GRAPH_ID) != graphs or set(events.GRAPH_ID) != graphs:
        raise ValueError("frozen v7 samples/events do not cover all 33 graphs")
    if EXCLUDED_GRAPH in set(samples.GRAPH_ID) or EXCLUDED_GRAPH in set(events.GRAPH_ID):
        raise ValueError("excluded graph leaked into v8 temporal domain")
    if samples.SAMPLE_ID.duplicated().any() or samples.duplicated(["GRAPH_ID", "FORECAST_TIME"]).any():
        raise ValueError("frozen sample index has duplicate ID or graph-origin")
    if events.EVENT_ID.duplicated().any():
        raise ValueError("frozen event index has duplicate EVENT_ID")
    if set(samples.EVENT_ID) - set(events.EVENT_ID):
        raise ValueError("sample references a missing retained event")
    if not set(samples.SPLIT).issubset(SPLIT_CODE) or not set(events.SPLIT).issubset(SPLIT_CODE):
        raise ValueError("unknown split label")
    event_split = events.set_index("EVENT_ID").SPLIT.to_dict()
    if any(event_split[event_id] != split for event_id, split in zip(samples.EVENT_ID, samples.SPLIT)):
        raise ValueError("sample split differs from its frozen event split")

    for field in ("INPUT_START", "FORECAST_TIME", "TARGET_START", "TARGET_END"):
        samples[field] = pd.to_datetime(samples[field], errors="raise")
    if not pd.to_numeric(samples.HISTORY_HOURS, errors="raise").eq(HISTORY_HOURS).all() or not pd.to_numeric(samples.FORECAST_HOURS, errors="raise").eq(FORECAST_HOURS).all():
        raise ValueError("sample history/forecast length changed")
    if not (samples.FORECAST_TIME - samples.INPUT_START).eq(pd.Timedelta(hours=HISTORY_HOURS - 1)).all():
        raise ValueError("INPUT_START/FORECAST_TIME is not an inclusive 24-hour history")
    if not (samples.TARGET_START - samples.FORECAST_TIME).eq(pd.Timedelta(hours=1)).all() or not (samples.TARGET_END - samples.FORECAST_TIME).eq(pd.Timedelta(hours=FORECAST_HOURS)).all():
        raise ValueError("forecast target window is not the next six hours")
    if samples.INPUT_START.min() < start or samples.TARGET_END.max() > end:
        raise ValueError("sample tensor window falls outside frozen global timeline")
    for split, boundaries in schema["time_splits"].items():
        subset = samples[samples.SPLIT.eq(split)]
        split_start, split_end = pd.Timestamp(boundaries["start"]), pd.Timestamp(boundaries["end"])
        if (subset.INPUT_START < split_start).any() or (subset.TARGET_END > split_end).any():
            raise ValueError(f"{split}: sample history/target crosses frozen split boundary")
        event_subset = events[events.SPLIT.eq(split)]
        event_start = pd.to_datetime(event_subset.EVENT_START, errors="raise")
        event_end = pd.to_datetime(event_subset.EVENT_END, errors="raise")
        if (event_start < split_start).any() or (event_end > split_end).any():
            raise ValueError(f"{split}: retained event crosses frozen split boundary")
    samples["_SOURCE_ORDER"] = np.arange(len(samples), dtype=np.int64)
    return samples, events, timeline


def load_graph_rain(
    graph_id: str,
    graph_nodes: pd.DataFrame,
    graph_root: Path,
    timeline: pd.DatetimeIndex,
) -> np.ndarray:
    coverage_path = graph_root / "rainfall/node_rainfall_coverage.csv"
    coverage = read_csv(coverage_path, dtype=str)
    coverage = coverage[coverage.GRAPH_ID.eq(graph_id)]
    expected_nodes = set(graph_nodes.NODE_ID)
    if set(coverage.NODE_ID) != expected_nodes or len(coverage) != len(graph_nodes):
        raise ValueError(f"{graph_id}: rainfall coverage does not match computational nodes")
    if not coverage.ZERO_SEMANTICS.eq("ABSENT_SPARSE_ROW_WITHIN_VALID_PERIOD_IS_0_MM").all():
        raise ValueError(f"{graph_id}: rainfall zero semantics changed")
    if pd.to_datetime(coverage.VALID_START).max() > timeline[0] or pd.to_datetime(coverage.VALID_END).min() < timeline[-1]:
        raise ValueError(f"{graph_id}: rainfall validity does not cover frozen timeline")
    node_index = dict(zip(graph_nodes.NODE_ID, graph_nodes.NODE_INDEX.astype(int)))
    matrix = np.zeros((len(timeline), len(graph_nodes)), dtype=np.float32)
    path = require_file(
        graph_root / f"rainfall/node_hourly_rain_sparse/graph_{graph_id}_hourly_sparse.csv",
        f"{graph_id} computational-unit rainfall",
    )
    frame = read_csv(path, dtype=str)
    if frame.empty:
        return matrix
    if frame.duplicated(["NODE_ID", "START_TIME"]).any():
        raise ValueError(f"{graph_id}: duplicate sparse node-hour rainfall")
    if set(frame.GRAPH_ID) != {graph_id} or set(frame.NODE_ID) - expected_nodes:
        raise ValueError(f"{graph_id}: sparse rainfall graph/node mismatch")
    start = pd.to_datetime(frame.START_TIME, errors="raise")
    end = pd.to_datetime(frame.END_TIME, errors="raise")
    if not (end - start).eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"{graph_id}: non-hourly rainfall interval")
    time_index = ((start - timeline[0]) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    if (time_index < 0).any() or (time_index >= len(timeline)).any():
        raise ValueError(f"{graph_id}: rainfall record outside frozen timeline")
    column_index = frame.NODE_ID.map(node_index).to_numpy(dtype=int)
    values = pd.to_numeric(frame.RAIN_MM, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{graph_id}: sparse rainfall must contain finite positive values only")
    matrix[time_index, column_index] = values.astype(np.float32)
    return matrix


def load_sparse_hydro_observations(
    graph_id: str,
    graph_mapping: pd.DataFrame,
    temporal_root: Path,
    timeline: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = require_file(temporal_root / f"dynamic/graph_{graph_id}_hourly.csv", f"{graph_id} frozen hourly Q/Z")
    frame = read_csv(
        path, dtype=str,
        usecols=["TIMESTAMP", "STATION_ID", "FLOW", "WATER_LEVEL", "FLOW_MASK", "WATER_LEVEL_MASK"],
    )
    station_to_obs = dict(zip(graph_mapping.STATION_ID, graph_mapping.OBS_INDEX.astype(int)))
    frame["STATION_ID"] = frame.STATION_ID.map(norm_id)
    frame = frame[frame.STATION_ID.isin(station_to_obs)].copy()
    frame["_OBS"] = frame.STATION_ID.map(station_to_obs).astype(int)
    timestamps = pd.to_datetime(frame.TIMESTAMP, errors="raise")
    frame["_TIME"] = ((timestamps - timeline[0]) / pd.Timedelta(hours=1)).astype(int)
    if frame.duplicated(["_TIME", "_OBS"]).any():
        raise ValueError(f"{graph_id}: duplicate observation station-hour")
    expected = len(timeline) * len(graph_mapping)
    if len(frame) != expected or frame._TIME.min() != 0 or frame._TIME.max() != len(timeline) - 1:
        raise ValueError(f"{graph_id}: frozen Q/Z does not provide the complete station-hour grid")
    frame = frame.sort_values(["_TIME", "_OBS"])
    expected_obs = np.tile(np.arange(len(graph_mapping)), len(timeline))
    if not np.array_equal(frame._OBS.to_numpy(), expected_obs):
        raise ValueError(f"{graph_id}: Q/Z observation order is incomplete")
    shape = (len(timeline), len(graph_mapping))
    q_mask_raw = pd.to_numeric(frame.FLOW_MASK, errors="raise").to_numpy(dtype=np.int8)
    z_mask_raw = pd.to_numeric(frame.WATER_LEVEL_MASK, errors="raise").to_numpy(dtype=np.int8)
    if not np.isin(q_mask_raw, [0, 1]).all() or not np.isin(z_mask_raw, [0, 1]).all():
        raise ValueError(f"{graph_id}: Q/Z masks are not boolean")
    q_mask = q_mask_raw.reshape(shape).astype(bool)
    z_mask = z_mask_raw.reshape(shape).astype(bool)
    q = pd.to_numeric(frame.FLOW, errors="coerce").to_numpy(dtype=np.float64, copy=True).reshape(shape)
    z = pd.to_numeric(frame.WATER_LEVEL, errors="coerce").to_numpy(dtype=np.float64, copy=True).reshape(shape)
    if np.isnan(q[q_mask]).any() or np.isnan(z[z_mask]).any():
        raise ValueError(f"{graph_id}: valid Q/Z mask points contain missing values")
    if np.isfinite(q[~q_mask]).any() or np.isfinite(z[~z_mask]).any():
        raise ValueError(f"{graph_id}: source Q/Z contains values where mask=0")
    if np.isinf(q[q_mask]).any() or np.isinf(z[z_mask]).any():
        raise ValueError(f"{graph_id}: source Q/Z contains infinity")
    q[~q_mask] = np.nan
    z[~z_mask] = np.nan
    return q.astype(np.float32), q_mask, z.astype(np.float32), z_mask


def tensorize_graph(
    graph_id: str,
    graph_samples: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_node_static: pd.DataFrame,
    graph_edge_static: pd.DataFrame,
    graph_mapping: pd.DataFrame,
    graph_root: Path,
    temporal_root: Path,
    timeline: pd.DatetimeIndex,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    node_count, edge_count, obs_count = len(graph_nodes), len(graph_edges), len(graph_mapping)
    rain = load_graph_rain(graph_id, graph_nodes, graph_root, timeline)
    q, q_mask, z, z_mask = load_sparse_hydro_observations(graph_id, graph_mapping, temporal_root, timeline)
    origins = ((graph_samples.FORECAST_TIME - timeline[0]) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    history_indices = origins[:, None] + np.arange(-(HISTORY_HOURS - 1), 1, dtype=int)[None, :]
    future_indices = origins[:, None] + np.arange(1, FORECAST_HOURS + 1, dtype=int)[None, :]
    if history_indices.min() < 0 or future_indices.max() >= len(timeline):
        raise ValueError(f"{graph_id}: tensor indices outside frozen timeline")

    history_rain = np.transpose(rain[history_indices], (0, 2, 1))
    future_rain = np.transpose(rain[future_indices], (0, 2, 1))
    q_history_mask = np.transpose(q_mask[history_indices], (0, 2, 1))
    z_history_mask = np.transpose(z_mask[history_indices], (0, 2, 1))
    q_history = np.transpose(q[history_indices], (0, 2, 1))
    z_history = np.transpose(z[history_indices], (0, 2, 1))
    q_target_mask = np.transpose(q_mask[future_indices], (0, 2, 1))
    q_target = np.transpose(q[future_indices], (0, 2, 1))
    z_future_mask = z_mask[future_indices]
    z_base_mask = z_mask[origins]
    z_target_mask_time_major = z_future_mask & z_base_mask[:, None, :]
    z_target_time_major = z[future_indices] - z[origins][:, None, :]
    z_target_time_major[~z_target_mask_time_major] = np.nan
    z_target = np.transpose(z_target_time_major, (0, 2, 1)).astype(np.float32)
    z_target_mask = np.transpose(z_target_mask_time_major, (0, 2, 1))

    for values, mask, label in (
        (q_history, q_history_mask, "q_history"),
        (z_history, z_history_mask, "z_history"),
        (q_target, q_target_mask, "q_target"),
        (z_target, z_target_mask, "z_target"),
    ):
        if np.isnan(values[mask]).any() or np.isinf(values[mask]).any() or np.isfinite(values[~mask]).any():
            raise ValueError(f"{graph_id}: {label} value/mask invariant failed")
    if not np.isfinite(history_rain).all() or not np.isfinite(future_rain).all() or (history_rain < 0).any() or (future_rain < 0).any():
        raise ValueError(f"{graph_id}: rainfall forcing is missing/nonfinite/negative")

    node_static_matrix = numeric_matrix(graph_node_static, NODE_STATIC_FEATURES, f"{graph_id} node static")
    incremental_area = pd.to_numeric(graph_node_static.incremental_area_km2, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(incremental_area).all() or (incremental_area <= 0).any():
        raise ValueError(f"{graph_id}: incremental/local runoff area is invalid")
    edge_static_matrix = numeric_matrix(graph_edge_static, EDGE_STATIC_FEATURES, f"{graph_id} edge static")
    edge_index = graph_edges[["FROM_NODE_INDEX", "TO_NODE_INDEX"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.int64).T
    if edge_index.shape != (2, edge_count):
        raise ValueError(f"{graph_id}: edge_index shape mismatch")
    if edge_count and (edge_index.min() < 0 or edge_index.max() >= node_count):
        raise ValueError(f"{graph_id}: edge_index endpoint outside node range")

    arrays = {
        "node_id": graph_nodes.NODE_ID.to_numpy(dtype=str),
        "node_static": node_static_matrix.astype(np.float32),
        "incremental_area_km2": incremental_area.astype(np.float32),
        "edge_index": edge_index,
        "edge_static": edge_static_matrix.astype(np.float32),
        "obs_station_id": graph_mapping.STATION_ID.to_numpy(dtype=str),
        "obs_node_index": graph_mapping.MAPPED_NODE_INDEX.to_numpy(dtype=np.int64),
        "sample_id": graph_samples.SAMPLE_ID.to_numpy(dtype=str),
        "split_code": graph_samples.SPLIT.map(SPLIT_CODE).to_numpy(dtype=np.uint8),
        "forecast_time_unix_hour": ((graph_samples.FORECAST_TIME - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)).astype(np.int64).to_numpy(),
        "history_rain": history_rain.astype(np.float32),
        "future_rain": future_rain.astype(np.float32),
        "q_history": q_history.astype(np.float32),
        "q_history_mask": q_history_mask,
        "z_history": z_history.astype(np.float32),
        "z_history_mask": z_history_mask,
        "q_target": q_target.astype(np.float32),
        "q_target_mask": q_target_mask,
        "z_target": z_target,
        "z_target_mask": z_target_mask,
    }
    expected_shapes = {
        "history_rain": (len(graph_samples), node_count, HISTORY_HOURS),
        "future_rain": (len(graph_samples), node_count, FORECAST_HOURS),
        "q_history": (len(graph_samples), obs_count, HISTORY_HOURS),
        "z_history": (len(graph_samples), obs_count, HISTORY_HOURS),
        "q_target": (len(graph_samples), obs_count, FORECAST_HOURS),
        "z_target": (len(graph_samples), obs_count, FORECAST_HOURS),
    }
    masked_keys = {"q_history", "z_history", "q_target", "z_target"}
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ValueError(f"{graph_id}: {key} expected {expected}, actual {arrays[key].shape}")
        if key in masked_keys and arrays[f"{key}_mask"].shape != expected:
            raise ValueError(
                f"{graph_id}: {key}_mask expected {expected}, "
                f"actual {arrays[f'{key}_mask'].shape}"
            )
    if not (q_target_mask | z_target_mask).any(axis=(1, 2)).all():
        raise ValueError(f"{graph_id}: a frozen sample unexpectedly has no Q or Z target supervision")

    index = graph_samples.copy()
    index["TENSOR_ROW"] = np.arange(len(index), dtype=np.int64)
    index["N_NODE"] = node_count
    index["N_OBS"] = obs_count
    index["Q_TARGET_VALID_COUNT"] = q_target_mask.sum(axis=(1, 2)).astype(int)
    index["Z_TARGET_VALID_COUNT"] = z_target_mask.sum(axis=(1, 2)).astype(int)
    return arrays, index


def update_coverage(coverage: dict, split: str, arrays: dict, selector: np.ndarray) -> None:
    record = coverage[split]
    count = int(selector.sum())
    if count == 0:
        return
    record["samples"] += count
    record["rain_total"] += int(arrays["history_rain"][selector].size + arrays["future_rain"][selector].size)
    record["rain_positive"] += int((arrays["history_rain"][selector] > 0).sum() + (arrays["future_rain"][selector] > 0).sum())
    for key in ("q_history", "z_history", "q_target", "z_target"):
        record[f"{key}_total"] += int(arrays[f"{key}_mask"][selector].size)
        record[f"{key}_valid"] += int(arrays[f"{key}_mask"][selector].sum())


def stats_by_station_result(
    stats: dict[str, RunningStats],
    global_result: dict,
) -> dict:
    output = {}
    for station in sorted(stats):
        result = stats[station].result(scale_floor=1e-6)
        if result["available"]:
            result["normalization_source"] = "TRAIN_PER_STATION"
            result["applied_mean"] = result["mean"]
            result["applied_scale"] = result["scale"]
        else:
            if not global_result["available"]:
                raise ValueError("TRAIN-global fallback statistics are unavailable")
            result["normalization_source"] = "TRAIN_GLOBAL_FALLBACK"
            result["applied_mean"] = global_result["mean"]
            result["applied_scale"] = global_result["scale"]
        output[station] = result
    return output


def percent(valid: int, total: int) -> float:
    return 100.0 * valid / total if total else 0.0


def write_report(
    path: Path,
    output: Path,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    mapping: pd.DataFrame,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    coverage: dict,
    normalization: dict,
    tensor_bytes: int,
) -> None:
    sample_counts = samples.SPLIT.value_counts().to_dict()
    event_counts = events.SPLIT.value_counts().to_dict()
    coverage_rows = []
    for split in ("TRAIN", "VALIDATION", "TEST"):
        row = coverage[split]
        coverage_rows.append(
            f"| {split} | {row['samples']} | "
            f"{percent(row['q_history_valid'], row['q_history_total']):.3f}% | "
            f"{percent(row['z_history_valid'], row['z_history_total']):.3f}% | "
            f"{percent(row['q_target_valid'], row['q_target_total']):.3f}% | "
            f"{percent(row['z_target_valid'], row['z_target_total']):.3f}% | "
            f"100.000% | {percent(row['rain_positive'], row['rain_total']):.3f}% |"
        )
    text = f"""# Model Dataset v8 — Hydrologic Computational Graph

## 构建结果

- Dataset: `{output}`
- Spatial source: `project/_hydrologic_graph_v1`（原样只读复用）
- Temporal source: `project/_model_dataset_v7_event_multitask`（只读继承 event、split、sample origin 和 Q/Z）
- Graphs: **{nodes.GRAPH_ID.nunique()}**
- Computational nodes: **{len(nodes)}**
- Directed river-reach edges: **{len(edges)}**
- Observation stations: **{len(mapping)}**（outlet **{int(mapping.IS_OUTLET_STATION.sum())}**；internal **{len(mapping)-int(mapping.IS_OUTLET_STATION.sum())}**）
- Events: **{len(events)}**（TRAIN {event_counts.get('TRAIN',0)}；VALIDATION {event_counts.get('VALIDATION',0)}；TEST {event_counts.get('TEST',0)}）
- Samples: **{len(samples)}**（TRAIN {sample_counts.get('TRAIN',0)}；VALIDATION {sample_counts.get('VALIDATION',0)}；TEST {sample_counts.get('TEST',0)}）
- Tensor files: **33** graph-grouped compressed NPZ；总大小 **{tensor_bytes/1024/1024:.2f} MiB**。

## Physics forcing 与 sparse observations

- `history_rain`: `[S, Nnode, 24]`, float32, mm
- `future_rain`: `[S, Nnode, 6]`, float32, mm
- `node_static`: `[Nnode, {len(NODE_STATIC_FEATURES)}]`, float32
- `incremental_area_km2`: `[Nnode]`, float32；它是 local runoff/unit catchment area，不是 upstream area
- `edge_index`: `[2, Nedge]`, int64
- `edge_static`: `[Nedge, {len(EDGE_STATIC_FEATURES)}]`, float32；字段为 `{EDGE_STATIC_FEATURES[0]}`, `{EDGE_STATIC_FEATURES[1]}`
- `obs_station_id`: `[Nobs]`；`obs_node_index`: `[Nobs]`，允许多个真实站映射到同一 computational node
- `q_history`, `z_history`: `[S, Nobs, 24]`, float32；缺测为 NaN，独立 boolean mask
- `q_target`: `[S, Nobs, 6]`, float32，原始 Q（m³/s）
- `z_target`: `[S, Nobs, 6]`, float32，保持既有目标语义 `ΔZ(t+h)=Z(t+h)-Z(t0)`
- 没有站点的 computational node 不存在任何 Q/Z 数组槽位；Q/Z 从未作为 `[Nnode,T]` dynamic feature 构建，也未以0伪造。
- 雨量是 physics forcing；冻结有效时间内稀疏文件无正雨量记录的小时为真实 `0 mm`。

## Supervision / forcing coverage

| split | samples | Q history | Z history | Q target | ΔZ target | rain forcing | positive rain |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(coverage_rows)}

Q/Z coverage 的分母是各 split 的 `sample × Nobs × hours`；不同任务独立缺测，mask=0 的任务不贡献监督。Rain forcing 在全部 sample/node/hour 上完整且有限。

## TRAIN-only normalization

- `metadata/dataset_contract.json` 内的 `normalization` 全部标记 `computed_from_split=TRAIN`。
- Rain statistics：仅使用 TRAIN sample 中实际暴露的 history+future forcing。
- Q/Z statistics：按 observation station 独立拟合，仅使用 TRAIN sample exposure；无该任务数据的站保留 `available=false`，不伪造 scale。
- `z_target` scale 按 TRAIN-only ΔZ 拟合。
- Node/edge static statistics：仅使用至少有一个 TRAIN sample 的 graph；VALIDATION/TEST 不参与拟合。

## QC

- 33 graph 集合与 `_hydrologic_graph_v1` 一致，`Q_61512000/61512000` 不存在：PASS
- 237 nodes、204 edges、39 station mappings 与 QC-PASS 空间事实一致：PASS
- 每图 DAG、唯一 outlet、edge endpoint、station-node index 一致：PASS
- 33 个出口观测站全部映射到各自 final outlet node：PASS
- v8 的 SAMPLE_ID/EVENT_ID/FORECAST_TIME/SPLIT 与 v7 的33图子集逐条一致：PASS
- history=24 h、forecast=6 h，事件和 split 未重建、未重分：PASS
- Rain tensor 全部有限、非负，forcing coverage=100%：PASS
- Q/Z 在 mask=1 时有限，在 mask=0 时为 NaN；未向无站节点扩展：PASS
- Node/edge static 与 incremental area 无 missing/nonfinite：PASS
- 所有 normalization/statistics 仅由 TRAIN 拟合：PASS
- Staging 完整验证后才原子发布：PASS

**FINAL QC STATUS: PASS**
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    graph_root = args.hydrologic_graph.expanduser().resolve()
    temporal_root = args.temporal_dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing formal dataset: {output}")
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    if "FINAL QC STATUS: PASS" not in require_file(graph_root / "BUILD_AND_QC.md", "hydrologic graph QC report").read_text(encoding="utf-8"):
        raise ValueError("source hydrologic graph is not QC PASS")

    graph_paths = {
        "node_catalog": require_file(graph_root / "graph/node_catalog.csv", "node catalog"),
        "edge_topology": require_file(graph_root / "graph/edge_topology.csv", "edge topology"),
        "node_static": require_file(graph_root / "graph/node_static_attributes.csv", "node static"),
        "edge_static": require_file(graph_root / "graph/edge_static_attributes.csv", "edge static"),
        "mapping": require_file(graph_root / "graph/station_observation_mapping.csv", "station mapping"),
    }
    schema_path = require_file(temporal_root / "metadata/feature_schema.json", "frozen v7 schema")
    source_sample_path = require_file(temporal_root / "samples/sample_index.csv", "frozen v7 samples")
    source_event_path = require_file(temporal_root / "events/hydrologic_events.csv", "frozen v7 events")

    nodes = read_csv(graph_paths["node_catalog"], dtype=str)
    edges = read_csv(graph_paths["edge_topology"], dtype=str)
    node_static = read_csv(graph_paths["node_static"], dtype=str)
    edge_static = read_csv(graph_paths["edge_static"], dtype=str)
    mapping = read_csv(graph_paths["mapping"], dtype=str)
    for frame in (nodes, edges, node_static, edge_static, mapping):
        frame["GRAPH_ID"] = frame.GRAPH_ID.map(norm_id)
    mapping["STATION_ID"] = mapping.STATION_ID.map(norm_id)
    graphs = validate_graph_facts(nodes, edges, node_static, edge_static, mapping)
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    source_samples_all = read_csv(source_sample_path, dtype=str)
    source_events_all = read_csv(source_event_path, dtype=str)
    samples, events, timeline = validate_temporal_contract(source_samples_all, source_events_all, schema, graphs)
    if len(samples) != 279_574 or len(events) != 2_807:
        raise ValueError("frozen 33-graph sample/event cardinality changed")

    stage.mkdir(parents=True)
    try:
        (stage / "graph").mkdir()
        (stage / "events").mkdir()
        (stage / "samples/tensors").mkdir(parents=True)
        (stage / "metadata").mkdir()

        # Preserve graph facts byte-for-byte except the mapping, which receives
        # an explicit graph-local OBS_INDEX used by tensor arrays.
        for key, name in (
            ("node_catalog", "node_catalog.csv"),
            ("edge_topology", "edge_topology.csv"),
            ("node_static", "node_static_attributes.csv"),
            ("edge_static", "edge_static_attributes.csv"),
        ):
            shutil.copy2(graph_paths[key], stage / "graph" / name)

        mapping_rows = []
        for graph_id in sorted(graphs):
            part = mapping[mapping.GRAPH_ID.eq(graph_id)].copy()
            part["IS_OUTLET_STATION"] = pd.to_numeric(part.IS_OUTLET_STATION, errors="raise").astype(int)
            part = part.sort_values(["IS_OUTLET_STATION", "STATION_ID"], ascending=[False, True]).reset_index(drop=True)
            part["OBS_INDEX"] = np.arange(len(part), dtype=int)
            mapping_rows.append(part)
        mapping_output = pd.concat(mapping_rows, ignore_index=True)
        mapping_output.to_csv(stage / "graph/station_observation_mapping.csv", index=False, encoding="utf-8-sig")

        event_output = events.drop(columns=[column for column in events.columns if column.startswith("_")], errors="ignore")
        event_output.to_csv(stage / "events/hydrologic_events.csv", index=False, encoding="utf-8-sig")

        coverage_template = {
            "samples": 0, "rain_total": 0, "rain_positive": 0,
            "q_history_total": 0, "q_history_valid": 0,
            "z_history_total": 0, "z_history_valid": 0,
            "q_target_total": 0, "q_target_valid": 0,
            "z_target_total": 0, "z_target_valid": 0,
        }
        coverage = {split: dict(coverage_template) for split in SPLIT_CODE}
        rain_stats = RunningStats()
        node_feature_stats = {feature: RunningStats() for feature in NODE_STATIC_FEATURES}
        incremental_area_stats = RunningStats()
        edge_feature_stats = {feature: RunningStats() for feature in EDGE_STATIC_FEATURES}
        q_history_stats = {station: RunningStats() for station in mapping_output.STATION_ID}
        z_history_stats = {station: RunningStats() for station in mapping_output.STATION_ID}
        q_target_stats = {station: RunningStats() for station in mapping_output.STATION_ID}
        dz_target_stats = {station: RunningStats() for station in mapping_output.STATION_ID}
        q_history_global = RunningStats()
        z_history_global = RunningStats()
        q_target_global = RunningStats()
        dz_target_global = RunningStats()
        sample_index_parts = []
        tensor_paths = []

        for number, graph_id in enumerate(sorted(graphs), 1):
            graph_nodes = nodes[nodes.GRAPH_ID.eq(graph_id)].copy()
            graph_nodes["NODE_INDEX"] = pd.to_numeric(graph_nodes.NODE_INDEX, errors="raise").astype(int)
            graph_nodes = graph_nodes.sort_values("NODE_INDEX")
            graph_edges = edges[edges.GRAPH_ID.eq(graph_id)].copy()
            for field in ("FROM_NODE_INDEX", "TO_NODE_INDEX"):
                graph_edges[field] = pd.to_numeric(graph_edges[field], errors="raise").astype(int)
            graph_edges = graph_edges.sort_values(["FROM_NODE_INDEX", "TO_NODE_INDEX"])
            graph_node_static = node_static[node_static.GRAPH_ID.eq(graph_id)].copy()
            graph_node_static["NODE_INDEX"] = pd.to_numeric(graph_node_static.NODE_INDEX, errors="raise").astype(int)
            graph_node_static = graph_node_static.sort_values("NODE_INDEX")
            graph_edge_static = edge_static[edge_static.GRAPH_ID.eq(graph_id)].copy()
            graph_edge_static["FROM_NODE_INDEX"] = pd.to_numeric(graph_edge_static.FROM_NODE_INDEX, errors="raise").astype(int)
            graph_edge_static["TO_NODE_INDEX"] = pd.to_numeric(graph_edge_static.TO_NODE_INDEX, errors="raise").astype(int)
            graph_edge_static = graph_edge_static.sort_values(["FROM_NODE_INDEX", "TO_NODE_INDEX"])
            if graph_nodes.NODE_ID.tolist() != graph_node_static.NODE_ID.tolist() or graph_edges.EDGE_ID.tolist() != graph_edge_static.EDGE_ID.tolist():
                raise ValueError(f"{graph_id}: topology/static row alignment failed")
            graph_mapping = mapping_output[mapping_output.GRAPH_ID.eq(graph_id)].sort_values("OBS_INDEX").copy()
            graph_mapping["MAPPED_NODE_INDEX"] = pd.to_numeric(graph_mapping.MAPPED_NODE_INDEX, errors="raise").astype(int)
            graph_samples = samples[samples.GRAPH_ID.eq(graph_id)].sort_values("_SOURCE_ORDER").copy()
            arrays, index = tensorize_graph(
                graph_id, graph_samples, graph_nodes, graph_edges,
                graph_node_static, graph_edge_static, graph_mapping,
                graph_root, temporal_root, timeline,
            )
            relative_tensor = f"samples/tensors/graph_{graph_id}.npz"
            tensor_path = stage / relative_tensor
            np.savez_compressed(tensor_path, **arrays)
            tensor_paths.append(tensor_path)
            index["TENSOR_FILE"] = relative_tensor
            sample_index_parts.append(index)

            for split in SPLIT_CODE:
                update_coverage(coverage, split, arrays, index.SPLIT.eq(split).to_numpy())
            train = index.SPLIT.eq("TRAIN").to_numpy()
            if not train.any():
                raise ValueError(f"{graph_id}: no frozen TRAIN samples")
            rain_stats.update(arrays["history_rain"][train])
            rain_stats.update(arrays["future_rain"][train])
            for feature_index, feature in enumerate(NODE_STATIC_FEATURES):
                node_feature_stats[feature].update(arrays["node_static"][:, feature_index])
            incremental_area_stats.update(arrays["incremental_area_km2"])
            for feature_index, feature in enumerate(EDGE_STATIC_FEATURES):
                edge_feature_stats[feature].update(arrays["edge_static"][:, feature_index])
            for obs_index, station_id in enumerate(arrays["obs_station_id"].tolist()):
                qh = arrays["q_history"][train, obs_index, :][arrays["q_history_mask"][train, obs_index, :]]
                zh = arrays["z_history"][train, obs_index, :][arrays["z_history_mask"][train, obs_index, :]]
                qt = arrays["q_target"][train, obs_index, :][arrays["q_target_mask"][train, obs_index, :]]
                dzt = arrays["z_target"][train, obs_index, :][arrays["z_target_mask"][train, obs_index, :]]
                q_history_stats[station_id].update(qh); q_history_global.update(qh)
                z_history_stats[station_id].update(zh); z_history_global.update(zh)
                q_target_stats[station_id].update(qt); q_target_global.update(qt)
                dz_target_stats[station_id].update(dzt); dz_target_global.update(dzt)
            print(
                f"[tensor {number:02d}/33] {graph_id}: samples={len(index)} "
                f"nodes={len(graph_nodes)} obs={len(graph_mapping)}",
                flush=True,
            )

        sample_output = pd.concat(sample_index_parts, ignore_index=True).sort_values("_SOURCE_ORDER")
        if sample_output.SAMPLE_ID.tolist() != samples.sort_values("_SOURCE_ORDER").SAMPLE_ID.tolist():
            raise ValueError("v8 sample ID/order differs from retained v7 domain")
        compare = sample_output[["SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "FORECAST_TIME", "SPLIT"]].copy()
        source_compare = samples.sort_values("_SOURCE_ORDER")[["SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "FORECAST_TIME", "SPLIT"]].reset_index(drop=True)
        if not compare.reset_index(drop=True).equals(source_compare):
            raise ValueError("v8 sample event/origin/split changed from retained v7 domain")
        output_columns = [
            "SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "OUTLET_ID", "INPUT_START",
            "FORECAST_TIME", "TARGET_START", "TARGET_END", "HISTORY_HOURS",
            "FORECAST_HOURS", "SPLIT", "N_NODE", "N_OBS",
            "Q_TARGET_VALID_COUNT", "Z_TARGET_VALID_COUNT", "TENSOR_FILE", "TENSOR_ROW",
        ]
        missing_output_columns = [field for field in output_columns if field not in sample_output.columns]
        if missing_output_columns:
            raise ValueError(f"sample output missing fields: {missing_output_columns}")
        sample_output[output_columns].to_csv(stage / "samples/sample_index.csv", index=False, encoding="utf-8-sig")

        q_history_global_result = q_history_global.result(scale_floor=1e-6)
        z_history_global_result = z_history_global.result(scale_floor=1e-6)
        q_target_global_result = q_target_global.result(scale_floor=1e-6)
        dz_target_global_result = dz_target_global.result(scale_floor=1e-6)
        normalization = {
            "computed_from_split": "TRAIN",
            "fit_scope": "TRAIN_SAMPLE_EXPOSURE_ONLY",
            "rain_mm": rain_stats.result(scale_floor=1e-6),
            "incremental_area_km2": incremental_area_stats.result(scale_floor=1e-6),
            "node_static": {feature: node_feature_stats[feature].result(scale_floor=1e-6) for feature in NODE_STATIC_FEATURES},
            "edge_static": {feature: edge_feature_stats[feature].result(scale_floor=1e-6) for feature in EDGE_STATIC_FEATURES},
            "q_history_global": q_history_global_result,
            "z_history_global": z_history_global_result,
            "q_target_global": q_target_global_result,
            "delta_z_target_global": dz_target_global_result,
            "q_history_by_station": stats_by_station_result(q_history_stats, q_history_global_result),
            "z_history_by_station": stats_by_station_result(z_history_stats, z_history_global_result),
            "q_target_by_station": stats_by_station_result(q_target_stats, q_target_global_result),
            "delta_z_target_by_station": stats_by_station_result(dz_target_stats, dz_target_global_result),
        }
        contract = {
            "contract": "hydrologic-computational-graph-sparse-observation-v1",
            "graph_count": 33,
            "computational_node_count": len(nodes),
            "edge_count": len(edges),
            "observation_station_count": len(mapping_output),
            "history_hours": HISTORY_HOURS,
            "forecast_hours": FORECAST_HOURS,
            "time_zone": schema.get("time_zone", "Asia/Shanghai"),
            "global_time_start": schema["global_time_start"],
            "global_time_end": schema["global_time_end"],
            "time_splits": schema["time_splits"],
            "split_codes": SPLIT_CODE,
            "temporal_domain": {
                "events": "events/hydrologic_events.csv",
                "samples": "samples/sample_index.csv",
                "source": str(temporal_root),
                "inheritance": "exact retained 33-graph subset; no event detection, resplit, or sample-origin change",
            },
            "graph_source": str(graph_root),
            "tensor_files": "samples/tensors/graph_{GRAPH_ID}.npz",
            "tensor_schema": {
                "history_rain": "float32 [S,Nnode,24], mm, dense physics forcing",
                "future_rain": "float32 [S,Nnode,6], mm, dense physics forcing",
                "node_static": f"float32 [Nnode,{len(NODE_STATIC_FEATURES)}]",
                "incremental_area_km2": "float32 [Nnode], physical local runoff area",
                "edge_index": "int64 [2,Nedge], upstream-to-downstream",
                "edge_static": f"float32 [Nedge,{len(EDGE_STATIC_FEATURES)}]",
                "obs_station_id": "unicode [Nobs]",
                "obs_node_index": "int64 [Nobs], station observation mapping; duplicates allowed",
                "q_history": "float32 [S,Nobs,24], NaN iff mask=false",
                "q_history_mask": "bool [S,Nobs,24]",
                "z_history": "float32 [S,Nobs,24], NaN iff mask=false",
                "z_history_mask": "bool [S,Nobs,24]",
                "q_target": "float32 [S,Nobs,6], raw Q m3/s, NaN iff mask=false",
                "q_target_mask": "bool [S,Nobs,6]",
                "z_target": "float32 [S,Nobs,6], delta Z from t0 in m, NaN iff mask=false",
                "z_target_mask": "bool [S,Nobs,6]",
            },
            "node_static_features": list(NODE_STATIC_FEATURES),
            "edge_static_features": list(EDGE_STATIC_FEATURES),
            "rain_zero_semantics": "within frozen global timeline, absent positive sparse record is 0 mm",
            "observation_semantics": "Q/Z exist only on Nobs mapped stations; no computational-node broadcasting or zero filling",
            "normalization": normalization,
        }
        (stage / "metadata/dataset_contract.json").write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

        # Independent on-disk validation: loading uses allow_pickle=False and
        # checks every array contract rather than trusting in-memory objects.
        if len(tensor_paths) != 33:
            raise ValueError("expected 33 tensor files")
        for tensor_path in tensor_paths:
            with np.load(tensor_path, allow_pickle=False) as tensor:
                required = {
                    "history_rain", "future_rain", "node_static", "incremental_area_km2",
                    "node_id", "edge_index", "edge_static", "obs_station_id", "obs_node_index",
                    "q_history", "q_history_mask", "z_history", "z_history_mask",
                    "q_target", "q_target_mask", "z_target", "z_target_mask",
                    "sample_id", "split_code", "forecast_time_unix_hour",
                }
                if set(tensor.files) != required:
                    raise ValueError(f"{tensor_path.name}: tensor keys differ from contract")
                for value_key, mask_key in (
                    ("q_history", "q_history_mask"), ("z_history", "z_history_mask"),
                    ("q_target", "q_target_mask"), ("z_target", "z_target_mask"),
                ):
                    values, masks = tensor[value_key], tensor[mask_key].astype(bool)
                    if values.shape != masks.shape or not np.isfinite(values[masks]).all() or np.isfinite(values[~masks]).any():
                        raise ValueError(f"{tensor_path.name}: {value_key} mask/value QC failed")
                if not np.isfinite(tensor["history_rain"]).all() or not np.isfinite(tensor["future_rain"]).all():
                    raise ValueError(f"{tensor_path.name}: rainfall forcing contains nonfinite data")

        tensor_bytes = sum(path.stat().st_size for path in tensor_paths)
        write_report(
            stage / "BUILD_AND_QC.md", output, nodes, edges,
            mapping_output.assign(IS_OUTLET_STATION=pd.to_numeric(mapping_output.IS_OUTLET_STATION)),
            sample_output, event_output, coverage, normalization, tensor_bytes,
        )
        if "FINAL QC STATUS: PASS" not in (stage / "BUILD_AND_QC.md").read_text(encoding="utf-8"):
            raise ValueError("final report did not record PASS")
        expected_files = {
            "graph/node_catalog.csv", "graph/edge_topology.csv",
            "graph/node_static_attributes.csv", "graph/edge_static_attributes.csv",
            "graph/station_observation_mapping.csv", "events/hydrologic_events.csv",
            "samples/sample_index.csv", "metadata/dataset_contract.json", "BUILD_AND_QC.md",
        }
        actual_files = {str(path.relative_to(stage)).replace("\\", "/") for path in stage.rglob("*") if path.is_file()}
        if not expected_files.issubset(actual_files) or len(actual_files) != len(expected_files) + 33:
            raise ValueError("formal output file set is incomplete or contains unexpected extras")
        stage.rename(output)
        print(json.dumps({
            "status": "PASS", "output": str(output), "graphs": 33,
            "nodes": len(nodes), "edges": len(edges), "stations": len(mapping_output),
            "events": len(event_output), "samples": len(sample_output),
            "split_samples": {key: int(value) for key, value in sample_output.SPLIT.value_counts().items()},
        }, ensure_ascii=False, indent=2))
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    main()
