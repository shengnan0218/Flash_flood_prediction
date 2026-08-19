"""V11 dataset views and event-balanced same-graph batching.

V11 preserves the frozen event/split/sample-origin facts but changes two runtime
properties only:
- rainfall warm-up is 72 h;
- Q/Z observation history used for assimilation remains 24 h.

TRAIN uses a deterministic event-balanced, phase-stratified sampler. VALIDATION
and TEST use the complete graph-grouped views.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from data.schema import topological_levels
from data.v8_schema import HydrologicGraphBatch
from datasets.hydrologic_graph_v8 import (
    HydrologicGraphV8Dataset,
    _norm_id,
)

CONTRACT_NAME_V11 = "hydrologic-computational-graph-72h-rain-24h-observation-v11"
RAIN_HISTORY_HOURS = 72
OBS_HISTORY_HOURS = 24
FORECAST_HOURS = 6
EVENT_PHASES = ("LOW", "RISING", "PEAK", "RECESSION")


class HydrologicGraphV11Dataset(HydrologicGraphV8Dataset):
    """Load V11 tensors without mutating the underlying on-disk dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        graph_id: str | None = None,
        future_rainfall_mode: str = "observed_hindcast",
        strict: bool = True,
        tensor_cache: dict[str, dict[str, np.ndarray]] | None = None,
        require_q_supervision: bool,
    ) -> None:
        # Do not call the V8 constructor: V11 has a different explicit contract.
        self.root = Path(root).expanduser().resolve()
        self.split = str(split).upper()
        if self.split not in {"TRAIN", "VALIDATION", "TEST"}:
            raise ValueError("split必须为TRAIN/VALIDATION/TEST")
        if future_rainfall_mode not in {"observed_hindcast", "zero", "persistence"}:
            raise ValueError(
                "future_rainfall_mode必须为observed_hindcast/zero/persistence"
            )
        self.future_rainfall_mode = future_rainfall_mode
        self.strict = bool(strict)
        self.require_q_supervision = bool(require_q_supervision)

        contract_path = self.root / "metadata/dataset_contract.json"
        sample_path = self.root / "samples/sample_index.csv"
        mapping_path = self.root / "graph/station_observation_mapping.csv"
        for path in (contract_path, sample_path, mapping_path):
            if not path.is_file():
                raise FileNotFoundError(f"v11正式数据缺少: {path}")

        self.contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        if self.contract.get("contract") != CONTRACT_NAME_V11:
            raise ValueError(
                f"不是v11 72h-rain contract: {self.contract.get('contract')!r}"
            )
        self.rain_history_hours = int(self.contract["rain_history_hours"])
        self.observation_history_hours = int(self.contract["observation_history_hours"])
        self.forecast_hours = int(self.contract["forecast_hours"])
        # Keep the inherited semantic name attached to Q/Z history only.
        self.history_hours = self.observation_history_hours
        if (
            self.rain_history_hours,
            self.observation_history_hours,
            self.forecast_hours,
        ) != (RAIN_HISTORY_HOURS, OBS_HISTORY_HOURS, FORECAST_HOURS):
            raise ValueError(
                "v11固定要求rain_history=72 h、Q/Z history=24 h、forecast=6 h"
            )

        samples = pd.read_csv(sample_path, encoding="utf-8-sig", dtype=str)
        required = {
            "SAMPLE_ID",
            "EVENT_ID",
            "GRAPH_ID",
            "FORECAST_TIME",
            "SPLIT",
            "TENSOR_FILE",
            "TENSOR_ROW",
            "N_NODE",
            "N_OBS",
            "Q_TARGET_VALID_COUNT",
            "EVENT_PHASE",
        }
        missing = required - set(samples.columns)
        if missing:
            raise ValueError(f"v11 sample_index缺少字段: {sorted(missing)}")
        samples["GRAPH_ID"] = samples["GRAPH_ID"].map(_norm_id)
        samples["SPLIT"] = samples["SPLIT"].str.upper()
        samples["EVENT_PHASE"] = samples["EVENT_PHASE"].str.upper()
        samples = samples[samples["SPLIT"].eq(self.split)].copy()
        if graph_id is not None:
            graph_id = _norm_id(graph_id)
            samples = samples[samples["GRAPH_ID"].eq(graph_id)].copy()
        if samples.empty:
            raise ValueError(f"{self.split}没有可用v11样本")

        for field in ("TENSOR_ROW", "N_NODE", "N_OBS", "Q_TARGET_VALID_COUNT"):
            samples[field] = pd.to_numeric(samples[field], errors="raise").astype(np.int64)
        if (samples["Q_TARGET_VALID_COUNT"] < 0).any():
            raise ValueError("Q_TARGET_VALID_COUNT不能为负")
        invalid_phase = ~samples["EVENT_PHASE"].isin(EVENT_PHASES)
        if invalid_phase.any():
            values = sorted(samples.loc[invalid_phase, "EVENT_PHASE"].unique().tolist())
            raise ValueError(f"v11 EVENT_PHASE非法: {values}")

        self.frozen_sample_count_before_q_filter = len(samples)
        self.q_supervised_sample_count = int(
            samples["Q_TARGET_VALID_COUNT"].gt(0).sum()
        )
        if self.require_q_supervision:
            samples = samples.loc[samples["Q_TARGET_VALID_COUNT"].gt(0)].copy()
            if samples.empty:
                raise ValueError(f"{self.split}没有Q监督窗口")
        self.samples = samples.reset_index(drop=True)

        mapping = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype=str)
        for field in ("GRAPH_ID", "STATION_ID", "MAPPED_NODE_INDEX"):
            if field not in mapping.columns:
                raise ValueError(f"station_observation_mapping缺少{field}")
        mapping["GRAPH_ID"] = mapping["GRAPH_ID"].map(_norm_id)
        mapping["STATION_ID"] = mapping["STATION_ID"].map(_norm_id)
        station_ids = sorted(mapping["STATION_ID"].unique().tolist())
        expected_station_count = int(self.contract["observation_station_count"])
        if len(station_ids) != expected_station_count:
            raise ValueError(
                f"全局观测站数量应为{expected_station_count}，实际={len(station_ids)}"
            )
        self.station_ids = tuple(station_ids)
        self.station_to_index = {
            station: index for index, station in enumerate(self.station_ids)
        }
        self.num_stations = len(self.station_ids)
        self.graph_ids = tuple(sorted(self.samples["GRAPH_ID"].unique().tolist()))
        self.event_ids = tuple(sorted(self.samples["EVENT_ID"].unique().tolist()))
        self._tensor_cache = tensor_cache if tensor_cache is not None else {}
        self.train_sampling_mode = (
            "v11_event_balanced_phase_stratified"
            if self.split == "TRAIN" and self.require_q_supervision
            else "v11_full_view_same_graph_batches"
        )

    @property
    def q_filter_removed_count(self) -> int:
        return self.frozen_sample_count_before_q_filter - len(self.samples)


def validate_v11_batch(batch: HydrologicGraphBatch) -> None:
    if batch.history_rain.ndim != 4:
        raise ValueError("v11 history_rain应为[B,72,Nnode,1]")
    b, hr, n, rain_dim = batch.history_rain.shape
    if hr != RAIN_HISTORY_HOURS or rain_dim != 1 or b <= 0 or n <= 0:
        raise ValueError("v11 72h rainfall warm-up shape错误")
    if tuple(batch.future_rain.shape) != (b, FORECAST_HOURS, n, 1):
        raise ValueError("v11 future_rain shape错误")
    if batch.node_static.ndim != 2 or batch.node_static.shape[0] != n:
        raise ValueError("v11 node_static shape错误")
    if tuple(batch.incremental_area_km2.shape) != (n,):
        raise ValueError("v11 incremental_area_km2 shape错误")
    if (
        batch.edge_index.dtype != torch.long
        or batch.edge_index.ndim != 2
        or batch.edge_index.shape[0] != 2
    ):
        raise ValueError("v11 edge_index必须为LongTensor [2,E]")
    edge_count = int(batch.edge_index.shape[1])
    if batch.edge_static.ndim != 2 or batch.edge_static.shape[0] != edge_count:
        raise ValueError("v11 edge_static shape错误")
    obs = int(batch.obs_node_index.numel())
    if (
        obs <= 0
        or batch.obs_node_index.dtype != torch.long
        or tuple(batch.obs_station_index.shape) != (obs,)
    ):
        raise ValueError("v11 observation mapping shape错误")
    if (batch.obs_node_index < 0).any() or (batch.obs_node_index >= n).any():
        raise ValueError("v11 obs_node_index越界")
    for name in ("q_history", "z_history", "q_mask", "z_mask"):
        if tuple(getattr(batch, name).shape) != (b, OBS_HISTORY_HOURS, obs):
            raise ValueError(f"v11 {name}必须为[B,24,Nobs]")
    for name in ("q_target", "z_target", "q_target_mask", "z_target_mask"):
        if tuple(getattr(batch, name).shape) != (b, FORECAST_HOURS, obs):
            raise ValueError(f"v11 {name}必须为[B,6,Nobs]")
    for name in ("q_mask", "z_mask", "q_target_mask", "z_target_mask"):
        if getattr(batch, name).dtype != torch.bool:
            raise ValueError(f"v11 {name}必须为BoolTensor")
    for name in (
        "history_rain",
        "future_rain",
        "node_static",
        "incremental_area_km2",
        "edge_static",
        "q_history",
        "z_history",
        "q_target",
        "z_target",
    ):
        if not torch.isfinite(getattr(batch, name)).all():
            raise ValueError(f"v11 {name}含NaN/Inf")
    if (batch.history_rain < 0).any() or (batch.future_rain < 0).any():
        raise ValueError("v11 rainfall forcing必须非负")
    if (batch.incremental_area_km2 <= 0).any():
        raise ValueError("v11 incremental_area_km2必须>0")
    topological_levels(batch.edge_index, n)


def collate_hydrologic_graph_v11(
    items: list[HydrologicGraphBatch],
) -> HydrologicGraphBatch:
    if not items:
        raise ValueError("不能collate空v11样本")
    graph_ids = [item.graph_id for item in items]
    if any(not isinstance(value, str) for value in graph_ids) or len(set(graph_ids)) != 1:
        raise ValueError(f"一个v11 batch只能来自同一GRAPH_ID，实际={graph_ids}")
    static_fields = (
        "node_static",
        "incremental_area_km2",
        "edge_index",
        "edge_static",
        "obs_node_index",
        "obs_station_index",
    )
    for name in static_fields:
        first = getattr(items[0], name)
        if any(not torch.equal(first, getattr(item, name)) for item in items[1:]):
            raise ValueError(f"同图v11 batch内{name}不一致")
    if any(item.obs_station_ids != items[0].obs_station_ids for item in items[1:]):
        raise ValueError("同图v11 batch内obs_station_ids不一致")

    kwargs: dict[str, Any] = {
        name: getattr(items[0], name) for name in static_fields
    }
    kwargs["obs_station_ids"] = items[0].obs_station_ids
    for name in (
        "history_rain",
        "future_rain",
        "q_history",
        "z_history",
        "q_mask",
        "z_mask",
        "q_target",
        "z_target",
        "q_target_mask",
        "z_target_mask",
        "sample_weight",
    ):
        values = [getattr(item, name) for item in items]
        if any(value is None for value in values):
            raise ValueError(f"v11 batch内{name}不能为None")
        kwargs[name] = torch.stack(values)
    for name in ("sample_id", "event_id", "graph_id", "forecast_time"):
        kwargs[name] = tuple(getattr(item, name) for item in items)
    batch = HydrologicGraphBatch(**kwargs)
    validate_v11_batch(batch)
    return batch


class EventBalancedV11BatchSampler(Sampler[list[int]]):
    """Select at most 8 origins/event/epoch, with 2-per-phase first priority."""

    def __init__(
        self,
        dataset: HydrologicGraphV11Dataset,
        batch_size: int,
        *,
        origins_per_event: int = 8,
        phase_quota: int = 2,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if dataset.split != "TRAIN" or not dataset.require_q_supervision:
            raise ValueError("event-balanced sampler只允许V11 Q-supervised TRAIN")
        if batch_size <= 0 or origins_per_event <= 0 or phase_quota <= 0:
            raise ValueError("batch_size/origins_per_event/phase_quota必须>0")
        if phase_quota * len(EVENT_PHASES) != origins_per_event:
            raise ValueError("正式V11固定四phase等额配额，phase_quota*4必须等于origins_per_event")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.origins_per_event = int(origins_per_event)
        self.phase_quota = int(phase_quota)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        by_event: dict[str, list[int]] = defaultdict(list)
        phase_by_event: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {phase: [] for phase in EVENT_PHASES}
        )
        graph_by_event: dict[str, str] = {}
        for index, row in dataset.samples.iterrows():
            event = str(row["EVENT_ID"])
            graph = str(row["GRAPH_ID"])
            phase = str(row["EVENT_PHASE"]).upper()
            if event in graph_by_event and graph_by_event[event] != graph:
                raise ValueError(f"EVENT_ID跨graph重复: {event}")
            graph_by_event[event] = graph
            by_event[event].append(int(index))
            phase_by_event[event][phase].append(int(index))
        if not by_event:
            raise ValueError("v11 TRAIN没有event-balanced候选")

        self.by_event = dict(by_event)
        self.phase_by_event = dict(phase_by_event)
        self.graph_by_event = graph_by_event
        self.selected_count_by_event = {
            event: min(self.origins_per_event, len(indices))
            for event, indices in self.by_event.items()
        }
        selected_by_graph: dict[str, int] = defaultdict(int)
        for event, count in self.selected_count_by_event.items():
            selected_by_graph[self.graph_by_event[event]] += count
        self.selected_count_by_graph = dict(selected_by_graph)
        self.selected_sample_count = int(sum(self.selected_count_by_event.values()))
        self.event_count = len(self.by_event)
        self._length = 0
        for count in self.selected_count_by_graph.values():
            self._length += (
                count // self.batch_size
                if self.drop_last
                else math.ceil(count / self.batch_size)
            )
        if self._length <= 0:
            raise ValueError("v11 event-balanced sampler没有batch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _select_event(self, event: str, rng: random.Random) -> list[int]:
        selected: list[int] = []
        selected_set: set[int] = set()
        for phase in EVENT_PHASES:
            candidates = list(self.phase_by_event[event][phase])
            rng.shuffle(candidates)
            for index in candidates[: self.phase_quota]:
                selected.append(index)
                selected_set.add(index)
        target = self.selected_count_by_event[event]
        if len(selected) < target:
            remaining = [
                index for index in self.by_event[event] if index not in selected_set
            ]
            rng.shuffle(remaining)
            selected.extend(remaining[: target - len(selected)])
        if len(selected) > target:
            rng.shuffle(selected)
            selected = selected[:target]
        if len(selected) != target or len(set(selected)) != len(selected):
            raise RuntimeError(f"{event}: v11 event sampling计数/去重失败")
        return selected

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        selected_by_graph: dict[str, list[int]] = defaultdict(list)
        events = sorted(self.by_event)
        rng.shuffle(events)
        for event in events:
            selected_by_graph[self.graph_by_event[event]].extend(
                self._select_event(event, rng)
            )
        graph_ids = sorted(selected_by_graph)
        rng.shuffle(graph_ids)
        for graph in graph_ids:
            indices = selected_by_graph[graph]
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        return self._length

    def audit(self) -> dict[str, Any]:
        phase_candidate_counts = {
            phase: int(
                sum(len(self.phase_by_event[event][phase]) for event in self.by_event)
            )
            for phase in EVENT_PHASES
        }
        return {
            "mode": "EVENT_BALANCED_PHASE_STRATIFIED",
            "event_count": self.event_count,
            "origins_per_event_max": self.origins_per_event,
            "phase_quota": self.phase_quota,
            "phases": list(EVENT_PHASES),
            "selected_samples_per_epoch": self.selected_sample_count,
            "candidate_samples": len(self.dataset),
            "candidate_phase_counts": phase_candidate_counts,
            "events_with_fewer_than_target_origins": int(
                sum(
                    len(indices) < self.origins_per_event
                    for indices in self.by_event.values()
                )
            ),
            "selected_count_by_graph": self.selected_count_by_graph,
        }


def build_hydrologic_graph_v11_loader(
    dataset: HydrologicGraphV11Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    event_balanced_train: bool = False,
    origins_per_event: int = 8,
    phase_quota: int = 2,
) -> DataLoader:
    if num_workers != 0:
        raise ValueError("当前v11 NPZ cache固定num_workers=0")
    if event_balanced_train:
        if not shuffle:
            raise ValueError("v11 event-balanced TRAIN要求shuffle=true")
        batch_sampler: Sampler[list[int]] = EventBalancedV11BatchSampler(
            dataset,
            batch_size,
            origins_per_event=origins_per_event,
            phase_quota=phase_quota,
            seed=seed,
        )
    else:
        # Reuse the proven same-graph grouping logic but swap in the V11 collate.
        from datasets.hydrologic_graph_v8 import GraphGroupedV8BatchSampler

        batch_sampler = GraphGroupedV8BatchSampler(
            dataset,
            batch_size,
            shuffle=shuffle,
            drop_last=False,
            seed=seed,
        )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_hydrologic_graph_v11,
    )
