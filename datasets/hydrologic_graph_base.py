"""hydrologic NPZ-backed loader for hydrologic computational graphs and sparse Q/Z observations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.hydrologic_schema import HydrologicGraphBatch, validate_hydrologic_batch


CONTRACT_NAME = "hydrologic-computational-graph-sparse-observation-v1"


def _norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


class HydrologicGraphBaseDataset(Dataset[HydrologicGraphBatch]):
    """Materialise only frozen hydrologic sample rows, keeping Nnode and Nobs separate."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        graph_id: str | None = None,
        future_rainfall_mode: str = "persistence",
        strict: bool = True,
        tensor_cache: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
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

        contract_path = self.root / "metadata/dataset_contract.json"
        sample_path = self.root / "samples/sample_index.csv"
        mapping_path = self.root / "graph/station_observation_mapping.csv"
        for path in (contract_path, sample_path, mapping_path):
            if not path.is_file():
                raise FileNotFoundError(f"hydrologic data缺少: {path}")
        self.contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        if self.contract.get("contract") != CONTRACT_NAME:
            raise ValueError(
                f"不是hydrologic hydrologic graph契约: {self.contract.get('contract')!r}"
            )
        self.history_hours = int(self.contract["history_hours"])
        self.forecast_hours = int(self.contract["forecast_hours"])
        if (self.history_hours, self.forecast_hours) != (24, 6):
            raise ValueError("fixed design requireshistory=24、forecast=6")

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
        }
        missing = required - set(samples.columns)
        if missing:
            raise ValueError(f"hydrologic sample_index缺少字段: {sorted(missing)}")
        samples["GRAPH_ID"] = samples["GRAPH_ID"].map(_norm_id)
        samples["SPLIT"] = samples["SPLIT"].str.upper()
        samples = samples[samples["SPLIT"].eq(self.split)].copy()
        if graph_id is not None:
            graph_id = _norm_id(graph_id)
            samples = samples[samples["GRAPH_ID"].eq(graph_id)].copy()
        if samples.empty:
            raise ValueError(f"{self.split}没有usable hydrologic samples")
        samples["TENSOR_ROW"] = pd.to_numeric(
            samples["TENSOR_ROW"], errors="raise"
        ).astype(np.int64)
        samples["N_NODE"] = pd.to_numeric(samples["N_NODE"], errors="raise").astype(
            np.int64
        )
        samples["N_OBS"] = pd.to_numeric(samples["N_OBS"], errors="raise").astype(
            np.int64
        )
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
        self.train_sampling_mode = "hydrologic_full_pass_same_graph_batches"
        self._tensor_cache = tensor_cache if tensor_cache is not None else {}

    def __len__(self) -> int:
        return len(self.samples)

    def graph_id_for_index(self, index: int) -> str:
        return str(self.samples.iloc[int(index)]["GRAPH_ID"])

    def _load_tensor_file(self, relative_name: str) -> dict[str, np.ndarray]:
        relative_name = str(relative_name).replace("\\", "/")
        cached = self._tensor_cache.get(relative_name)
        if cached is not None:
            return cached
        path = (self.root / relative_name).resolve()
        if self.root not in path.parents:
            raise ValueError(f"TENSOR_FILE越出dataset root: {relative_name}")
        if not path.is_file():
            raise FileNotFoundError(f"hydrologic tensor不存在: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required = {
            "node_static",
            "incremental_area_km2",
            "edge_index",
            "edge_static",
            "obs_station_id",
            "obs_node_index",
            "sample_id",
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
        expected = {
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
        if set(arrays) != expected:
            missing = required - set(arrays)
            raise ValueError(
                f"{path.name}: NPZ key与hydrologic契约不一致，missing={sorted(missing)}"
            )
        self._tensor_cache[relative_name] = arrays
        return arrays

    @staticmethod
    def _masked_finite(
        values: np.ndarray, mask: np.ndarray, label: str
    ) -> torch.Tensor:
        values = np.asarray(values, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError(f"{label}: value/mask shape不一致")
        if not np.isfinite(values[mask]).all():
            raise ValueError(f"{label}: 有效mask内含NaN/Inf")
        safe = np.where(mask, values, 0.0)
        return torch.from_numpy(safe.astype(np.float32, copy=False))

    def __getitem__(self, index: int) -> HydrologicGraphBatch:
        row = self.samples.iloc[int(index)]
        arrays = self._load_tensor_file(row["TENSOR_FILE"])
        tensor_row = int(row["TENSOR_ROW"])
        sample_ids = arrays["sample_id"]
        if tensor_row < 0 or tensor_row >= len(sample_ids):
            raise IndexError(f"TENSOR_ROW越界: {tensor_row}")
        if str(sample_ids[tensor_row]) != str(row["SAMPLE_ID"]):
            raise ValueError(
                f"{row['SAMPLE_ID']}: sample_index与NPZ tensor row不一致"
            )

        node_static = torch.from_numpy(
            np.asarray(arrays["node_static"], dtype=np.float32)
        )
        area = torch.from_numpy(
            np.asarray(arrays["incremental_area_km2"], dtype=np.float32)
        )
        edge_index = torch.from_numpy(
            np.asarray(arrays["edge_index"], dtype=np.int64)
        ).long()
        edge_static = torch.from_numpy(
            np.asarray(arrays["edge_static"], dtype=np.float32)
        )
        obs_nodes = torch.from_numpy(
            np.asarray(arrays["obs_node_index"], dtype=np.int64)
        ).long()
        obs_station_ids = tuple(_norm_id(value) for value in arrays["obs_station_id"])
        obs_station_index = torch.tensor(
            [self.station_to_index[station] for station in obs_station_ids],
            dtype=torch.long,
        )

        history_rain_np = np.asarray(
            arrays["history_rain"][tensor_row], dtype=np.float32
        )
        stored_future_np = np.asarray(
            arrays["future_rain"][tensor_row], dtype=np.float32
        )
        if not np.isfinite(history_rain_np).all() or (history_rain_np < 0).any():
            raise ValueError(f"{row['SAMPLE_ID']}: history_rain非法")
        if not np.isfinite(stored_future_np).all() or (stored_future_np < 0).any():
            raise ValueError(f"{row['SAMPLE_ID']}: future_rain非法")
        if self.future_rainfall_mode == "observed_hindcast":
            future_rain_np = stored_future_np
        elif self.future_rainfall_mode == "zero":
            future_rain_np = np.zeros_like(stored_future_np)
        else:
            future_rain_np = np.repeat(
                history_rain_np[:, -1:], self.forecast_hours, axis=1
            )
        history_rain = torch.from_numpy(history_rain_np.T.copy()).unsqueeze(-1)
        future_rain = torch.from_numpy(future_rain_np.T.copy()).unsqueeze(-1)

        q_history_mask_np = np.asarray(
            arrays["q_history_mask"][tensor_row], dtype=bool
        )
        z_history_mask_np = np.asarray(
            arrays["z_history_mask"][tensor_row], dtype=bool
        )
        q_target_mask_np = np.asarray(
            arrays["q_target_mask"][tensor_row], dtype=bool
        )
        z_target_mask_np = np.asarray(
            arrays["z_target_mask"][tensor_row], dtype=bool
        )
        q_history = self._masked_finite(
            arrays["q_history"][tensor_row], q_history_mask_np, "q_history"
        ).transpose(0, 1)
        z_history = self._masked_finite(
            arrays["z_history"][tensor_row], z_history_mask_np, "z_history"
        ).transpose(0, 1)
        q_target = self._masked_finite(
            arrays["q_target"][tensor_row], q_target_mask_np, "q_target"
        ).transpose(0, 1)
        z_target = self._masked_finite(
            arrays["z_target"][tensor_row], z_target_mask_np, "z_target"
        ).transpose(0, 1)
        q_mask = torch.from_numpy(q_history_mask_np.T.copy())
        z_mask = torch.from_numpy(z_history_mask_np.T.copy())
        q_target_mask = torch.from_numpy(q_target_mask_np.T.copy())
        z_target_mask = torch.from_numpy(z_target_mask_np.T.copy())

        n_node = int(row["N_NODE"])
        n_obs = int(row["N_OBS"])
        if node_static.shape[0] != n_node or len(obs_station_ids) != n_obs:
            raise ValueError(f"{row['SAMPLE_ID']}: N_NODE/N_OBS与NPZ不一致")

        return HydrologicGraphBatch(
            history_rain=history_rain,
            future_rain=future_rain,
            node_static=node_static,
            incremental_area_km2=area,
            edge_index=edge_index,
            edge_static=edge_static,
            obs_node_index=obs_nodes,
            obs_station_index=obs_station_index,
            q_history=q_history,
            z_history=z_history,
            q_mask=q_mask,
            z_mask=z_mask,
            q_target=q_target,
            z_target=z_target,
            q_target_mask=q_target_mask,
            z_target_mask=z_target_mask,
            obs_station_ids=obs_station_ids,
            sample_id=str(row["SAMPLE_ID"]),
            event_id=str(row["EVENT_ID"]),
            graph_id=str(row["GRAPH_ID"]),
            forecast_time=str(row["FORECAST_TIME"]),
            sample_weight=torch.tensor(1.0, dtype=torch.float32),
        )


def collate_hydrologic_graph_base(
    items: list[HydrologicGraphBatch],
) -> HydrologicGraphBatch:
    if not items:
        raise ValueError("不能collate空hydrologic样本")
    graph_ids = [item.graph_id for item in items]
    if any(not isinstance(value, str) for value in graph_ids) or len(set(graph_ids)) != 1:
        raise ValueError(f"一个hydrologic batch只能来自同一GRAPH_ID，实际={graph_ids}")

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
            raise ValueError(f"同图hydrologic batch内{name}不一致")
    if any(item.obs_station_ids != items[0].obs_station_ids for item in items[1:]):
        raise ValueError("同图hydrologic batch内obs_station_ids不一致")

    stack_fields = (
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
    )
    kwargs: dict[str, Any] = {
        name: getattr(items[0], name) for name in static_fields
    }
    kwargs["obs_station_ids"] = items[0].obs_station_ids
    for name in stack_fields:
        values = [getattr(item, name) for item in items]
        if any(value is None for value in values):
            raise ValueError(f"hydrologic batch内{name}不能为None")
        kwargs[name] = torch.stack(values)
    for name in ("sample_id", "event_id", "graph_id", "forecast_time"):
        kwargs[name] = tuple(getattr(item, name) for item in items)
    batch = HydrologicGraphBatch(**kwargs)
    validate_hydrologic_batch(batch)
    return batch


class GraphGroupedBatchSampler(Sampler[list[int]]):
    """Deterministic same-graph batching for variable Nnode/Nobs graphs."""

    def __init__(
        self,
        dataset: HydrologicGraphBaseDataset,
        batch_size: int,
        shuffle: bool,
        *,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size必须>0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(self.seed)
        groups: dict[str, list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            groups[dataset.graph_id_for_index(index)].append(index)
        self.groups = dict(groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for indices in self.groups.values():
            ordered = list(indices)
            if self.shuffle:
                order = torch.randperm(
                    len(ordered), generator=self.generator
                ).tolist()
                ordered = [ordered[position] for position in order]
            for start in range(0, len(ordered), self.batch_size):
                batch = ordered[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[position] for position in order]
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.groups.values())
        return sum(
            (len(indices) + self.batch_size - 1) // self.batch_size
            for indices in self.groups.values()
        )


def build_hydrologic_graph_base_loader(
    dataset: HydrologicGraphBaseDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> DataLoader[HydrologicGraphBatch]:
    sampler = GraphGroupedBatchSampler(
        dataset,
        batch_size,
        shuffle,
        drop_last=False,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_hydrologic_graph_base,
        generator=sampler.generator,
    )
