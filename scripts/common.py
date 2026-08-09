"""Shared, split-safe construction for debug and formal Hunan runs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from config import load_config, validate_config
from data.device import resolve_device, seed_everything
from datasets import SyntheticEventDataset, collate_graph_events
from models import HybridFloodModel


_SPLIT_SEED_OFFSET = {"TRAIN": 0, "VALIDATION": 10_000, "TEST": 20_000}
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HUNAN_REQUIRED_FILES = (
    "graph/node_catalog.csv",
    "graph/edge_topology.csv",
    "graph/node_static_attributes.csv",
    "graph/edge_static_attributes.csv",
    "events/flood_events_all.csv",
    "events/flood_events_final.csv",
    "events/sample_index.csv",
    "events/data_split.csv",
    "events/target_variable_by_graph.csv",
    "metadata/feature_schema.json",
    "metadata/normalization_stats.json",
    "metadata/dataset_summary.csv",
    "metadata/source_manifest.json",
    "metadata/build_log.txt",
    "qc/dynamic_coverage.csv",
    "qc/event_exclusion.csv",
    "qc/hydro_file_selection.csv",
    "qc/hydro_load_audit.csv",
    "qc/rain_source_coverage.csv",
    "qc/sample_rejection.csv",
)
_CONTRACT_HASH_FILES = tuple(
    name
    for name in _HUNAN_REQUIRED_FILES
    if name
    not in {
        "metadata/build_log.txt",
        "qc/dynamic_coverage.csv",
        "qc/hydro_file_selection.csv",
        "qc/hydro_load_audit.csv",
        "qc/rain_source_coverage.csv",
    }
)


def _runtime_config(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(load_config(config_path))
    if dataset_root is not None:
        root = Path(dataset_root).expanduser()
        cfg["data"]["dataset_root"] = str(root.resolve())
    elif cfg["data"]["dataset_root"] is not None:
        root = Path(cfg["data"]["dataset_root"]).expanduser()
        cfg["data"]["dataset_root"] = str(
            root.resolve() if root.is_absolute() else (_PROJECT_ROOT / root).resolve()
        )
    if graph_id is not None:
        cfg["data"]["graph_id"] = graph_id
    for key in ("checkpoint", "log_csv"):
        path = Path(cfg["training"][key]).expanduser()
        cfg["training"][key] = str(
            path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()
        )
    return validate_config(cfg)


def _normalise_split(split: str) -> str:
    name = split.upper()
    aliases = {"VAL": "VALIDATION", "VALID": "VALIDATION"}
    name = aliases.get(name, name)
    if name not in _SPLIT_SEED_OFFSET:
        raise ValueError("split 必须是 TRAIN、VALIDATION 或 TEST")
    return name


def _synthetic_loader(cfg: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
    seed = int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split]
    dataset = SyntheticEventDataset(
        cfg["debug_num_events"],
        cfg["history_length"],
        cfg["forecast_horizon"],
        cfg["dynamic_dim"],
        cfg["node_static_dim"],
        cfg["edge_static_dim"],
        seed,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        collate_fn=collate_graph_events,
        generator=generator,
    )


def _hunan_loader(
    cfg: dict[str, Any],
    split: str,
    shuffle: bool,
    dynamic_cache: dict | None = None,
) -> Any:
    try:
        from datasets import HunanGraphEventDataset, build_hunan_loader
    except ImportError as exc:
        raise RuntimeError(
            "湖南正式数据适配器不可用；请确认 datasets.hunan 已安装在当前 project 中"
        ) from exc

    data_cfg = cfg["data"]
    root = Path(data_cfg["dataset_root"]).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"湖南正式 _model_dataset 根目录不存在: {root}")
    missing = [name for name in _HUNAN_REQUIRED_FILES if not (root / name).is_file()]
    dynamic_dir = root / "dynamic"
    if not dynamic_dir.is_dir() or not any(dynamic_dir.glob("graph_*_hourly.csv")):
        missing.append("dynamic/graph_<BASIN_ID>_hourly.csv")
    if missing:
        raise FileNotFoundError(
            "湖南正式 _model_dataset 结构不完整，缺少: " + ", ".join(missing)
        )
    target = data_cfg["target_variable"]
    dataset = HunanGraphEventDataset(
        root,
        split,
        history_hours=cfg["history_length"],
        forecast_hours=cfg["forecast_horizon"],
        graph_id=data_cfg["graph_id"],
        target_variables=target,
        normalize_dynamic=data_cfg["normalize_dynamic"],
        future_rainfall_mode=data_cfg["future_rainfall_mode"],
        use_observation_masks=data_cfg["use_observation_masks"],
        strict=data_cfg["strict_validation"],
        dynamic_cache=dynamic_cache,
    )
    if len(dataset) == 0:
        raise ValueError(f"{split} 划分没有可用样本")
    dimensions = {
        "dynamic_dim": dataset.dynamic_dim,
        "node_static_dim": dataset.node_static_dim,
        "edge_static_dim": dataset.edge_static_dim,
        "history_length": dataset.history_hours,
        "forecast_horizon": dataset.forecast_hours,
    }
    mismatches = [
        f"{key}: config={cfg[key]}, data={actual}"
        for key, actual in dimensions.items()
        if cfg[key] != actual
    ]
    if mismatches:
        raise ValueError("配置与 feature_schema/正式表维度不一致: " + "; ".join(mismatches))
    return build_hunan_loader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=False,
        seed=cfg["seed"] + _SPLIT_SEED_OFFSET[split],
    )


def _make_loader(
    cfg: dict[str, Any],
    split: str,
    shuffle: bool,
    dynamic_cache: dict | None = None,
) -> Any:
    if split != "TRAIN" and shuffle:
        raise ValueError(f"{split} loader 禁止 shuffle")
    if cfg["data"]["mode"] == "synthetic":
        return _synthetic_loader(cfg, split, shuffle)
    return _hunan_loader(cfg, split, shuffle, dynamic_cache)


def _dataset_nodes(loader: Any) -> int:
    dataset = loader.dataset
    # Formal multi-graph datasets expose global station indices.  The model's
    # observation head is sized once to this global catalogue while each batch
    # still contains only one unpadded graph.
    stations = getattr(dataset, "num_stations", None)
    if stations is not None:
        return int(stations)
    try:
        return int(dataset.num_nodes)
    except (AttributeError, ValueError):
        pass
    counts = getattr(dataset, "graph_node_counts", None)
    if isinstance(counts, dict):
        unique = set(counts.values())
        if len(unique) == 1:
            return int(next(iter(unique)))
        graphs = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        raise ValueError(f"数据集缺少全局 num_stations，无法对齐多河网站点参数: {graphs}")
    sample = dataset[0]
    return int(sample.node_static.shape[0])


def _ensure_matching_graph(train_loader: Any, validation_loader: Any) -> None:
    train_ids = set(getattr(train_loader.dataset, "graph_ids", ()))
    validation_ids = set(getattr(validation_loader.dataset, "graph_ids", ()))
    unseen = validation_ids - train_ids
    if train_ids and unseen:
        raise ValueError(
            "VALIDATION 包含 TRAIN 从未出现的 GRAPH_ID，无法训练其站点观测参数: "
            f"{sorted(unseen)}"
        )
    train_stations = tuple(getattr(train_loader.dataset, "station_ids", ()))
    validation_stations = tuple(getattr(validation_loader.dataset, "station_ids", ()))
    if train_stations and train_stations != validation_stations:
        raise ValueError("TRAIN 与 VALIDATION 的全局 STATION_ID 映射不一致")
    train_events = set(getattr(train_loader.dataset, "event_ids", ()))
    validation_events = set(getattr(validation_loader.dataset, "event_ids", ()))
    overlap = train_events & validation_events
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(
            "TRAIN 与 VALIDATION 存在事件级泄漏，重复 EVENT_ID: "
            f"{preview}{' ...' if len(overlap) > len(preview) else ''}"
        )


def _contract_digest(root: Path) -> str:
    """Hash the small authoritative artifacts that define tensor semantics.

    Hourly CSVs can be province-scale, so their provenance must be represented
    by ``source_manifest.json`` rather than being re-read solely for hashing.
    """
    digest = hashlib.sha256()
    for relative_name in _CONTRACT_HASH_FILES:
        path = root / relative_name
        digest.update(relative_name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _runtime_metadata(loader: Any, mode: str) -> dict[str, Any]:
    dataset = loader.dataset
    if mode == "synthetic":
        return {
            "loss_scales": {"discharge": 1.0, "water_level": 1.0},
            "data_contract": {"format_version": 1, "mode": "synthetic"},
        }
    root = Path(dataset.root)
    target_mapping = {
        graph_id: sorted(values)
        for graph_id, values in sorted(dataset.target_variables_by_graph.items())
    }
    return {
        # Huber loss is calculated on errors divided by TRAIN-only standard
        # deviation.  Reported MAE remains in physical m3/s and metres.
        "loss_scales": {
            "discharge": float(dataset.normalization["FLOW"].std),
            "water_level": float(dataset.normalization["WATER_LEVEL"].std),
        },
        "data_contract": {
            "format_version": 1,
            "mode": "hunan",
            "station_ids": list(dataset.station_ids),
            "graph_ids": sorted(dataset.graph_ids),
            "target_variables_by_graph": target_mapping,
            "artifact_sha256": _contract_digest(root),
        },
    }


def setup_training(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> tuple[dict[str, Any], HybridFloodModel, Any, Any, torch.device]:
    """Build independent TRAIN and VALIDATION loaders plus one model."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    seed_everything(cfg["seed"])
    dynamic_cache: dict = {}
    train_loader = _make_loader(
        cfg, cfg["data"]["train_split"], shuffle=True, dynamic_cache=dynamic_cache
    )
    validation_loader = _make_loader(
        cfg,
        cfg["data"]["validation_split"],
        shuffle=False,
        dynamic_cache=dynamic_cache,
    )
    _ensure_matching_graph(train_loader, validation_loader)
    nodes = _dataset_nodes(train_loader)
    if _dataset_nodes(validation_loader) != nodes:
        raise ValueError("TRAIN 与 VALIDATION 的节点数不一致")
    cfg["_runtime"] = _runtime_metadata(train_loader, cfg["data"]["mode"])
    model = HybridFloodModel(cfg, nodes)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> tuple[dict[str, Any], HybridFloodModel, Any, torch.device]:
    """Build a deterministic, never-shuffled evaluation loader."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    seed_everything(cfg["seed"])
    split = _normalise_split(split)
    loader = _make_loader(cfg, split, shuffle=False)
    cfg["_runtime"] = _runtime_metadata(loader, cfg["data"]["mode"])
    model = HybridFloodModel(cfg, _dataset_nodes(loader))
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device


def validate_checkpoint_config(
    checkpoint: dict[str, Any], cfg: dict[str, Any], *, resume: bool = False
) -> None:
    """Reject checkpoints trained under an incompatible tensor/model contract."""
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        if cfg["data"]["mode"] == "hunan":
            raise ValueError(
                "正式湖南权重缺少训练配置和数据契约，无法安全确认站点映射"
            )
        return
    keys = (
        "runoff_mode",
        "routing_mode",
        "history_length",
        "forecast_horizon",
        "dynamic_dim",
        "node_static_dim",
        "edge_static_dim",
        "hidden_dim",
        "solver",
        "physical_bounds",
    )
    mismatches = [key for key in keys if saved.get(key) != cfg.get(key)]
    saved_data = saved.get("data")
    if isinstance(saved_data, dict):
        for key in (
            "mode",
            "target_variable",
            "use_observation_masks",
            "future_rainfall_mode",
        ):
            if saved_data.get(key) != cfg["data"].get(key):
                mismatches.append(f"data.{key}")
        saved_graph = saved_data.get("graph_id")
        current_graph = cfg["data"].get("graph_id")
        if resume:
            if saved_graph != current_graph:
                mismatches.append("data.graph_id")
        elif saved_graph is not None and saved_graph != current_graph:
            # A model trained on all graphs may be evaluated on a selected graph,
            # but a graph-specific model must never be presented as province-wide.
            mismatches.append("data.graph_id")
        if resume and saved_data.get("dataset_root") != cfg["data"].get("dataset_root"):
            mismatches.append("data.dataset_root")
    if cfg["data"]["mode"] == "hunan":
        saved_contract = saved.get("_runtime", {}).get("data_contract")
        current_contract = cfg.get("_runtime", {}).get("data_contract")
        if not isinstance(saved_contract, dict):
            raise ValueError(
                "正式湖南checkpoint缺少数据契约指纹；无法确认站点参数映射，"
                "请使用当前版本重新训练"
            )
        if not isinstance(current_contract, dict):
            raise ValueError("当前正式数据尚未生成运行时数据契约")
        for key in ("station_ids", "target_variables_by_graph", "artifact_sha256"):
            if saved_contract.get(key) != current_contract.get(key):
                mismatches.append(f"data_contract.{key}")
        saved_graphs = set(saved_contract.get("graph_ids", ()))
        current_graphs = set(current_contract.get("graph_ids", ()))
        if (resume and saved_graphs != current_graphs) or (
            not resume and not current_graphs.issubset(saved_graphs)
        ):
            mismatches.append("data_contract.graph_ids")
    if mismatches:
        raise ValueError(
            "checkpoint 与当前配置不兼容: " + ", ".join(sorted(set(mismatches)))
        )


def setup(config_path: str | Path):
    """Compatibility helper for legacy synthetic scripts; returns the TRAIN loader."""
    if load_config(config_path)["data"]["mode"] != "synthetic":
        raise RuntimeError(
            "legacy setup() 仅用于 synthetic 调试；正式数据请使用 "
            "setup_training() 或 setup_evaluation()"
        )
    cfg, model, train_loader, _validation_loader, device = setup_training(config_path)
    return cfg, model, train_loader, device
