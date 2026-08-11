"""Shared, split-safe construction for debug and formal Hunan runs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
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
    return _runtime_config_from_mapping(
        load_config(config_path), dataset_root=dataset_root, graph_id=graph_id
    )


def _runtime_config_from_mapping(
    source: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> dict[str, Any]:
    """Resolve one already-validated config without writing a temporary YAML."""

    cfg = deepcopy(source)
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


def _runtime_metadata(
    loader: Any,
    cfg: dict[str, Any],
    *,
    q_scale_dataset: Any | None = None,
) -> dict[str, Any]:
    dataset = loader.dataset
    mode = cfg["data"]["mode"]
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
    loss_scales: dict[str, Any] = {
        "discharge": float(dataset.normalization["FLOW"].std),
        "water_level": float(dataset.normalization["WATER_LEVEL"].std),
    }
    runtime: dict[str, Any] = {
        # Huber loss is calculated on errors divided by TRAIN-only standard
        # deviation.  Reported MAE remains in physical m3/s and metres.
        "loss_scales": loss_scales,
        "data_contract": {
            "format_version": 1,
            "mode": "hunan",
            "station_ids": list(dataset.station_ids),
            "graph_ids": sorted(dataset.graph_ids),
            "target_variables_by_graph": target_mapping,
            "artifact_sha256": _contract_digest(root),
        },
    }
    if cfg["loss"]["q_scale_mode"] == "per_graph":
        source = dataset if q_scale_dataset is None else q_scale_dataset
        raw_statistics = source.train_q_supervision_statistics()
        expected_flow_graphs = {
            graph_id
            for graph_id in source.graph_ids
            if "FLOW" in source.target_variables_by_graph[graph_id]
        }
        if set(raw_statistics) != expected_flow_graphs:
            raise ValueError(
                "TRAIN逐图Q统计未完整覆盖FLOW监督graph: "
                f"expected={sorted(expected_flow_graphs)}, "
                f"actual={sorted(raw_statistics)}"
            )
        floor = float(cfg["loss"]["q_scale_floor_m3s"])
        graph_audit: dict[str, dict[str, Any]] = {}
        graph_scales: dict[str, float] = {}
        for graph_id in sorted(source.graph_ids):
            if graph_id not in raw_statistics:
                graph_audit[graph_id] = {
                    "status": "NOT_APPLICABLE_NO_FLOW_SUPERVISION",
                    "valid_unique_point_count": 0,
                    "mean_m3s": None,
                    "std_m3s": None,
                    "q_loss_scale_m3s": None,
                    "floor_applied": False,
                }
                continue
            statistics = raw_statistics[graph_id]
            raw_std = float(statistics["std_m3s"])
            used_scale = max(raw_std, floor)
            floor_applied = raw_std < floor
            graph_scales[graph_id] = used_scale
            graph_audit[graph_id] = {
                "status": "APPLIED",
                **statistics,
                "q_loss_scale_m3s": used_scale,
                "floor_applied": floor_applied,
            }
        loss_scales["discharge_by_graph"] = graph_scales
        runtime["q_scale_audit"] = {
            "computed_from_split": "TRAIN",
            "source": "unique TRAIN outlet-Q supervision timestamps",
            "std_definition": "population std (ddof=0)",
            "q_scale_floor_m3s": floor,
            "graphs": graph_audit,
        }
    return runtime


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
    cfg["_runtime"] = _runtime_metadata(train_loader, cfg)
    model = HybridFloodModel(cfg, nodes)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_training_from_config(
    source: dict[str, Any],
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> tuple[dict[str, Any], HybridFloodModel, Any, Any, torch.device]:
    """Build TRAIN/VALIDATION only from an in-memory HPO trial config."""

    cfg = _runtime_config_from_mapping(
        validate_config(deepcopy(source)),
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
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
    cfg["_runtime"] = _runtime_metadata(train_loader, cfg)
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
    q_scale_dataset = None
    dynamic_cache: dict | None = None
    if (
        cfg["data"]["mode"] == "hunan"
        and cfg["loss"]["q_scale_mode"] == "per_graph"
    ):
        # Standalone VALIDATION/TEST loss must reuse TRAIN scales.  Constructing
        # this deterministic TRAIN view never fits on the evaluated split.
        dynamic_cache = {}
        train_scale_loader = _make_loader(
            cfg,
            cfg["data"]["train_split"],
            shuffle=False,
            dynamic_cache=dynamic_cache,
        )
        q_scale_dataset = train_scale_loader.dataset
    loader = _make_loader(
        cfg, split, shuffle=False, dynamic_cache=dynamic_cache
    )
    cfg["_runtime"] = _runtime_metadata(
        loader, cfg, q_scale_dataset=q_scale_dataset
    )
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
    saved_loss = saved.get("loss") if isinstance(saved.get("loss"), dict) else {}
    current_loss = cfg.get("loss", {})
    saved_q_scale_mode = saved_loss.get("q_scale_mode", "global")
    current_q_scale_mode = current_loss.get("q_scale_mode", "global")
    if saved_q_scale_mode != current_q_scale_mode:
        mismatches.append("loss.q_scale_mode")
    if current_q_scale_mode == "per_graph":
        saved_floor = saved_loss.get("q_scale_floor_m3s")
        current_floor = current_loss.get("q_scale_floor_m3s")
        if saved_floor != current_floor:
            mismatches.append("loss.q_scale_floor_m3s")
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
        if current_q_scale_mode == "per_graph":
            saved_q_scales = (
                saved.get("_runtime", {})
                .get("loss_scales", {})
                .get("discharge_by_graph")
            )
            current_q_scales = (
                cfg.get("_runtime", {})
                .get("loss_scales", {})
                .get("discharge_by_graph")
            )
            if not isinstance(saved_q_scales, dict) or not isinstance(
                current_q_scales, dict
            ):
                mismatches.append("loss_scales.discharge_by_graph")
            else:
                saved_scale_graphs = set(saved_q_scales)
                current_scale_graphs = set(current_q_scales)
                if (resume and saved_scale_graphs != current_scale_graphs) or (
                    not resume
                    and not current_scale_graphs.issubset(saved_scale_graphs)
                ):
                    mismatches.append("loss_scales.discharge_by_graph")
                for graph_id in current_scale_graphs & saved_scale_graphs:
                    saved_value = float(saved_q_scales[graph_id])
                    current_value = float(current_q_scales[graph_id])
                    if not (
                        math.isfinite(saved_value)
                        and math.isfinite(current_value)
                        and math.isclose(
                            saved_value,
                            current_value,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ):
                        mismatches.append(
                            f"loss_scales.discharge_by_graph.{graph_id}"
                        )
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
