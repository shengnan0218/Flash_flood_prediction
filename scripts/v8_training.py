"""Training/evaluation setup for the v8 hydrologic-graph sparse-observation contract."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from data.device import resolve_device, seed_everything
from datasets.hydrologic_graph_v8 import (
    CONTRACT_NAME,
    HydrologicGraphV8Dataset,
    build_hydrologic_graph_v8_loader,
)
from models.hydrologic_graph_v8 import HydrologicGraphV8Model


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPLIT_SEED_OFFSET = {"TRAIN": 0, "VALIDATION": 10_000, "TEST": 20_000}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path: str | Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path in stack:
        raise ValueError(
            "配置继承存在循环: "
            + " -> ".join(str(value) for value in (*stack, path))
        )
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("配置顶层必须是mapping")
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError("_base_必须是非空文件名")
    return _merge(_load_yaml(path.parent / base_name, (*stack, path)), raw)


def _resolve_root(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else _PROJECT_ROOT / path).resolve()


def _dataset_contract_name(root: Path) -> str | None:
    path = root / "metadata/dataset_contract.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw.get("contract")


def is_v8_requested(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
) -> bool:
    if dataset_root is not None:
        root = _resolve_root(dataset_root)
        if _dataset_contract_name(root) == CONTRACT_NAME:
            return True
        return "model_dataset_v8_hydrologic_graph" in str(root).lower()
    raw = _load_yaml(config_path)
    value = raw.get("data", {}).get("dataset_root")
    if not isinstance(value, str) or not value.strip():
        return False
    root = _resolve_root(value)
    return (
        _dataset_contract_name(root) == CONTRACT_NAME
        or "model_dataset_v8_hydrologic_graph" in value.lower()
    )


def _runtime_config(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> dict[str, Any]:
    cfg = _load_yaml(config_path)
    if dataset_root is not None:
        cfg["data"]["dataset_root"] = str(_resolve_root(dataset_root))
    else:
        cfg["data"]["dataset_root"] = str(_resolve_root(cfg["data"]["dataset_root"]))
    if graph_id is not None:
        cfg["data"]["graph_id"] = str(graph_id).strip()
    for key in ("checkpoint", "log_csv", "final_checkpoint"):
        value = cfg["training"].get(key)
        if value is None:
            continue
        cfg["training"][key] = str(_resolve_root(value))
    _validate_v8_config(cfg)
    return cfg


def _validate_v8_config(cfg: dict[str, Any]) -> None:
    data = cfg.get("data", {})
    if data.get("mode") != "hunan":
        raise ValueError("v8正式配置要求data.mode=hunan")
    if data.get("dataset_type") != "continuous":
        raise ValueError("v8正式配置要求dataset_type=continuous")
    if data.get("target_variable") != "BOTH":
        raise ValueError("v8同时监督Q/Delta-Z，target_variable必须为BOTH")
    if not bool(data.get("strict_validation", False)):
        raise ValueError("v8要求strict_validation=true")
    if not bool(data.get("use_observation_masks", False)):
        raise ValueError("v8要求use_observation_masks=true")
    if data.get("future_rainfall_mode") not in {
        "observed_hindcast",
        "zero",
        "persistence",
    }:
        raise ValueError("v8 future_rainfall_mode非法")
    if (int(cfg.get("history_length", -1)), int(cfg.get("forecast_horizon", -1))) != (
        24,
        6,
    ):
        raise ValueError("v8固定history=24、forecast=6")
    if int(cfg.get("node_static_dim", -1)) != 10 or int(
        cfg.get("edge_static_dim", -1)
    ) != 2:
        raise ValueError("v8固定node_static_dim=10、edge_static_dim=2")
    if cfg.get("runoff_mode") not in {"pure_lstm", "water_balance_lstm"}:
        raise ValueError("未知runoff_mode")
    if cfg.get("routing_mode") not in {"pure_gnn", "kinematic_wave_gnn"}:
        raise ValueError("未知routing_mode")
    state = cfg.get("state_initialization", {})
    if not bool(state.get("enabled", False)) or state.get("mode") != "forecast_origin":
        raise ValueError("四组v8实验统一要求history-informed forecast-origin initialization")

    loss = cfg.get("loss", {})
    expected = {
        "mode": "multitask",
        "q_scale_mode": "per_station",
        "z_target_mode": "delta_from_t0",
        "delta_z_scale_mode": "per_station",
    }
    mismatch = {
        key: loss.get(key) for key, value in expected.items() if loss.get(key) != value
    }
    if mismatch:
        raise ValueError(f"v8 loss normalization/target contract不一致: {mismatch}")
    numeric_expected = {
        "discharge_weight": 1.0,
        "water_level_weight": 1.0,
        "q_point_weight": 1.0,
        "q_peak_weight": 0.25,
        "q_volume_weight": 0.25,
        "z_level_weight": 1.0,
        "z_slope_weight": 0.25,
    }
    wrong = {
        key: loss.get(key)
        for key, value in numeric_expected.items()
        if not math.isclose(float(loss.get(key, float("nan"))), value)
    }
    if wrong:
        raise ValueError(f"v8分层loss权重不一致: {wrong}")
    weights = cfg.get("loss_weights", {})
    if (
        not math.isclose(float(weights.get("discharge", float("nan"))), 1.0)
        or not math.isclose(float(weights.get("water_level", float("nan"))), 1.0)
    ):
        raise ValueError("v8 Q:Z主任务权重固定1:1")
    if bool(cfg.get("train_sampling", {}).get("enabled", False)):
        raise ValueError("v8首个正式四组对照固定full-pass TRAIN，不启用weighted sampling")
    if cfg.get("validation_selection", {}).get("mode") != "val_loss":
        raise ValueError("v8当前checkpoint selection固定val_loss")
    if bool(cfg.get("hyperparameter_optimization", {}).get("enabled", False)):
        raise ValueError("v8四组正式基线禁止同时启用HPO")
    if int(cfg.get("num_workers", 0)) != 0:
        raise ValueError("当前v8 NPZ cache正式配置固定num_workers=0")
    if int(cfg.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size必须>0")
    if int(cfg.get("hidden_dim", 0)) <= 0:
        raise ValueError("hidden_dim必须>0")


def _applied_stats(
    mapping: dict[str, Any], station_ids: tuple[str, ...], label: str
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    scales: list[float] = []
    for station in station_ids:
        raw = mapping.get(station)
        if not isinstance(raw, dict):
            raise ValueError(f"{label}: 缺少station={station}")
        mean = float(raw["applied_mean"])
        scale = float(raw["applied_scale"])
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{label}/{station}: applied normalization非法")
        means.append(mean)
        scales.append(scale)
    return means, scales


def _attach_runtime(cfg: dict[str, Any], dataset: HydrologicGraphV8Dataset) -> None:
    contract = dataset.contract
    root = dataset.root
    if contract.get("contract") != CONTRACT_NAME:
        raise ValueError("v8 dataset contract错误")
    if int(contract.get("graph_count", -1)) != 33:
        raise ValueError("v8 graph_count必须为33")
    if int(contract.get("computational_node_count", -1)) != 237:
        raise ValueError("v8 computational_node_count必须为237")
    if int(contract.get("edge_count", -1)) != 204:
        raise ValueError("v8 edge_count必须为204")
    if int(contract.get("observation_station_count", -1)) != 39:
        raise ValueError("v8 observation_station_count必须为39")
    report = root / "BUILD_AND_QC.md"
    if not report.is_file() or "FINAL QC STATUS: PASS" not in report.read_text(
        encoding="utf-8"
    ):
        raise ValueError("v8正式数据没有FINAL QC STATUS: PASS")

    normal = contract.get("normalization")
    if (
        not isinstance(normal, dict)
        or normal.get("computed_from_split") != "TRAIN"
        or normal.get("fit_scope") != "TRAIN_SAMPLE_EXPOSURE_ONLY"
    ):
        raise ValueError("v8 normalization必须为TRAIN-only exposure")
    station_ids = dataset.station_ids
    qh_mean, qh_scale = _applied_stats(
        normal["q_history_by_station"], station_ids, "q_history_by_station"
    )
    zh_mean, zh_scale = _applied_stats(
        normal["z_history_by_station"], station_ids, "z_history_by_station"
    )
    qt_mean, qt_scale = _applied_stats(
        normal["q_target_by_station"], station_ids, "q_target_by_station"
    )
    dz_mean, dz_scale = _applied_stats(
        normal["delta_z_target_by_station"],
        station_ids,
        "delta_z_target_by_station",
    )
    static_names = tuple(contract["node_static_features"])
    node_mean = [
        float(normal["node_static"][name]["mean"]) for name in static_names
    ]
    node_scale = [
        float(normal["node_static"][name]["scale"]) for name in static_names
    ]
    if any(not math.isfinite(value) for value in (*node_mean, *node_scale)) or any(
        value <= 0 for value in node_scale
    ):
        raise ValueError("v8 node static normalization非法")
    rain_mean = float(normal["rain_mm"]["mean"])
    rain_scale = float(normal["rain_mm"]["scale"])
    if not math.isfinite(rain_mean) or not math.isfinite(rain_scale) or rain_scale <= 0:
        raise ValueError("v8 rain normalization非法")

    contract_bytes = (root / "metadata/dataset_contract.json").read_bytes()
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    cfg["_runtime"] = {
        "v8_station_count": len(station_ids),
        "v8_station_ids": list(station_ids),
        "v8_normalization": {
            "rain_mean": rain_mean,
            "rain_scale": rain_scale,
            "node_static_mean": node_mean,
            "node_static_scale": node_scale,
            "q_history_mean": qh_mean,
            "q_history_scale": qh_scale,
            "z_history_mean": zh_mean,
            "z_history_scale": zh_scale,
            "q_target_mean": qt_mean,
            "q_target_scale": qt_scale,
            "dz_target_mean": dz_mean,
            "dz_target_scale": dz_scale,
        },
        "loss_scales": {
            "discharge": float(normal["q_target_global"]["scale"]),
            "water_level": float(normal["delta_z_target_global"]["scale"]),
        },
        "data_contract": {
            "format_version": 8,
            "contract": CONTRACT_NAME,
            "artifact_sha256": contract_sha,
            "graph_count": int(contract["graph_count"]),
            "computational_node_count": int(contract["computational_node_count"]),
            "edge_count": int(contract["edge_count"]),
            "observation_station_count": int(contract["observation_station_count"]),
            "station_ids": list(station_ids),
        },
    }


def _ensure_split_compatibility(
    train_dataset: HydrologicGraphV8Dataset,
    validation_dataset: HydrologicGraphV8Dataset,
) -> None:
    if train_dataset.station_ids != validation_dataset.station_ids:
        raise ValueError("TRAIN/VALIDATION全局station catalogue不一致")
    unseen = set(validation_dataset.graph_ids) - set(train_dataset.graph_ids)
    if unseen:
        raise ValueError(f"VALIDATION包含TRAIN未出现graph: {sorted(unseen)}")
    overlap = set(train_dataset.event_ids) & set(validation_dataset.event_ids)
    if overlap:
        raise ValueError(f"TRAIN/VALIDATION EVENT_ID泄漏: {sorted(overlap)[:10]}")


def setup_v8_training(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    cfg = _runtime_config(
        config_path, dataset_root=dataset_root, graph_id=graph_id
    )
    seed_everything(int(cfg["seed"]))
    root = cfg["data"]["dataset_root"]
    tensor_cache: dict[str, dict[str, Any]] = {}
    train_dataset = HydrologicGraphV8Dataset(
        root,
        cfg["data"]["train_split"],
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        tensor_cache=tensor_cache,
    )
    validation_dataset = HydrologicGraphV8Dataset(
        root,
        cfg["data"]["validation_split"],
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        tensor_cache=tensor_cache,
    )
    _ensure_split_compatibility(train_dataset, validation_dataset)
    _attach_runtime(cfg, train_dataset)
    train_loader = build_hydrologic_graph_v8_loader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["TRAIN"],
    )
    validation_loader = build_hydrologic_graph_v8_loader(
        validation_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["VALIDATION"],
    )
    model = HydrologicGraphV8Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_v8_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    split = str(split).upper()
    if split not in {"VALIDATION", "TEST"}:
        raise ValueError("v8 evaluation split必须为VALIDATION/TEST")
    cfg = _runtime_config(
        config_path, dataset_root=dataset_root, graph_id=graph_id
    )
    seed_everything(int(cfg["seed"]))
    dataset = HydrologicGraphV8Dataset(
        cfg["data"]["dataset_root"],
        split,
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
    )
    _attach_runtime(cfg, dataset)
    loader = build_hydrologic_graph_v8_loader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split],
    )
    model = HydrologicGraphV8Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device


def validate_v8_checkpoint_config(
    checkpoint: dict[str, Any],
    cfg: dict[str, Any],
    *,
    resume: bool = False,
) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("v8正式checkpoint缺少训练config")
    for key in (
        "runoff_mode",
        "routing_mode",
        "history_length",
        "forecast_horizon",
        "node_static_dim",
        "edge_static_dim",
        "hidden_dim",
        "solver",
        "physical_bounds",
    ):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"checkpoint与当前v8配置不兼容: {key}")
    for key in (
        "discharge_weight",
        "water_level_weight",
        "q_point_weight",
        "q_peak_weight",
        "q_volume_weight",
        "z_level_weight",
        "z_slope_weight",
        "q_scale_mode",
        "z_target_mode",
        "delta_z_scale_mode",
    ):
        if saved.get("loss", {}).get(key) != cfg.get("loss", {}).get(key):
            raise ValueError(f"checkpoint与当前v8 loss不兼容: loss.{key}")
    saved_contract = saved.get("_runtime", {}).get("data_contract", {})
    current_contract = cfg.get("_runtime", {}).get("data_contract", {})
    if saved_contract.get("artifact_sha256") != current_contract.get("artifact_sha256"):
        raise ValueError("checkpoint与当前v8 dataset_contract.json不一致")
    if saved_contract.get("station_ids") != current_contract.get("station_ids"):
        raise ValueError("checkpoint与当前v8 station catalogue不一致")
    if resume:
        saved_graph = saved.get("data", {}).get("graph_id")
        current_graph = cfg.get("data", {}).get("graph_id")
        if saved_graph != current_graph:
            raise ValueError("resume要求data.graph_id与checkpoint完全一致")
