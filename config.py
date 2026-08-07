"""Strict YAML configuration loading with explicit inheritance."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is incomplete or internally inconsistent."""


_ROOT_KEYS = {
    "seed",
    "runoff_mode",
    "routing_mode",
    "history_length",
    "forecast_horizon",
    "dynamic_dim",
    "node_static_dim",
    "edge_static_dim",
    "hidden_dim",
    "device",
    "gpu_id",
    "batch_size",
    "gradient_accumulation_steps",
    "amp",
    "gradient_checkpointing",
    "num_workers",
    "pin_memory",
    "debug_mode",
    "debug_num_events",
    "debug_max_batches",
    "data",
    "physical_bounds",
    "solver",
    "loss_weights",
    "optimizer",
    "training",
    "transfer_learning",
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = (
            _merge(out[key], value)
            if isinstance(value, dict) and isinstance(out.get(key), dict)
            else value
        )
    return out


def _load(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"配置继承存在循环: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置顶层必须是映射: {path}")
    base_name = raw.pop("_base_", None)
    if base_name is None:
        return raw
    if not isinstance(base_name, str) or not base_name.strip():
        raise ConfigError(f"_base_ 必须是非空文件名: {path}")
    return _merge(_load(path.parent / base_name, (*stack, path)), raw)


def _mapping(cfg: Mapping[str, Any], key: str, allowed: set[str]) -> Mapping[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} 必须是映射")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"{key} 含未知字段: {', '.join(sorted(unknown))}")
    missing = allowed - set(value)
    if missing:
        raise ConfigError(f"{key} 缺少字段: {', '.join(sorted(missing))}")
    return value


def _bool(cfg: Mapping[str, Any], key: str) -> None:
    if not isinstance(cfg.get(key), bool):
        raise ConfigError(f"{key} 必须是 true/false")


def _int(cfg: Mapping[str, Any], key: str, minimum: int = 0) -> None:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{key} 必须是 >= {minimum} 的整数")


def _number(
    cfg: Mapping[str, Any], key: str, minimum: float = 0.0, *, strictly: bool = False
) -> None:
    value = cfg.get(key)
    valid_type = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    valid_range = value > minimum if valid_type and strictly else valid_type and value >= minimum
    if not valid_range:
        symbol = ">" if strictly else ">="
        raise ConfigError(f"{key} 必须是 {symbol} {minimum} 的数值")


def _enum(cfg: Mapping[str, Any], key: str, choices: set[str]) -> None:
    value = cfg.get(key)
    if value not in choices:
        raise ConfigError(f"{key} 必须是 {sorted(choices)} 之一，实际为 {value!r}")


def _bounds(cfg: Mapping[str, Any], key: str, *, positive: bool = True) -> None:
    value = cfg.get(key)
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{key} 必须是 [下界, 上界]")
    low, high = value
    if any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(float(x))
        for x in value
    ):
        raise ConfigError(f"{key} 上下界必须是数值")
    if (positive and low <= 0) or low >= high:
        raise ConfigError(f"{key} 必须满足 0 < 下界 < 上界")


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete merged configuration and return it unchanged."""
    unknown = set(cfg) - _ROOT_KEYS
    if unknown:
        raise ConfigError(f"配置含未知顶层字段: {', '.join(sorted(unknown))}")
    missing = _ROOT_KEYS - set(cfg)
    if missing:
        raise ConfigError(f"配置缺少顶层字段: {', '.join(sorted(missing))}")

    _int(cfg, "seed")
    _enum(cfg, "runoff_mode", {"pure_lstm", "water_balance_lstm"})
    _enum(cfg, "routing_mode", {"pure_gnn", "kinematic_wave_gnn"})
    for key in (
        "history_length",
        "forecast_horizon",
        "dynamic_dim",
        "node_static_dim",
        "edge_static_dim",
        "hidden_dim",
        "batch_size",
        "gradient_accumulation_steps",
        "debug_num_events",
        "debug_max_batches",
    ):
        _int(cfg, key, 1)
    _int(cfg, "gpu_id")
    _int(cfg, "num_workers")
    for key in ("amp", "gradient_checkpointing", "pin_memory", "debug_mode"):
        _bool(cfg, key)
    if cfg["gradient_checkpointing"]:
        raise ConfigError(
            "gradient_checkpointing尚未实现，不能设置为true后静默忽略"
        )
    _enum(cfg, "device", {"auto", "cpu", "cuda"})
    if cfg["device"] == "cpu" and cfg["amp"]:
        raise ConfigError("device=cpu 时 amp 必须为 false")

    data = _mapping(
        cfg,
        "data",
        {
            "mode",
            "dataset_root",
            "graph_id",
            "target_variable",
            "normalize_dynamic",
            "strict_validation",
            "use_observation_masks",
            "future_rainfall_mode",
            "train_split",
            "validation_split",
            "test_split",
        },
    )
    _enum(data, "mode", {"synthetic", "hunan"})
    _enum(data, "target_variable", {"AUTO", "FLOW", "WATER_LEVEL", "BOTH"})
    _enum(
        data,
        "future_rainfall_mode",
        {"observed_hindcast", "zero", "persistence"},
    )
    for key in ("normalize_dynamic", "strict_validation", "use_observation_masks"):
        _bool(data, key)
    graph_id = data["graph_id"]
    if graph_id is not None and (not isinstance(graph_id, str) or not graph_id.strip()):
        raise ConfigError("data.graph_id 必须是非空字符串或 null")
    splits = (data["train_split"], data["validation_split"], data["test_split"])
    if splits != ("TRAIN", "VALIDATION", "TEST"):
        raise ConfigError("正式划分必须固定为 TRAIN / VALIDATION / TEST")
    root = data["dataset_root"]
    if data["mode"] == "hunan" and (not isinstance(root, str) or not root.strip()):
        raise ConfigError("data.mode=hunan 时必须提供非空 data.dataset_root")
    if data["mode"] == "synthetic" and root is not None:
        raise ConfigError("合成调试模式的 data.dataset_root 必须为 null")
    if data["mode"] == "synthetic" and graph_id is not None:
        raise ConfigError("合成调试模式的 data.graph_id 必须为 null")
    if data["mode"] == "synthetic" and data["target_variable"] == "AUTO":
        raise ConfigError("合成调试模式不支持 target_variable=AUTO")
    if data["mode"] == "hunan":
        if cfg["node_static_dim"] != 10 or cfg["edge_static_dim"] != 2:
            raise ConfigError("湖南正式数据固定要求 node_static_dim=10、edge_static_dim=2")
        if cfg["debug_mode"]:
            raise ConfigError("湖南正式数据禁止 debug_mode=true，以免截断训练批次")
        if not data["strict_validation"]:
            raise ConfigError("湖南正式数据要求 data.strict_validation=true")
        if not data["use_observation_masks"]:
            raise ConfigError("湖南正式数据要求 data.use_observation_masks=true")
        if cfg["num_workers"] != 0:
            raise ConfigError(
                "当前CSV内存数据层在Windows正式模式要求num_workers=0，"
                "避免worker复制整省动态张量"
            )

    bounds = _mapping(cfg, "physical_bounds", {"width", "manning_n"})
    _bounds(bounds, "width")
    _bounds(bounds, "manning_n")

    solver = _mapping(
        cfg,
        "solver",
        {
            "dx",
            "cfl",
            "maximum_substeps",
            "minimum_slope",
            "minimum_length",
            "seconds_per_step",
        },
    )
    for key in ("dx", "minimum_slope", "minimum_length", "seconds_per_step"):
        _number(solver, key, strictly=True)
    _number(solver, "cfl", strictly=True)
    if solver["cfl"] > 1:
        raise ConfigError("solver.cfl 必须在 (0, 1] 范围内")
    _int(solver, "maximum_substeps", 1)
    if data["mode"] == "hunan" and solver["seconds_per_step"] != 3600:
        raise ConfigError("湖南逐时正式数据要求 solver.seconds_per_step=3600")

    weights = _mapping(cfg, "loss_weights", {"discharge", "water_level"})
    for key in ("discharge", "water_level"):
        _number(weights, key)
    if weights["discharge"] + weights["water_level"] <= 0:
        raise ConfigError("至少一个 loss_weights 必须大于 0")
    if data["target_variable"] == "FLOW" and weights["discharge"] <= 0:
        raise ConfigError("target_variable=FLOW 时 discharge loss weight 必须大于 0")
    if data["target_variable"] == "FLOW" and weights["water_level"] != 0:
        raise ConfigError("target_variable=FLOW 时 water_level loss weight 必须为 0")
    if data["target_variable"] == "WATER_LEVEL" and weights["water_level"] <= 0:
        raise ConfigError("target_variable=WATER_LEVEL 时 water_level loss weight 必须大于 0")
    if data["target_variable"] == "WATER_LEVEL" and weights["discharge"] != 0:
        raise ConfigError("target_variable=WATER_LEVEL 时 discharge loss weight 必须为 0")
    if data["target_variable"] in {"AUTO", "BOTH"} and (
        weights["discharge"] <= 0 or weights["water_level"] <= 0
    ):
        raise ConfigError("target_variable=AUTO/BOTH 时两个 loss weight 都必须大于 0")

    optimizer = _mapping(cfg, "optimizer", {"name", "lr", "weight_decay"})
    _enum(optimizer, "name", {"adamw"})
    _number(optimizer, "lr", strictly=True)
    _number(optimizer, "weight_decay")

    training = _mapping(
        cfg,
        "training",
        {"epochs", "patience", "gradient_clip", "checkpoint", "log_csv"},
    )
    _int(training, "epochs", 1)
    _int(training, "patience", 1)
    _number(training, "gradient_clip", strictly=True)
    for key in ("checkpoint", "log_csv"):
        if not isinstance(training[key], str) or not training[key].strip():
            raise ConfigError(f"training.{key} 必须是非空路径")

    transfer = _mapping(
        cfg,
        "transfer_learning",
        {"strategy", "stages", "full_finetune_lr"},
    )
    strategies = {
        "observation_only",
        "observation_and_edge_parameters",
        "full_finetune",
    }
    _enum(transfer, "strategy", strategies)
    stages = transfer["stages"]
    if not isinstance(stages, list) or not stages or any(x not in strategies for x in stages):
        raise ConfigError("transfer_learning.stages 含无效策略")
    _number(transfer, "full_finetune_lr", strictly=True)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, inherit and strictly validate one YAML configuration."""
    return validate_config(_load(Path(path), ()))
