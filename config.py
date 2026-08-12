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
    "loss",
    "validation_selection",
    "hyperparameter_optimization",
    "optimizer",
    "train_sampling",
    "state_initialization",
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
    # Backward-compatible default for legacy/programmatically constructed P1/P2
    # configurations.  Canonical YAMLs also declare this block explicitly.
    if "state_initialization" not in cfg:
        cfg["state_initialization"] = {
            "enabled": False,
            "mode": "forecast_origin",
        }

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
            "dataset_type",
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
    _enum(data, "dataset_type", {"event", "continuous"})
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
    if data["dataset_type"] == "continuous" and (
        data["mode"] != "hunan" or data["target_variable"] != "BOTH"
    ):
        raise ConfigError(
            "continuous dataset要求data.mode=hunan且target_variable=BOTH"
        )
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

    state_initialization = _mapping(
        cfg,
        "state_initialization",
        {"enabled", "mode"},
    )
    _bool(state_initialization, "enabled")
    _enum(state_initialization, "mode", {"forecast_origin"})
    if state_initialization["enabled"]:
        if data["dataset_type"] != "continuous":
            raise ConfigError("state_initialization仅允许continuous dataset")
        if cfg["runoff_mode"] != "water_balance_lstm":
            raise ConfigError(
                "state_initialization要求runoff_mode=water_balance_lstm"
            )
        if cfg["routing_mode"] != "kinematic_wave_gnn":
            raise ConfigError(
                "state_initialization要求routing_mode=kinematic_wave_gnn"
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
            "integration_scheme",
            "implicit_iterations",
            "implicit_residual_tolerance",
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
    _enum(solver, "integration_scheme", {"backward_euler"})
    _int(solver, "implicit_iterations", 1)
    _number(solver, "implicit_residual_tolerance", strictly=True)
    if solver["implicit_residual_tolerance"] > 1:
        raise ConfigError("solver.implicit_residual_tolerance 必须在 (0, 1] 范围内")
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

    loss = _mapping(
        cfg,
        "loss",
        {
            "mode",
            "q_scale_mode",
            "q_scale_floor_m3s",
            "z_target_mode",
            "delta_z_scale_mode",
            "delta_z_scale_floor_m",
            "discharge_weight",
            "water_level_weight",
            "q_point_weight",
            "q_peak_weight",
            "q_volume_weight",
            "z_level_weight",
            "z_slope_weight",
        },
    )
    _enum(loss, "mode", {"legacy", "multitask"})
    _enum(loss, "q_scale_mode", {"global", "per_graph"})
    _number(loss, "q_scale_floor_m3s", strictly=True)
    _enum(loss, "z_target_mode", {"absolute", "delta_from_t0"})
    _enum(loss, "delta_z_scale_mode", {"global", "per_station"})
    _number(loss, "delta_z_scale_floor_m", strictly=True)
    for key in (
        "discharge_weight",
        "water_level_weight",
        "q_point_weight",
        "q_peak_weight",
        "q_volume_weight",
        "z_level_weight",
        "z_slope_weight",
    ):
        _number(loss, key)
    if loss["mode"] == "multitask":
        fixed = {
            "discharge_weight": 2.0,
            "water_level_weight": 1.0,
            "q_point_weight": 1.0,
            "z_level_weight": 1.0,
        }
        changed = {
            key: loss[key]
            for key, expected in fixed.items()
            if float(loss[key]) != expected
        }
        if changed:
            raise ConfigError(
                "multitask固定要求Q:Z=2:1且q_point/z_level权重为1，"
                f"不符合项={changed}"
            )
    if loss["q_scale_mode"] == "per_graph" and (
        loss["mode"] != "multitask" or data["mode"] != "hunan"
    ):
        raise ConfigError(
            "loss.q_scale_mode=per_graph仅允许湖南正式multitask loss"
        )
    if loss["z_target_mode"] == "delta_from_t0" and (
        data["dataset_type"] != "continuous"
        or loss["delta_z_scale_mode"] != "per_station"
    ):
        raise ConfigError(
            "delta_from_t0要求continuous dataset和TRAIN-only per_station ΔZ scale"
        )
    if data["dataset_type"] == "continuous" and loss["z_target_mode"] != "delta_from_t0":
        raise ConfigError("continuous P2必须使用delta_from_t0水位监督")

    selection = _mapping(
        cfg,
        "validation_selection",
        {
            "mode",
            "q_nse_weight",
            "q_kge_weight",
            "q_peak_weight",
            "q_volume_weight",
            "z_level_weight",
            "z_slope_weight",
            "efficiency_clip_min",
            "efficiency_clip_max",
        },
    )
    _enum(selection, "mode", {"val_loss", "composite"})
    selection_weight_names = (
        "q_nse_weight",
        "q_kge_weight",
        "q_peak_weight",
        "q_volume_weight",
        "z_level_weight",
        "z_slope_weight",
    )
    for key in selection_weight_names:
        _number(selection, key)
    total_selection_weight = sum(float(selection[key]) for key in selection_weight_names)
    if not math.isclose(total_selection_weight, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ConfigError(
            "validation_selection六项权重之和必须为1，"
            f"实际={total_selection_weight}"
        )
    for key in ("efficiency_clip_min", "efficiency_clip_max"):
        value = selection[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ConfigError(f"validation_selection.{key}必须是有限数值")
    if selection["efficiency_clip_min"] >= selection["efficiency_clip_max"]:
        raise ConfigError("validation_selection efficiency clip要求min < max")
    if (
        loss["mode"] == "multitask"
        and selection["mode"] != "composite"
        and data["dataset_type"] != "continuous"
    ):
        raise ConfigError("multitask loss必须使用composite validation selection")

    hpo = _mapping(
        cfg,
        "hyperparameter_optimization",
        {
            "enabled",
            "n_trials",
            "output_dir",
            "sampler",
            "pruner",
            "search_space",
        },
    )
    _bool(hpo, "enabled")
    _int(hpo, "n_trials", 1)
    if not isinstance(hpo["output_dir"], str) or not hpo["output_dir"].strip():
        raise ConfigError("hyperparameter_optimization.output_dir必须是非空路径")
    sampler = _mapping(hpo, "sampler", {"name", "seed"})
    _enum(sampler, "name", {"tpe"})
    _int(sampler, "seed")
    pruner = _mapping(
        hpo,
        "pruner",
        {"name", "n_startup_trials", "n_warmup_steps", "interval_steps"},
    )
    _enum(pruner, "name", {"median"})
    _int(pruner, "n_startup_trials")
    _int(pruner, "n_warmup_steps")
    _int(pruner, "interval_steps", 1)
    search_space = _mapping(
        hpo,
        "search_space",
        {
            "learning_rate",
            "weight_decay",
            "hidden_dim",
            "q_peak_weight",
            "q_volume_weight",
            "z_slope_weight",
        },
    )
    for key in (
        "learning_rate",
        "weight_decay",
        "q_peak_weight",
        "q_volume_weight",
        "z_slope_weight",
    ):
        spec = _mapping(search_space, key, {"type", "low", "high"})
        _enum(spec, "type", {"log_float"})
        _number(spec, "low", strictly=True)
        _number(spec, "high", strictly=True)
        if spec["low"] >= spec["high"]:
            raise ConfigError(f"search_space.{key}要求low < high")
    hidden_space = _mapping(search_space, "hidden_dim", {"type", "choices"})
    _enum(hidden_space, "type", {"categorical"})
    choices = hidden_space["choices"]
    if (
        not isinstance(choices, list)
        or not choices
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in choices)
        or len(set(choices)) != len(choices)
    ):
        raise ConfigError("search_space.hidden_dim.choices必须是不重复的正整数列表")
    if hpo["enabled"] and (
        loss["mode"] != "multitask"
        or selection["mode"] != "composite"
        or cfg["runoff_mode"] != "water_balance_lstm"
        or cfg["routing_mode"] != "kinematic_wave_gnn"
    ):
        raise ConfigError("HPO仅允许E4 multitask + composite selection配置")

    optimizer = _mapping(cfg, "optimizer", {"name", "lr", "weight_decay"})
    _enum(optimizer, "name", {"adamw"})
    _number(optimizer, "lr", strictly=True)
    _number(optimizer, "weight_decay")

    sampling = _mapping(
        cfg,
        "train_sampling",
        {
            "enabled",
            "response_strength",
            "response_cap",
            "minimum_weight",
            "maximum_weight",
        },
    )
    _bool(sampling, "enabled")
    for key in (
        "response_strength",
        "response_cap",
        "minimum_weight",
        "maximum_weight",
    ):
        _number(sampling, key, strictly=True)
    if sampling["minimum_weight"] > sampling["maximum_weight"]:
        raise ConfigError("train_sampling要求minimum_weight<=maximum_weight")
    if sampling["enabled"] and data["dataset_type"] != "continuous":
        raise ConfigError("TRAIN weighted sampling仅允许continuous dataset")

    raw_training = cfg.get("training")
    if isinstance(raw_training, dict):
        # Backward-compatible defaults for programmatically constructed legacy
        # configs. P2 overrides both fields explicitly.
        raw_training.setdefault("early_stopping", True)
        raw_training.setdefault("final_checkpoint", None)
    training = _mapping(
        cfg,
        "training",
        {
            "epochs",
            "patience",
            "early_stopping",
            "gradient_clip",
            "checkpoint",
            "final_checkpoint",
            "log_csv",
        },
    )
    _int(training, "epochs", 1)
    _int(training, "patience", 1)
    _bool(training, "early_stopping")
    _number(training, "gradient_clip", strictly=True)
    for key in ("checkpoint", "log_csv"):
        if not isinstance(training[key], str) or not training[key].strip():
            raise ConfigError(f"training.{key} 必须是非空路径")
    final_checkpoint = training["final_checkpoint"]
    if final_checkpoint is not None and (
        not isinstance(final_checkpoint, str) or not final_checkpoint.strip()
    ):
        raise ConfigError("training.final_checkpoint必须是非空路径或null")
    if not training["early_stopping"] and final_checkpoint is None:
        raise ConfigError("取消early stopping时必须显式设置final_checkpoint")

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
