"""Training/evaluation setup for v9 using the frozen v8 Hunan data contract."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from data.device import resolve_device, seed_everything
from datasets.hydrologic_graph_v8 import (
    HydrologicGraphV8Dataset,
    build_hydrologic_graph_v8_loader,
)
from models.hydrologic_graph_v9 import HydrologicGraphV9Model
from scripts.v8_training import (
    _SPLIT_SEED_OFFSET,
    _attach_runtime,
    _ensure_split_compatibility,
    _load_yaml,
    _resolve_root,
)


def is_v9_requested(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
) -> bool:
    del dataset_root
    raw = _load_yaml(config_path)
    return str(raw.get("model_version", "")).lower() == "v9"


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
        if value is not None:
            cfg["training"][key] = str(_resolve_root(value))

    temporal = cfg.get("temporal", {})
    forcing_step = float(temporal.get("forcing_step_seconds", float("nan")))
    if not math.isfinite(forcing_step) or forcing_step <= 0:
        raise ValueError("v9 temporal.forcing_step_seconds必须>0")
    # One source of truth for physical dt: routing follows the forcing cadence.
    cfg.setdefault("solver", {})["seconds_per_step"] = forcing_step
    _validate_v9_config(cfg)
    return cfg


def _validate_v9_config(cfg: dict[str, Any]) -> None:
    if str(cfg.get("model_version", "")).lower() != "v9":
        raise ValueError("v9配置必须显式model_version: v9")
    data = cfg.get("data", {})
    if data.get("mode") != "hunan":
        raise ValueError("当前v9 Hunan配置要求data.mode=hunan")
    if data.get("dataset_type") != "event":
        raise ValueError("v9 Hunan数据事实是event-domain，dataset_type必须为event")
    if data.get("target_variable") != "BOTH":
        raise ValueError("v9同时监督Q/Delta-Z，target_variable必须为BOTH")
    if not bool(data.get("strict_validation", False)):
        raise ValueError("v9要求strict_validation=true")
    if not bool(data.get("use_observation_masks", False)):
        raise ValueError("v9要求use_observation_masks=true")
    if data.get("future_rainfall_mode") not in {
        "observed_hindcast",
        "zero",
        "persistence",
    }:
        raise ValueError("v9 future_rainfall_mode非法")

    temporal = cfg.get("temporal", {})
    history_duration = int(temporal.get("history_duration_seconds", 0))
    forecast_duration = int(temporal.get("forecast_duration_seconds", 0))
    forcing_step = int(temporal.get("forcing_step_seconds", 0))
    target_step = int(temporal.get("target_step_seconds", 0))
    if min(history_duration, forecast_duration, forcing_step, target_step) <= 0:
        raise ValueError("v9 temporal durations/steps必须全部>0")
    if history_duration % forcing_step or forecast_duration % forcing_step:
        raise ValueError("history/forecast duration必须整除forcing_step_seconds")
    if target_step < forcing_step or target_step % forcing_step:
        raise ValueError("target_step_seconds必须是forcing_step_seconds的整数倍")
    if forecast_duration % target_step:
        raise ValueError("forecast_duration_seconds必须整除target_step_seconds")
    history_steps = history_duration // forcing_step
    internal_forecast_steps = forecast_duration // forcing_step
    target_steps = forecast_duration // target_step
    if int(cfg.get("history_length", -1)) != history_steps:
        raise ValueError("history_length必须等于history_duration/forcing_step")
    if int(cfg.get("forecast_horizon", -1)) != target_steps:
        raise ValueError("forecast_horizon必须等于forecast_duration/target_step")
    # The currently frozen Hunan v8 tensors are hourly 24->6.  The v9 model
    # itself supports different forcing/target cadence; a Zhejiang/minute loader
    # can later provide the corresponding longer tensors without changing v9.
    if (history_steps, internal_forecast_steps, target_steps) != (24, 6, 6):
        raise ValueError(
            "当前_hydrologic_graph_v8数据只提供24个history和6个future内部步；"
            "分钟级浙江数据需要新的同契约loader/tensor，而无需改v9模型"
        )

    if int(cfg.get("node_static_dim", -1)) != 10 or int(cfg.get("edge_static_dim", -1)) != 2:
        raise ValueError("v9 Hunan固定node_static_dim=10、edge_static_dim=2")
    if cfg.get("runoff_mode") not in {"pure_lstm", "water_balance_lstm"}:
        raise ValueError("v9未知runoff_mode")
    if cfg.get("routing_mode") not in {"pure_gnn", "kinematic_wave_gnn"}:
        raise ValueError("v9未知routing_mode")

    warmup = cfg.get("warmup", {})
    if not bool(warmup.get("enabled", False)):
        raise ValueError("v9必须启用history runoff+routing warm-up")
    if warmup.get("initial_state") != "static_prior":
        raise ValueError("v9 warm-up起点固定使用static_prior，避免重复编码同一24h history")
    state = cfg.get("state_initialization", {})
    if not bool(state.get("enabled", False)) or state.get("mode") != "sequential_warmup":
        raise ValueError("v9 state_initialization必须为sequential_warmup")

    z_head = cfg.get("z_head", {})
    expected_z = {
        "mode": "explicit_state_mlp",
        "incremental_output": True,
        "detach_hydraulic_features": True,
        "use_z0": True,
        "use_recent_trend": True,
        "use_q_features": True,
        "use_channel_state": True,
    }
    mismatch_z = {
        key: z_head.get(key)
        for key, expected in expected_z.items()
        if z_head.get(key) != expected
    }
    if mismatch_z:
        raise ValueError(f"v9 Z head冻结设计不一致: {mismatch_z}")
    trend_windows = z_head.get("trend_windows_seconds")
    if not isinstance(trend_windows, list) or len(trend_windows) != 3:
        raise ValueError("v9 z_head.trend_windows_seconds必须含3个物理时长")
    if any(int(value) <= 0 or int(value) > history_duration for value in trend_windows):
        raise ValueError("v9 Z trend window必须位于history duration内")

    loss = cfg.get("loss", {})
    expected_loss = {
        "mode": "multitask",
        "q_scale_mode": "per_station",
        "z_target_mode": "delta_from_t0",
        "delta_z_scale_mode": "per_station",
    }
    mismatch_loss = {
        key: loss.get(key)
        for key, expected in expected_loss.items()
        if loss.get(key) != expected
    }
    if mismatch_loss:
        raise ValueError(f"v9 loss normalization/target contract不一致: {mismatch_loss}")
    numeric_expected = {
        "discharge_weight": 1.0,
        "water_level_weight": 1.0,
        "q_point_weight": 1.0,
        "q_peak_weight": 0.25,
        "q_volume_weight": 0.25,
        "z_level_weight": 1.0,
        "z_slope_weight": 0.25,
        "qz_consistency_weight": 0.0,
    }
    wrong = {
        key: loss.get(key)
        for key, expected in numeric_expected.items()
        if not math.isclose(float(loss.get(key, float("nan"))), expected)
    }
    if wrong:
        raise ValueError(f"v9 loss权重不一致: {wrong}")
    weights = cfg.get("loss_weights", {})
    if (
        not math.isclose(float(weights.get("discharge", float("nan"))), 1.0)
        or not math.isclose(float(weights.get("water_level", float("nan"))), 1.0)
    ):
        raise ValueError("v9 Q:Z主任务权重固定1:1")
    if bool(cfg.get("train_sampling", {}).get("enabled", False)):
        raise ValueError("v9四组对照固定full-pass TRAIN")
    if cfg.get("validation_selection", {}).get("mode") != "val_loss":
        raise ValueError("v9 checkpoint selection固定val_loss")
    if bool(cfg.get("hyperparameter_optimization", {}).get("enabled", False)):
        raise ValueError("v9正式四组禁止同时启用HPO")
    if bool(cfg.get("training", {}).get("early_stopping", True)):
        raise ValueError("v9本轮固定完整训练，不启用early stopping")
    if int(cfg.get("training", {}).get("epochs", 0)) != 100:
        raise ValueError("v9本轮固定训练100 epochs")
    if int(cfg.get("num_workers", 0)) != 0:
        raise ValueError("当前v9 NPZ cache正式配置固定num_workers=0")
    if int(cfg.get("batch_size", 0)) <= 0 or int(cfg.get("hidden_dim", 0)) <= 0:
        raise ValueError("v9 batch_size/hidden_dim必须>0")


def setup_v9_training(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    cfg = _runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
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
    cfg["_runtime"]["model_version"] = "v9"
    cfg["_runtime"]["temporal"] = dict(cfg["temporal"])
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
    model = HydrologicGraphV9Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_v9_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    split = str(split).upper()
    if split not in {"VALIDATION", "TEST"}:
        raise ValueError("v9 evaluation split必须为VALIDATION/TEST")
    cfg = _runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    dataset = HydrologicGraphV8Dataset(
        cfg["data"]["dataset_root"],
        split,
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
    )
    _attach_runtime(cfg, dataset)
    cfg["_runtime"]["model_version"] = "v9"
    cfg["_runtime"]["temporal"] = dict(cfg["temporal"])
    loader = build_hydrologic_graph_v8_loader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split],
    )
    model = HydrologicGraphV9Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device


def validate_v9_checkpoint_config(
    checkpoint: dict[str, Any],
    cfg: dict[str, Any],
    *,
    resume: bool = False,
) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ValueError("v9正式checkpoint缺少训练config")
    for key in (
        "model_version",
        "runoff_mode",
        "routing_mode",
        "history_length",
        "forecast_horizon",
        "node_static_dim",
        "edge_static_dim",
        "hidden_dim",
        "temporal",
        "warmup",
        "state_initialization",
        "z_head",
        "solver",
        "physical_bounds",
    ):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"checkpoint与当前v9配置不兼容: {key}")
    for key in (
        "discharge_weight",
        "water_level_weight",
        "q_point_weight",
        "q_peak_weight",
        "q_volume_weight",
        "z_level_weight",
        "z_slope_weight",
        "qz_consistency_weight",
        "q_scale_mode",
        "z_target_mode",
        "delta_z_scale_mode",
    ):
        if saved.get("loss", {}).get(key) != cfg.get("loss", {}).get(key):
            raise ValueError(f"checkpoint与当前v9 loss不兼容: loss.{key}")
    saved_contract = saved.get("_runtime", {}).get("data_contract", {})
    current_contract = cfg.get("_runtime", {}).get("data_contract", {})
    if saved_contract.get("artifact_sha256") != current_contract.get("artifact_sha256"):
        raise ValueError("checkpoint与当前v9 dataset_contract.json不一致")
    if saved_contract.get("station_ids") != current_contract.get("station_ids"):
        raise ValueError("checkpoint与当前v9 station catalogue不一致")
    if resume and saved.get("data", {}).get("graph_id") != cfg.get("data", {}).get("graph_id"):
        raise ValueError("resume要求data.graph_id与checkpoint完全一致")
