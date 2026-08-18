"""Training/evaluation setup for v9 using the frozen v8 Hunan data contract."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

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


V9_TIME_SEMANTICS = {
    "rain_interval_anchor": "interval_start",
    "hydro_hour_bin_anchor": "start_label_last_observation_within_bin",
    "forecast_origin_anchor": "end_of_last_history_bin",
}

# Zhejiang transfer must not reuse Hunan station-specific embeddings/statistics.
_V9_TRANSFER_EXCLUDED_EXACT = {
    "q_history_mean",
    "q_history_scale",
    "z_history_mean",
    "z_history_scale",
    "q_target_mean",
    "q_target_scale",
    "dz_target_scale",
}
_V9_TRANSFER_EXCLUDED_SUFFIX = "station_embedding.weight"


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
        cfg["data"]["dataset_root"] = str(
            _resolve_root(cfg["data"]["dataset_root"])
        )
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
    # One source of truth for physical dt: routing follows forcing cadence.
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
    # Current frozen Hunan tensors are hourly 24->6.  The model itself is
    # duration/dt aware; a Zhejiang minute loader must provide matching tensors.
    if (history_steps, internal_forecast_steps, target_steps) != (24, 6, 6):
        raise ValueError(
            "当前_hydrologic_graph_v8数据只提供24个history和6个future内部步；"
            "分钟级浙江数据需要新的同契约loader/tensor，而无需重写v9模型"
        )
    for key, expected in V9_TIME_SEMANTICS.items():
        if temporal.get(key) != expected:
            raise ValueError(
                f"v9 temporal.{key}必须为{expected!r}，实际={temporal.get(key)!r}"
            )

    if int(cfg.get("node_static_dim", -1)) != 10 or int(
        cfg.get("edge_static_dim", -1)
    ) != 2:
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
    correction = cfg.get("state_correction", {})
    expected_correction = {
        "enabled": True,
        "mode": "v8_history_residual_after_warmup",
        "use_qz_history": True,
        "correct_channel_storage": True,
    }
    mismatch_correction = {
        key: correction.get(key)
        for key, expected in expected_correction.items()
        if correction.get(key) != expected
    }
    if mismatch_correction:
        raise ValueError(f"v9 state correction冻结设计不一致: {mismatch_correction}")
    hidden_residual_scale = float(correction.get("hidden_residual_scale", float("nan")))
    storage_log_scale = float(correction.get("storage_log_scale", float("nan")))
    if (
        not math.isfinite(hidden_residual_scale)
        or not 0 < hidden_residual_scale <= 1.0
        or not math.isfinite(storage_log_scale)
        or not 0 < storage_log_scale <= 1.0
    ):
        raise ValueError("v9 state correction尺度必须位于(0,1]")

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
    if any(
        int(value) <= 0 or int(value) > history_duration
        for value in trend_windows
    ):
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
    for key in ("q_scale_floor_m3s", "delta_z_scale_floor_m"):
        value = float(loss.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"v9 loss.{key}必须>0")
    weights = cfg.get("loss_weights", {})
    if (
        not math.isclose(float(weights.get("discharge", float("nan"))), 1.0)
        or not math.isclose(
            float(weights.get("water_level", float("nan"))), 1.0
        )
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


def _validate_dataset_time_contract(
    cfg: dict[str, Any], dataset: HydrologicGraphV8Dataset
) -> None:
    """Make the historical hourly-bin convention explicit and auditable.

    Step11 rain is [start,end) and stored under interval start.  The frozen
    hourly hydro table keeps the last observation within the same labelled hour.
    Therefore a history label t represents the bin ending at t+1h for physical
    warm-up purposes, and the forecast origin is the end of the last history bin.
    No one-hour tensor shift is performed.
    """
    declared = dataset.contract.get("timestamp_semantics")
    if isinstance(declared, Mapping):
        mismatches = {
            key: declared.get(key)
            for key, expected in V9_TIME_SEMANTICS.items()
            if declared.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"dataset timestamp semantics与v9不一致: {mismatches}")
    cfg.setdefault("_runtime", {})["timestamp_semantics"] = {
        **V9_TIME_SEMANTICS,
        "dataset_declared": isinstance(declared, Mapping),
        "interpretation": (
            "history label t is the [t,t+dt) bin; forecast origin is the end "
            "of the final history bin; targets advance by target_step_seconds"
        ),
    }


def _apply_v9_scale_floors(cfg: dict[str, Any]) -> None:
    """Enforce YAML scale floors even though the frozen v8 builder used 1e-6."""
    runtime = cfg.get("_runtime", {})
    normal = runtime.get("v8_normalization")
    if not isinstance(normal, dict):
        raise ValueError("v9 runtime normalization缺失")
    station_ids = list(runtime.get("v8_station_ids", ()))
    q_floor = float(cfg["loss"]["q_scale_floor_m3s"])
    dz_floor = float(cfg["loss"]["delta_z_scale_floor_m"])
    raw_q = [float(value) for value in normal["q_target_scale"]]
    raw_dz = [float(value) for value in normal["dz_target_scale"]]
    if len(raw_q) != len(station_ids) or len(raw_dz) != len(station_ids):
        raise ValueError("v9 station scale数量与station catalogue不一致")
    applied_q = [max(value, q_floor) for value in raw_q]
    applied_dz = [max(value, dz_floor) for value in raw_dz]
    normal["q_target_scale"] = applied_q
    normal["dz_target_scale"] = applied_dz
    stations = {
        station: {
            "q_raw_scale_m3s": raw_q[index],
            "q_applied_scale_m3s": applied_q[index],
            "q_floor_applied": raw_q[index] < q_floor,
            "delta_z_raw_scale_m": raw_dz[index],
            "delta_z_applied_scale_m": applied_dz[index],
            "delta_z_floor_applied": raw_dz[index] < dz_floor,
        }
        for index, station in enumerate(station_ids)
    }
    runtime["target_scale_audit"] = {
        "computed_from_split": "TRAIN",
        "source": "v8 dataset_contract per-station scales with v9 runtime floors",
        "q_scale_floor_m3s": q_floor,
        "delta_z_scale_floor_m": dz_floor,
        "stations": stations,
    }


def _finalize_v9_runtime(
    cfg: dict[str, Any], dataset: HydrologicGraphV8Dataset
) -> None:
    _attach_runtime(cfg, dataset)
    _validate_dataset_time_contract(cfg, dataset)
    _apply_v9_scale_floors(cfg)
    cfg["_runtime"]["model_version"] = "v9"
    cfg["_runtime"]["temporal"] = dict(cfg["temporal"])


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
    _finalize_v9_runtime(cfg, train_dataset)
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
    _finalize_v9_runtime(cfg, dataset)
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
        "state_correction",
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
        "q_scale_floor_m3s",
        "z_target_mode",
        "delta_z_scale_mode",
        "delta_z_scale_floor_m",
    ):
        if saved.get("loss", {}).get(key) != cfg.get("loss", {}).get(key):
            raise ValueError(f"checkpoint与当前v9 loss不兼容: loss.{key}")
    saved_contract = saved.get("_runtime", {}).get("data_contract", {})
    current_contract = cfg.get("_runtime", {}).get("data_contract", {})
    if saved_contract.get("artifact_sha256") != current_contract.get("artifact_sha256"):
        raise ValueError("checkpoint与当前v9 dataset_contract.json不一致")
    if saved_contract.get("station_ids") != current_contract.get("station_ids"):
        raise ValueError("checkpoint与当前v9 station catalogue不一致")
    if resume and saved.get("data", {}).get("graph_id") != cfg.get("data", {}).get(
        "graph_id"
    ):
        raise ValueError("resume要求data.graph_id与checkpoint完全一致")


def extract_v9_transferable_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return Hunan->Zhejiang transferable weights only.

    Station embeddings and all station-specific normalization buffers are
    deliberately excluded.  A Zhejiang adapter must create its own station
    catalogue and TRAIN-only normalization before loading this dictionary with
    strict=False and explicitly auditing missing/unexpected keys.
    """
    state = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("v9 transfer checkpoint缺少model state_dict")
    transferable: dict[str, Any] = {}
    for key, value in state.items():
        if key in _V9_TRANSFER_EXCLUDED_EXACT:
            continue
        if key.endswith(_V9_TRANSFER_EXCLUDED_SUFFIX):
            continue
        transferable[key] = value
    if not transferable:
        raise ValueError("v9 transfer state_dict为空")
    return transferable
