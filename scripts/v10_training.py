"""Formal training/evaluation setup for v10 Q-only hydrologic forecasting."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from data.device import resolve_device, seed_everything
from datasets.hydrologic_graph_v8 import HydrologicGraphV8Dataset, build_hydrologic_graph_v8_loader
from models.hydrologic_graph_v10 import HydrologicGraphV10Model
from scripts.v10_rating import fit_train_only_linear_ratings
from scripts.v8_training import (
    _SPLIT_SEED_OFFSET,
    _attach_runtime,
    _ensure_split_compatibility,
    _load_yaml,
    _resolve_root,
)


V10_TIME_SEMANTICS = {
    "rain_interval_anchor": "interval_start",
    "hydro_hour_bin_anchor": "start_label_last_observation_within_bin",
    "forecast_origin_anchor": "end_of_last_history_bin",
}

_V10_TRANSFER_EXCLUDED_EXACT = {
    "q_history_mean",
    "q_history_scale",
    "z_history_mean",
    "z_history_scale",
    "q_target_mean",
    "q_target_scale",
    "rating.slope",
    "rating.intercept",
    "rating.available",
    "rating.pair_count",
    "rating.q_min_m3s",
    "rating.q_max_m3s",
}
_V10_TRANSFER_EXCLUDED_SUFFIX = "station_embedding.weight"


def is_v10_requested(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
) -> bool:
    del dataset_root
    raw = _load_yaml(config_path)
    return str(raw.get("model_version", "")).lower() == "v10"


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
    forcing_step = float(cfg.get("temporal", {}).get("forcing_step_seconds", float("nan")))
    if not math.isfinite(forcing_step) or forcing_step <= 0:
        raise ValueError("v10 temporal.forcing_step_seconds必须>0")
    cfg.setdefault("solver", {})["seconds_per_step"] = forcing_step
    _validate_v10_config(cfg)
    return cfg


def _validate_v10_config(cfg: dict[str, Any]) -> None:
    if str(cfg.get("model_version", "")).lower() != "v10":
        raise ValueError("v10配置必须显式model_version: v10")
    data = cfg.get("data", {})
    expected_data = {
        "mode": "hunan",
        "dataset_type": "event",
        "target_variable": "Q",
        "strict_validation": True,
        "use_observation_masks": True,
        "future_rainfall_mode": "observed_hindcast",
    }
    wrong_data = {k: data.get(k) for k, v in expected_data.items() if data.get(k) != v}
    if wrong_data:
        raise ValueError(f"v10 data contract不一致: {wrong_data}")

    temporal = cfg.get("temporal", {})
    history_duration = int(temporal.get("history_duration_seconds", 0))
    forecast_duration = int(temporal.get("forecast_duration_seconds", 0))
    forcing_step = int(temporal.get("forcing_step_seconds", 0))
    target_step = int(temporal.get("target_step_seconds", 0))
    if min(history_duration, forecast_duration, forcing_step, target_step) <= 0:
        raise ValueError("v10 temporal duration/step必须>0")
    if history_duration % forcing_step or forecast_duration % forcing_step:
        raise ValueError("v10 history/forecast duration必须整除forcing step")
    if target_step < forcing_step or target_step % forcing_step or forecast_duration % target_step:
        raise ValueError("v10 target cadence与forcing/forecast duration不兼容")
    history_steps = history_duration // forcing_step
    internal_steps = forecast_duration // forcing_step
    target_steps = forecast_duration // target_step
    if (history_steps, internal_steps, target_steps) != (24, 6, 6):
        raise ValueError("当前冻结Hunan v8 tensor只支持24 h warm-up和6 h forecast")
    if int(cfg.get("history_length", -1)) != history_steps or int(cfg.get("forecast_horizon", -1)) != target_steps:
        raise ValueError("v10 history_length/forecast_horizon与physical temporal contract不一致")
    for key, expected in V10_TIME_SEMANTICS.items():
        if temporal.get(key) != expected:
            raise ValueError(f"v10 temporal.{key}必须为{expected!r}")

    if int(cfg.get("node_static_dim", -1)) != 10 or int(cfg.get("edge_static_dim", -1)) != 2:
        raise ValueError("v10 Hunan固定node_static_dim=10、edge_static_dim=2")
    if cfg.get("runoff_mode") != "water_balance_lstm":
        raise ValueError("正式E4 v10固定water_balance_lstm runoff")
    if cfg.get("routing_mode") != "kinematic_wave_gnn":
        raise ValueError("正式E4 v10固定kinematic_wave_gnn routing")
    state = cfg.get("state_initialization", {})
    if state.get("enabled") is not True or state.get("mode") != "sequential_warmup":
        raise ValueError("v10必须sequential_warmup")
    warmup = cfg.get("warmup", {})
    if warmup.get("enabled") is not True or warmup.get("initial_state") != "static_prior":
        raise ValueError("v10 warm-up必须从static_prior开始")
    correction = cfg.get("state_correction", {})
    expected_correction = {
        "enabled": True,
        "mode": "v8_history_residual_after_warmup",
        "use_qz_history": True,
        "propagate_upstream": True,
        "correct_channel_storage": True,
        "additive_storage_from_q_residual": True,
    }
    wrong_correction = {
        k: correction.get(k) for k, v in expected_correction.items() if correction.get(k) != v
    }
    if wrong_correction:
        raise ValueError(f"v10 state assimilation不一致: {wrong_correction}")
    for key in ("hidden_residual_scale", "storage_log_scale"):
        value = float(correction.get(key, float("nan")))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"v10 state_correction.{key}必须位于(0,1]")
    if float(correction.get("max_additive_storage_hours", 0)) <= 0:
        raise ValueError("v10 max_additive_storage_hours必须>0")

    stage = cfg.get("stage_output", {})
    expected_stage = {
        "enabled": True,
        "method": "train_only_station_linear_rating",
        "require_all_outlet_stations": True,
        "origin_residual_correction": True,
        "q0_source": "observed_if_available_else_assimilated_model",
        "z0_source": "exact_forecast_origin_observation_only",
        "allow_backward_z_search": False,
    }
    wrong_stage = {k: stage.get(k) for k, v in expected_stage.items() if stage.get(k) != v}
    if wrong_stage:
        raise ValueError(f"v10 stage_output不一致: {wrong_stage}")
    if int(stage.get("min_unique_train_pairs", 0)) < 2:
        raise ValueError("v10 stage_output.min_unique_train_pairs必须>=2")
    if "z_head" in cfg:
        raise ValueError("v10配置不得包含独立z_head")

    loss = cfg.get("loss", {})
    expected_loss = {"mode": "q_only", "q_scale_mode": "per_station"}
    wrong_loss = {k: loss.get(k) for k, v in expected_loss.items() if loss.get(k) != v}
    if wrong_loss:
        raise ValueError(f"v10 loss contract不一致: {wrong_loss}")
    for key, expected in {
        "q_point_weight": 1.0,
        "q_peak_weight": 0.25,
        "q_volume_weight": 0.25,
    }.items():
        if not math.isclose(float(loss.get(key, float("nan"))), expected):
            raise ValueError(f"v10 loss.{key}必须为{expected}")
    floor = float(loss.get("q_scale_floor_m3s", float("nan")))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("v10 q_scale_floor_m3s必须>0")
    forbidden_z_loss = {
        "water_level_weight", "z_level_weight", "z_slope_weight",
        "z_target_mode", "delta_z_scale_mode", "delta_z_scale_floor_m",
        "qz_consistency_weight",
    }
    present = sorted(forbidden_z_loss & set(loss))
    if present:
        raise ValueError(f"v10 Q-only loss不得包含Z任务项: {present}")

    if cfg.get("validation_selection", {}).get("mode") != "val_loss":
        raise ValueError("v10 checkpoint selection固定Q-only val_loss")
    if bool(cfg.get("train_sampling", {}).get("enabled", False)):
        raise ValueError("v10固定full-pass TRAIN")
    if bool(cfg.get("hyperparameter_optimization", {}).get("enabled", False)):
        raise ValueError("v10正式训练禁止同时HPO")
    training = cfg.get("training", {})
    if bool(training.get("early_stopping", True)) or int(training.get("epochs", 0)) != 100:
        raise ValueError("v10固定完整训练100 epochs且不early-stop")
    if int(cfg.get("num_workers", -1)) != 0:
        raise ValueError("当前v10 NPZ cache固定num_workers=0")
    if int(cfg.get("batch_size", 0)) <= 0 or int(cfg.get("hidden_dim", 0)) <= 0:
        raise ValueError("v10 batch_size/hidden_dim必须>0")


def _validate_dataset_time_contract(cfg: dict[str, Any], dataset: HydrologicGraphV8Dataset) -> None:
    declared = dataset.contract.get("timestamp_semantics")
    if isinstance(declared, Mapping):
        wrong = {
            key: declared.get(key)
            for key, expected in V10_TIME_SEMANTICS.items()
            if declared.get(key) != expected
        }
        if wrong:
            raise ValueError(f"dataset timestamp semantics与v10不一致: {wrong}")
    cfg.setdefault("_runtime", {})["timestamp_semantics"] = {
        **V10_TIME_SEMANTICS,
        "dataset_declared": isinstance(declared, Mapping),
        "whole_hour_tensor_shift": False,
        "interpretation": (
            "rain/hydro labels are hourly bins; forecast origin is end of final history bin; "
            "targets advance by target_step_seconds"
        ),
    }


def _finalize_v10_runtime(cfg: dict[str, Any], dataset: HydrologicGraphV8Dataset) -> None:
    _attach_runtime(cfg, dataset)
    _validate_dataset_time_contract(cfg, dataset)
    runtime = cfg["_runtime"]
    q_floor = float(cfg["loss"]["q_scale_floor_m3s"])
    raw_q = [float(v) for v in runtime["v8_normalization"]["q_target_scale"]]
    applied_q = [max(v, q_floor) for v in raw_q]
    runtime["v8_normalization"]["q_target_scale"] = applied_q
    runtime["target_scale_audit"] = {
        "computed_from_split": "TRAIN",
        "supervised_target": "Q_ONLY",
        "q_scale_floor_m3s": q_floor,
        "stations": {
            station: {
                "raw_scale_m3s": raw_q[i],
                "applied_scale_m3s": applied_q[i],
                "floor_applied": raw_q[i] < q_floor,
            }
            for i, station in enumerate(runtime["v8_station_ids"])
        },
    }
    runtime["v10_rating_curves"] = fit_train_only_linear_ratings(
        cfg["data"]["dataset_root"],
        tuple(runtime["v8_station_ids"]),
        min_unique_pairs=int(cfg["stage_output"]["min_unique_train_pairs"]),
        require_all_outlet_stations=bool(cfg["stage_output"]["require_all_outlet_stations"]),
    )
    runtime["model_version"] = "v10"
    runtime["supervised_target"] = "Q_ONLY"
    runtime["stage_prediction"] = {
        "learned_head": False,
        "method": "station TRAIN-only linear rating + forecast-origin Z residual correction",
        "future_z_used": False,
    }
    runtime["temporal"] = dict(cfg["temporal"])


def setup_v10_training(
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
        root, cfg["data"]["train_split"], graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"], tensor_cache=tensor_cache,
    )
    validation_dataset = HydrologicGraphV8Dataset(
        root, cfg["data"]["validation_split"], graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"], tensor_cache=tensor_cache,
    )
    _ensure_split_compatibility(train_dataset, validation_dataset)
    _finalize_v10_runtime(cfg, train_dataset)
    train_loader = build_hydrologic_graph_v8_loader(
        train_dataset, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["TRAIN"],
    )
    validation_loader = build_hydrologic_graph_v8_loader(
        validation_dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["VALIDATION"],
    )
    model = HydrologicGraphV10Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_v10_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    split = str(split).upper()
    if split not in {"VALIDATION", "TEST"}:
        raise ValueError("v10 evaluation split必须为VALIDATION/TEST")
    cfg = _runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    dataset = HydrologicGraphV8Dataset(
        cfg["data"]["dataset_root"], split, graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
    )
    _finalize_v10_runtime(cfg, dataset)
    loader = build_hydrologic_graph_v8_loader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split],
    )
    model = HydrologicGraphV10Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device


def validate_v10_checkpoint_config(
    checkpoint: Mapping[str, Any], cfg: dict[str, Any], *, resume: bool = False
) -> None:
    saved = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    if not isinstance(saved, Mapping):
        raise ValueError("v10 checkpoint缺少训练config")
    for key in (
        "model_version", "runoff_mode", "routing_mode", "history_length",
        "forecast_horizon", "node_static_dim", "edge_static_dim", "hidden_dim",
        "temporal", "warmup", "state_initialization", "state_correction",
        "stage_output", "solver", "physical_bounds", "loss",
    ):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"checkpoint与当前v10配置不兼容: {key}")
    saved_runtime = saved.get("_runtime", {})
    current_runtime = cfg.get("_runtime", {})
    for label, key in (
        ("dataset", "data_contract"),
        ("rating", "v10_rating_curves"),
    ):
        old = saved_runtime.get(key, {})
        new = current_runtime.get(key, {})
        if old.get("artifact_sha256") != new.get("artifact_sha256"):
            raise ValueError(f"checkpoint与当前v10 {label} artifact不一致")
    if saved_runtime.get("v8_station_ids") != current_runtime.get("v8_station_ids"):
        raise ValueError("checkpoint与当前v10 station catalogue不一致")
    if resume and saved.get("data", {}).get("graph_id") != cfg.get("data", {}).get("graph_id"):
        raise ValueError("resume要求v10 data.graph_id完全一致")


def extract_v10_transferable_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude all Hunan station-specific normalization/rating/embedding state."""
    state = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("v10 transfer checkpoint缺少model state_dict")
    transferable: dict[str, Any] = {}
    for key, value in state.items():
        if key in _V10_TRANSFER_EXCLUDED_EXACT:
            continue
        if key.endswith(_V10_TRANSFER_EXCLUDED_SUFFIX):
            continue
        transferable[key] = value
    if not transferable:
        raise ValueError("v10 transferable state_dict为空")
    return transferable
