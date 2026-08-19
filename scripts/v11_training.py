"""Formal training/evaluation setup for V11 generalization-focused forecasting."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from data.device import resolve_device, seed_everything
from datasets.hydrologic_graph_v11 import (
    CONTRACT_NAME_V11,
    EVENT_PHASES,
    HydrologicGraphV11Dataset,
    build_hydrologic_graph_v11_loader,
)
from models.hydrologic_graph_v11 import HydrologicGraphV11Model
from scripts.v10_rating import fit_train_only_linear_ratings
from scripts.v8_training import (
    _SPLIT_SEED_OFFSET,
    _applied_stats,
    _ensure_split_compatibility,
    _load_yaml,
    _resolve_root,
)

V11_TIME_SEMANTICS = {
    "rain_interval_anchor": "interval_start",
    "hydro_hour_bin_anchor": "start_label_last_observation_within_bin",
    "forecast_origin_anchor": "end_of_last_history_bin",
}

_V11_TRANSFER_EXCLUDED_EXACT = {
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
_V11_TRANSFER_EXCLUDED_SUFFIX = "station_embedding.weight"


def is_v11_requested(
    config_path: str | Path,
    dataset_root: str | Path | None = None,
) -> bool:
    del dataset_root
    raw = _load_yaml(config_path)
    return str(raw.get("model_version", "")).lower() == "v11"


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
    forcing_step = float(
        cfg.get("temporal", {}).get("forcing_step_seconds", float("nan"))
    )
    if not math.isfinite(forcing_step) or forcing_step <= 0:
        raise ValueError("v11 temporal.forcing_step_seconds必须>0")
    cfg.setdefault("solver", {})["seconds_per_step"] = forcing_step
    _validate_v11_config(cfg)
    return cfg


def _validate_v11_config(cfg: dict[str, Any]) -> None:
    if str(cfg.get("model_version", "")).lower() != "v11":
        raise ValueError("v11配置必须显式model_version: v11")
    data = cfg.get("data", {})
    expected_data = {
        "mode": "hunan",
        "dataset_type": "event",
        "target_variable": "Q",
        "strict_validation": True,
        "use_observation_masks": True,
        "future_rainfall_mode": "observed_hindcast",
    }
    wrong = {
        key: data.get(key)
        for key, value in expected_data.items()
        if data.get(key) != value
    }
    if wrong:
        raise ValueError(f"v11 data contract不一致: {wrong}")

    temporal = cfg.get("temporal", {})
    forcing = int(temporal.get("forcing_step_seconds", 0))
    target = int(temporal.get("target_step_seconds", 0))
    rain_history = int(temporal.get("rain_history_duration_seconds", 0))
    observation_history = int(
        temporal.get("observation_history_duration_seconds", 0)
    )
    forecast = int(temporal.get("forecast_duration_seconds", 0))
    if min(forcing, target, rain_history, observation_history, forecast) <= 0:
        raise ValueError("v11 temporal duration/step必须>0")
    if any(value % forcing for value in (rain_history, observation_history, forecast)):
        raise ValueError("v11 history/forecast duration必须整除forcing step")
    if target < forcing or target % forcing or forecast % target:
        raise ValueError("v11 target cadence与forcing/forecast不兼容")
    if (
        rain_history // forcing,
        observation_history // forcing,
        forecast // target,
    ) != (72, 24, 6):
        raise ValueError("正式V11固定72h rain warm-up、24h Q/Z history、6h forecast")
    if int(cfg.get("history_length", -1)) != 72:
        raise ValueError("v11 history_length必须表示72h rainfall warm-up")
    if int(cfg.get("observation_history_length", -1)) != 24:
        raise ValueError("v11 observation_history_length必须为24")
    if int(cfg.get("forecast_horizon", -1)) != 6:
        raise ValueError("v11 forecast_horizon必须为6")
    for key, expected in V11_TIME_SEMANTICS.items():
        if temporal.get(key) != expected:
            raise ValueError(f"v11 temporal.{key}必须为{expected!r}")

    if int(cfg.get("node_static_dim", -1)) != 10 or int(
        cfg.get("edge_static_dim", -1)
    ) != 2:
        raise ValueError("v11 Hunan固定node_static_dim=10、edge_static_dim=2")
    if cfg.get("runoff_mode") != "water_balance_lstm":
        raise ValueError("正式E4 V11固定water_balance_lstm runoff")
    if cfg.get("routing_mode") != "kinematic_wave_gnn":
        raise ValueError("正式E4 V11固定kinematic_wave_gnn routing")

    warmup = cfg.get("warmup", {})
    if (
        warmup.get("enabled") is not True
        or warmup.get("initial_state") != "static_prior"
        or int(warmup.get("rainfall_history_hours", -1)) != 72
        or int(warmup.get("observation_history_hours", -1)) != 24
    ):
        raise ValueError("v11 warm-up必须为72h rain / 24h observation / static_prior")
    state = cfg.get("state_initialization", {})
    if state.get("enabled") is not True or state.get("mode") != "sequential_warmup":
        raise ValueError("v11必须sequential_warmup")
    correction = cfg.get("state_correction", {})
    expected_correction = {
        "enabled": True,
        "mode": "v8_history_residual_after_warmup",
        "use_qz_history": True,
        "qz_history_hours": 24,
        "propagate_upstream": True,
        "correct_channel_storage": True,
        "additive_storage_from_q_residual": True,
    }
    wrong = {
        key: correction.get(key)
        for key, value in expected_correction.items()
        if correction.get(key) != value
    }
    if wrong:
        raise ValueError(f"v11 state assimilation不一致: {wrong}")
    for key in ("hidden_residual_scale", "storage_log_scale"):
        value = float(correction.get(key, float("nan")))
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"v11 state_correction.{key}必须位于(0,1]")
    if float(correction.get("max_additive_storage_hours", 0)) <= 0:
        raise ValueError("v11 max_additive_storage_hours必须>0")

    stage = cfg.get("stage_output", {})
    expected_stage = {
        "enabled": True,
        "method": "train_only_station_linear_rating",
        "require_all_outlet_stations": True,
        "origin_residual_correction": True,
        "q0_source": "final_history_bin_observed_if_available_else_assimilated_model",
        "z0_source": "final_history_bin_observation_only",
        "allow_backward_z_search": False,
    }
    wrong = {
        key: stage.get(key)
        for key, value in expected_stage.items()
        if stage.get(key) != value
    }
    if wrong:
        raise ValueError(f"v11 stage_output不一致: {wrong}")
    if int(stage.get("min_unique_train_pairs", 0)) < 2:
        raise ValueError("v11 stage_output.min_unique_train_pairs必须>=2")
    if "z_head" in cfg:
        raise ValueError("v11正式配置不得包含独立z_head")

    loss = cfg.get("loss", {})
    if loss.get("mode") != "q_only_high_flow" or loss.get("q_scale_mode") != "per_station":
        raise ValueError("v11 loss必须为q_only_high_flow + per_station scale")
    expected_loss = {
        "q_point_weight": 1.0,
        "q_high_flow_weight": 0.25,
        "q_volume_weight": 0.25,
        "high_flow_lower_quantile": 0.80,
        "high_flow_upper_quantile": 0.99,
        "high_flow_max_multiplier": 3.0,
    }
    wrong = {
        key: loss.get(key)
        for key, value in expected_loss.items()
        if not math.isclose(float(loss.get(key, float("nan"))), value)
    }
    if wrong:
        raise ValueError(f"v11 loss固定设计不一致: {wrong}")
    if "q_peak_weight" in loss:
        raise ValueError("v11禁止sliding-window q_peak loss")
    q_floor = float(loss.get("q_scale_floor_m3s", float("nan")))
    if not math.isfinite(q_floor) or q_floor <= 0:
        raise ValueError("v11 q_scale_floor_m3s必须>0")

    sampling = cfg.get("train_sampling", {})
    expected_sampling = {
        "enabled": True,
        "mode": "event_balanced_phase_stratified",
        "origins_per_event": 8,
        "phase_quota": 2,
    }
    wrong = {
        key: sampling.get(key)
        for key, value in expected_sampling.items()
        if sampling.get(key) != value
    }
    if wrong:
        raise ValueError(f"v11 event-balanced sampling不一致: {wrong}")
    if tuple(str(value).upper() for value in sampling.get("phases", ())) != EVENT_PHASES:
        raise ValueError(f"v11 phases必须严格为{EVENT_PHASES}")

    if cfg.get("validation_selection", {}).get("mode") != "val_loss":
        raise ValueError("v11 checkpoint selection固定Q-only val_loss")
    if bool(cfg.get("hyperparameter_optimization", {}).get("enabled", False)):
        raise ValueError("v11正式训练禁止同时HPO")
    training = cfg.get("training", {})
    if bool(training.get("early_stopping", True)) or int(training.get("epochs", 0)) != 100:
        raise ValueError("v11固定100 epochs且不early-stop")
    if int(cfg.get("num_workers", -1)) != 0:
        raise ValueError("当前v11 NPZ cache固定num_workers=0")
    if int(cfg.get("batch_size", 0)) <= 0 or int(cfg.get("hidden_dim", 0)) <= 0:
        raise ValueError("v11 batch_size/hidden_dim必须>0")


def _attach_v11_runtime(
    cfg: dict[str, Any], dataset: HydrologicGraphV11Dataset
) -> None:
    contract = dataset.contract
    root = dataset.root
    if contract.get("contract") != CONTRACT_NAME_V11:
        raise ValueError("v11 dataset contract错误")
    for key, expected in {
        "graph_count": 33,
        "computational_node_count": 237,
        "edge_count": 204,
        "observation_station_count": 39,
        "rain_history_hours": 72,
        "observation_history_hours": 24,
        "forecast_hours": 6,
    }.items():
        if int(contract.get(key, -1)) != expected:
            raise ValueError(f"v11 dataset {key}应为{expected}")
    report = root / "BUILD_AND_QC.md"
    if not report.is_file() or "FINAL QC STATUS: PASS" not in report.read_text(
        encoding="utf-8"
    ):
        raise ValueError("v11正式数据没有FINAL QC STATUS: PASS")
    antecedent = contract.get("antecedent_rainfall", {})
    if (
        int(antecedent.get("hours", -1)) != 72
        or bool(antecedent.get("zero_padding_outside_valid_period", True))
        or antecedent.get("coverage_check")
        != "PER_NODE_FAIL_IF_REQUIRED_HOUR_OUTSIDE_VALID_PERIOD"
    ):
        raise ValueError("v11 antecedent rainfall coverage/zero semantics不符合设计")
    observation = contract.get("observation_history", {})
    if (
        int(observation.get("hours", -1)) != 24
        or bool(observation.get("extended_to_72h", True))
    ):
        raise ValueError("v11 Q/Z observation history被错误扩展")

    normal = contract.get("normalization")
    if (
        not isinstance(normal, Mapping)
        or normal.get("computed_from_split") != "TRAIN"
        or normal.get("fit_scope") != "TRAIN_SAMPLE_EXPOSURE_ONLY_V11_72H_RAIN"
    ):
        raise ValueError("v11 normalization必须为TRAIN-only 72h exposure")
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
        normal["delta_z_target_by_station"], station_ids, "delta_z_target_by_station"
    )
    static_names = tuple(contract["node_static_features"])
    node_mean = [float(normal["node_static"][name]["mean"]) for name in static_names]
    node_scale = [float(normal["node_static"][name]["scale"]) for name in static_names]
    if any(not math.isfinite(value) for value in (*node_mean, *node_scale)) or any(
        value <= 0 for value in node_scale
    ):
        raise ValueError("v11 node static normalization非法")
    rain_mean = float(normal["rain_mm"]["mean"])
    rain_scale = float(normal["rain_mm"]["scale"])
    if not math.isfinite(rain_mean) or not math.isfinite(rain_scale) or rain_scale <= 0:
        raise ValueError("v11 rain normalization非法")

    q_floor = float(cfg["loss"]["q_scale_floor_m3s"])
    applied_q = [max(float(value), q_floor) for value in qt_scale]
    high_flow = contract.get("high_flow_quantiles")
    if not isinstance(high_flow, Mapping):
        raise ValueError("v11 contract缺少TRAIN-only high-flow quantiles")
    if (
        high_flow.get("fit_split") != "TRAIN"
        or high_flow.get("deduplication_key")
        != "STATION_ID+PHYSICAL_TARGET_UNIX_HOUR"
        or int(high_flow.get("duplicate_value_conflict_count", -1)) != 0
        or high_flow.get("outlet_missing_threshold")
    ):
        raise ValueError("v11 high-flow quantile provenance/coverage不合法")
    if not math.isclose(float(high_flow.get("lower_quantile", -1)), 0.80) or not math.isclose(
        float(high_flow.get("upper_quantile", -1)), 0.99
    ):
        raise ValueError("v11 high-flow quantile与正式P80/P99不一致")

    contract_sha = hashlib.sha256(
        (root / "metadata/dataset_contract.json").read_bytes()
    ).hexdigest()
    runtime = cfg["_runtime"] = {
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
            "q_target_scale": applied_q,
            "dz_target_mean": dz_mean,
            "dz_target_scale": dz_scale,
        },
        "target_scale_audit": {
            "computed_from_split": "TRAIN",
            "supervised_target": "Q_ONLY",
            "q_scale_floor_m3s": q_floor,
            "stations": {
                station: {
                    "raw_scale_m3s": float(qt_scale[index]),
                    "applied_scale_m3s": applied_q[index],
                    "floor_applied": float(qt_scale[index]) < q_floor,
                }
                for index, station in enumerate(station_ids)
            },
        },
        "data_contract": {
            "format_version": 11,
            "contract": CONTRACT_NAME_V11,
            "artifact_sha256": contract_sha,
            "graph_count": 33,
            "computational_node_count": 237,
            "edge_count": 204,
            "observation_station_count": 39,
            "station_ids": list(station_ids),
        },
        "v11_high_flow_quantiles": dict(high_flow),
        "model_version": "v11",
        "supervised_target": "Q_ONLY",
        "temporal": dict(cfg["temporal"]),
        "history_design": {
            "rainfall_physical_warmup_hours": 72,
            "qz_assimilation_history_hours": 24,
            "qz_extended_to_72h": False,
            "antecedent_rain_zero_padded_outside_valid_period": False,
        },
    }

    # Rating calibration uses exactly the V10 method and TRAIN-only deduplicated
    # physical Q/Z pairs. Official V11 runtime stores its own artifact key; a
    # private V10 alias is supplied solely because the preserved V10 constructor
    # creates the identical non-trainable rating module.
    ratings = fit_train_only_linear_ratings(
        cfg["data"]["dataset_root"],
        tuple(station_ids),
        min_unique_pairs=int(cfg["stage_output"]["min_unique_train_pairs"]),
        require_all_outlet_stations=bool(
            cfg["stage_output"]["require_all_outlet_stations"]
        ),
    )
    runtime["v11_rating_curves"] = ratings
    runtime["v10_rating_curves"] = ratings
    runtime["stage_prediction"] = {
        "learned_head": False,
        "method": "station TRAIN-only linear rating + final-history-bin Z residual correction",
        "future_z_used": False,
        "q0_source": cfg["stage_output"]["q0_source"],
        "z0_source": cfg["stage_output"]["z0_source"],
    }
    runtime["timestamp_semantics"] = {
        **V11_TIME_SEMANTICS,
        "rain_history_hours": 72,
        "observation_history_hours": 24,
        "whole_hour_tensor_shift": False,
        "history_qz_anchor_semantics": (
            "final Q/Z history tensor value is the retained representative observation "
            "inside the final hourly bin, not guaranteed an exact end-of-bin instant"
        ),
    }


def _view_audit(dataset: HydrologicGraphV11Dataset) -> dict[str, Any]:
    return {
        "split": dataset.split,
        "require_q_supervision": dataset.require_q_supervision,
        "frozen_sample_count": dataset.frozen_sample_count_before_q_filter,
        "active_sample_count": len(dataset),
        "q_supervised_sample_count": dataset.q_supervised_sample_count,
        "q_filter_removed_count": dataset.q_filter_removed_count,
    }


def setup_v11_training(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    cfg = _runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    root = cfg["data"]["dataset_root"]
    tensor_cache: dict[str, dict[str, Any]] = {}
    train_dataset = HydrologicGraphV11Dataset(
        root,
        cfg["data"]["train_split"],
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        tensor_cache=tensor_cache,
        require_q_supervision=True,
    )
    validation_dataset = HydrologicGraphV11Dataset(
        root,
        cfg["data"]["validation_split"],
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        tensor_cache=tensor_cache,
        require_q_supervision=True,
    )
    _ensure_split_compatibility(train_dataset, validation_dataset)
    _attach_v11_runtime(cfg, train_dataset)
    cfg["_runtime"]["q_supervision_views"] = {
        "train": _view_audit(train_dataset),
        "validation": _view_audit(validation_dataset),
    }
    sampling = cfg["train_sampling"]
    train_loader = build_hydrologic_graph_v11_loader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["TRAIN"],
        event_balanced_train=True,
        origins_per_event=int(sampling["origins_per_event"]),
        phase_quota=int(sampling["phase_quota"]),
    )
    audit = getattr(train_loader.batch_sampler, "audit", None)
    if not callable(audit):
        raise ValueError("v11 TRAIN loader没有event-balanced sampler audit")
    cfg["_runtime"]["event_balanced_sampling"] = audit()
    validation_loader = build_hydrologic_graph_v11_loader(
        validation_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET["VALIDATION"],
        event_balanced_train=False,
    )
    model = HydrologicGraphV11Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_v11_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    split = str(split).upper()
    if split not in {"VALIDATION", "TEST"}:
        raise ValueError("v11 evaluation split必须为VALIDATION/TEST")
    cfg = _runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    dataset = HydrologicGraphV11Dataset(
        cfg["data"]["dataset_root"],
        split,
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        require_q_supervision=False,
    )
    _attach_v11_runtime(cfg, dataset)
    cfg["_runtime"]["evaluation_view"] = _view_audit(dataset)
    loader = build_hydrologic_graph_v11_loader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + _SPLIT_SEED_OFFSET[split],
        event_balanced_train=False,
    )
    model = HydrologicGraphV11Model(cfg)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device


def validate_v11_checkpoint_config(
    checkpoint: Mapping[str, Any], cfg: dict[str, Any], *, resume: bool = False
) -> None:
    saved = checkpoint.get("config") if isinstance(checkpoint, Mapping) else None
    if not isinstance(saved, Mapping):
        raise ValueError("v11 checkpoint缺少训练config")
    for key in (
        "model_version",
        "runoff_mode",
        "routing_mode",
        "history_length",
        "observation_history_length",
        "forecast_horizon",
        "node_static_dim",
        "edge_static_dim",
        "hidden_dim",
        "temporal",
        "warmup",
        "state_initialization",
        "state_correction",
        "stage_output",
        "solver",
        "physical_bounds",
        "loss",
        "train_sampling",
    ):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"checkpoint与当前v11配置不兼容: {key}")
    old_runtime = saved.get("_runtime", {})
    new_runtime = cfg.get("_runtime", {})
    for label, key in (
        ("dataset", "data_contract"),
        ("rating", "v11_rating_curves"),
        ("high_flow", "v11_high_flow_quantiles"),
    ):
        old = old_runtime.get(key, {})
        new = new_runtime.get(key, {})
        if old.get("artifact_sha256") != new.get("artifact_sha256"):
            raise ValueError(f"checkpoint与当前v11 {label} artifact不一致")
    if old_runtime.get("v8_station_ids") != new_runtime.get("v8_station_ids"):
        raise ValueError("checkpoint与当前v11 station catalogue不一致")
    if resume and saved.get("data", {}).get("graph_id") != cfg.get("data", {}).get("graph_id"):
        raise ValueError("resume要求v11 data.graph_id完全一致")


def extract_v11_transferable_state_dict(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    state = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("v11 transfer checkpoint缺少model state_dict")
    transferable: dict[str, Any] = {}
    for key, value in state.items():
        if key in _V11_TRANSFER_EXCLUDED_EXACT:
            continue
        if key.endswith(_V11_TRANSFER_EXCLUDED_SUFFIX):
            continue
        transferable[key] = value
    if not transferable:
        raise ValueError("v11 transferable state_dict为空")
    return transferable
