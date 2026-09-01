"""Only supported Hunan training/evaluation setup."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from data.device import resolve_device, seed_everything
from datasets.hydrologic_graph import (
    CONTRACT_NAME,
    HydrologicGraphDataset,
    build_hydrologic_graph_loader,
)
from models.hydrologic_model import HydrologicModel
from scripts.rating import fit_train_only_linear_ratings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEED = {"TRAIN": 0, "VALIDATION": 10_000, "TEST": 20_000}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load_yaml(path: str | Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path in stack:
        raise ValueError("cyclic config inheritance")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("_base_", None)
    return raw if base is None else _merge(load_yaml(path.parent / base, (*stack, path)), raw)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _applied(mapping: Mapping[str, Any], stations: tuple[str, ...], label: str):
    means, scales = [], []
    for station in stations:
        record = mapping.get(station)
        if not isinstance(record, Mapping):
            raise ValueError(f"{label}: missing station {station}")
        mean, scale = float(record["applied_mean"]), float(record["applied_scale"])
        if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{label}/{station}: invalid normalization")
        means.append(mean); scales.append(scale)
    return means, scales


def _fit_log_rain(dataset: HydrologicGraphDataset) -> dict[str, float | int]:
    """Fit log1p-rain moments from TRAIN sample exposure only."""

    if dataset.split != "TRAIN":
        raise ValueError("log-rain normalization must be fitted from TRAIN")
    count = 0
    total = 0.0
    squared = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for relative_name, group in dataset.samples.groupby("TENSOR_FILE", sort=True):
        arrays = dataset._load_tensor_file(str(relative_name))
        rows = group["TENSOR_ROW"].to_numpy(dtype=np.int64)
        for key in ("history_rain", "future_rain"):
            values = np.log1p(
                np.asarray(arrays[key][rows], dtype=np.float64)
            )
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"TRAIN {key} is invalid for log1p normalization")
            count += int(values.size)
            total += float(values.sum(dtype=np.float64))
            squared += float(np.square(values).sum(dtype=np.float64))
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    if count < 2:
        raise ValueError("TRAIN log-rain normalization has insufficient values")
    mean = total / count
    variance = max(squared / count - mean * mean, 0.0)
    scale = math.sqrt(variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TRAIN log-rain normalization has zero/invalid scale")
    return {
        "transform": "log1p",
        "fit_split": "TRAIN",
        "count": count,
        "mean": mean,
        "scale": scale,
        "min": minimum,
        "max": maximum,
    }


def runtime_config(config_path: str | Path, *, dataset_root=None, graph_id=None):
    cfg = load_yaml(config_path)
    if cfg.get("model") != "hydrologic_lstm_graph_gate":
        raise ValueError("only model=hydrologic_lstm_graph_gate is supported")
    if cfg.get("runoff_mode") not in {"pure_lstm", "water_balance_lstm"}:
        raise ValueError("invalid runoff_mode")
    if cfg.get("routing_mode") not in {"pure_gnn", "muskingum_gnn"}:
        raise ValueError("invalid routing_mode")
    if "state_correction" in cfg or "observation_encoder" in cfg:
        raise ValueError("state correction and observation encoders are removed")
    cfg["data"]["dataset_root"] = str(_resolve(dataset_root or cfg["data"]["dataset_root"]))
    if graph_id is not None:
        cfg["data"]["graph_id"] = str(graph_id).strip()
    for key in ("checkpoint", "final_checkpoint", "log_csv"):
        cfg["training"][key] = str(_resolve(cfg["training"][key]))
    cfg["solver"]["seconds_per_step"] = float(cfg["temporal"]["forcing_step_seconds"])
    if (int(cfg["history_length"]), int(cfg["observation_history_length"]), int(cfg["forecast_horizon"])) != (72, 24, 6):
        raise ValueError("fixed design is 72 h rain, 24 h Q/Z, 6 h forecast")
    return cfg


def attach_runtime(cfg: dict[str, Any], dataset: HydrologicGraphDataset) -> None:
    if dataset.split != "TRAIN":
        raise ValueError("runtime artifacts must be attached from TRAIN")
    contract = dataset.contract
    if contract.get("contract") != CONTRACT_NAME:
        raise ValueError("wrong hydrologic dataset contract")
    normal = contract.get("normalization")
    if not isinstance(normal, Mapping) or normal.get("computed_from_split") != "TRAIN":
        raise ValueError("normalization must be TRAIN-only")
    stations = dataset.station_ids
    q_mean, q_scale = _applied(normal["q_target_by_station"], stations, "Q target")
    floor = float(cfg["loss"]["q_scale_floor_m3s"])
    q_scale = [max(value, floor) for value in q_scale]
    static_names = tuple(contract["node_static_features"])
    node_mean = [float(normal["node_static"][name]["mean"]) for name in static_names]
    node_scale = [float(normal["node_static"][name]["scale"]) for name in static_names]
    edge_names = tuple(contract["edge_static_features"])
    edge_mean = [float(normal["edge_static"][name]["mean"]) for name in edge_names]
    edge_scale = [float(normal["edge_static"][name]["scale"]) for name in edge_names]
    if any(not math.isfinite(value) for value in (*node_mean, *edge_mean)):
        raise ValueError("node/edge normalization means must be finite")
    if any(not math.isfinite(value) or value <= 0 for value in (*node_scale, *edge_scale)):
        raise ValueError("node/edge normalization scales must be finite and positive")
    log_rain = _fit_log_rain(dataset)
    high_flow = contract.get("high_flow_quantiles")
    if not isinstance(high_flow, Mapping) or high_flow.get("fit_split") != "TRAIN":
        raise ValueError("TRAIN-only high-flow thresholds missing")
    ratings = fit_train_only_linear_ratings(
        cfg["data"]["dataset_root"], stations,
        min_unique_pairs=int(cfg["stage_output"]["min_unique_train_pairs"]),
        require_all_outlet_stations=True,
    )
    contract_path = dataset.root / "metadata/dataset_contract.json"
    cfg["_runtime"] = {
        "station_ids": list(stations),
        "normalization": {
            "log_rain_mean": float(log_rain["mean"]),
            "log_rain_scale": float(log_rain["scale"]),
            "node_static_mean": node_mean,
            "node_static_scale": node_scale,
            "edge_static_mean": edge_mean,
            "edge_static_scale": edge_scale,
            "q_target_mean": q_mean,
            "q_target_scale": q_scale,
        },
        "log_rain_normalization": log_rain,
        "high_flow_quantiles": dict(high_flow),
        "rating_curves": ratings,
        "data_contract": {
            "contract": CONTRACT_NAME,
            "artifact_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "station_ids": list(stations),
        },
        "supervised_target": "Q_ONLY",
        "architecture": "runoff LSTM -> graph routing -> Q0 reliability gate",
        "state_correction": False,
    }


def _dataset(cfg, split, *, cache=None, require_q_supervision=True):
    return HydrologicGraphDataset(
        cfg["data"]["dataset_root"], split,
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        tensor_cache=cache,
        require_q_supervision=require_q_supervision,
    )


def _loader(cfg, dataset, split, *, balanced=False):
    sampling = cfg["train_sampling"]
    return build_hydrologic_graph_loader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=split == "TRAIN",
        num_workers=int(cfg["num_workers"]), pin_memory=bool(cfg["pin_memory"]),
        seed=int(cfg["seed"]) + SPLIT_SEED[split], event_balanced_train=balanced,
        origins_per_event=int(sampling["origins_per_event"]),
        phase_quota=int(sampling["phase_quota"]),
    )


def setup_training(config_path, *, dataset_root=None, graph_id=None):
    cfg = runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    cache = {}
    train = _dataset(cfg, "TRAIN", cache=cache)
    validation = _dataset(cfg, "VALIDATION", cache=cache)
    if set(train.event_ids) & set(validation.event_ids):
        raise ValueError("TRAIN/VALIDATION event leakage")
    attach_runtime(cfg, train)
    model = HydrologicModel(cfg)
    return cfg, model, _loader(cfg, train, "TRAIN", balanced=True), _loader(cfg, validation, "VALIDATION"), resolve_device(cfg["device"], cfg["gpu_id"])


def setup_evaluation(config_path, *, split="TEST", dataset_root=None, graph_id=None):
    split = str(split).upper()
    if split not in {"VALIDATION", "TEST"}:
        raise ValueError("split must be VALIDATION or TEST")
    cfg = runtime_config(config_path, dataset_root=dataset_root, graph_id=graph_id)
    seed_everything(int(cfg["seed"]))
    cache = {}
    train_reference = _dataset(cfg, "TRAIN", cache=cache)
    dataset = _dataset(cfg, split, cache=cache, require_q_supervision=False)
    attach_runtime(cfg, train_reference)
    if dataset.station_ids != train_reference.station_ids:
        raise ValueError("evaluation station catalogue differs from TRAIN")
    return cfg, HydrologicModel(cfg), _loader(cfg, dataset, split), resolve_device(cfg["device"], cfg["gpu_id"])


def validate_checkpoint(checkpoint: Mapping[str, Any], cfg: Mapping[str, Any], *, resume=False):
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint config missing")
    for key in (
        "model", "runoff_mode", "routing_mode", "history_length",
        "forecast_horizon", "node_static_dim", "edge_static_dim",
        "input_preprocessing", "water_balance", "muskingum_routing",
        "output_head", "loss", "validation_selection",
    ):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"checkpoint incompatible: {key}")
    if saved.get("_runtime", {}).get("data_contract", {}).get("artifact_sha256") != cfg.get("_runtime", {}).get("data_contract", {}).get("artifact_sha256"):
        raise ValueError("checkpoint dataset contract mismatch")
