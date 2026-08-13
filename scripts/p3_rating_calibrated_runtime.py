"""Split-safe construction for P3 with a frozen TRAIN-calibrated Q->Z function.

This runtime preserves the existing P3 hydrology and Q0-informed state
initialization, but removes the trainable Z residual shortcut. Paired Q/Z data
are used only before training to calibrate the station observation function.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data.device import resolve_device, seed_everything
from models.p3_rating_calibrated import P3RatingCalibratedModel
from scripts.common import (
    _dataset_nodes,
    _ensure_matching_graph,
    _make_loader,
    _normalise_split,
    _runtime_config,
    _runtime_metadata,
)
from scripts.p3_rating_calibration import fit_train_monotone_rating_statistics
from scripts.p3_rating_aligned_runtime import (
    _enable_aligned_dataset,
    _inject_train_facts,
)


def _fit_train_facts(
    train_dataset: Any, cfg: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(train_dataset, "split", None) != "TRAIN":
        raise ValueError("P3 calibrated input/rating只能从TRAIN dataset拟合")
    input_statistics = train_dataset.fit_aligned_input_statistics(
        q_scale_floor_m3s=float(cfg["loss"]["q_scale_floor_m3s"]),
        delta_z_scale_floor_m=float(cfg["loss"]["delta_z_scale_floor_m"]),
    )
    rating_statistics = fit_train_monotone_rating_statistics(train_dataset)
    return input_statistics, rating_statistics


def _augment_runtime(
    cfg: dict[str, Any],
    input_statistics: dict[str, Any],
    rating_statistics: dict[str, Any],
) -> None:
    runtime = cfg.setdefault("_runtime", {})
    # Keep p3_rating_aligned=true so the already-audited Q0-informed state
    # initialization and legacy-consistency disabling remain unchanged.
    runtime["p3_rating_aligned"] = True
    runtime["p3_rating_calibrated"] = True
    runtime["z_observation_semantics"] = (
        "TRAIN-paired Q/Z calibrate a frozen monotone station rating function; "
        "no trainable neural Z residual exists; Z loss backpropagates through "
        "the frozen rating function into Q"
    )
    runtime["input_normalization"] = input_statistics
    runtime["rating_curves"] = rating_statistics


def _configure_model(
    cfg: dict[str, Any], loader: Any, rating_statistics: dict[str, Any]
) -> P3RatingCalibratedModel:
    model = P3RatingCalibratedModel(cfg, _dataset_nodes(loader))
    model.configure_rating_curves(rating_statistics, loader.dataset.station_ids)
    return model


def _validate_contract(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("state_initialization", {}).get("enabled", False)):
        raise ValueError("P3 calibrated rating要求state_initialization.enabled=true")
    if cfg.get("loss", {}).get("z_target_mode") != "delta_from_t0":
        raise ValueError("P3 calibrated rating要求z_target_mode=delta_from_t0")
    if cfg.get("data", {}).get("dataset_type") != "continuous":
        raise ValueError("P3 calibrated rating要求continuous tensor contract")
    if not bool(cfg.get("data", {}).get("normalize_dynamic", False)):
        raise ValueError("P3 calibrated rating要求normalize_dynamic=true")


def setup_training_rating_calibrated(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> tuple[dict[str, Any], P3RatingCalibratedModel, Any, Any, Any]:
    """Build TRAIN/VALIDATION with TRAIN-only normalization and rating calibration."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    _validate_contract(cfg)
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
    _enable_aligned_dataset(train_loader.dataset)
    _enable_aligned_dataset(validation_loader.dataset)
    input_statistics, rating_statistics = _fit_train_facts(train_loader.dataset, cfg)
    _inject_train_facts(train_loader.dataset, input_statistics)
    _inject_train_facts(validation_loader.dataset, input_statistics)

    nodes = _dataset_nodes(train_loader)
    if _dataset_nodes(validation_loader) != nodes:
        raise ValueError("TRAIN 与 VALIDATION 的节点数不一致")
    cfg["_runtime"] = _runtime_metadata(train_loader, cfg)
    _augment_runtime(cfg, input_statistics, rating_statistics)
    model = _configure_model(cfg, train_loader, rating_statistics)
    if model.independent_z_head is not None:
        raise RuntimeError("P3 calibrated模型错误地保留了neural Z residual head")
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_evaluation_rating_calibrated(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
    sample_index_path: str | Path | None = None,
) -> tuple[dict[str, Any], P3RatingCalibratedModel, Any, Any]:
    """Build deterministic evaluation reusing only TRAIN-fitted facts."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    _validate_contract(cfg)
    split = _normalise_split(split)
    seed_everything(cfg["seed"])
    dynamic_cache: dict = {}
    train_loader = _make_loader(
        cfg,
        cfg["data"]["train_split"],
        shuffle=False,
        dynamic_cache=dynamic_cache,
    )
    loader = _make_loader(
        cfg,
        split,
        shuffle=False,
        dynamic_cache=dynamic_cache,
        sample_index_path=sample_index_path,
    )
    _enable_aligned_dataset(train_loader.dataset)
    _enable_aligned_dataset(loader.dataset)
    input_statistics, rating_statistics = _fit_train_facts(train_loader.dataset, cfg)
    _inject_train_facts(train_loader.dataset, input_statistics)
    _inject_train_facts(loader.dataset, input_statistics)
    cfg["_runtime"] = _runtime_metadata(
        loader, cfg, q_scale_dataset=train_loader.dataset
    )
    _augment_runtime(cfg, input_statistics, rating_statistics)
    model = _configure_model(cfg, loader, rating_statistics)
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, loader, device
