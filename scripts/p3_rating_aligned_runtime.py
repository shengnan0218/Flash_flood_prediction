"""Split-safe construction for the revised rating-aligned P3 experiment.

This module is intentionally separate from ``scripts.common`` so historical
P2/P3/E4 entry points retain their exact runtime semantics.  The revised P3
fits every new data-derived quantity on TRAIN only, then injects the frozen
result into VALIDATION/TEST:

* station-aware FLOW / relative-Z input normalization;
* station-specific Q->Z rating curves.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from data.device import resolve_device, seed_everything
from models import HybridFloodModel
from scripts.common import (
    _dataset_nodes,
    _ensure_matching_graph,
    _make_loader,
    _normalise_split,
    _runtime_config,
    _runtime_metadata,
)


def _enable_aligned_dataset(dataset: Any) -> None:
    if not hasattr(dataset, "dynamic_normalization_mode"):
        raise TypeError("rating-aligned P3要求continuous_sampling.HunanContinuousDataset")
    dataset.dynamic_normalization_mode = "train_aligned"
    if not bool(getattr(dataset, "normalize_dynamic", False)):
        raise ValueError("rating-aligned P3要求data.normalize_dynamic=true")


def _fit_train_facts(train_dataset: Any, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if getattr(train_dataset, "split", None) != "TRAIN":
        raise ValueError("P3 aligned input/rating只能从TRAIN dataset拟合")
    input_statistics = train_dataset.fit_aligned_input_statistics(
        q_scale_floor_m3s=float(cfg["loss"]["q_scale_floor_m3s"]),
        delta_z_scale_floor_m=float(cfg["loss"]["delta_z_scale_floor_m"]),
    )
    rating_statistics = train_dataset.train_rating_curve_statistics()
    return input_statistics, rating_statistics


def _inject_train_facts(dataset: Any, input_statistics: dict[str, Any]) -> None:
    _enable_aligned_dataset(dataset)
    dataset.set_aligned_input_statistics(input_statistics)


def _augment_runtime(
    cfg: dict[str, Any],
    input_statistics: dict[str, Any],
    rating_statistics: dict[str, Any],
) -> None:
    runtime = cfg.setdefault("_runtime", {})
    runtime["p3_rating_aligned"] = True
    runtime["input_normalization"] = input_statistics
    runtime["rating_curves"] = rating_statistics


def _configure_model(
    cfg: dict[str, Any],
    loader: Any,
    rating_statistics: dict[str, Any],
) -> HybridFloodModel:
    model = HybridFloodModel(cfg, _dataset_nodes(loader))
    model.configure_rating_curves(rating_statistics, loader.dataset.station_ids)
    return model


def setup_training_rating_aligned(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> tuple[dict[str, Any], HybridFloodModel, Any, Any, Any]:
    """Build revised-P3 TRAIN/VALIDATION with TRAIN-only normalization/rating."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    if not bool(cfg.get("state_initialization", {}).get("enabled", False)):
        raise ValueError("rating-aligned P3要求state_initialization.enabled=true")
    if cfg.get("loss", {}).get("z_target_mode") != "delta_from_t0":
        raise ValueError("rating-aligned P3要求z_target_mode=delta_from_t0")
    if cfg.get("data", {}).get("dataset_type") != "continuous":
        raise ValueError("rating-aligned P3要求continuous tensor contract")
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
    device = resolve_device(cfg["device"], cfg["gpu_id"])
    return cfg, model, train_loader, validation_loader, device


def setup_evaluation_rating_aligned(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
    sample_index_path: str | Path | None = None,
) -> tuple[dict[str, Any], HybridFloodModel, Any, Any]:
    """Build deterministic revised-P3 evaluation with TRAIN-only fitted facts."""
    cfg = _runtime_config(config_path, dataset_root, graph_id)
    if not bool(cfg.get("state_initialization", {}).get("enabled", False)):
        raise ValueError("rating-aligned P3要求state_initialization.enabled=true")
    if cfg.get("loss", {}).get("z_target_mode") != "delta_from_t0":
        raise ValueError("rating-aligned P3要求z_target_mode=delta_from_t0")
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
