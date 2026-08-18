"""Formal v9 runtime entrypoint.

The frozen v8 dataset/runtime contract remains owned by ``scripts.v9_training``.
This adapter swaps only the active model implementation to the mass-aware
forecast-origin assimilation variant, so train/evaluate/validate share one
unambiguous formal v9 path.
"""
from __future__ import annotations

from pathlib import Path

from models.hydrologic_graph_v9_assimilated import HydrologicGraphV9Model
from scripts import v9_training as _legacy_runtime

V9_TIME_SEMANTICS = _legacy_runtime.V9_TIME_SEMANTICS
is_v9_requested = _legacy_runtime.is_v9_requested
validate_v9_checkpoint_config = _legacy_runtime.validate_v9_checkpoint_config
extract_v9_transferable_state_dict = _legacy_runtime.extract_v9_transferable_state_dict


def _record_actual_hourly_semantics(cfg: dict) -> None:
    """Expose what the frozen builder actually did without inventing a 1 h shift."""
    runtime = cfg.setdefault("_runtime", {})
    semantics = runtime.setdefault("timestamp_semantics", {})
    semantics.update(
        {
            "hydro_hour_selection_actual": (
                "one representative observation retained within each labelled "
                "hour after completeness/fetch-time ordering"
            ),
            "whole_hour_tensor_shift_applied": False,
            "time_alignment_caveat": (
                "Q/Z hourly labels are bins, not guaranteed exact end-of-hour "
                "instantaneous states; current v9 therefore keeps the frozen "
                "same-bin/next-bin indexing and does not shift tensors by one hour"
            ),
        }
    )


def setup_v9_training(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    cfg, _old_model, train_loader, validation_loader, device = (
        _legacy_runtime.setup_v9_training(
            config_path,
            dataset_root=dataset_root,
            graph_id=graph_id,
        )
    )
    _record_actual_hourly_semantics(cfg)
    model = HydrologicGraphV9Model(cfg)
    return cfg, model, train_loader, validation_loader, device


def setup_v9_evaluation(
    config_path: str | Path,
    *,
    split: str = "TEST",
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
):
    cfg, _old_model, loader, device = _legacy_runtime.setup_v9_evaluation(
        config_path,
        split=split,
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
    _record_actual_hourly_semantics(cfg)
    model = HydrologicGraphV9Model(cfg)
    return cfg, model, loader, device
