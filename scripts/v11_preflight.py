"""Read-only global preflight for formal V11 implementation and data semantics."""
from __future__ import annotations

import gc
from typing import Any

from datasets.hydrologic_graph_v11 import (
    EVENT_PHASES,
    HydrologicGraphV11Dataset,
)
from scripts.v11_training import setup_v11_evaluation, setup_v11_training


def _summary(dataset: Any) -> dict[str, Any]:
    return {
        "samples": len(dataset),
        "events": len(getattr(dataset, "event_ids", ())),
        "graphs": list(getattr(dataset, "graph_ids", ())),
        "observation_station_count": len(getattr(dataset, "station_ids", ())),
    }


def _full_view(cfg: dict[str, Any], split: str) -> HydrologicGraphV11Dataset:
    return HydrologicGraphV11Dataset(
        cfg["data"]["dataset_root"],
        split,
        graph_id=cfg["data"].get("graph_id"),
        future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
        strict=cfg["data"]["strict_validation"],
        require_q_supervision=False,
    )


def validate_v11_dataset(
    config_path: str,
    *,
    dataset_root: str | None,
    graph_id: str | None,
) -> dict[str, Any]:
    cfg, model, train_loader, validation_loader, _device = setup_v11_training(
        config_path, dataset_root=dataset_root, graph_id=graph_id
    )
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset
    train_probe = next(iter(train_loader))
    validation_probe = next(iter(validation_loader))

    if train_dataset.station_ids != validation_dataset.station_ids:
        raise ValueError("v11 TRAIN/VALIDATION station catalogue不一致")
    if set(validation_dataset.graph_ids) - set(train_dataset.graph_ids):
        raise ValueError("v11 VALIDATION出现TRAIN学习域未见graph")

    frozen_train = _full_view(cfg, cfg["data"]["train_split"])
    frozen_validation = _full_view(cfg, cfg["data"]["validation_split"])
    if frozen_train.station_ids != frozen_validation.station_ids:
        raise ValueError("v11完整TRAIN/VALIDATION station catalogue不一致")
    overlap_tv = sorted(set(frozen_train.event_ids) & set(frozen_validation.event_ids))
    if overlap_tv:
        raise ValueError(f"v11冻结TRAIN/VALIDATION EVENT_ID泄漏: {overlap_tv[:20]}")

    runtime = cfg["_runtime"]
    views = runtime.get("q_supervision_views", {})
    for label, dataset in (("train", train_dataset), ("validation", validation_dataset)):
        view = views.get(label, {})
        if view.get("require_q_supervision") is not True:
            raise ValueError(f"v11 {label}学习视图未强制Q监督")
        frozen = int(view.get("frozen_sample_count", -1))
        active = int(view.get("active_sample_count", -1))
        removed = int(view.get("q_filter_removed_count", -1))
        if min(frozen, active, removed) < 0 or active + removed != frozen:
            raise ValueError(f"v11 {label} Q-only view计数不守恒: {view}")
        if active != len(dataset) or active != int(view.get("q_supervised_sample_count", -1)):
            raise ValueError(f"v11 {label} active view不是纯Q-supervised")

    # Audit the actual epoch-0 event-balanced sampling plan, not merely config.
    sampler = train_loader.batch_sampler
    audit_fn = getattr(sampler, "audit", None)
    if not callable(audit_fn):
        raise ValueError("v11 TRAIN没有EventBalancedV11BatchSampler audit")
    sampling = audit_fn()
    if (
        sampling.get("mode") != "EVENT_BALANCED_PHASE_STRATIFIED"
        or int(sampling.get("origins_per_event_max", -1)) != 8
        or int(sampling.get("phase_quota", -1)) != 2
        or tuple(sampling.get("phases", ())) != EVENT_PHASES
    ):
        raise ValueError(f"v11实际sampler设计错误: {sampling}")
    if int(sampling.get("event_count", -1)) != len(train_dataset.event_ids):
        raise ValueError("v11 sampler event_count与Q-supervised TRAIN event不一致")

    selected: list[int] = []
    for indices in sampler:
        if not indices:
            raise ValueError("v11 sampler产生空batch")
        graphs = {train_dataset.graph_id_for_index(index) for index in indices}
        if len(graphs) != 1:
            raise ValueError(f"v11 sampler batch混入多个graph: {graphs}")
        selected.extend(int(index) for index in indices)
    if len(selected) != int(sampling["selected_samples_per_epoch"]):
        raise ValueError("v11 sampler实际selected count与audit不一致")
    if len(selected) != len(set(selected)):
        raise ValueError("v11 sampler同一epoch重复选择同一forecast origin")
    selected_frame = train_dataset.samples.iloc[selected]
    counts = selected_frame.groupby("EVENT_ID").size()
    candidate_counts = train_dataset.samples.groupby("EVENT_ID").size()
    expected = candidate_counts.clip(upper=8)
    if not counts.reindex(expected.index, fill_value=0).equals(expected):
        raise ValueError("v11 sampler没有做到每event最多8且候选不足时全部使用")
    if set(selected_frame["EVENT_PHASE"].unique()) - set(EVENT_PHASES):
        raise ValueError("v11 sampler使用非法EVENT_PHASE")

    high_flow = runtime.get("v11_high_flow_quantiles", {})
    high_flow_audit = {
        key: high_flow.get(key)
        for key in (
            "method",
            "fit_split",
            "deduplication_key",
            "lower_quantile",
            "upper_quantile",
            "station_count",
            "available_station_count",
            "outlet_station_count",
            "outlet_missing_threshold",
            "unique_pair_count",
            "duplicate_value_conflict_count",
            "artifact_sha256",
        )
    }
    if (
        high_flow.get("fit_split") != "TRAIN"
        or high_flow.get("deduplication_key")
        != "STATION_ID+PHYSICAL_TARGET_UNIX_HOUR"
        or int(high_flow.get("duplicate_value_conflict_count", -1)) != 0
        or high_flow.get("outlet_missing_threshold")
    ):
        raise ValueError("v11 high-flow threshold不是无泄漏TRAIN-only artifact")

    ratings = runtime.get("v11_rating_curves", {})
    if (
        ratings.get("fit_split") != "TRAIN"
        or ratings.get("q0_required_for_curve_fit") is not False
        or int(ratings.get("duplicate_value_conflict_count", -1)) != 0
        or ratings.get("outlet_missing_curve")
    ):
        raise ValueError("v11 rating curve provenance/coverage错误")

    state_keys = tuple(model.state_dict().keys())
    forbidden = [
        key
        for key in state_keys
        if key.startswith("z_head.")
        or key.startswith("node_context_projection.")
        or key == "dz_target_scale"
    ]
    if forbidden:
        raise ValueError(f"v11仍包含独立Z-head状态: {forbidden[:10]}")
    parameter_names = tuple(name for name, _ in model.named_parameters())
    if any(name.startswith("rating.") for name in parameter_names):
        raise ValueError("v11 rating被错误注册为trainable parameter")
    if "q_peak_weight" in cfg["loss"] or "q_high_flow_weight" not in cfg["loss"]:
        raise ValueError("v11 loss仍含window peak或缺high-flow objective")

    for label, probe in (("TRAIN", train_probe), ("VALIDATION", validation_probe)):
        if probe.history_rain.shape[1] != 72:
            raise ValueError(f"{label}: v11 rainfall warm-up不是72h")
        if probe.q_history.shape[1] != 24 or probe.z_history.shape[1] != 24:
            raise ValueError(f"{label}: v11 Q/Z assimilation history不是24h")
        if probe.future_rain.shape[1] != 6 or probe.q_target.shape[1] != 6:
            raise ValueError(f"{label}: v11 forecast不是6h")
        if not probe.q_target_mask.any():
            raise ValueError(f"{label}: Q-supervised学习batch没有Q target")

    contract = train_dataset.contract
    antecedent = contract.get("antecedent_rainfall", {})
    if bool(antecedent.get("zero_padding_outside_valid_period", True)):
        raise ValueError("v11 antecedent rain错误允许valid period外zero padding")
    observation = contract.get("observation_history", {})
    if bool(observation.get("extended_to_72h", True)):
        raise ValueError("v11 Q/Z history错误扩到72h")

    train_station_ids = frozen_train.station_ids
    frozen_train_graphs = set(frozen_train.graph_ids)
    frozen_train_events = set(frozen_train.event_ids)
    frozen_validation_events = set(frozen_validation.event_ids)
    data_sha = runtime["data_contract"]["artifact_sha256"]
    rating_sha = ratings.get("artifact_sha256")
    high_flow_sha = high_flow.get("artifact_sha256")
    del model, train_loader, validation_loader
    gc.collect()

    test_cfg, test_model, test_loader, _test_device = setup_v11_evaluation(
        config_path,
        split="TEST",
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
    test_dataset = test_loader.dataset
    test_probe = next(iter(test_loader))
    overlaps = {
        "train_validation": overlap_tv,
        "train_test": sorted(frozen_train_events & set(test_dataset.event_ids)),
        "validation_test": sorted(frozen_validation_events & set(test_dataset.event_ids)),
    }
    if any(overlaps.values()):
        raise ValueError(f"v11冻结事件级split泄漏: {overlaps}")
    if test_dataset.station_ids != train_station_ids:
        raise ValueError("v11 TEST与TRAIN station catalogue不一致")
    if set(test_dataset.graph_ids) - frozen_train_graphs:
        raise ValueError("v11 TEST包含TRAIN未出现graph")
    test_runtime = test_cfg["_runtime"]
    if test_runtime["data_contract"]["artifact_sha256"] != data_sha:
        raise ValueError("v11 preflight期间dataset contract发生变化")
    if test_runtime["v11_rating_curves"].get("artifact_sha256") != rating_sha:
        raise ValueError("v11 TRAIN/TEST rating artifact不一致")
    if test_runtime["v11_high_flow_quantiles"].get("artifact_sha256") != high_flow_sha:
        raise ValueError("v11 TRAIN/TEST high-flow artifact不一致")
    evaluation_view = test_runtime.get("evaluation_view", {})
    if evaluation_view.get("require_q_supervision") is not False:
        raise ValueError("v11 final TEST必须保留完整冻结split")
    if int(evaluation_view.get("active_sample_count", -1)) != int(
        evaluation_view.get("frozen_sample_count", -2)
    ):
        raise ValueError("v11 final TEST错误过滤了无Q但可评价Z的窗口")
    if test_probe.history_rain.shape[1] != 72 or test_probe.q_history.shape[1] != 24:
        raise ValueError("v11 TEST probe 72h/24h history contract错误")

    result = {
        "dataset_root": cfg["data"]["dataset_root"],
        "model_version": "v11",
        "data_contract": runtime["data_contract"],
        "supervised_target": "Q_ONLY",
        "history_design": runtime["history_design"],
        "stage_prediction": runtime["stage_prediction"],
        "train": _summary(train_dataset),
        "validation": _summary(validation_dataset),
        "frozen_train": _summary(frozen_train),
        "frozen_validation": _summary(frozen_validation),
        "test": _summary(test_dataset),
        "q_supervision_views": views,
        "event_balanced_sampling": {
            **sampling,
            "epoch0_actual_selected_samples": len(selected),
            "epoch0_duplicate_sample_count": len(selected) - len(set(selected)),
            "epoch0_event_count": int(counts.size),
            "epoch0_max_origins_per_event": int(counts.max()),
            "epoch0_min_origins_per_event": int(counts.min()),
        },
        "high_flow_quantile_audit": high_flow_audit,
        "rating_curve_audit": {
            key: ratings.get(key)
            for key in (
                "method",
                "fit_split",
                "deduplication_key",
                "q0_required_for_curve_fit",
                "min_unique_train_pairs",
                "candidate_pair_occurrences",
                "unique_pair_count",
                "duplicate_value_conflict_count",
                "station_count",
                "available_station_count",
                "outlet_station_count",
                "outlet_missing_curve",
                "artifact_sha256",
            )
        },
        "model_contract_audit": {
            "independent_z_head_present": False,
            "rating_trainable_parameter_present": False,
            "rating_buffers_present": all(
                key in state_keys
                for key in ("rating.slope", "rating.intercept", "rating.available")
            ),
            "window_peak_loss_present": False,
            "high_flow_loss_present": True,
        },
        "target_scale_audit": runtime.get("target_scale_audit"),
        "timestamp_semantics": runtime.get("timestamp_semantics"),
        "evaluation_view": evaluation_view,
        "event_overlap": overlaps,
        "probe_batch_shapes": {
            "train_history_rain": list(train_probe.history_rain.shape),
            "train_q_history": list(train_probe.q_history.shape),
            "train_future_rain": list(train_probe.future_rain.shape),
            "train_q_target": list(train_probe.q_target.shape),
            "test_history_rain": list(test_probe.history_rain.shape),
            "test_q_history": list(test_probe.q_history.shape),
        },
        "status": "VALID",
    }
    del test_model, test_loader
    return result
