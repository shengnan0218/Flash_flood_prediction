"""Read-only preflight validation for formal Hunan V10/V9/V8 datasets."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from datasets.hydrologic_graph_v10 import HydrologicGraphV10Dataset
from scripts.v10_training import (
    is_v10_requested,
    setup_v10_evaluation,
    setup_v10_training,
)
from scripts.v8_training import is_v8_requested, setup_v8_evaluation, setup_v8_training
from scripts.v9_active import is_v9_requested, setup_v9_evaluation, setup_v9_training


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    graph_ids = list(getattr(dataset, "graph_ids", ()))
    event_ids = list(getattr(dataset, "event_ids", ()))
    result: dict[str, Any] = {
        "samples": len(dataset),
        "events": len(event_ids),
        "graphs": graph_ids,
    }
    station_ids = getattr(dataset, "station_ids", None)
    if station_ids is not None:
        result["observation_station_count"] = len(station_ids)
    return result


def _version_setup(version: str):
    if version == "v10":
        return setup_v10_training, setup_v10_evaluation
    if version == "v9":
        return setup_v9_training, setup_v9_evaluation
    if version == "v8":
        return setup_v8_training, setup_v8_evaluation
    raise ValueError(f"未知formal hydrologic graph版本: {version}")


def _validate_hydrologic_graph_dataset(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None,
    graph_id: str | None,
    version: str,
) -> dict[str, Any]:
    setup_train, setup_eval = _version_setup(version)
    cfg, model, train_loader, validation_loader, _device = setup_train(
        config_path, dataset_root=dataset_root, graph_id=graph_id
    )
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset

    # Materialise Q-supervised probes now, before releasing the shared TRAIN NPZ
    # cache.  This avoids a second full setup/rating scan later in preflight.
    train_probe = next(iter(train_loader))
    validation_probe = next(iter(validation_loader))

    if train_dataset.station_ids != validation_dataset.station_ids:
        raise ValueError("TRAIN/VALIDATION station catalogue不一致")
    unseen_validation_learning = set(validation_dataset.graph_ids) - set(
        train_dataset.graph_ids
    )
    if unseen_validation_learning:
        raise ValueError(
            "V10/V9/V8 VALIDATION包含TRAIN学习域未出现graph: "
            f"{sorted(unseen_validation_learning)}"
        )

    # V10 learning views are Q-filtered, so frozen split leakage must be checked
    # on separate unfiltered read-only views.  V8/V9 datasets are already full.
    if version == "v10":
        frozen_train = HydrologicGraphV10Dataset(
            cfg["data"]["dataset_root"],
            cfg["data"]["train_split"],
            graph_id=cfg["data"].get("graph_id"),
            future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
            strict=cfg["data"]["strict_validation"],
            require_q_supervision=False,
        )
        frozen_validation = HydrologicGraphV10Dataset(
            cfg["data"]["dataset_root"],
            cfg["data"]["validation_split"],
            graph_id=cfg["data"].get("graph_id"),
            future_rainfall_mode=cfg["data"]["future_rainfall_mode"],
            strict=cfg["data"]["strict_validation"],
            require_q_supervision=False,
        )
    else:
        frozen_train = train_dataset
        frozen_validation = validation_dataset

    frozen_train_events = set(frozen_train.event_ids)
    frozen_validation_events = set(frozen_validation.event_ids)
    overlap_tv = sorted(frozen_train_events & frozen_validation_events)
    if overlap_tv:
        raise ValueError(f"冻结TRAIN/VALIDATION EVENT_ID泄漏: {overlap_tv[:20]}")
    if frozen_train.station_ids != frozen_validation.station_ids:
        raise ValueError("冻结TRAIN/VALIDATION station catalogue不一致")
    unseen_validation_frozen = set(frozen_validation.graph_ids) - set(
        frozen_train.graph_ids
    )
    if unseen_validation_frozen:
        raise ValueError(
            f"冻结VALIDATION包含TRAIN未出现graph: {sorted(unseen_validation_frozen)}"
        )

    runtime = cfg.get("_runtime", {})
    result: dict[str, Any] = {
        "dataset_root": cfg["data"]["dataset_root"],
        "model_version": version,
        "data_contract": dict(runtime.get("data_contract", {})),
        "dataset_type": cfg["data"].get("dataset_type"),
        "supervised_target": runtime.get(
            "supervised_target", cfg["data"].get("target_variable")
        ),
        "stage_prediction": runtime.get("stage_prediction"),
        "history_length": int(cfg["history_length"]),
        "forecast_horizon": int(cfg["forecast_horizon"]),
        "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
        "train": _dataset_summary(train_dataset),
        "validation": _dataset_summary(validation_dataset),
        "frozen_train": _dataset_summary(frozen_train),
        "frozen_validation": _dataset_summary(frozen_validation),
        "target_scale_audit": runtime.get("target_scale_audit"),
        "timestamp_semantics": runtime.get("timestamp_semantics"),
    }
    if version == "v10":
        # Make the distinction between the frozen V8 split and the Q-only
        # learning/selection view explicit.  No sample is physically deleted.
        result["q_supervision_views"] = runtime.get("q_supervision_views")
        views = result["q_supervision_views"] or {}
        for label in ("train", "validation"):
            view = views.get(label, {})
            if view.get("require_q_supervision") is not True:
                raise ValueError(f"v10 {label}学习视图没有强制Q监督")
            frozen = int(view.get("frozen_sample_count", -1))
            active = int(view.get("active_sample_count", -1))
            removed = int(view.get("q_filter_removed_count", -1))
            if min(frozen, active, removed) < 0 or active + removed != frozen:
                raise ValueError(f"v10 {label} Q-only样本视图计数不守恒: {view}")
            if active != int(view.get("q_supervised_sample_count", -1)):
                raise ValueError(f"v10 {label} active样本并非全部Q-supervised")
        if int(views["train"]["frozen_sample_count"]) != len(frozen_train):
            raise ValueError("v10 TRAIN Q-view frozen计数与完整冻结视图不一致")
        if int(views["validation"]["frozen_sample_count"]) != len(frozen_validation):
            raise ValueError("v10 VALIDATION Q-view frozen计数与完整冻结视图不一致")

        ratings = runtime.get("v10_rating_curves", {})
        result["rating_curve_audit"] = {
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
        }
        if ratings.get("fit_split") != "TRAIN":
            raise ValueError("v10 rating curve不是TRAIN-only")
        if ratings.get("q0_required_for_curve_fit") is not False:
            raise ValueError("v10 rating curve拟合不应错误依赖Q0")
        if ratings.get("duplicate_value_conflict_count") != 0:
            raise ValueError("v10 rating TRAIN重叠窗口有冲突")
        if ratings.get("outlet_missing_curve"):
            raise ValueError("v10存在无rating curve的outlet")
        state_keys = tuple(model.state_dict().keys())
        forbidden = [
            key
            for key in state_keys
            if key.startswith("z_head.")
            or key.startswith("node_context_projection.")
            or key == "dz_target_scale"
        ]
        if forbidden:
            raise ValueError(f"v10仍包含独立Z-head状态: {forbidden[:10]}")
        parameter_names = tuple(name for name, _ in model.named_parameters())
        if any(name.startswith("rating.") for name in parameter_names):
            raise ValueError("v10 rating curve被错误注册为可训练参数")
        result["model_contract_audit"] = {
            "independent_z_head_present": False,
            "rating_trainable_parameter_present": False,
            "rating_buffers_present": all(
                key in state_keys
                for key in ("rating.slope", "rating.intercept", "rating.available")
            ),
        }

    train_station_ids = frozen_train.station_ids
    frozen_train_graphs = set(frozen_train.graph_ids)
    del model, train_loader, validation_loader
    gc.collect()

    test_cfg, test_model, test_loader, _test_device = setup_eval(
        config_path, split="TEST", dataset_root=dataset_root, graph_id=graph_id
    )
    test_dataset = test_loader.dataset
    test_probe = next(iter(test_loader))
    frozen_test_events = set(test_dataset.event_ids)
    overlaps = {
        "train_validation": overlap_tv,
        "train_test": sorted(frozen_train_events & frozen_test_events),
        "validation_test": sorted(frozen_validation_events & frozen_test_events),
    }
    if any(overlaps.values()):
        raise ValueError(f"冻结事件级split泄漏: {overlaps}")
    if test_dataset.station_ids != train_station_ids:
        raise ValueError("冻结TEST与TRAIN station catalogue不一致")
    unseen_test = set(test_dataset.graph_ids) - frozen_train_graphs
    if unseen_test:
        raise ValueError(f"冻结TEST包含TRAIN未出现graph: {sorted(unseen_test)}")
    if (
        test_cfg.get("_runtime", {})
        .get("data_contract", {})
        .get("artifact_sha256")
        != result["data_contract"].get("artifact_sha256")
    ):
        raise ValueError("preflight期间dataset_contract.json发生变化")
    if version == "v10":
        if (
            test_cfg.get("_runtime", {})
            .get("v10_rating_curves", {})
            .get("artifact_sha256")
            != cfg.get("_runtime", {})
            .get("v10_rating_curves", {})
            .get("artifact_sha256")
        ):
            raise ValueError("TRAIN/TEST setup得到不同rating curve artifact")
        evaluation_view = test_cfg.get("_runtime", {}).get("evaluation_view", {})
        if evaluation_view.get("require_q_supervision") is not False:
            raise ValueError("v10 final TEST评价必须保留完整冻结split")
        if int(evaluation_view.get("active_sample_count", -1)) != int(
            evaluation_view.get("frozen_sample_count", -2)
        ):
            raise ValueError("v10 final TEST评价错误过滤了冻结样本")
        result["evaluation_view"] = evaluation_view

    for label, probe in (
        ("TRAIN", train_probe),
        ("VALIDATION", validation_probe),
        ("TEST", test_probe),
    ):
        if probe.history_rain.ndim != 4 or probe.future_rain.ndim != 4:
            raise ValueError(f"{label}: rain tensor维度错误")
        if probe.q_target.shape != probe.z_target.shape:
            raise ValueError(f"{label}: frozen Q/Z target shape不一致")
        if probe.q_target_mask.shape != probe.q_target.shape:
            raise ValueError(f"{label}: Q target mask shape错误")
        if probe.z_target_mask.shape != probe.z_target.shape:
            raise ValueError(f"{label}: Z evaluation target mask shape错误")
    if not train_probe.q_target_mask.any() or not validation_probe.q_target_mask.any():
        raise ValueError("v10 TRAIN/VALIDATION probe不应出现无Q监督batch")

    result["test"] = _dataset_summary(test_dataset)
    result["event_overlap"] = overlaps
    result["probe_batch_shapes"] = {
        "train_history_rain": list(train_probe.history_rain.shape),
        "train_future_rain": list(train_probe.future_rain.shape),
        "train_q_target": list(train_probe.q_target.shape),
    }
    result["status"] = "VALID"
    del test_model, test_loader
    return result


def validate_dataset(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
    qc_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if qc_output_dir is not None:
        raise ValueError("V10/V9/V8使用冻结hydrologic-graph QC，不写legacy qc_output_dir")
    if is_v10_requested(config_path, dataset_root):
        version = "v10"
    elif is_v9_requested(config_path, dataset_root):
        version = "v9"
    elif is_v8_requested(config_path, dataset_root):
        version = "v8"
    else:
        raise ValueError("正式preflight只保留V10/V9/V8；legacy/P2/P3已退役")
    return _validate_hydrologic_graph_dataset(
        config_path, dataset_root=dataset_root, graph_id=graph_id, version=version
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="只读校验湖南正式V10/V9/V8 model dataset；默认V10"
    )
    parser.add_argument(
        "--config", default=str(root / "configs" / "hunan_e4_v10.yaml")
    )
    parser.add_argument("--dataset-root", help="覆盖冻结model dataset根目录")
    parser.add_argument("--graph-id", help="可选：只校验一个GRAPH_ID")
    parser.add_argument("--output", help="可选JSON报告路径")
    args = parser.parse_args()
    result = validate_dataset(
        args.config, dataset_root=args.dataset_root, graph_id=args.graph_id
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
