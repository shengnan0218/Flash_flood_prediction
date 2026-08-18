"""Read-only preflight validation for formal Hunan model datasets."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from audit_model_dataset import audit as audit_structural_dataset
from dataset_quality import build_dataset_quality_audit, enforce_strict_quality
from scripts.common import setup_evaluation, setup_training
from scripts.v8_training import (
    is_v8_requested,
    setup_v8_evaluation,
    setup_v8_training,
)
from scripts.v9_training import (
    is_v9_requested,
    setup_v9_evaluation,
    setup_v9_training,
)


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    graph_ids = list(getattr(dataset, "graph_ids", ()))
    event_ids = list(getattr(dataset, "event_ids", ()))
    result: dict[str, Any] = {
        "samples": len(dataset),
        "events": len(event_ids),
        "graphs": graph_ids,
    }
    graph_node_counts = getattr(dataset, "graph_node_counts", None)
    if graph_node_counts is not None:
        result["graph_node_counts"] = dict(graph_node_counts)
    station_ids = getattr(dataset, "station_ids", None)
    if station_ids is not None:
        result["observation_station_count"] = len(station_ids)
    return result


def _validate_hydrologic_graph_dataset(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None,
    graph_id: str | None,
    version: str,
) -> dict[str, Any]:
    if version == "v9":
        setup_train = setup_v9_training
        setup_eval = setup_v9_evaluation
    elif version == "v8":
        setup_train = setup_v8_training
        setup_eval = setup_v8_evaluation
    else:  # pragma: no cover
        raise ValueError(f"未知hydrologic graph版本: {version}")

    cfg, model, train_loader, validation_loader, _device = setup_train(
        config_path,
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset
    train_events = set(train_dataset.event_ids)
    validation_events = set(validation_dataset.event_ids)
    overlap_tv = sorted(train_events & validation_events)
    if overlap_tv:
        raise ValueError(f"TRAIN/VALIDATION EVENT_ID泄漏: {overlap_tv[:20]}")
    if train_dataset.station_ids != validation_dataset.station_ids:
        raise ValueError("TRAIN/VALIDATION station catalogue不一致")
    unseen_validation = set(validation_dataset.graph_ids) - set(train_dataset.graph_ids)
    if unseen_validation:
        raise ValueError(
            f"VALIDATION包含TRAIN未出现graph: {sorted(unseen_validation)}"
        )

    runtime_contract = dict(cfg.get("_runtime", {}).get("data_contract", {}))
    result: dict[str, Any] = {
        "dataset_root": cfg["data"]["dataset_root"],
        "model_version": version,
        "data_contract": runtime_contract,
        "dataset_type": cfg["data"].get("dataset_type"),
        "history_length": int(cfg["history_length"]),
        "forecast_horizon": int(cfg["forecast_horizon"]),
        "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
        "train": _dataset_summary(train_dataset),
        "validation": _dataset_summary(validation_dataset),
        "target_scale_audit": cfg.get("_runtime", {}).get("target_scale_audit"),
        "timestamp_semantics": cfg.get("_runtime", {}).get("timestamp_semantics"),
    }

    # Release TRAIN/VALIDATION references before opening TEST NPZ views.
    del model, train_loader, validation_loader
    gc.collect()

    test_cfg, test_model, test_loader, _test_device = setup_eval(
        config_path,
        split="TEST",
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
    test_dataset = test_loader.dataset
    test_events = set(test_dataset.event_ids)
    overlaps = {
        "train_validation": overlap_tv,
        "train_test": sorted(train_events & test_events),
        "validation_test": sorted(validation_events & test_events),
    }
    if any(overlaps.values()):
        raise ValueError(f"事件级split泄漏: {overlaps}")
    if test_dataset.station_ids != train_dataset.station_ids:
        raise ValueError("TEST与TRAIN station catalogue不一致")
    unseen_test = set(test_dataset.graph_ids) - set(train_dataset.graph_ids)
    if unseen_test:
        raise ValueError(f"TEST包含TRAIN未出现graph: {sorted(unseen_test)}")
    test_contract = test_cfg.get("_runtime", {}).get("data_contract", {})
    if test_contract.get("artifact_sha256") != runtime_contract.get("artifact_sha256"):
        raise ValueError("preflight期间dataset_contract.json发生变化")

    # Materialise one same-graph batch per split so NPZ tensor keys, masks and
    # collate contracts are actually exercised without starting training.
    train_probe = next(iter(setup_train(
        config_path,
        dataset_root=dataset_root,
        graph_id=graph_id,
    )[2]))
    validation_probe = next(iter(setup_train(
        config_path,
        dataset_root=dataset_root,
        graph_id=graph_id,
    )[3]))
    test_probe = next(iter(test_loader))
    for label, probe in (
        ("TRAIN", train_probe),
        ("VALIDATION", validation_probe),
        ("TEST", test_probe),
    ):
        if probe.history_rain.ndim != 4 or probe.future_rain.ndim != 4:
            raise ValueError(f"{label}: hydrologic graph batch rain维度错误")
        if probe.q_target.shape != probe.z_target.shape:
            raise ValueError(f"{label}: Q/Z target shape不一致")
        if probe.q_target_mask.shape != probe.q_target.shape:
            raise ValueError(f"{label}: Q target mask shape错误")
        if probe.z_target_mask.shape != probe.z_target.shape:
            raise ValueError(f"{label}: Z target mask shape错误")

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
    """Validate every split and return a JSON-safe dataset summary."""
    if is_v9_requested(config_path, dataset_root):
        if qc_output_dir is not None:
            raise ValueError("v9使用冻结hydrologic-graph QC，不写legacy qc_output_dir")
        return _validate_hydrologic_graph_dataset(
            config_path,
            dataset_root=dataset_root,
            graph_id=graph_id,
            version="v9",
        )
    if is_v8_requested(config_path, dataset_root):
        if qc_output_dir is not None:
            raise ValueError("v8使用冻结hydrologic-graph QC，不写legacy qc_output_dir")
        return _validate_hydrologic_graph_dataset(
            config_path,
            dataset_root=dataset_root,
            graph_id=graph_id,
            version="v8",
        )

    cfg, model, train_loader, validation_loader, _device = setup_training(
        config_path, dataset_root=dataset_root, graph_id=graph_id
    )
    train_dataset = train_loader.dataset
    validation_dataset = validation_loader.dataset
    train_events = set(train_dataset.event_ids)
    validation_events = set(validation_dataset.event_ids)
    train_contract = dict(cfg["_runtime"]["data_contract"])
    train_graphs = set(train_contract["graph_ids"])
    train_stations = tuple(train_contract["station_ids"])
    target_mapping = {
        key: sorted(value)
        for key, value in train_dataset.target_variables_by_graph.items()
    }
    resolved_root = Path(cfg["data"]["dataset_root"]).expanduser().resolve()
    quality_audit = build_dataset_quality_audit(resolved_root)
    written_quality_files = (
        quality_audit.write(qc_output_dir) if qc_output_dir is not None else {}
    )
    structural_audit = audit_structural_dataset(resolved_root)
    normalization_audit = structural_audit["normalization"]
    if normalization_audit.get("computed_from_split") != "TRAIN":
        raise ValueError("normalization_stats.json必须声明computed_from_split=TRAIN")
    if not normalization_audit.get("matches"):
        raise ValueError(
            "normalization_stats.json与TRAIN输入窗口重算结果不一致: "
            f"{normalization_audit.get('mismatches', [])[:10]}"
        )
    if bool(cfg.get("data", {}).get("strict_validation", True)):
        enforce_strict_quality(quality_audit)
    result: dict[str, Any] = {
        "dataset_root": cfg["data"]["dataset_root"],
        "history_hours": cfg["history_length"],
        "forecast_hours": cfg["forecast_horizon"],
        "dynamic_features": list(train_dataset.dynamic_features),
        "node_static_features": list(train_dataset.node_static_features),
        "edge_static_features": list(train_dataset.edge_static_features),
        "num_stations": train_dataset.num_stations,
        "target_variables_by_graph": target_mapping,
        "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
        "artifact_sha256": train_contract["artifact_sha256"],
        "train": _dataset_summary(train_dataset),
        "validation": _dataset_summary(validation_dataset),
        "metadata_qc_files": dict(train_dataset.artifact_status),
        "qc_row_counts": dict(train_dataset.qc_status["row_counts"]),
        "dataset_quality_audit": quality_audit.summary,
        "quality_qc_files_written": written_quality_files,
        "normalization_recompute_audit": normalization_audit,
    }

    del model, train_loader, validation_loader, train_dataset, validation_dataset
    gc.collect()

    test_cfg, test_model, test_loader, _test_device = setup_evaluation(
        config_path,
        split="TEST",
        dataset_root=dataset_root,
        graph_id=graph_id,
    )
    test_dataset = test_loader.dataset
    test_events = set(test_dataset.event_ids)
    overlaps = {
        "train_validation": sorted(train_events & validation_events),
        "train_test": sorted(train_events & test_events),
        "validation_test": sorted(validation_events & test_events),
    }
    if any(overlaps.values()):
        raise ValueError(f"事件级split泄漏: {overlaps}")
    test_contract = test_cfg["_runtime"]["data_contract"]
    test_graphs = set(test_contract["graph_ids"])
    if not test_graphs.issubset(train_graphs):
        raise ValueError(
            "TEST包含TRAIN从未训练的GRAPH_ID，站点观测参数无效: "
            f"{sorted(test_graphs - train_graphs)}"
        )
    if tuple(test_contract["station_ids"]) != train_stations:
        raise ValueError("TEST与TRAIN的全局STATION_ID映射不一致")
    if test_contract["artifact_sha256"] != train_contract["artifact_sha256"]:
        raise ValueError("验证期间正式数据契约文件发生变化")
    result["test"] = _dataset_summary(test_dataset)
    result["event_overlap"] = overlaps
    result["status"] = "VALID"
    del test_model
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="只读校验湖南正式model dataset，不启动训练"
    )
    parser.add_argument(
        "--config", default=str(project_root / "configs" / "hunan_e4.yaml")
    )
    parser.add_argument("--dataset-root", help="覆盖model dataset根目录")
    parser.add_argument("--graph-id", help="可选：只校验一个GRAPH_ID")
    parser.add_argument("--output", help="可选JSON报告路径")
    parser.add_argument(
        "--qc-output-dir",
        help="legacy数据可选QC输出目录；v8/v9使用冻结QC，不接受此参数",
    )
    args = parser.parse_args()
    result = validate_dataset(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
        qc_output_dir=args.qc_output_dir,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
