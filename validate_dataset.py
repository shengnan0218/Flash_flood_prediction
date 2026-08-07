"""Read-only preflight validation for the formal Hunan model dataset."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from scripts.common import setup_evaluation, setup_training


def _dataset_summary(dataset: Any) -> dict[str, Any]:
    return {
        "samples": len(dataset),
        "events": len(dataset.event_ids),
        "graphs": list(dataset.graph_ids),
        "graph_node_counts": dict(dataset.graph_node_counts),
    }


def validate_dataset(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    graph_id: str | None = None,
) -> dict[str, Any]:
    """Validate every split and return a JSON-safe dataset summary."""

    cfg, _model, train_loader, validation_loader, _device = setup_training(
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
    }

    # Release the shared TRAIN/VALIDATION hourly tensors before reading TEST.
    del _model, train_loader, validation_loader, train_dataset, validation_dataset
    gc.collect()

    test_cfg, _test_model, test_loader, _test_device = setup_evaluation(
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
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="只读校验湖南正式 _model_dataset，不启动训练"
    )
    parser.add_argument(
        "--config", default=str(project_root / "configs" / "hunan_e4.yaml")
    )
    parser.add_argument("--dataset-root", help="覆盖 _model_dataset 根目录")
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
