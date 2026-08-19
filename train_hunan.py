"""Train formal Hunan hydrologic-graph models; V10 is the default entry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.v10_training import (
    is_v10_requested,
    setup_v10_training,
    validate_v10_checkpoint_config,
)
from scripts.v8_training import is_v8_requested, setup_v8_training, validate_v8_checkpoint_config
from scripts.v9_active import is_v9_requested, setup_v9_training, validate_v9_checkpoint_config
from trainers.v10_trainer import V10Trainer
from trainers.v8_trainer import V8Trainer
from trainers.v9_trainer import V9Trainer


def _event_count(loader) -> int:
    event_ids = getattr(loader.dataset, "event_ids", None)
    return len(event_ids) if event_ids is not None else len(loader.dataset)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="湖南正式训练入口；默认E4 V10，显式配置仍可复现V9/V8"
    )
    parser.add_argument("--config", default=str(root / "configs" / "hunan_e4_v10.yaml"))
    parser.add_argument("--dataset-root", help="覆盖配置中的冻结model dataset根目录")
    parser.add_argument("--graph-id", help="可选：只训练指定GRAPH_ID")
    parser.add_argument("--resume", help="恢复完整训练状态（含optimizer）")
    parser.add_argument(
        "--overwrite", action="store_true", help="显式允许全新训练覆盖同名输出"
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")

    if is_v10_requested(args.config, args.dataset_root):
        cfg, model, train_loader, validation_loader, device = setup_v10_training(
            args.config, dataset_root=args.dataset_root, graph_id=args.graph_id
        )
        trainer = V10Trainer(model, cfg, device)
        checkpoint_validator = validate_v10_checkpoint_config
    elif is_v9_requested(args.config, args.dataset_root):
        cfg, model, train_loader, validation_loader, device = setup_v9_training(
            args.config, dataset_root=args.dataset_root, graph_id=args.graph_id
        )
        trainer = V9Trainer(model, cfg, device)
        checkpoint_validator = validate_v9_checkpoint_config
    elif is_v8_requested(args.config, args.dataset_root):
        cfg, model, train_loader, validation_loader, device = setup_v8_training(
            args.config, dataset_root=args.dataset_root, graph_id=args.graph_id
        )
        trainer = V8Trainer(model, cfg, device)
        checkpoint_validator = validate_v8_checkpoint_config
    else:
        raise ValueError(
            "正式train_hunan入口只保留V10/V9/V8；P3/P2及更早legacy配置已退役"
        )

    print(
        json.dumps(
            {
                "mode": cfg["data"]["mode"],
                "model_version": cfg.get("model_version", "v8"),
                "dataset_root": cfg["data"]["dataset_root"],
                "data_contract": cfg.get("_runtime", {}).get("data_contract", {}).get("contract"),
                "runoff_mode": cfg["runoff_mode"],
                "routing_mode": cfg["routing_mode"],
                "supervised_target": cfg.get("_runtime", {}).get(
                    "supervised_target", cfg["data"]["target_variable"]
                ),
                "stage_prediction": cfg.get("_runtime", {}).get("stage_prediction"),
                "dataset_type": cfg["data"].get("dataset_type"),
                "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
                "temporal": cfg.get("temporal"),
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "train_events": _event_count(train_loader),
                "validation_events": _event_count(validation_loader),
                "early_stopping": bool(cfg["training"].get("early_stopping", True)),
                "train_graphs": list(getattr(train_loader.dataset, "graph_ids", ())),
                "validation_graphs": list(getattr(validation_loader.dataset, "graph_ids", ())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.resume:
        checkpoint = trainer.resume_checkpoint(args.resume)
        checkpoint_validator(checkpoint, cfg, resume=True)
    trainer.fit(train_loader, validation_loader, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
