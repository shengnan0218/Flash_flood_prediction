"""Train formal Hunan datasets with VALIDATION selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import setup_training, validate_checkpoint_config
from scripts.v8_training import (
    is_v8_requested,
    setup_v8_training,
    validate_v8_checkpoint_config,
)
from scripts.v9_active import (
    is_v9_requested,
    setup_v9_training,
    validate_v9_checkpoint_config,
)
from trainers import Trainer
from trainers.v8_trainer import V8Trainer
from trainers.v9_trainer import V9Trainer


def _event_count(loader) -> int:
    event_ids = getattr(loader.dataset, "event_ids", None)
    return len(event_ids) if event_ids is not None else len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="湖南洪水模型训练；默认正式入口为E4 v9，仍可显式切换E1-E4/v8/legacy配置"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "configs" / "hunan_e4_v9.yaml"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        help="覆盖配置中的model dataset根目录",
    )
    parser.add_argument(
        "--graph-id",
        help="可选：只训练指定GRAPH_ID；默认训练全部河网",
    )
    parser.add_argument(
        "--resume",
        help="显式恢复完整训练状态（包含optimizer）；不用于只读评估",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许全新训练覆盖同名输出；不能与--resume同时使用",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")

    use_v9 = is_v9_requested(args.config, args.dataset_root)
    use_v8 = False if use_v9 else is_v8_requested(args.config, args.dataset_root)
    if use_v9:
        cfg, model, train_loader, validation_loader, device = setup_v9_training(
            args.config,
            dataset_root=args.dataset_root,
            graph_id=args.graph_id,
        )
        trainer = V9Trainer(model, cfg, device)
        checkpoint_validator = validate_v9_checkpoint_config
    elif use_v8:
        cfg, model, train_loader, validation_loader, device = setup_v8_training(
            args.config,
            dataset_root=args.dataset_root,
            graph_id=args.graph_id,
        )
        trainer = V8Trainer(model, cfg, device)
        checkpoint_validator = validate_v8_checkpoint_config
    else:
        cfg, model, train_loader, validation_loader, device = setup_training(
            args.config,
            dataset_root=args.dataset_root,
            graph_id=args.graph_id,
        )
        trainer = Trainer(model, cfg, device)
        checkpoint_validator = validate_checkpoint_config

    print(
        json.dumps(
            {
                "mode": cfg["data"]["mode"],
                "model_version": cfg.get("model_version", "legacy_or_v8"),
                "dataset_root": cfg["data"]["dataset_root"],
                "data_contract": (
                    cfg.get("_runtime", {})
                    .get("data_contract", {})
                    .get("contract", "legacy")
                ),
                "runoff_mode": cfg["runoff_mode"],
                "routing_mode": cfg["routing_mode"],
                "target_variable": cfg["data"]["target_variable"],
                "dataset_type": cfg["data"].get("dataset_type", "event"),
                "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
                "temporal": cfg.get("temporal"),
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "train_events": _event_count(train_loader),
                "validation_events": _event_count(validation_loader),
                "train_weighted_sampling": bool(
                    cfg.get("train_sampling", {}).get("enabled", False)
                ),
                "train_sampling_mode": getattr(
                    train_loader.dataset,
                    "train_sampling_mode",
                    "legacy_or_unweighted",
                ),
                "early_stopping": bool(
                    cfg["training"].get("early_stopping", True)
                ),
                "train_graphs": list(
                    getattr(train_loader.dataset, "graph_ids", ())
                ),
                "validation_graphs": list(
                    getattr(validation_loader.dataset, "graph_ids", ())
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.resume:
        checkpoint = trainer.resume_checkpoint(args.resume)
        checkpoint_validator(checkpoint, cfg, resume=True)
    trainer.fit(
        train_loader,
        validation_loader,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
