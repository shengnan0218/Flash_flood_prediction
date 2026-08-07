"""Train on formal Hunan TRAIN events and select checkpoints on VALIDATION only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import setup_training, validate_checkpoint_config
from trainers import Trainer


def _event_count(loader) -> int:
    event_ids = getattr(loader.dataset, "event_ids", None)
    return len(event_ids) if event_ids is not None else len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="湖南洪水模型训练；TRAIN 与 VALIDATION 按事件严格隔离"
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parent / "configs" / "hunan_e4.yaml")
    )
    parser.add_argument(
        "--dataset-root",
        help="覆盖配置中的 _model_dataset 根目录（含 graph/dynamic/events/metadata/qc）",
    )
    parser.add_argument(
        "--graph-id",
        help="可选：只训练指定 GRAPH_ID；默认由同图批采样器训练全部河网",
    )
    parser.add_argument(
        "--resume",
        help="显式恢复完整训练状态（包含 optimizer）；不用于只读评估",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许全新训练覆盖同名输出；不能与--resume同时使用",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")
    cfg, model, train_loader, validation_loader, device = setup_training(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    print(
        json.dumps(
            {
                "mode": cfg["data"]["mode"],
                "dataset_root": cfg["data"]["dataset_root"],
                "target_variable": cfg["data"]["target_variable"],
                "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "train_events": _event_count(train_loader),
                "validation_events": _event_count(validation_loader),
                "train_graphs": list(getattr(train_loader.dataset, "graph_ids", ())),
                "validation_graphs": list(
                    getattr(validation_loader.dataset, "graph_ids", ())
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    trainer = Trainer(model, cfg, device)
    if args.resume:
        checkpoint = trainer.resume_checkpoint(args.resume)
        validate_checkpoint_config(checkpoint, cfg, resume=True)
    trainer.fit(train_loader, validation_loader, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
