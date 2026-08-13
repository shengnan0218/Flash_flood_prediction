"""Train the explicit rating-aligned P3 without changing historical entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import validate_checkpoint_config
from scripts.p3_rating_aligned_runtime import setup_training_rating_aligned
from trainers import Trainer


def _event_count(loader) -> int:
    event_ids = getattr(loader.dataset, "event_ids", None)
    return len(event_ids) if event_ids is not None else len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="湖南P3 rating-aligned训练：TRAIN-only输入归一化、Q0锚定、rating(Q)->ΔZ主干"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent
            / "configs"
            / "hunan_p3_state_init_single_q611e0340_event_domain_rating_aligned.yaml"
        ),
    )
    parser.add_argument("--dataset-root", help="覆盖配置中的数据集根目录")
    parser.add_argument("--graph-id", help="可选：只训练指定GRAPH_ID")
    parser.add_argument("--resume", help="显式恢复该rating-aligned P3完整训练状态")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许全新训练覆盖同名rating-aligned输出",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")

    cfg, model, train_loader, validation_loader, device = setup_training_rating_aligned(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    rating = cfg["_runtime"]["rating_curves"]
    station_fits = rating.get("stations", {})
    usable = {
        station: values
        for station, values in station_fits.items()
        if bool(values.get("usable_linear", False))
    }
    print(
        json.dumps(
            {
                "mode": cfg["data"]["mode"],
                "p3_rating_aligned": True,
                "dataset_root": cfg["data"]["dataset_root"],
                "graph_id": cfg["data"].get("graph_id"),
                "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
                "input_normalization": "TRAIN-only station-aware FLOW + relative Z(t)-Z(t0)",
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "train_events": _event_count(train_loader),
                "validation_events": _event_count(validation_loader),
                "train_weighted_sampling": bool(
                    cfg.get("train_sampling", {}).get("enabled", False)
                ),
                "train_sampling_mode": getattr(
                    train_loader.dataset, "train_sampling_mode", "legacy_or_unweighted"
                ),
                "rating_station_count": len(station_fits),
                "usable_linear_rating_station_count": len(usable),
                "rating_fits": usable,
                "early_stopping": bool(cfg["training"].get("early_stopping", True)),
                "checkpoint": cfg["training"]["checkpoint"],
                "final_checkpoint": cfg["training"].get("final_checkpoint"),
                "log_csv": cfg["training"]["log_csv"],
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
