"""Train P3 with a frozen TRAIN-calibrated monotone Q->Z observation function."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common import validate_checkpoint_config
from scripts.p3_rating_calibrated_runtime import setup_training_rating_calibrated
from trainers import Trainer


def _event_count(loader) -> int:
    event_ids = getattr(loader.dataset, "event_ids", None)
    return len(event_ids) if event_ids is not None else len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "湖南P3 calibrated-rating训练：TRAIN-only输入归一化、Q0-informed state init、"
            "冻结低自由度单调Q->Z标定函数、无neural Z residual"
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent
            / "configs"
            / "hunan_p3_state_init_single_q611e0340_event_domain_rating_calibrated.yaml"
        ),
    )
    parser.add_argument("--dataset-root", help="覆盖配置中的数据集根目录")
    parser.add_argument("--graph-id", help="可选：只训练指定GRAPH_ID")
    parser.add_argument("--resume", help="显式恢复该calibrated-rating P3完整训练状态")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="显式允许全新训练覆盖同名calibrated-rating输出",
    )
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume与--overwrite不能同时使用")

    cfg, model, train_loader, validation_loader, device = setup_training_rating_calibrated(
        args.config,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    station_fits = cfg["_runtime"]["rating_curves"].get("stations", {})
    usable = {
        station: values
        for station, values in station_fits.items()
        if bool(values.get("usable_calibrated", False))
    }
    unusable = sorted(set(station_fits) - set(usable))
    if unusable:
        raise RuntimeError(
            "calibrated-rating P3要求每个active outlet都有TRAIN-only可用rating curve；"
            f"当前不可用站点={unusable}"
        )

    sampling_mode = getattr(
        train_loader.dataset, "train_sampling_mode", "legacy_or_unweighted"
    )
    loss_weight_mode = getattr(
        train_loader.dataset, "train_loss_weight_mode", "unit_or_legacy"
    )
    weight_statistics = (
        train_loader.dataset.event_loss_weight_statistics()
        if hasattr(train_loader.dataset, "event_loss_weight_statistics")
        else {"mode": loss_weight_mode}
    )
    sampler_name = type(train_loader.batch_sampler).__name__
    weighted_cfg = bool(cfg.get("train_sampling", {}).get("enabled", False))
    replacement_sampling = "Weighted" in sampler_name
    if sampling_mode == "event_full_pass":
        if weighted_cfg or replacement_sampling:
            raise RuntimeError(
                "hydrologic event-domain TRAIN必须full-pass且无replacement sampling；"
                f"config_weighted={weighted_cfg}, sampler={sampler_name}"
            )
        if loss_weight_mode != "deterministic_graph_event_window_balance":
            raise RuntimeError(
                "hydrologic event-domain TRAIN缺少确定性graph/event/window loss balance；"
                f"actual={loss_weight_mode}"
            )

    print(
        json.dumps(
            {
                "mode": cfg["data"]["mode"],
                "p3_rating_aligned": True,
                "p3_rating_calibrated": True,
                "dataset_root": cfg["data"]["dataset_root"],
                "graph_id": cfg["data"].get("graph_id"),
                "future_rainfall_mode": cfg["data"]["future_rainfall_mode"],
                "input_normalization": "TRAIN-only station-aware FLOW + relative Z(t)-Z(t0)",
                "z_observation": (
                    "frozen TRAIN-calibrated low-DOF monotone piecewise-linear rating(Q); "
                    "global TRAIN OLS-slope extrapolation; no neural residual"
                ),
                "train_samples": len(train_loader.dataset),
                "validation_samples": len(validation_loader.dataset),
                "train_events": _event_count(train_loader),
                "validation_events": _event_count(validation_loader),
                "train_sampling_mode": sampling_mode,
                "train_batch_sampler": sampler_name,
                "train_weighted_sampling": weighted_cfg,
                "train_replacement_sampling": replacement_sampling,
                "train_loss_weight_mode": loss_weight_mode,
                "train_loss_weight_statistics": weight_statistics,
                "calibrated_rating_station_count": len(station_fits),
                "usable_calibrated_rating_station_count": len(usable),
                "rating_fits": usable,
                "neural_z_residual_head": model.independent_z_head is not None,
                "early_stopping": bool(cfg["training"].get("early_stopping", True)),
                "checkpoint": cfg["training"]["checkpoint"],
                "final_checkpoint": cfg["training"].get("final_checkpoint"),
                "log_csv": cfg["training"]["log_csv"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if model.independent_z_head is not None:
        raise RuntimeError("calibrated-rating P3禁止neural Z residual head")
    trainer = Trainer(model, cfg, device)
    if args.resume:
        checkpoint = trainer.resume_checkpoint(args.resume)
        validate_checkpoint_config(checkpoint, cfg, resume=True)
    trainer.fit(train_loader, validation_loader, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
