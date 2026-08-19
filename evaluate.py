"""Evaluate formal Hunan V10/V9/V8 models on VALIDATION or TEST."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics.v10_station_evaluation import evaluate_v10_station_aware
from metrics.v8_station_evaluation import evaluate_v8_station_aware
from metrics.v9_station_evaluation import evaluate_v9_station_aware
from scripts.v10_training import (
    is_v10_requested,
    setup_v10_evaluation,
    validate_v10_checkpoint_config,
)
from scripts.v8_training import is_v8_requested, setup_v8_evaluation, validate_v8_checkpoint_config
from scripts.v9_active import is_v9_requested, setup_v9_evaluation, validate_v9_checkpoint_config
from trainers.v10_trainer import V10Trainer
from trainers.v8_trainer import V8Trainer
from trainers.v9_trainer import V9Trainer


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="湖南正式评价入口；默认E4 V10，显式配置仍可复现V9/V8"
    )
    parser.add_argument("--config", default=str(root / "configs" / "hunan_e4_v10.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", help="覆盖冻结model dataset根目录")
    parser.add_argument("--graph-id", help="可选：只评价指定GRAPH_ID")
    parser.add_argument("--split", choices=["VALIDATION", "TEST"], default="TEST")
    parser.add_argument("--output-dir", help="station-aware评价明细输出目录")
    parser.add_argument("--output", help="可选顶层JSON结果路径")
    args = parser.parse_args()

    if is_v10_requested(args.config, args.dataset_root):
        setup_fn = setup_v10_evaluation
        trainer_cls = V10Trainer
        validator = validate_v10_checkpoint_config
        evaluator = evaluate_v10_station_aware
        version = "v10"
    elif is_v9_requested(args.config, args.dataset_root):
        setup_fn = setup_v9_evaluation
        trainer_cls = V9Trainer
        validator = validate_v9_checkpoint_config
        evaluator = evaluate_v9_station_aware
        version = "v9"
    elif is_v8_requested(args.config, args.dataset_root):
        setup_fn = setup_v8_evaluation
        trainer_cls = V8Trainer
        validator = validate_v8_checkpoint_config
        evaluator = evaluate_v8_station_aware
        version = "v8"
    else:
        raise ValueError(
            "正式evaluate入口只保留V10/V9/V8；P3/P2及更早legacy评价路径已退役"
        )

    cfg, model, loader, device = setup_fn(
        args.config,
        split=args.split,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    trainer = trainer_cls(model, cfg, device)
    checkpoint = trainer.load_weights(args.checkpoint)
    validator(checkpoint, cfg)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = args.output_dir or (
        checkpoint_path.parent
        / f"{checkpoint_path.stem}_{args.split.lower()}_{version}_evaluation"
    )
    evaluation = evaluator(
        trainer,
        loader,
        output_dir,
        split=args.split,
        checkpoint=args.checkpoint,
    )
    result = {
        "split": args.split,
        "model_version": cfg.get("model_version", version),
        "samples": len(loader.dataset),
        "graphs": list(getattr(loader.dataset, "graph_ids", ())),
        "supervised_target": cfg.get("_runtime", {}).get(
            "supervised_target", cfg["data"]["target_variable"]
        ),
        "stage_prediction": cfg.get("_runtime", {}).get("stage_prediction"),
        "data_contract": cfg.get("_runtime", {}).get("data_contract", {}).get("contract"),
        "checkpoint": str(checkpoint_path),
        "evaluation_dir": evaluation.get(
            "output_dir", str(Path(output_dir).expanduser().resolve())
        ),
        "evaluation_files": evaluation["files"],
        "metrics": evaluation["summary"],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
