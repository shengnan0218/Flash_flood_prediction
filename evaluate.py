"""Evaluate the single supported hydrologic architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics.forecast_evaluation import evaluate_forecast
from scripts.training import setup_evaluation, validate_checkpoint
from trainers.hydrologic_trainer import HydrologicTrainer


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Hunan hydrologic evaluation")
    parser.add_argument("--config", default=str(root / "configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--graph-id")
    parser.add_argument("--split", choices=["VALIDATION", "TEST"], default="TEST")
    parser.add_argument("--output-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg, model, loader, device = setup_evaluation(
        args.config, split=args.split, dataset_root=args.dataset_root, graph_id=args.graph_id
    )
    trainer = HydrologicTrainer(model, cfg, device)
    checkpoint = trainer.load_weights(args.checkpoint)
    validate_checkpoint(checkpoint, cfg)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = args.output_dir or checkpoint_path.parent / f"{checkpoint_path.stem}_{args.split.lower()}_evaluation"
    evaluation = evaluate_forecast(
        trainer, loader, output_dir, split=args.split, checkpoint=args.checkpoint
    )
    result = {
        "split": args.split,
        "architecture": cfg["_runtime"]["architecture"],
        "runoff_mode": cfg["runoff_mode"],
        "routing_mode": cfg["routing_mode"],
        "state_correction": False,
        "samples": len(loader.dataset),
        "checkpoint": str(checkpoint_path),
        "evaluation_files": evaluation["files"],
        "metrics": evaluation["summary"],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
