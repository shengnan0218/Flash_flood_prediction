"""Evaluate persistence, full-route, and gated-route Q without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics.output_decomposition import evaluate_output_decomposition
from scripts.training import setup_evaluation, validate_checkpoint
from trainers.hydrologic_trainer import HydrologicTrainer


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Evaluate Q0 persistence, full routed Delta-Q, and learned gate"
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/e4_water_balance_lstm_muskingum_gnn.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--graph-id")
    parser.add_argument("--split", choices=["VALIDATION", "TEST"], default="VALIDATION")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg, model, loader, device = setup_evaluation(
        args.config, split=args.split, dataset_root=args.dataset_root, graph_id=args.graph_id
    )
    trainer = HydrologicTrainer(model, cfg, device)
    checkpoint = trainer.load_weights(args.checkpoint)
    validate_checkpoint(checkpoint, cfg)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = args.output_dir or (
        checkpoint_path.parent / f"{checkpoint_path.stem}_{args.split.lower()}_decomposition"
    )
    result = evaluate_output_decomposition(
        trainer, loader, output_dir, split=args.split, checkpoint=checkpoint_path
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
