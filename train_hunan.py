"""Train one of the four controlled Hunan hydrologic experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.training import setup_training, validate_checkpoint
from trainers.hydrologic_trainer import HydrologicTrainer


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Hunan LSTM-GNN-FC training")
    parser.add_argument("--config", default=str(root / "configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--graph-id")
    parser.add_argument("--resume")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    cfg, model, train_loader, validation_loader, device = setup_training(
        args.config, dataset_root=args.dataset_root, graph_id=args.graph_id
    )
    trainer = HydrologicTrainer(model, cfg, device)
    print(json.dumps({
        "architecture": cfg["_runtime"]["architecture"],
        "runoff_mode": cfg["runoff_mode"],
        "routing_mode": cfg["routing_mode"],
        "state_correction": False,
        "train_samples": len(train_loader.dataset),
        "train_batches_per_epoch": len(train_loader),
        "validation_samples": len(validation_loader.dataset),
        "device": str(device),
    }, ensure_ascii=False, indent=2))
    if args.resume:
        checkpoint = trainer.resume_checkpoint(args.resume)
        validate_checkpoint(checkpoint, cfg, resume=True)
    trainer.fit(train_loader, validation_loader, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
