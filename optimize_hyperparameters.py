"""Explicit opt-in entry point for E4 shared-parameter Optuna search."""
from __future__ import annotations

import argparse
from pathlib import Path

from optimization import run_optimization


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4 multitask Optuna TPE + MedianPruner（默认不启动）"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "configs" / "hunan_e4_multitask.yaml"
        ),
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument(
        "--enable",
        action="store_true",
        help="显式确认启动HPO；没有此标志程序会拒绝运行",
    )
    args = parser.parse_args()
    run_optimization(
        args.config,
        dataset_root=args.dataset_root,
        n_trials=args.n_trials,
        explicitly_enabled=args.enable,
    )


if __name__ == "__main__":
    main()
