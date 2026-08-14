"""Evaluate model weights on the independent Hunan VALIDATION/TEST split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.common import setup_evaluation, validate_checkpoint_config
from scripts.v8_training import (
    is_v8_requested,
    setup_v8_evaluation,
    validate_v8_checkpoint_config,
)
from trainers import Trainer
from trainers.v8_trainer import V8Trainer
from metrics.p2_event_evaluation import evaluate_p2_flood_events


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _contains_none(value) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="默认在独立TEST划分评估；自动识别legacy与v8数据契约"
    )
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parent / "configs" / "hunan_e4.yaml"
        ),
    )
    parser.add_argument(
        "--event-sample-index",
        help="legacy P2 test_flood_event_samples.csv；v8不使用此参数",
    )
    parser.add_argument(
        "--output-dir",
        help="legacy P2 Q/Z/Delta-Z逐点和分组指标输出目录",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-root",
        help="覆盖配置中的model dataset根目录",
    )
    parser.add_argument("--graph-id", help="评估指定GRAPH_ID")
    parser.add_argument(
        "--split",
        choices=["VALIDATION", "TEST"],
        default="TEST",
    )
    parser.add_argument("--output", help="可选JSON指标输出路径")
    parser.add_argument(
        "--diagnostics-dir",
        help="legacy正式诊断输出目录；v8当前不调用旧station=node诊断器",
    )
    args = parser.parse_args()

    use_v8 = is_v8_requested(args.config, args.dataset_root)
    if use_v8:
        if args.event_sample_index:
            parser.error("v8已冻结event/sample domain，不接受--event-sample-index")
        cfg, model, loader, device = setup_v8_evaluation(
            args.config,
            split=args.split,
            dataset_root=args.dataset_root,
            graph_id=args.graph_id,
        )
        trainer = V8Trainer(model, cfg, device)
        checkpoint_validator = validate_v8_checkpoint_config
    else:
        cfg, model, loader, device = setup_evaluation(
            args.config,
            split=args.split,
            dataset_root=args.dataset_root,
            graph_id=args.graph_id,
            sample_index_path=args.event_sample_index,
        )
        trainer = Trainer(model, cfg, device)
        checkpoint_validator = validate_checkpoint_config

    checkpoint = trainer.load_weights(args.checkpoint)
    checkpoint_validator(checkpoint, cfg)

    if args.event_sample_index:
        if args.split != "TEST":
            parser.error("--event-sample-index只允许--split TEST")
        if cfg["data"].get("dataset_type") != "continuous":
            parser.error("--event-sample-index要求continuous P2配置")
        output_dir = args.output_dir or (
            Path(args.checkpoint).expanduser().resolve().parent
            / "hunan_p2_flood_event_test"
        )
        result = {
            "split": "TEST",
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "event_sample_index": str(
                Path(args.event_sample_index).expanduser().resolve()
            ),
            "metrics": evaluate_p2_flood_events(
                model, loader, device, output_dir
            ),
        }
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return

    # The legacy validation-diagnostics accumulator assumes gauge=node.  v8
    # keeps Nobs sparse and therefore uses the generic physical-unit metrics
    # here; station-aware v8 diagnostics can be added without changing training.
    formal_legacy_evaluation = (
        cfg.get("data", {}).get("mode") == "hunan" and not use_v8
    )
    metrics = trainer.evaluate(
        loader,
        include_group_details=True,
        include_validation_diagnostics=formal_legacy_evaluation,
        include_diagnostic_details=formal_legacy_evaluation,
    )
    validation_diagnostics = metrics.pop("_validation_diagnostics", None)
    serialised_metrics = _json_safe(metrics)
    result = {
        "split": args.split,
        "samples": len(loader.dataset),
        "graphs": list(getattr(loader.dataset, "graph_ids", ())),
        "target_variable": cfg["data"]["target_variable"],
        "data_contract": (
            cfg.get("_runtime", {}).get("data_contract", {}).get(
                "contract", "legacy"
            )
        ),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "metrics": serialised_metrics,
    }
    if validation_diagnostics is not None:
        if args.diagnostics_dir:
            diagnostics_dir = Path(args.diagnostics_dir).expanduser().resolve()
        elif args.output:
            output_stem = Path(args.output).expanduser().resolve().with_suffix("")
            diagnostics_dir = output_stem.with_name(
                f"{output_stem.name}_diagnostics"
            )
        else:
            checkpoint_path = Path(args.checkpoint).expanduser().resolve()
            diagnostics_dir = checkpoint_path.parent / (
                f"{checkpoint_path.stem}_{args.split.lower()}_diagnostics"
            )
        written = validation_diagnostics.write(
            diagnostics_dir,
            split=args.split,
            context={
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "split": args.split,
            },
        )
        result["diagnostics_dir"] = str(diagnostics_dir)
        result["diagnostic_files"] = written
    if _contains_none(serialised_metrics):
        result["metric_note"] = (
            "null表示该目标没有有效标签，或指标因零方差等条件无定义；"
            "请结合*_valid_count解读"
        )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
