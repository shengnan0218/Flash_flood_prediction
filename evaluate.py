"""Evaluate model weights on the independent Hunan TEST split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.common import (
    setup_evaluation,
    validate_checkpoint_config,
)
from trainers import Trainer


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
        description="默认仅在 TEST 事件上评估；不会恢复 optimizer 状态"
    )
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parent / "configs" / "hunan_e4.yaml")
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-root",
        help="覆盖配置中的 _model_dataset 根目录（含 graph/dynamic/events/metadata/qc）",
    )
    parser.add_argument("--graph-id", help="评估指定 GRAPH_ID")
    parser.add_argument(
        "--split",
        choices=["VALIDATION", "TEST"],
        default="TEST",
        help="默认使用从未参与拟合或早停的 TEST 划分",
    )
    parser.add_argument("--output", help="可选 JSON 指标输出路径")
    args = parser.parse_args()

    cfg, model, loader, device = setup_evaluation(
        args.config,
        split=args.split,
        dataset_root=args.dataset_root,
        graph_id=args.graph_id,
    )
    trainer = Trainer(model, cfg, device)
    checkpoint = trainer.load_weights(args.checkpoint)
    validate_checkpoint_config(checkpoint, cfg)
    metrics = trainer.evaluate(loader, include_group_details=True)
    serialised_metrics = _json_safe(metrics)
    result = {
        "split": args.split,
        "samples": len(loader.dataset),
        "graphs": list(getattr(loader.dataset, "graph_ids", ())),
        "target_variable": cfg["data"]["target_variable"],
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "metrics": serialised_metrics,
    }
    if _contains_none(serialised_metrics):
        result["metric_note"] = (
            "null 表示该目标没有有效标签，或该指标因零方差等条件无定义；"
            "请结合 *_valid_count 解读"
        )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
