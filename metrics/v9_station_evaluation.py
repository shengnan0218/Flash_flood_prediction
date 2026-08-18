"""Cadence-aware wrapper around the frozen v8 station-aware evaluator.

The v8 evaluator is numerically correct for the current Hunan hourly target
cadence.  v9 keeps those aggregation rules but converts lead/timing metadata
using temporal.target_step_seconds so future Zhejiang sub-hour targets are not
mislabelled as hours.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from metrics.v8_station_evaluation import evaluate_v8_station_aware


_TIMING_SUFFIXES = ("_peak_timing_mae_h", "_peak_timing_bias_h")


def _scale_timing_in_mapping(value: Any, factor: float) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if (
                isinstance(item, (int, float))
                and any(str(key).endswith(suffix) for suffix in _TIMING_SUFFIXES)
            ):
                result[key] = float(item) * factor
            else:
                result[key] = _scale_timing_in_mapping(item, factor)
        return result
    if isinstance(value, list):
        return [_scale_timing_in_mapping(item, factor) for item in value]
    return value


def _scale_timing_csv(path: Path, factor: float) -> None:
    if not path.is_file():
        return
    frame = pd.read_csv(path, encoding="utf-8-sig")
    changed = False
    for column in frame.columns:
        if any(str(column).endswith(suffix) for suffix in _TIMING_SUFFIXES):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
            changed = True
    if changed:
        frame.to_csv(path, index=False, encoding="utf-8-sig")


def evaluate_v9_station_aware(
    trainer: Any,
    loader: Any,
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    result = evaluate_v8_station_aware(
        trainer,
        loader,
        output_dir,
        split=split,
        checkpoint=checkpoint,
    )
    temporal = trainer.cfg.get("temporal", {})
    target_step_seconds = float(temporal.get("target_step_seconds", 3600.0))
    if target_step_seconds <= 0:
        raise ValueError("v9 evaluation target_step_seconds必须>0")
    step_hours = target_step_seconds / 3600.0

    files = {key: Path(path) for key, path in result["files"].items()}
    for key in ("station_metrics", "graph_metrics", "event_station_metrics"):
        _scale_timing_csv(files[key], step_hours)

    lead_path = files["lead_time_metrics"]
    lead = pd.read_csv(lead_path, encoding="utf-8-sig")
    ordinal = pd.to_numeric(lead["LEAD_HOUR"], errors="raise")
    lead["LEAD_SECONDS"] = ordinal * target_step_seconds
    lead["LEAD_MINUTES"] = lead["LEAD_SECONDS"] / 60.0
    lead["LEAD_HOURS"] = lead["LEAD_SECONDS"] / 3600.0
    # Preserve the historical column while making it physically correct.
    lead["LEAD_HOUR"] = lead["LEAD_HOURS"]
    lead.to_csv(lead_path, index=False, encoding="utf-8-sig")

    summary = _scale_timing_in_mapping(result["summary"], step_hours)
    summary.setdefault("target_semantics", {})["target_step_seconds"] = target_step_seconds
    summary["target_semantics"]["lead_time_columns"] = (
        "LEAD_SECONDS/LEAD_MINUTES/LEAD_HOURS are physical lead times"
    )
    result["summary"] = summary

    summary_path = files["summary"]
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["summary"] = summary
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
