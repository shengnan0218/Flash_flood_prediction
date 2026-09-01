"""Read-only Q0 / routed-base / final-output forecast decomposition."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch

from metrics.flood_metrics import masked_regression_sums, regression_metrics


METHODS = ("persistence", "routed_base", "final")
METHOD_LABELS = {
    "persistence": "Q0 persistence",
    "routed_base": "Q0 + routed Delta-Q (FC disabled)",
    "final": "final Q output",
}


def _empty_sums() -> dict[str, float | int]:
    return {
        "count": 0,
        "absolute_error": 0.0,
        "squared_error": 0.0,
        "error": 0.0,
        "prediction": 0.0,
        "target": 0.0,
        "prediction_squared": 0.0,
        "target_squared": 0.0,
        "cross": 0.0,
    }


def _merge(destination: dict[str, float | int], source: dict[str, float | int]) -> None:
    for key in destination:
        destination[key] = destination[key] + source[key]


def _new_group() -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        method: {"q": _empty_sums(), "delta_q": _empty_sums()}
        for method in METHODS
    }


def _update_group(
    group: dict[str, dict[str, dict[str, float | int]]],
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    q0: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    for method, prediction in predictions.items():
        _merge(group[method]["q"], masked_regression_sums(prediction, target, mask))
        _merge(
            group[method]["delta_q"],
            masked_regression_sums(prediction - q0, target - q0, mask),
        )


def _skill(sums: dict[str, float | int], persistence: dict[str, float | int]) -> float:
    count = int(sums["count"])
    model_sse = float(sums["squared_error"])
    persistence_sse = float(persistence["squared_error"])
    if count == 0 or persistence_sse <= 0.0:
        return float("nan")
    return 1.0 - model_sse / persistence_sse


def _report(
    method: str,
    group: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, float | int | str]:
    q = regression_metrics(group[method]["q"])
    delta = regression_metrics(group[method]["delta_q"])
    return {
        "METHOD": method,
        "METHOD_LABEL": METHOD_LABELS[method],
        "Q0_VALID_COUNT": int(q["valid_count"]),
        "Q_MAE": float(q["mae"]),
        "Q_RMSE": float(q["rmse"]),
        "Q_BIAS": float(q["bias"]),
        "Q_NSE": float(q["nse"]),
        "Q_KGE": float(q["kge"]),
        "SKILL_OVER_PERSISTENCE": _skill(
            group[method]["q"], group["persistence"]["q"]
        ),
        "DELTA_Q_RMSE": float(delta["rmse"]),
        "DELTA_Q_NSE": float(delta["nse"]),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@torch.no_grad()
def evaluate_output_decomposition(
    trainer: Any,
    loader: Iterable[Any],
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Evaluate persistence, routed base and final Q on one Q0-valid subset.

    routed_base is exactly the output head's Q0-anchored physical base with
    the final non-negative clamp retained and the FC correction disabled.  The
    three methods therefore differ only in the information they add after Q0.
    """
    model = trainer.model
    device = trainer.device
    model.eval()
    overall = _new_group()
    by_lead: dict[int, dict[str, dict[str, dict[str, float | int]]]] = {}
    by_station: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        diagnostics = output.get("diagnostics", {})
        base = diagnostics.get("q_residual_base_m3s")
        if not isinstance(base, torch.Tensor):
            raise KeyError(
                "三分解评价需要diagnostics['q_residual_base_m3s']"
            )
        if base.shape != output["q"].shape:
            raise ValueError("q_residual_base_m3s与最终Q预测形状不一致")

        target = batch.q_target
        q0 = batch.q_history[:, -1]
        q0_mask = batch.q_mask[:, -1].bool()
        valid = batch.q_target_mask.bool() & q0_mask.unsqueeze(1)
        q0_expanded = q0.unsqueeze(1).expand_as(target)
        predictions = {
            "persistence": q0_expanded,
            "routed_base": torch.relu(base),
            "final": output["q"],
        }
        _update_group(overall, predictions, target, q0_expanded, valid)

        for horizon in range(target.shape[1]):
            lead = horizon + 1
            group = by_lead.setdefault(lead, _new_group())
            _update_group(
                group,
                {name: value[:, horizon] for name, value in predictions.items()},
                target[:, horizon],
                q0,
                valid[:, horizon],
            )

        station_ids = tuple(str(value).strip() for value in batch.obs_station_ids)
        if len(station_ids) != target.shape[2]:
            raise ValueError("obs_station_ids数量与Q observation维度不一致")
        for index, station_id in enumerate(station_ids):
            group = by_station.setdefault(station_id, _new_group())
            _update_group(
                group,
                {name: value[:, :, index] for name, value in predictions.items()},
                target[:, :, index],
                q0[:, index],
                valid[:, :, index],
            )

    if not int(overall["persistence"]["q"]["count"]):
        raise ValueError("三分解评价没有Q0和Q target同时有效的样本")

    summary_methods = {method: _report(method, overall) for method in METHODS}
    lead_rows = [
        {"LEAD_HOUR": lead, **_report(method, group)}
        for lead, group in sorted(by_lead.items())
        for method in METHODS
    ]
    station_rows = [
        {"STATION_ID": station, **_report(method, group)}
        for station, group in sorted(by_station.items())
        for method in METHODS
    ]

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = str(split).lower()
    lead_path = output_dir / f"{split_name}_q_decomposition_by_lead.csv"
    station_path = output_dir / f"{split_name}_q_decomposition_by_station.csv"
    summary_path = output_dir / f"{split_name}_q_decomposition_summary.json"
    pd.DataFrame(lead_rows).to_csv(lead_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(station_rows).to_csv(station_path, index=False, encoding="utf-8-sig")
    summary = {
        "split": str(split).upper(),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "q0_subset_definition": "Q target valid and final-history-bin Q0 observed",
        "methods": summary_methods,
        "interpretation": {
            "persistence": "Q0 held unchanged through all forecast leads",
            "routed_base": "relu(Q0 + routed_Q(t+h) - routed_Q(t0)); FC correction disabled",
            "final": "current model output after the bounded FC correction",
        },
        "files": {
            "by_lead": str(lead_path),
            "by_station": str(station_path),
            "summary": str(summary_path),
        },
    }
    safe_summary = _json_safe(summary)
    summary_path.write_text(
        json.dumps(safe_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return safe_summary
