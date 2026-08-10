"""Robust higher-is-better validation checkpoint selection."""
from __future__ import annotations

import math
from typing import Any, Mapping


def bounded_efficiency(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    """Clip NSE/KGE to a finite interval and map it linearly to [0, 1]."""

    if not all(math.isfinite(item) for item in (value, lower, upper)) or lower >= upper:
        raise ValueError("efficiency value/clip必须有限且lower < upper")
    clipped = min(max(value, lower), upper)
    return (clipped - lower) / (upper - lower)


def bounded_error_skill(error: float, scale: float = 1.0) -> float:
    """Convert a non-negative error to ``1 / (1 + error / scale)``."""

    if not math.isfinite(error) or error < 0:
        raise ValueError(f"error必须为有限非负数，实际={error}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"error scale必须为有限正数，实际={scale}")
    return 1.0 / (1.0 + error / scale)


def validation_selection_score(
    summary_metrics: Mapping[str, Any],
    loss_scales: Mapping[str, Any],
    selection_cfg: Mapping[str, Any],
) -> dict[str, float]:
    """Return the fixed six-component VALIDATION-only composite score.

    Q NSE/KGE are graph-level medians after shortest-lead deduplication.  Peak
    and volume use event-level median absolute relative errors.  Absolute Z
    and true first-difference Z use station-level median MAE divided by the
    TRAIN-only water-level standard deviation.  All six skills are bounded in
    [0, 1] and the configured weights sum to one, so larger is always better.
    """

    required = {
        "q_graph_nse_median",
        "q_graph_kge_median",
        "q_event_absolute_relative_peak_error_median",
        "q_event_absolute_relative_volume_error_median",
        "z_station_mae_median",
        "z_slope_station_mae_median",
    }
    missing = required - set(summary_metrics)
    if missing:
        raise KeyError(f"validation diagnostics缺少selection字段: {sorted(missing)}")
    values = {name: float(summary_metrics[name]) for name in required}
    error_names = {
        "q_event_absolute_relative_peak_error_median",
        "q_event_absolute_relative_volume_error_median",
        "z_station_mae_median",
        "z_slope_station_mae_median",
    }
    nonfinite = {
        name: values[name]
        for name in error_names
        if not math.isfinite(values[name])
    }
    if nonfinite:
        raise FloatingPointError(
            "validation selection error component无定义；不会静默重分配权重: "
            f"{nonfinite}"
        )
    z_scale = float(loss_scales.get("water_level", float("nan")))
    if not math.isfinite(z_scale) or z_scale <= 0:
        raise ValueError(f"TRAIN-only water_level scale必须为有限正数，实际={z_scale}")
    lower = float(selection_cfg["efficiency_clip_min"])
    upper = float(selection_cfg["efficiency_clip_max"])
    nse_defined = math.isfinite(values["q_graph_nse_median"])
    kge_defined = math.isfinite(values["q_graph_kge_median"])
    components = {
        "q_nse_skill": (
            bounded_efficiency(values["q_graph_nse_median"], lower, upper)
            if nse_defined
            else 0.0
        ),
        "q_kge_skill": (
            bounded_efficiency(values["q_graph_kge_median"], lower, upper)
            if kge_defined
            else 0.0
        ),
        "q_peak_skill": bounded_error_skill(
            values["q_event_absolute_relative_peak_error_median"]
        ),
        "q_volume_skill": bounded_error_skill(
            values["q_event_absolute_relative_volume_error_median"]
        ),
        "z_level_skill": bounded_error_skill(values["z_station_mae_median"], z_scale),
        "z_slope_skill": bounded_error_skill(
            values["z_slope_station_mae_median"], z_scale
        ),
    }
    weights = {
        "q_nse_skill": float(selection_cfg["q_nse_weight"]),
        "q_kge_skill": float(selection_cfg["q_kge_weight"]),
        "q_peak_skill": float(selection_cfg["q_peak_weight"]),
        "q_volume_skill": float(selection_cfg["q_volume_weight"]),
        "z_level_skill": float(selection_cfg["z_level_weight"]),
        "z_slope_skill": float(selection_cfg["z_slope_weight"]),
    }
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError(f"validation selection weights必须有限非负: {weights}")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"validation selection weights之和必须为1: {weights}")
    score = sum(weights[name] * components[name] for name in components)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise FloatingPointError(f"validation_selection_score越界/非有限: {score}")
    return {
        "validation_selection_score": score,
        "validation_selection_q_nse_defined": float(nse_defined),
        "validation_selection_q_kge_defined": float(kge_defined),
        **{
            f"validation_selection_{name}": value
            for name, value in components.items()
        },
    }
