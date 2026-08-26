"""Hydrologic station, persistence, lead-time and event-peak evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from metrics.station_evaluation import evaluate_station_aware
from metrics.station_evaluation_core import _batch_strings, _json_safe, _normalise_id


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float | int]:
    pred = np.asarray(pred, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(obs)
    pred = pred[valid]
    obs = obs[valid]
    count = int(obs.size)
    if not count:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "nse": float("nan"),
        }
    error = pred - obs
    denominator = float(np.square(obs - obs.mean()).sum())
    return {
        "count": count,
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
        "nse": (
            1.0 - float(np.square(error).sum()) / denominator
            if denominator > 0
            else float("nan")
        ),
    }


def _skill(model: np.ndarray, persistence: np.ndarray, obs: np.ndarray) -> float:
    model = np.asarray(model, dtype=np.float64)
    persistence = np.asarray(persistence, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    valid = np.isfinite(model) & np.isfinite(persistence) & np.isfinite(obs)
    if not valid.any():
        return float("nan")
    model_sse = float(np.square(model[valid] - obs[valid]).sum())
    persistence_sse = float(np.square(persistence[valid] - obs[valid]).sum())
    return (
        1.0 - model_sse / persistence_sse
        if persistence_sse > 0
        else float("nan")
    )


def _unix_hour(value: str) -> int:
    timestamp = pd.to_datetime(value, errors="raise")
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)
    offset = (
        timestamp - pd.Timestamp("1970-01-01 00:00:00")
    ) / pd.Timedelta(hours=1)
    rounded = round(float(offset))
    if not math.isfinite(float(offset)) or not math.isclose(
        float(offset), rounded, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(f"hydrologic evaluation FORECAST_TIME不是整小时: {value}")
    return int(rounded)


def _deduplicate_truth(points: pd.DataFrame) -> pd.DataFrame:
    key = ["GRAPH_ID", "STATION_ID", "EVENT_ID", "TARGET_UNIX_HOUR"]
    duplicate = points[points.duplicated(key, keep=False)]
    if not duplicate.empty:
        for _, group in duplicate.groupby(key, sort=False):
            values = group["Q_TRUE"].to_numpy(np.float64)
            if np.nanmax(values) - np.nanmin(values) > 1.0e-5:
                raise ValueError("hydrologic evaluation重叠窗口Q truth物理时刻冲突")
    return points.drop_duplicates(key, keep="first")[key + ["Q_TRUE"]].copy()


@torch.no_grad()
def _forecast_skill(
    trainer: Any,
    loader: Iterable[Any],
    output_dir: Path,
    *,
    split: str,
) -> dict[str, Any]:
    """Second final-only pass for diagnostics that require Q0 and physical time."""
    model = trainer.model
    device = trainer.device
    model.eval()
    target_step_seconds = int(trainer.cfg["temporal"]["target_step_seconds"])
    if target_step_seconds != 3600:
        raise ValueError("当前Hunan Hydrologic fixed-lead evaluator要求1h target cadence")

    rows: list[dict[str, Any]] = []
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        q_pred = output["q"].detach().float().cpu().numpy()
        q_true = batch.q_target.detach().float().cpu().numpy()
        q_mask = batch.q_target_mask.detach().cpu().numpy().astype(bool)
        q_history = batch.q_history.detach().float().cpu().numpy()
        q_history_mask = batch.q_mask.detach().cpu().numpy().astype(bool)
        batch_size, horizon, obs_count = q_pred.shape
        if q_true.shape != q_pred.shape or q_mask.shape != q_pred.shape:
            raise ValueError("hydrologic evaluation Q prediction/target/mask shape不一致")
        graph_ids = _batch_strings(batch.graph_id, batch_size, "graph_id")
        event_ids = _batch_strings(batch.event_id, batch_size, "event_id")
        sample_ids = _batch_strings(batch.sample_id, batch_size, "sample_id")
        forecast_times = _batch_strings(
            batch.forecast_time, batch_size, "forecast_time"
        )
        stations = tuple(_normalise_id(value) for value in batch.obs_station_ids)
        if len(stations) != obs_count:
            raise ValueError("hydrologic evaluation station count与Q tensor不一致")
        origin_hours = [_unix_hour(value) for value in forecast_times]
        q0 = q_history[:, -1]
        q0_mask = q_history_mask[:, -1]
        for b, h, o in np.argwhere(q_mask):
            observed_q0 = float(q0[b, o]) if q0_mask[b, o] else float("nan")
            true = float(q_true[b, h, o])
            pred = float(q_pred[b, h, o])
            rows.append(
                {
                    "GRAPH_ID": _normalise_id(graph_ids[b]),
                    "STATION_ID": stations[o],
                    "EVENT_ID": _normalise_id(event_ids[b]),
                    "SAMPLE_ID": str(sample_ids[b]),
                    "LEAD_HOUR": int(h) + 1,
                    "TARGET_UNIX_HOUR": int(origin_hours[b]) + int(h) + 1,
                    "Q_TRUE": true,
                    "Q_PRED": pred,
                    "Q0_OBS": observed_q0,
                    "Q0_OBS_VALID": bool(q0_mask[b, o]),
                    "DELTA_Q_TRUE": (
                        true - observed_q0 if q0_mask[b, o] else float("nan")
                    ),
                    "DELTA_Q_PRED": (
                        pred - observed_q0 if q0_mask[b, o] else float("nan")
                    ),
                }
            )
    points = pd.DataFrame(rows)
    if points.empty:
        raise ValueError("hydrologic final evaluation没有有效Q points")

    lead_rows: list[dict[str, Any]] = []
    for lead, group in points.groupby("LEAD_HOUR", sort=True):
        all_model = _metrics(
            group["Q_PRED"].to_numpy(), group["Q_TRUE"].to_numpy()
        )
        q0_group = group[group["Q0_OBS_VALID"]].copy()
        model_q0 = _metrics(
            q0_group["Q_PRED"].to_numpy(), q0_group["Q_TRUE"].to_numpy()
        )
        persistence = _metrics(
            q0_group["Q0_OBS"].to_numpy(), q0_group["Q_TRUE"].to_numpy()
        )
        delta = _metrics(
            q0_group["DELTA_Q_PRED"].to_numpy(),
            q0_group["DELTA_Q_TRUE"].to_numpy(),
        )
        lead_rows.append(
            {
                "LEAD_HOUR": int(lead),
                "MODEL_Q_COUNT": all_model["count"],
                "MODEL_Q_MAE": all_model["mae"],
                "MODEL_Q_RMSE": all_model["rmse"],
                "MODEL_Q_NSE": all_model["nse"],
                "Q0_OBS_COUNT": int(len(q0_group)),
                "MODEL_Q0_SUBSET_RMSE": model_q0["rmse"],
                "MODEL_Q0_SUBSET_NSE": model_q0["nse"],
                "PERSISTENCE_RMSE": persistence["rmse"],
                "PERSISTENCE_NSE": persistence["nse"],
                "SKILL_OVER_PERSISTENCE": _skill(
                    q0_group["Q_PRED"].to_numpy(),
                    q0_group["Q0_OBS"].to_numpy(),
                    q0_group["Q_TRUE"].to_numpy(),
                ),
                "DELTA_Q_RMSE": delta["rmse"],
                "DELTA_Q_NSE": delta["nse"],
            }
        )

    truth = _deduplicate_truth(points)
    reference: dict[tuple[str, str, str], tuple[float, tuple[int, ...], int]] = {}
    for key, group in truth.groupby(
        ["GRAPH_ID", "STATION_ID", "EVENT_ID"], sort=False
    ):
        obs = group["Q_TRUE"].to_numpy(np.float64)
        times = group["TARGET_UNIX_HOUR"].to_numpy(np.int64)
        peak = float(obs.max())
        peak_times = tuple(
            int(value)
            for value in times[
                np.isclose(obs, peak, rtol=0.0, atol=1.0e-5)
            ].tolist()
        )
        reference[(str(key[0]), str(key[1]), str(key[2]))] = (
            peak,
            peak_times,
            int(len(group)),
        )

    peak_rows: list[dict[str, Any]] = []
    for key, group in points.groupby(
        ["GRAPH_ID", "STATION_ID", "EVENT_ID", "LEAD_HOUR"], sort=False
    ):
        graph, station, event, lead = (
            str(key[0]), str(key[1]), str(key[2]), int(key[3])
        )
        duplicate = group[group.duplicated(["TARGET_UNIX_HOUR"], keep=False)]
        if not duplicate.empty:
            for _, part in duplicate.groupby("TARGET_UNIX_HOUR", sort=False):
                if (
                    part["Q_TRUE"].max() - part["Q_TRUE"].min() > 1.0e-5
                    or part["Q_PRED"].max() - part["Q_PRED"].min() > 1.0e-5
                ):
                    raise ValueError(
                        "hydrologic fixed-lead event series存在冲突duplicate"
                    )
        group = group.drop_duplicates(["TARGET_UNIX_HOUR"], keep="first").sort_values(
            "TARGET_UNIX_HOUR"
        )
        ref_peak, ref_peak_times, ref_count = reference[(graph, station, event)]
        pred = group["Q_PRED"].to_numpy(np.float64)
        obs = group["Q_TRUE"].to_numpy(np.float64)
        times = group["TARGET_UNIX_HOUR"].to_numpy(np.int64)
        predicted_peak = float(pred.max())
        predicted_peak_time = int(
            times[
                np.flatnonzero(
                    np.isclose(pred, predicted_peak, rtol=0.0, atol=1.0e-5)
                )[0]
            ]
        )
        time_set = set(int(value) for value in times.tolist())
        covered = any(value in time_set for value in ref_peak_times)
        if covered:
            signed = predicted_peak - ref_peak
            relative = signed / ref_peak if ref_peak != 0 else float("nan")
            nearest = min(
                ref_peak_times,
                key=lambda value: abs(predicted_peak_time - value),
            )
            timing = float(predicted_peak_time - nearest)
        else:
            signed = relative = timing = float("nan")
        event_metrics = _metrics(pred, obs)
        peak_rows.append(
            {
                "GRAPH_ID": graph,
                "STATION_ID": station,
                "EVENT_ID": event,
                "LEAD_HOUR": lead,
                "REFERENCE_Q_POINT_COUNT": ref_count,
                "LEAD_SERIES_POINT_COUNT": int(len(group)),
                "REFERENCE_PEAK_COVERED": bool(covered),
                "OBS_EVENT_PEAK_M3S": ref_peak,
                "PRED_EVENT_PEAK_M3S": predicted_peak,
                "PEAK_SIGNED_ERROR_M3S": signed,
                "PEAK_ABS_RELATIVE_ERROR": (
                    abs(relative) if math.isfinite(relative) else float("nan")
                ),
                "PEAK_RELATIVE_ERROR": relative,
                "PEAK_RATIO": (
                    predicted_peak / ref_peak if ref_peak != 0 else float("nan")
                ),
                "PEAK_TIMING_ERROR_H": timing,
                "PEAK_TIMING_ABS_ERROR_H": (
                    abs(timing) if math.isfinite(timing) else float("nan")
                ),
                "EVENT_Q_NSE": event_metrics["nse"],
                "EVENT_Q_RMSE": event_metrics["rmse"],
            }
        )

    lead_frame = pd.DataFrame(lead_rows)
    peak_frame = pd.DataFrame(peak_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_path = output_dir / f"{str(split).lower()}_persistence_deltaq_metrics.csv"
    peak_path = output_dir / f"{str(split).lower()}_fixed_lead_event_peak_metrics.csv"
    lead_frame.to_csv(lead_path, index=False, encoding="utf-8-sig")
    peak_frame.to_csv(peak_path, index=False, encoding="utf-8-sig")

    peak_summary: list[dict[str, Any]] = []
    for lead, group in peak_frame.groupby("LEAD_HOUR", sort=True):
        covered = group[group["REFERENCE_PEAK_COVERED"]]
        abs_relative = covered["PEAK_ABS_RELATIVE_ERROR"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        ratio = covered["PEAK_RATIO"].replace([np.inf, -np.inf], np.nan).dropna()
        timing = covered["PEAK_TIMING_ABS_ERROR_H"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        event_nse = covered["EVENT_Q_NSE"].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        peak_summary.append(
            {
                "lead_hour": int(lead),
                "event_group_count": int(len(group)),
                "reference_peak_covered_count": int(len(covered)),
                "peak_abs_relative_error_median": (
                    float(abs_relative.median()) if len(abs_relative) else float("nan")
                ),
                "peak_ratio_median": (
                    float(ratio.median()) if len(ratio) else float("nan")
                ),
                "peak_timing_mae_h": (
                    float(timing.mean()) if len(timing) else float("nan")
                ),
                "event_q_nse_median": (
                    float(event_nse.median()) if len(event_nse) else float("nan")
                ),
            }
        )
    return {
        "lead_persistence_deltaq": lead_frame.to_dict("records"),
        "fixed_lead_event_peak": peak_summary,
        "files": {
            "persistence_deltaq_metrics": str(lead_path),
            "fixed_lead_event_peak_metrics": str(peak_path),
        },
    }


@torch.no_grad()
def evaluate_forecast(
    trainer: Any,
    loader: Iterable[Any],
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    payload = evaluate_station_aware(
        trainer,
        loader,
        output_dir,
        split=split,
        checkpoint=checkpoint,
    )
    skill = _forecast_skill(
        trainer,
        loader,
        output_dir,
        split=split,
    )
    summary = payload["summary"]
    summary["model"] = "hydrologic_lstm_gnn_fc"
    summary["history_design"] = {
        "rainfall_physical_warmup_hours": 72,
        "q0_anchor_history_hours": 24,
        "qz_extended_to_72h": False,
        "state_correction": False,
    }
    # Evaluation setup intentionally does not instantiate the TRAIN sampler;
    # report the formal configured design rather than a missing runtime object.
    summary["training_design"] = {
        "train_sampling": dict(trainer.cfg["train_sampling"]),
        "loss": (
            "Q point + TRAIN-only high-flow weighted point + volume; "
            "no window-max peak loss"
        ),
    }
    summary["persistence_and_delta_q"] = skill["lead_persistence_deltaq"]
    summary["fixed_lead_event_peak"] = skill["fixed_lead_event_peak"]
    summary["rating_curve_audit"] = trainer.cfg.get("_runtime", {}).get(
        "rating_curves", {}
    )
    summary["high_flow_quantile_audit"] = trainer.cfg.get("_runtime", {}).get(
        "high_flow_quantiles", {}
    )
    payload["files"].update(skill["files"])
    summary_path = Path(payload["files"]["summary"])
    summary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload
