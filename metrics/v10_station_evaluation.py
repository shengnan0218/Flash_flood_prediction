"""Final station/event/graph evaluation for V10 Q-only + derived stage."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from metrics.v8_station_evaluation import (
    _array_hydrograph,
    _array_sums,
    _batch_strings,
    _empty_hydrograph,
    _empty_regression,
    _hydrograph_fields,
    _json_safe,
    _load_station_metadata,
    _macro_hydrograph,
    _macro_regression,
    _merge,
    _normalise_id,
    _regression_fields,
)


def _empty_extrapolation() -> dict[str, int]:
    return {
        "raw_future_count": 0,
        "raw_future_outside_train_q_range": 0,
        "corrected_future_count": 0,
        "corrected_future_outside_train_q_range": 0,
        "origin_count": 0,
        "origin_outside_train_q_range": 0,
    }


def _merge_extrapolation(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += int(source[key])


def _extrapolation_fields(values: dict[str, int]) -> dict[str, float | int]:
    result: dict[str, float | int] = dict(values)
    for prefix in ("raw_future", "corrected_future", "origin"):
        count = int(values[f"{prefix}_count"])
        outside = int(values[f"{prefix}_outside_train_q_range"])
        result[f"{prefix}_outside_train_q_range_fraction"] = (
            outside / count if count else float("nan")
        )
    return result


@torch.no_grad()
def evaluate_v10_station_aware(
    trainer: Any,
    loader: Iterable[Any],
    output_dir: str | Path,
    *,
    split: str,
    checkpoint: str | Path,
) -> dict[str, Any]:
    """Evaluate learned Q and explicitly masked non-neural stage in one pass."""
    model = trainer.model
    device = trainer.device
    model.eval()
    dataset = loader.dataset
    station_meta, outlet_by_graph = _load_station_metadata(dataset)
    horizon = int(trainer.cfg["forecast_horizon"])

    station_q = defaultdict(_empty_regression)
    station_dz = defaultdict(_empty_regression)
    station_z_abs = defaultdict(_empty_regression)
    station_z_raw = defaultdict(_empty_regression)
    station_q_hydro = defaultdict(_empty_hydrograph)
    station_extrapolation = defaultdict(_empty_extrapolation)

    graph_q = defaultdict(_empty_regression)
    graph_dz = defaultdict(_empty_regression)
    graph_z_abs = defaultdict(_empty_regression)
    graph_outlet_q = defaultdict(_empty_regression)
    graph_outlet_dz = defaultdict(_empty_regression)
    graph_outlet_z_abs = defaultdict(_empty_regression)
    graph_outlet_q_hydro = defaultdict(_empty_hydrograph)

    event_q = defaultdict(_empty_regression)
    event_dz = defaultdict(_empty_regression)
    event_z_abs = defaultdict(_empty_regression)
    event_q_hydro = defaultdict(_empty_hydrograph)
    event_sample_ids: dict[tuple[str, str, str], set[str]] = defaultdict(set)

    q_global = _empty_regression()
    dz_global = _empty_regression()
    z_abs_global = _empty_regression()
    z_raw_global = _empty_regression()
    q_hydro_global = _empty_hydrograph()
    outlet_q_global = _empty_regression()
    outlet_dz_global = _empty_regression()
    outlet_z_abs_global = _empty_regression()
    outlet_q_hydro_global = _empty_hydrograph()
    extrapolation_global = _empty_extrapolation()

    lead_q = [_empty_regression() for _ in range(horizon)]
    lead_dz = [_empty_regression() for _ in range(horizon)]
    lead_z_abs = [_empty_regression() for _ in range(horizon)]
    lead_outlet_q = [_empty_regression() for _ in range(horizon)]
    lead_outlet_dz = [_empty_regression() for _ in range(horizon)]
    lead_outlet_z_abs = [_empty_regression() for _ in range(horizon)]

    sample_total = 0
    stage_target_valid = 0
    stage_corrected_valid = 0
    raw_stage_valid = 0
    q0_observed_total = 0
    z0_observed_total = 0

    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        q_pred = output["q"].detach().float().cpu().numpy()
        q_true = batch.q_target.detach().float().cpu().numpy()
        q_mask = batch.q_target_mask.detach().cpu().numpy().astype(bool)
        dz_pred = output["z_delta"].detach().float().cpu().numpy()
        z_abs_pred = output["z_abs"].detach().float().cpu().numpy()
        z_raw_pred = output["z_rating_raw_abs"].detach().float().cpu().numpy()
        q0_analysis = output["q0_analysis"].detach().float().cpu().numpy()
        stage_available = output["z_available_mask"].detach().cpu().numpy().astype(bool)
        raw_available = (
            output["z_rating_raw_available_mask"].detach().cpu().numpy().astype(bool)
        )
        dz_true = batch.z_target.detach().float().cpu().numpy()
        z_target_mask = batch.z_target_mask.detach().cpu().numpy().astype(bool)
        z0 = batch.z_history[:, -1].detach().float().cpu().numpy()
        z0_mask = batch.z_mask[:, -1].detach().cpu().numpy().astype(bool)
        q0_mask = batch.q_mask[:, -1].detach().cpu().numpy().astype(bool)
        z_abs_true = z0[:, None, :] + dz_true
        corrected_mask = z_target_mask & stage_available
        raw_mask = z_target_mask & raw_available & z0_mask[:, None, :]

        batch_size = int(q_pred.shape[0])
        if q_pred.shape != q_true.shape or dz_pred.shape != dz_true.shape:
            raise ValueError("v10 evaluation prediction/target shape不一致")
        graph_ids = _batch_strings(batch.graph_id, batch_size, "graph_id")
        event_ids = _batch_strings(batch.event_id, batch_size, "event_id")
        sample_ids = _batch_strings(batch.sample_id, batch_size, "sample_id")
        graph_id = _normalise_id(graph_ids[0])
        if any(_normalise_id(value) != graph_id for value in graph_ids):
            raise ValueError("v10 evaluation batch混入多个graph")

        local_station_ids = tuple(_normalise_id(value) for value in batch.obs_station_ids)
        outlet_station = outlet_by_graph[graph_id]
        outlet_matches = [
            index for index, station in enumerate(local_station_ids)
            if station == outlet_station
        ]
        if len(outlet_matches) != 1:
            raise ValueError(
                f"{graph_id}: v10 evaluation必须唯一定位outlet={outlet_station}，"
                f"实际matches={outlet_matches}"
            )
        outlet_index = outlet_matches[0]

        # Rating-domain diagnostics use predicted Q/Q0 rather than truth.  They
        # never clamp or modify outputs; they only expose TRAIN-range extrapolation.
        local_station_index = batch.obs_station_index.detach().long().cpu()
        q_min = model.rating.q_min_m3s.detach().cpu()[local_station_index].numpy()
        q_max = model.rating.q_max_m3s.detach().cpu()[local_station_index].numpy()
        rating_available_station = (
            model.rating.available.detach().cpu()[local_station_index].numpy().astype(bool)
        )
        future_outside = (q_pred < q_min[None, None, :]) | (
            q_pred > q_max[None, None, :]
        )
        origin_outside = (q0_analysis < q_min[None, :]) | (
            q0_analysis > q_max[None, :]
        )
        raw_domain_mask = np.broadcast_to(
            rating_available_station[None, None, :], q_pred.shape
        )
        corrected_domain_mask = stage_available
        origin_domain_mask = np.broadcast_to(
            rating_available_station[None, :], q0_analysis.shape
        )
        _merge_extrapolation(
            extrapolation_global,
            {
                "raw_future_count": int(raw_domain_mask.sum()),
                "raw_future_outside_train_q_range": int(
                    (raw_domain_mask & future_outside).sum()
                ),
                "corrected_future_count": int(corrected_domain_mask.sum()),
                "corrected_future_outside_train_q_range": int(
                    (corrected_domain_mask & future_outside).sum()
                ),
                "origin_count": int(origin_domain_mask.sum()),
                "origin_outside_train_q_range": int(
                    (origin_domain_mask & origin_outside).sum()
                ),
            },
        )

        _merge(q_global, _array_sums(q_pred, q_true, q_mask))
        _merge(dz_global, _array_sums(dz_pred, dz_true, corrected_mask))
        _merge(z_abs_global, _array_sums(z_abs_pred, z_abs_true, corrected_mask))
        _merge(z_raw_global, _array_sums(z_raw_pred, z_abs_true, raw_mask))
        _merge(q_hydro_global, _array_hydrograph(q_pred, q_true, q_mask))
        _merge(graph_q[graph_id], _array_sums(q_pred, q_true, q_mask))
        _merge(graph_dz[graph_id], _array_sums(dz_pred, dz_true, corrected_mask))
        _merge(
            graph_z_abs[graph_id],
            _array_sums(z_abs_pred, z_abs_true, corrected_mask),
        )

        oq_pred = q_pred[:, :, outlet_index : outlet_index + 1]
        oq_true = q_true[:, :, outlet_index : outlet_index + 1]
        oq_mask = q_mask[:, :, outlet_index : outlet_index + 1]
        odz_pred = dz_pred[:, :, outlet_index : outlet_index + 1]
        odz_true = dz_true[:, :, outlet_index : outlet_index + 1]
        oz_pred = z_abs_pred[:, :, outlet_index : outlet_index + 1]
        oz_true = z_abs_true[:, :, outlet_index : outlet_index + 1]
        ostage_mask = corrected_mask[:, :, outlet_index : outlet_index + 1]
        _merge(outlet_q_global, _array_sums(oq_pred, oq_true, oq_mask))
        _merge(outlet_dz_global, _array_sums(odz_pred, odz_true, ostage_mask))
        _merge(outlet_z_abs_global, _array_sums(oz_pred, oz_true, ostage_mask))
        _merge(outlet_q_hydro_global, _array_hydrograph(oq_pred, oq_true, oq_mask))
        _merge(graph_outlet_q[graph_id], _array_sums(oq_pred, oq_true, oq_mask))
        _merge(graph_outlet_dz[graph_id], _array_sums(odz_pred, odz_true, ostage_mask))
        _merge(
            graph_outlet_z_abs[graph_id],
            _array_sums(oz_pred, oz_true, ostage_mask),
        )
        _merge(
            graph_outlet_q_hydro[graph_id],
            _array_hydrograph(oq_pred, oq_true, oq_mask),
        )

        for lead in range(horizon):
            _merge(
                lead_q[lead],
                _array_sums(q_pred[:, lead], q_true[:, lead], q_mask[:, lead]),
            )
            _merge(
                lead_dz[lead],
                _array_sums(
                    dz_pred[:, lead], dz_true[:, lead], corrected_mask[:, lead]
                ),
            )
            _merge(
                lead_z_abs[lead],
                _array_sums(
                    z_abs_pred[:, lead], z_abs_true[:, lead], corrected_mask[:, lead]
                ),
            )
            _merge(
                lead_outlet_q[lead],
                _array_sums(
                    q_pred[:, lead, outlet_index],
                    q_true[:, lead, outlet_index],
                    q_mask[:, lead, outlet_index],
                ),
            )
            _merge(
                lead_outlet_dz[lead],
                _array_sums(
                    dz_pred[:, lead, outlet_index],
                    dz_true[:, lead, outlet_index],
                    corrected_mask[:, lead, outlet_index],
                ),
            )
            _merge(
                lead_outlet_z_abs[lead],
                _array_sums(
                    z_abs_pred[:, lead, outlet_index],
                    z_abs_true[:, lead, outlet_index],
                    corrected_mask[:, lead, outlet_index],
                ),
            )

        for obs, station_id in enumerate(local_station_ids):
            _merge(
                station_q[station_id],
                _array_sums(q_pred[:, :, obs], q_true[:, :, obs], q_mask[:, :, obs]),
            )
            _merge(
                station_q_hydro[station_id],
                _array_hydrograph(
                    q_pred[:, :, obs : obs + 1],
                    q_true[:, :, obs : obs + 1],
                    q_mask[:, :, obs : obs + 1],
                ),
            )
            _merge(
                station_dz[station_id],
                _array_sums(
                    dz_pred[:, :, obs],
                    dz_true[:, :, obs],
                    corrected_mask[:, :, obs],
                ),
            )
            _merge(
                station_z_abs[station_id],
                _array_sums(
                    z_abs_pred[:, :, obs],
                    z_abs_true[:, :, obs],
                    corrected_mask[:, :, obs],
                ),
            )
            _merge(
                station_z_raw[station_id],
                _array_sums(
                    z_raw_pred[:, :, obs], z_abs_true[:, :, obs], raw_mask[:, :, obs]
                ),
            )
            _merge_extrapolation(
                station_extrapolation[station_id],
                {
                    "raw_future_count": int(raw_domain_mask[:, :, obs].sum()),
                    "raw_future_outside_train_q_range": int(
                        (raw_domain_mask[:, :, obs] & future_outside[:, :, obs]).sum()
                    ),
                    "corrected_future_count": int(
                        corrected_domain_mask[:, :, obs].sum()
                    ),
                    "corrected_future_outside_train_q_range": int(
                        (
                            corrected_domain_mask[:, :, obs]
                            & future_outside[:, :, obs]
                        ).sum()
                    ),
                    "origin_count": int(origin_domain_mask[:, obs].sum()),
                    "origin_outside_train_q_range": int(
                        (origin_domain_mask[:, obs] & origin_outside[:, obs]).sum()
                    ),
                },
            )

            for sample_index in range(batch_size):
                event_id = str(event_ids[sample_index])
                event_key = (graph_id, event_id, station_id)
                _merge(
                    event_q[event_key],
                    _array_sums(
                        q_pred[sample_index, :, obs],
                        q_true[sample_index, :, obs],
                        q_mask[sample_index, :, obs],
                    ),
                )
                _merge(
                    event_dz[event_key],
                    _array_sums(
                        dz_pred[sample_index, :, obs],
                        dz_true[sample_index, :, obs],
                        corrected_mask[sample_index, :, obs],
                    ),
                )
                _merge(
                    event_z_abs[event_key],
                    _array_sums(
                        z_abs_pred[sample_index, :, obs],
                        z_abs_true[sample_index, :, obs],
                        corrected_mask[sample_index, :, obs],
                    ),
                )
                _merge(
                    event_q_hydro[event_key],
                    _array_hydrograph(
                        q_pred[sample_index : sample_index + 1, :, obs : obs + 1],
                        q_true[sample_index : sample_index + 1, :, obs : obs + 1],
                        q_mask[sample_index : sample_index + 1, :, obs : obs + 1],
                    ),
                )
                event_sample_ids[event_key].add(str(sample_ids[sample_index]))

        sample_total += batch_size
        stage_target_valid += int(z_target_mask.sum())
        stage_corrected_valid += int(corrected_mask.sum())
        raw_stage_valid += int(raw_mask.sum())
        q0_observed_total += int(q0_mask.sum())
        z0_observed_total += int(z0_mask.sum())

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_name = str(split).lower()

    station_rows: list[dict[str, Any]] = []
    for (graph_id, station_id), meta in sorted(station_meta.items()):
        if station_id not in station_q and station_id not in station_dz:
            continue
        station_rows.append(
            {
                "GRAPH_ID": graph_id,
                "STATION_ID": station_id,
                "STATION_ROLE": meta["station_role"],
                "IS_OUTLET_STATION": int(meta["is_outlet_station"]),
                **_regression_fields("q", station_q[station_id]),
                **_hydrograph_fields("q", station_q_hydro[station_id]),
                **_regression_fields(
                    "delta_z_rating_corrected", station_dz[station_id]
                ),
                **_regression_fields(
                    "z_abs_rating_corrected", station_z_abs[station_id]
                ),
                **_regression_fields(
                    "z_abs_rating_unanchored", station_z_raw[station_id]
                ),
                **_extrapolation_fields(station_extrapolation[station_id]),
            }
        )

    graph_rows: list[dict[str, Any]] = []
    for graph_id in sorted(graph_q):
        graph_rows.append(
            {
                "GRAPH_ID": graph_id,
                "OUTLET_STATION_ID": outlet_by_graph.get(graph_id, ""),
                **_regression_fields("all_station_q", graph_q[graph_id]),
                **_regression_fields("all_station_delta_z", graph_dz[graph_id]),
                **_regression_fields("all_station_z_abs", graph_z_abs[graph_id]),
                **_regression_fields("outlet_q", graph_outlet_q[graph_id]),
                **_hydrograph_fields("outlet_q", graph_outlet_q_hydro[graph_id]),
                **_regression_fields("outlet_delta_z", graph_outlet_dz[graph_id]),
                **_regression_fields("outlet_z_abs", graph_outlet_z_abs[graph_id]),
            }
        )

    event_rows: list[dict[str, Any]] = []
    for graph_id, event_id, station_id in sorted(event_q):
        meta = station_meta.get((graph_id, station_id))
        if meta is None:
            raise ValueError(
                f"v10 event evaluation缺station metadata: {graph_id}/{station_id}"
            )
        key = (graph_id, event_id, station_id)
        event_rows.append(
            {
                "GRAPH_ID": graph_id,
                "EVENT_ID": event_id,
                "STATION_ID": station_id,
                "STATION_ROLE": meta["station_role"],
                "IS_OUTLET_STATION": int(meta["is_outlet_station"]),
                "SAMPLE_COUNT": len(event_sample_ids[key]),
                **_regression_fields("q", event_q[key]),
                **_hydrograph_fields("q", event_q_hydro[key]),
                **_regression_fields("delta_z_rating_corrected", event_dz[key]),
                **_regression_fields("z_abs_rating_corrected", event_z_abs[key]),
            }
        )

    step_seconds = float(
        trainer.cfg.get("temporal", {}).get("target_step_seconds", 3600.0)
    )
    lead_rows: list[dict[str, Any]] = []
    for lead in range(horizon):
        lead_rows.append(
            {
                "LEAD_INDEX": lead + 1,
                "LEAD_SECONDS": (lead + 1) * step_seconds,
                "LEAD_MINUTES": (lead + 1) * step_seconds / 60.0,
                "LEAD_HOURS": (lead + 1) * step_seconds / 3600.0,
                **_regression_fields("all_station_q", lead_q[lead]),
                **_regression_fields("all_station_delta_z", lead_dz[lead]),
                **_regression_fields("all_station_z_abs", lead_z_abs[lead]),
                **_regression_fields("outlet_q", lead_outlet_q[lead]),
                **_regression_fields("outlet_delta_z", lead_outlet_dz[lead]),
                **_regression_fields("outlet_z_abs", lead_outlet_z_abs[lead]),
            }
        )

    station_path = output_dir / f"{split_name}_station_metrics.csv"
    graph_path = output_dir / f"{split_name}_graph_metrics.csv"
    event_path = output_dir / f"{split_name}_event_station_metrics.csv"
    lead_path = output_dir / f"{split_name}_lead_time_metrics.csv"
    summary_path = output_dir / f"{split_name}_summary.json"
    pd.DataFrame(station_rows).to_csv(station_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(graph_rows).to_csv(graph_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(event_rows).to_csv(event_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(lead_rows).to_csv(lead_path, index=False, encoding="utf-8-sig")

    outlet_station_rows = [
        row for row in station_rows if int(row["IS_OUTLET_STATION"]) == 1
    ]
    summary = {
        "model_version": "v10",
        "supervised_forecast_target": "Q_ONLY",
        "stage_prediction_method": (
            "Q-derived TRAIN-only station linear rating + final-history-bin Z residual correction"
        ),
        "history_anchor_semantics": (
            "Q0/Z0 are retained observations in the final history hourly bin; "
            "not guaranteed exact end-of-bin instantaneous values"
        ),
        "future_z_observation_used": False,
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "split": str(split).upper(),
        "sample_count": sample_total,
        "q_all_stations_pooled": {
            **_regression_fields("q", q_global),
            **_hydrograph_fields("q", q_hydro_global),
        },
        "q_outlets_pooled": {
            **_regression_fields("q", outlet_q_global),
            **_hydrograph_fields("q", outlet_q_hydro_global),
        },
        "derived_stage_all_stations_pooled": {
            **_regression_fields("delta_z_rating_corrected", dz_global),
            **_regression_fields("z_abs_rating_corrected", z_abs_global),
            **_regression_fields("z_abs_rating_unanchored", z_raw_global),
        },
        "derived_stage_outlets_pooled": {
            **_regression_fields("delta_z_rating_corrected", outlet_dz_global),
            **_regression_fields("z_abs_rating_corrected", outlet_z_abs_global),
        },
        "station_macro": {
            "q": _macro_regression(station_rows, "q"),
            "q_hydrograph": _macro_hydrograph(station_rows, "q"),
            "delta_z_rating_corrected": _macro_regression(
                station_rows, "delta_z_rating_corrected"
            ),
            "z_abs_rating_corrected": _macro_regression(
                station_rows, "z_abs_rating_corrected"
            ),
        },
        "outlet_station_macro": {
            "q": _macro_regression(outlet_station_rows, "q"),
            "q_hydrograph": _macro_hydrograph(outlet_station_rows, "q"),
            "delta_z_rating_corrected": _macro_regression(
                outlet_station_rows, "delta_z_rating_corrected"
            ),
            "z_abs_rating_corrected": _macro_regression(
                outlet_station_rows, "z_abs_rating_corrected"
            ),
        },
        "derived_stage_coverage": {
            "z_target_valid_count": stage_target_valid,
            "corrected_stage_valid_count": stage_corrected_valid,
            "corrected_stage_coverage_of_z_target": (
                stage_corrected_valid / stage_target_valid if stage_target_valid else 0.0
            ),
            "raw_rating_stage_valid_count": raw_stage_valid,
            "final_history_bin_q_observed_count": q0_observed_total,
            "final_history_bin_z_observed_count": z0_observed_total,
        },
        "rating_extrapolation": _extrapolation_fields(extrapolation_global),
        "event_station_group_count": len(event_rows),
        "rating_curve_audit": trainer.cfg.get("_runtime", {}).get(
            "v10_rating_curves", {}
        ),
        "evaluation_view": trainer.cfg.get("_runtime", {}).get("evaluation_view"),
    }
    payload = {
        "summary": summary,
        "files": {
            "station_metrics": str(station_path),
            "graph_metrics": str(graph_path),
            "event_station_metrics": str(event_path),
            "lead_time_metrics": str(lead_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return payload
