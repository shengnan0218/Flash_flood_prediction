"""TRAIN-only station rating calibration for current hydrologic model.

The current hydrologic data remain the source of truth.  Curves are fitted from unique
TRAIN physical target timestamps where both absolute Q and Delta-Z are valid.
Absolute Z is reconstructed as Z(t0)+Delta-Z.  Forecast-origin Q0 is deliberately
NOT required for fitting; it is only an operational anchor used by the current model.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def _bool(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y"}


def _fit_ols(q: np.ndarray, z: np.ndarray) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if q.ndim != 1 or z.shape != q.shape or q.size < 2:
        raise ValueError("rating OLS要求至少2个一维Q/Z配对点")
    q_mean = float(q.mean())
    z_mean = float(z.mean())
    denominator = float(np.square(q - q_mean).sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("rating TRAIN Q无方差")
    slope = float(((q - q_mean) * (z - z_mean)).sum() / denominator)
    intercept = float(z_mean - slope * q_mean)
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0:
        raise ValueError("rating要求TRAIN-only线性Q-Z斜率为有限正数")
    prediction = slope * q + intercept
    error = prediction - z
    target_variance = float(np.square(z - z_mean).sum())
    nse = (
        1.0 - float(np.square(error).sum()) / target_variance
        if target_variance > 0
        else float("nan")
    )
    corr = (
        float(np.corrcoef(q, z)[0, 1])
        if q.size >= 2 and float(np.std(q)) > 0 and float(np.std(z)) > 0
        else float("nan")
    )
    return {
        "slope_m_per_m3s": slope,
        "intercept_m": intercept,
        "train_rmse_m": float(np.sqrt(np.square(error).mean())),
        "train_mae_m": float(np.abs(error).mean()),
        "train_nse": nse,
        "train_corr": corr,
        "q_min_m3s": float(q.min()),
        "q_max_m3s": float(q.max()),
        "z_min_m": float(z.min()),
        "z_max_m": float(z.max()),
    }


def fit_train_only_linear_ratings(
    dataset_root: str | Path,
    station_ids: tuple[str, ...],
    *,
    min_unique_pairs: int,
    require_all_outlet_stations: bool,
) -> dict[str, Any]:
    """Fit deterministic station linear ratings from the full frozen TRAIN split."""
    root = Path(dataset_root).expanduser().resolve()
    if min_unique_pairs < 2:
        raise ValueError("stage_output.min_unique_train_pairs必须>=2")
    sample_path = root / "samples/sample_index.csv"
    mapping_path = root / "graph/station_observation_mapping.csv"
    if not sample_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError("rating需要hydrologic sample_index和station mapping")

    samples = pd.read_csv(sample_path, encoding="utf-8-sig", dtype=str)
    required = {"SPLIT", "TENSOR_FILE", "TENSOR_ROW", "GRAPH_ID"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"rating sample_index缺字段: {sorted(missing)}")
    samples["SPLIT"] = samples["SPLIT"].str.upper()
    samples = samples[samples["SPLIT"].eq("TRAIN")].copy()
    samples["TENSOR_ROW"] = pd.to_numeric(samples["TENSOR_ROW"], errors="raise").astype(np.int64)
    if samples.empty:
        raise ValueError("rating没有TRAIN样本")

    station_catalogue = tuple(_norm_id(value) for value in station_ids)
    if len(set(station_catalogue)) != len(station_catalogue):
        raise ValueError("globalstation catalogue含重复ID")
    known = set(station_catalogue)
    paired: dict[tuple[str, int], tuple[float, float]] = {}
    candidate_occurrences = 0
    duplicate_conflicts = 0

    for relative_name, group in samples.groupby("TENSOR_FILE", sort=True):
        relative = str(relative_name).replace("\\", "/")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"rating非法TENSOR_FILE: {relative}")
        with np.load(path, allow_pickle=False) as archive:
            required_keys = {
                "obs_station_id", "forecast_time_unix_hour", "q_target",
                "q_target_mask", "z_history", "z_history_mask", "z_target",
                "z_target_mask",
            }
            missing_keys = required_keys - set(archive.files)
            if missing_keys:
                raise ValueError(f"{path.name}: 缺少rating所需字段{sorted(missing_keys)}")
            local_stations = tuple(_norm_id(v) for v in archive["obs_station_id"].tolist())
            if set(local_stations) - known:
                raise ValueError(f"{path.name}: station不在全局目录")
            q = archive["q_target"]
            q_mask = archive["q_target_mask"].astype(bool)
            z_history = archive["z_history"]
            z_history_mask = archive["z_history_mask"].astype(bool)
            dz = archive["z_target"]
            dz_mask = archive["z_target_mask"].astype(bool)
            origin = archive["forecast_time_unix_hour"].astype(np.int64)
            if q.shape != dz.shape or q.ndim != 3:
                raise ValueError(f"{path.name}: rating target tensor shape非法")
            sample_count, obs_count, horizon = q.shape
            if len(local_stations) != obs_count or z_history.shape[:2] != (sample_count, obs_count):
                raise ValueError(f"{path.name}: rating observation shape非法")
            for row in group.itertuples(index=False):
                index = int(row.TENSOR_ROW)
                if not 0 <= index < sample_count:
                    raise IndexError(f"{path.name}: TENSOR_ROW越界={index}")
                z0 = z_history[index, :, -1].astype(np.float64)
                z0_mask = z_history_mask[index, :, -1].astype(bool)
                for obs, station in enumerate(local_stations):
                    for lead in range(horizon):
                        if not (q_mask[index, obs, lead] and dz_mask[index, obs, lead]):
                            continue
                        if not z0_mask[obs] or not np.isfinite(z0[obs]):
                            raise ValueError("rating发现Delta-Z有效但Z0无效")
                        q_value = float(q[index, obs, lead])
                        z_value = float(z0[obs] + dz[index, obs, lead])
                        if not math.isfinite(q_value) or not math.isfinite(z_value):
                            raise ValueError("rating有效Q/Z含NaN/Inf")
                        candidate_occurrences += 1
                        key = (station, int(origin[index]) + lead + 1)
                        previous = paired.get(key)
                        if previous is not None:
                            if abs(previous[0] - q_value) > 1.0e-5 or abs(previous[1] - z_value) > 1.0e-5:
                                duplicate_conflicts += 1
                            continue
                        paired[key] = (q_value, z_value)
    if duplicate_conflicts:
        raise ValueError(f"rating重叠TRAIN窗口存在{duplicate_conflicts}个矛盾Q/Z值")

    by_station: dict[str, list[tuple[float, float]]] = {station: [] for station in station_catalogue}
    for (station, _timestamp), values in paired.items():
        by_station[station].append(values)

    station_statistics: dict[str, dict[str, Any]] = {}
    for station in station_catalogue:
        values = by_station[station]
        record: dict[str, Any] = {
            "available": False,
            "unique_train_pair_count": len(values),
        }
        if len(values) >= min_unique_pairs:
            q_values = np.asarray([item[0] for item in values], dtype=np.float64)
            z_values = np.asarray([item[1] for item in values], dtype=np.float64)
            try:
                record.update(_fit_ols(q_values, z_values))
                record["available"] = True
            except ValueError as exc:
                record["fit_error"] = str(exc)
        station_statistics[station] = record

    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype=str)
    for field in ("STATION_ID", "IS_OUTLET_STATION"):
        if field not in mapping.columns:
            raise ValueError(f"rating mapping缺少{field}")
    mapping["STATION_ID"] = mapping["STATION_ID"].map(_norm_id)
    outlet_stations = sorted(
        mapping.loc[mapping["IS_OUTLET_STATION"].map(_bool), "STATION_ID"].unique().tolist()
    )
    missing_outlets = [
        station for station in outlet_stations
        if station not in station_statistics or not station_statistics[station]["available"]
    ]
    if require_all_outlet_stations and missing_outlets:
        raise ValueError(
            "stage output要求所有outlet有TRAIN-only rating curve，缺失="
            f"{missing_outlets}"
        )

    payload: dict[str, Any] = {
        "method": "TRAIN_ONLY_STATION_LINEAR_OLS",
        "fit_split": "TRAIN",
        "deduplication_key": "STATION_ID+PHYSICAL_TARGET_UNIX_HOUR",
        "absolute_z_reconstruction": "Z(t0)+Delta-Z(t+h)",
        "q0_required_for_curve_fit": False,
        "min_unique_train_pairs": int(min_unique_pairs),
        "candidate_pair_occurrences": int(candidate_occurrences),
        "unique_pair_count": int(len(paired)),
        "duplicate_value_conflict_count": int(duplicate_conflicts),
        "station_count": len(station_catalogue),
        "available_station_count": sum(bool(v["available"]) for v in station_statistics.values()),
        "outlet_station_count": len(outlet_stations),
        "outlet_missing_curve": missing_outlets,
        "stations": station_statistics,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=True)
    payload["artifact_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload
