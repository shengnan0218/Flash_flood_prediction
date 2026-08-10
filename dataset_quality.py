"""Deterministic event-overlap and water-level quality audits.

The upstream event builder is intentionally not reimplemented here.  These
audits consume the formal dataset contract, expose evidence needed to repair
the upstream data, and provide strict pre-training gates without rewriting
EVENT_ID, split, sample, or normalization artifacts in place.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


EVENT_OVERLAP_FIELDS = (
    "GRAPH_ID",
    "EVENT_ID_A",
    "EVENT_ID_B",
    "target_station_id",
    "target_variable",
    "hydro_start_A",
    "hydro_end_A",
    "hydro_start_B",
    "hydro_end_B",
    "overlap_hours",
    "overlap_fraction",
    "official_hydro_overlap_hours",
    "hydro_gap_hours",
    "same_observed_peak_time",
    "peak_time_A",
    "peak_time_B",
    "observed_peak_A",
    "observed_peak_B",
    "split_A",
    "split_B",
    "cross_split",
    "status",
    "reason",
    "suggested_action",
)

WATER_LEVEL_AUDIT_FIELDS = (
    "station_id",
    "graph_ids",
    "split",
    "valid_count",
    "min_z",
    "max_z",
    "mean_z",
    "std_z",
    "train_min_z",
    "train_max_z",
    "out_of_train_range_count",
    "out_of_train_range_fraction",
    "normalization_train_min_z",
    "normalization_train_max_z",
    "out_of_normalization_range_count",
    "out_of_normalization_range_fraction",
    "max_abs_delta_z",
    "train_jump_outer_fence_m",
    "suspicious_jump_count",
    "train_event_count",
    "train_reference_shift_event_count",
    "train_reference_shift_event_ids",
    "max_train_event_median_shift_m",
    "normalization_computed_from_split",
    "qc_status",
    "qc_reason",
)


@dataclass(frozen=True)
class DatasetQualityAudit:
    event_hydrograph_overlap: pd.DataFrame
    water_level_station_audit: pd.DataFrame
    summary: dict[str, Any]

    def write(self, output_dir: str | Path) -> dict[str, str]:
        """Atomically write the two QC tables and their compact summary."""

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        tables = {
            "event_hydrograph_overlap.csv": self.event_hydrograph_overlap,
            "water_level_station_audit.csv": self.water_level_station_audit,
        }
        written: dict[str, str] = {}
        for filename, frame in tables.items():
            path = output / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(temporary, index=False, encoding="utf-8")
            temporary.replace(path)
            written[filename] = str(path)
        summary_path = output / "dataset_quality_audit_summary.json"
        temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
        written[summary_path.name] = str(summary_path)
        return written


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"数据质量审计缺少文件: {path}")
    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={
                "EVENT_ID": str,
                "GRAPH_ID": str,
                "OUTLET_ID": str,
                "STATION_ID": str,
                "SAMPLE_ID": str,
            },
            low_memory=False,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"数据质量审计不接受无表头空CSV: {path}") from exc


def _parse_times(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")


def _mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()
    return numeric.eq(1) | text.isin({"true", "t", "yes", "y"})


def _iso(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).isoformat(sep=" ")


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _hour_set(rows: pd.DataFrame) -> frozenset[pd.Timestamp]:
    hours: set[pd.Timestamp] = set()
    for start, end in zip(rows["TARGET_START"], rows["TARGET_END"]):
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("sample_index包含无效TARGET_START/TARGET_END")
        hours.update(pd.date_range(start, end, freq="h"))
    return frozenset(hours)


def _validate_event_references(
    events: pd.DataFrame, splits: pd.DataFrame, samples: pd.DataFrame
) -> pd.DataFrame:
    event_ids = set(events["EVENT_ID"].astype(str))
    split_ids = set(splits["EVENT_ID"].astype(str))
    sample_ids = set(samples["EVENT_ID"].astype(str))
    stale_samples = sorted(sample_ids - event_ids)
    stale_splits = sorted(split_ids - event_ids)
    missing_splits = sorted(event_ids - split_ids)
    if stale_samples:
        raise ValueError(f"sample_index存在stale EVENT_ID: {stale_samples[:10]}")
    if stale_splits or missing_splits:
        raise ValueError(
            "data_split与正式事件表外键不一致: "
            f"stale={stale_splits[:10]}, missing={missing_splits[:10]}"
        )
    split_map = splits.set_index("EVENT_ID")["SPLIT"].astype(str).str.upper()
    mapped = samples["EVENT_ID"].map(split_map)
    sample_split = samples["SPLIT"].astype(str).str.upper()
    mismatch = mapped.ne(sample_split)
    if mismatch.any():
        examples = samples.loc[mismatch, ["SAMPLE_ID", "EVENT_ID", "SPLIT"]]
        raise ValueError(
            "sample_index与data_split的SPLIT不一致: "
            + examples.head(10).to_dict(orient="records").__repr__()
        )
    result = events.copy()
    authoritative = result["EVENT_ID"].map(split_map)
    if "SPLIT" in result:
        given = result["SPLIT"].astype(str).str.upper()
        if given.ne(authoritative).any():
            raise ValueError("flood_events_final与data_split的SPLIT不一致")
    result["SPLIT"] = authoritative
    return result


def _target_variable(value: Any) -> str:
    names = {
        item.strip().upper()
        for item in str(value).replace("+", ";").replace(",", ";").split(";")
        if item.strip()
    }
    if "BOTH" in names or "FLOW" in names:
        return "FLOW"
    if "WATER_LEVEL" in names:
        return "WATER_LEVEL"
    raise ValueError(f"不支持的TARGET_VARIABLE={value!r}")


def _observed_series(
    root: Path,
    graph_id: str,
    basin_id: str,
    station_id: str,
    variable: str,
) -> tuple[pd.Series, int]:
    frame = _read_csv(root / "dynamic" / f"graph_{basin_id}_hourly.csv")
    _parse_times(frame, ("TIMESTAMP",))
    values = pd.to_numeric(frame[variable], errors="coerce")
    valid_mask = _mask(frame[f"{variable}_MASK"])
    station_mask = frame["STATION_ID"].astype(str).eq(str(station_id))
    invalid_count = int((valid_mask & station_mask & ~values.map(math.isfinite)).sum())
    valid = valid_mask & station_mask & values.map(math.isfinite) & frame["TIMESTAMP"].notna()
    series = pd.Series(
        values.loc[valid].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame.loc[valid, "TIMESTAMP"]),
        dtype=float,
    ).sort_index()
    if series.index.has_duplicates:
        duplicated = series.index[series.index.duplicated()].unique()[:5].tolist()
        raise ValueError(
            f"GRAPH_ID={graph_id}, STATION_ID={station_id}存在重复动态时间: {duplicated}"
        )
    return series, invalid_count


def _event_overlap_audit(
    root: Path,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    target_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = {
        str(row.GRAPH_ID): _target_variable(row.TARGET_VARIABLE)
        for row in target_mapping.itertuples(index=False)
    }
    event_info: dict[str, dict[str, Any]] = {}
    for graph_id, graph_samples in samples.groupby("GRAPH_ID", sort=True):
        graph_id = str(graph_id)
        graph_events = events[events["GRAPH_ID"].astype(str).eq(graph_id)]
        if graph_events.empty:
            raise ValueError(f"sample_index引用没有正式事件的GRAPH_ID={graph_id}")
        station_ids = graph_events["OUTLET_ID"].astype(str).unique()
        if len(station_ids) != 1:
            raise ValueError(f"GRAPH_ID={graph_id}存在多个target OUTLET_ID")
        station_id = station_ids[0]
        basin_ids = graph_events["BASIN_ID"].astype(str).unique()
        if len(basin_ids) != 1:
            raise ValueError(f"GRAPH_ID={graph_id}存在多个BASIN_ID")
        basin_id = basin_ids[0]
        variable = mapping.get(graph_id)
        if variable is None:
            raise ValueError(f"target_variable_by_graph.csv缺少GRAPH_ID={graph_id}")
        series, _invalid = _observed_series(
            root, graph_id, basin_id, station_id, variable
        )
        event_rows = graph_events.set_index("EVENT_ID")
        for event_id, event_samples in graph_samples.groupby("EVENT_ID", sort=True):
            event_id = str(event_id)
            if event_id not in event_rows.index:
                raise ValueError(f"sample_index引用未知EVENT_ID={event_id}")
            row = event_rows.loc[event_id]
            hours = _hour_set(event_samples)
            observed = series[series.index.isin(hours)]
            peak_time: pd.Timestamp | None = None
            peak_value: float | None = None
            if not observed.empty:
                peak_value = float(observed.max())
                peak_time = pd.Timestamp(observed[observed.eq(peak_value)].index.min())
            event_info[event_id] = {
                "graph_id": graph_id,
                "station_id": station_id,
                "variable": variable,
                "split": str(row["SPLIT"]).upper(),
                "hours": hours,
                "valid_hours": frozenset(pd.Timestamp(item) for item in observed.index),
                "target_start": min(hours),
                "target_end": max(hours),
                "hydro_start": row["HYDRO_START"],
                "hydro_end": row["HYDRO_END"],
                "peak_time": peak_time,
                "peak_value": peak_value,
            }

    rows: list[dict[str, Any]] = []
    review_gap_hours = int(
        pd.to_numeric(samples["FORECAST_HOURS"], errors="raise").max()
    )
    for graph_id in sorted({item["graph_id"] for item in event_info.values()}):
        ordered_ids = sorted(
            (event_id for event_id, item in event_info.items() if item["graph_id"] == graph_id),
            key=lambda event_id: (event_info[event_id]["target_start"], event_id),
        )
        for left_index, event_a in enumerate(ordered_ids):
            a = event_info[event_a]
            for right_index in range(left_index + 1, len(ordered_ids)):
                event_b = ordered_ids[right_index]
                b = event_info[event_b]
                interval_overlap = b["target_start"] <= a["target_end"]
                adjacent = right_index == left_index + 1
                if not interval_overlap and not adjacent:
                    break
                shared = a["valid_hours"] & b["valid_hours"]
                overlap_hours = len(shared)
                denominator = min(len(a["valid_hours"]), len(b["valid_hours"]))
                overlap_fraction = overlap_hours / denominator if denominator else None
                official_overlap: float | None = None
                if pd.notna(a["hydro_start"]) and pd.notna(a["hydro_end"]) and pd.notna(b["hydro_start"]) and pd.notna(b["hydro_end"]):
                    official_overlap = max(
                        0.0,
                        (
                            min(a["hydro_end"], b["hydro_end"])
                            - max(a["hydro_start"], b["hydro_start"])
                        ).total_seconds()
                        / 3600.0,
                    )
                hydro_gap: float | None = None
                if pd.notna(a["hydro_end"]) and pd.notna(b["hydro_start"]):
                    hydro_gap = (
                        b["hydro_start"] - a["hydro_end"]
                    ).total_seconds() / 3600.0
                same_peak = bool(
                    a["peak_time"] is not None
                    and b["peak_time"] is not None
                    and a["peak_time"] == b["peak_time"]
                )
                cross_split = a["split"] != b["split"]
                must_merge = bool(
                    (overlap_hours > 0 and same_peak)
                    or (official_overlap is not None and official_overlap > 0)
                )
                if must_merge and cross_split:
                    status = "CROSS_SPLIT_LEAKAGE"
                    reason = "SAME_CONTINUOUS_RESPONSE_CROSSES_SPLIT"
                    action = "REBUILD_EVENTS_MERGE_PROCESS_THEN_RERUN_DETERMINISTIC_SPLIT"
                elif must_merge:
                    status = "MUST_MERGE"
                    reason = (
                        "SHARED_OBSERVED_PEAK_AND_TARGET_HOURS"
                        if overlap_hours > 0 and same_peak
                        else "OFFICIAL_HYDRO_WINDOWS_OVERLAP"
                    )
                    action = "MERGE_IN_UPSTREAM_EVENT_BUILDER_AND_REBUILD_DEPENDENCIES"
                elif overlap_hours > 0:
                    status = "REVIEW"
                    reason = "SHARED_TARGET_HOURS_WITH_DIFFERENT_OBSERVED_PEAKS"
                    action = "REVIEW_CONTINUOUS_HYDROGRAPH_AND_UPSTREAM_RECESSION_RULE"
                elif hydro_gap is not None and 0 <= hydro_gap <= review_gap_hours:
                    status = "REVIEW"
                    reason = "HYDRO_GAP_NOT_LONGER_THAN_FORECAST_HORIZON"
                    action = "REVIEW_RECESSION_COMPLETION_BEFORE_NEXT_EVENT"
                else:
                    status = "OK"
                    reason = "NO_SHARED_RESPONSE_EVIDENCE"
                    action = "NONE"
                rows.append(
                    {
                        "GRAPH_ID": graph_id,
                        "EVENT_ID_A": event_a,
                        "EVENT_ID_B": event_b,
                        "target_station_id": a["station_id"],
                        "target_variable": a["variable"],
                        "hydro_start_A": _iso(a["hydro_start"]),
                        "hydro_end_A": _iso(a["hydro_end"]),
                        "hydro_start_B": _iso(b["hydro_start"]),
                        "hydro_end_B": _iso(b["hydro_end"]),
                        "overlap_hours": overlap_hours,
                        "overlap_fraction": overlap_fraction,
                        "official_hydro_overlap_hours": official_overlap,
                        "hydro_gap_hours": hydro_gap,
                        "same_observed_peak_time": same_peak,
                        "peak_time_A": _iso(a["peak_time"]),
                        "peak_time_B": _iso(b["peak_time"]),
                        "observed_peak_A": a["peak_value"],
                        "observed_peak_B": b["peak_value"],
                        "split_A": a["split"],
                        "split_B": b["split"],
                        "cross_split": cross_split,
                        "status": status,
                        "reason": reason,
                        "suggested_action": action,
                    }
                )
    frame = pd.DataFrame(rows, columns=EVENT_OVERLAP_FIELDS)
    failure_rows = frame[frame["status"].isin({"MUST_MERGE", "CROSS_SPLIT_LEAKAGE"})]
    parent = {event_id: event_id for event_id in events["EVENT_ID"].astype(str)}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for row in failure_rows.itertuples(index=False):
        union(str(row.EVENT_ID_A), str(row.EVENT_ID_B))
    components: dict[str, list[str]] = {}
    for event_id in parent:
        components.setdefault(find(event_id), []).append(event_id)
    merged_components = [values for values in components.values() if len(values) > 1]
    reduction_by_graph: dict[str, int] = {}
    reduction_by_split: dict[str, int] = {}
    event_lookup = events.set_index("EVENT_ID")
    for component in merged_components:
        graph = str(event_lookup.loc[component[0], "GRAPH_ID"])
        reduction_by_graph[graph] = reduction_by_graph.get(graph, 0) + len(component) - 1
        splits = {str(event_lookup.loc[event_id, "SPLIT"]) for event_id in component}
        if len(splits) == 1:
            split = next(iter(splits))
            reduction_by_split[split] = reduction_by_split.get(split, 0) + len(component) - 1
    split_counts = {
        str(key): int(value) for key, value in events["SPLIT"].value_counts().items()
    }
    projected_split_counts = {
        split: count - reduction_by_split.get(split, 0)
        for split, count in split_counts.items()
    }
    reduction = sum(len(component) - 1 for component in merged_components)
    summary = {
        "event_count_before": int(len(events)),
        "event_count_after_provisional_merge": int(len(events) - reduction),
        "provisional_merge_note": (
            "Counts describe deterministic connected components of MUST_MERGE evidence; "
            "the repository does not contain the authoritative upstream event builder, "
            "so no EVENT_ID or split artifact was rewritten."
        ),
        "event_split_counts_before": split_counts,
        "event_split_counts_after_provisional_merge": projected_split_counts,
        "provisional_split_count_note": (
            "These counts only collapse same-split duplicate components under the "
            "existing assignments. They are not the official post-rebuild split; "
            "the deterministic split strategy must be rerun on merged real events."
        ),
        "overlapping_event_pair_count": int((frame["overlap_hours"] > 0).sum()),
        "must_merge_pair_count": int((frame["status"] == "MUST_MERGE").sum()),
        "cross_split_leakage_pair_count": int(
            (frame["status"] == "CROSS_SPLIT_LEAKAGE").sum()
        ),
        "review_pair_count": int((frame["status"] == "REVIEW").sum()),
        "merged_component_count": len(merged_components),
        "event_reduction_by_graph": dict(sorted(reduction_by_graph.items())),
    }
    return frame, summary


def _outer_fence(values: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 4:
        return None
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    return q1 - 3.0 * iqr, q3 + 3.0 * iqr


def _hourly_absolute_differences(series: pd.Series) -> pd.Series:
    ordered = series.sort_index()
    time_delta = ordered.index.to_series().diff()
    differences = ordered.diff().abs()
    return differences[time_delta.eq(pd.Timedelta(1, unit="h"))].dropna()


def _water_level_audit(
    root: Path,
    events: pd.DataFrame,
    samples: pd.DataFrame,
    target_mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats_raw = json.loads(
        (root / "metadata" / "normalization_stats.json").read_text(
            encoding="utf-8-sig"
        )
    )
    provenance = str(
        stats_raw.get("computed_from_split", stats_raw.get("split", ""))
    ).upper()
    container = stats_raw.get("features", stats_raw)
    if not isinstance(container, dict) or "WATER_LEVEL" not in container:
        raise ValueError("normalization_stats.json缺少WATER_LEVEL")
    level_stats = container["WATER_LEVEL"]
    normalization_min = float(level_stats["min"])
    normalization_max = float(level_stats["max"])

    mapping = {
        str(row.GRAPH_ID): str(row.TARGET_VARIABLE).upper()
        for row in target_mapping.itertuples(index=False)
    }
    water_graphs = {
        graph_id
        for graph_id, value in mapping.items()
        if "WATER_LEVEL" in value or "BOTH" in value
    }
    rows: list[dict[str, Any]] = []
    station_failures: set[str] = set()
    for graph_id in sorted(water_graphs & set(samples["GRAPH_ID"].astype(str))):
        graph_events = events[events["GRAPH_ID"].astype(str).eq(graph_id)].copy()
        if graph_events.empty:
            continue
        station_ids = graph_events["OUTLET_ID"].astype(str).unique()
        if len(station_ids) != 1:
            raise ValueError(f"GRAPH_ID={graph_id}存在多个水位target station")
        station_id = station_ids[0]
        basin_ids = graph_events["BASIN_ID"].astype(str).unique()
        if len(basin_ids) != 1:
            raise ValueError(f"GRAPH_ID={graph_id}存在多个BASIN_ID")
        basin_id = basin_ids[0]
        series, invalid_masked_count = _observed_series(
            root, graph_id, basin_id, station_id, "WATER_LEVEL"
        )
        graph_samples = samples[samples["GRAPH_ID"].astype(str).eq(graph_id)]
        split_values: dict[str, pd.Series] = {}
        event_records: list[dict[str, Any]] = []
        for event_id, event_samples in graph_samples.groupby("EVENT_ID", sort=True):
            hours = _hour_set(event_samples)
            values = series[series.index.isin(hours)]
            if not values.empty:
                event_records.append(
                    {
                        "EVENT_ID": str(event_id),
                        "SPLIT": str(event_samples["SPLIT"].iloc[0]).upper(),
                        "EVENT_TIME": min(hours),
                        "MIN": float(values.min()),
                        "MAX": float(values.max()),
                        "MEDIAN": float(values.median()),
                    }
                )
        event_frame = pd.DataFrame(event_records)
        for split, split_samples in graph_samples.groupby("SPLIT", sort=True):
            hours: set[pd.Timestamp] = set()
            for _, event_samples in split_samples.groupby("EVENT_ID", sort=True):
                hours.update(_hour_set(event_samples))
            split_values[str(split).upper()] = series[series.index.isin(hours)]
        train_values = split_values.get("TRAIN", pd.Series(dtype=float))
        train_min = _finite_or_none(train_values.min()) if not train_values.empty else None
        train_max = _finite_or_none(train_values.max()) if not train_values.empty else None
        train_differences = _hourly_absolute_differences(train_values)
        jump_fence = _outer_fence(train_differences)
        jump_threshold = jump_fence[1] if jump_fence is not None else None

        train_events = (
            event_frame[event_frame["SPLIT"].eq("TRAIN")].sort_values("EVENT_TIME")
            if not event_frame.empty
            else pd.DataFrame()
        )
        reference_shift_ids: list[str] = []
        maximum_event_shift: float | None = None
        if not train_events.empty:
            medians = train_events["MEDIAN"]
            median_fence = _outer_fence(medians)
            if median_fence is not None:
                lower, upper = median_fence
                reference_shift_ids = sorted(
                    train_events.loc[
                        train_events["MAX"].lt(lower)
                        | train_events["MIN"].gt(upper),
                        "EVENT_ID",
                    ].astype(str)
                )
            shifts = train_events["MEDIAN"].diff().abs().dropna()
            maximum_event_shift = _finite_or_none(shifts.max()) if not shifts.empty else None

        station_reasons: list[str] = []
        station_status = "OK"
        if provenance not in {"TRAIN", "TRAINING"}:
            station_status = "FAIL"
            station_reasons.append("NORMALIZATION_PROVENANCE_NOT_TRAIN")
        if invalid_masked_count:
            station_status = "FAIL"
            station_reasons.append(f"MASKED_NONFINITE_WATER_LEVEL={invalid_masked_count}")
        if train_values.empty:
            station_status = "FAIL"
            station_reasons.append("NO_TRAIN_WATER_LEVEL_REFERENCE")
        if reference_shift_ids:
            station_status = "FAIL"
            station_reasons.append(
                "TRAIN_WATER_LEVEL_REFERENCE_SHIFT_EVENTS="
                + ";".join(reference_shift_ids)
            )
        if station_status == "FAIL":
            station_failures.add(station_id)

        for split in ("TRAIN", "VALIDATION", "TEST"):
            if split not in split_values:
                continue
            values = split_values[split]
            count = len(values)
            out_station = (
                values.lt(float(train_min)) | values.gt(float(train_max))
                if train_min is not None and train_max is not None
                else pd.Series(False, index=values.index)
            )
            out_normalization = values.lt(normalization_min) | values.gt(normalization_max)
            differences = _hourly_absolute_differences(values)
            suspicious = (
                differences.gt(float(jump_threshold))
                if jump_threshold is not None
                else pd.Series(False, index=differences.index)
            )
            status = station_status
            reasons = list(station_reasons)
            if status != "FAIL" and int(out_station.sum()):
                status = "REVIEW"
                reasons.append("OUTSIDE_STATION_TRAIN_RANGE")
            if status != "FAIL" and int(out_normalization.sum()):
                status = "REVIEW"
                reasons.append("OUTSIDE_GLOBAL_TRAIN_NORMALIZATION_RANGE")
            if status != "FAIL" and int(suspicious.sum()):
                status = "REVIEW"
                reasons.append("HOURLY_JUMP_EXCEEDS_TRAIN_TUKEY_OUTER_FENCE")
            rows.append(
                {
                    "station_id": station_id,
                    "graph_ids": graph_id,
                    "split": split,
                    "valid_count": count,
                    "min_z": _finite_or_none(values.min()) if count else None,
                    "max_z": _finite_or_none(values.max()) if count else None,
                    "mean_z": _finite_or_none(values.mean()) if count else None,
                    "std_z": _finite_or_none(values.std(ddof=0)) if count else None,
                    "train_min_z": train_min,
                    "train_max_z": train_max,
                    "out_of_train_range_count": int(out_station.sum()),
                    "out_of_train_range_fraction": int(out_station.sum()) / count if count else None,
                    "normalization_train_min_z": normalization_min,
                    "normalization_train_max_z": normalization_max,
                    "out_of_normalization_range_count": int(out_normalization.sum()),
                    "out_of_normalization_range_fraction": int(out_normalization.sum()) / count if count else None,
                    "max_abs_delta_z": _finite_or_none(differences.max()) if not differences.empty else None,
                    "train_jump_outer_fence_m": jump_threshold,
                    "suspicious_jump_count": int(suspicious.sum()),
                    "train_event_count": int(len(train_events)),
                    "train_reference_shift_event_count": len(reference_shift_ids),
                    "train_reference_shift_event_ids": ";".join(reference_shift_ids),
                    "max_train_event_median_shift_m": maximum_event_shift,
                    "normalization_computed_from_split": provenance,
                    "qc_status": status,
                    "qc_reason": ";".join(reasons) if reasons else "NONE",
                }
            )
    frame = pd.DataFrame(rows, columns=WATER_LEVEL_AUDIT_FIELDS)
    summary = {
        "water_level_station_count": int(frame["station_id"].nunique()) if not frame.empty else 0,
        "water_level_station_split_row_count": int(len(frame)),
        "water_level_fail_station_count": len(station_failures),
        "water_level_fail_stations": sorted(station_failures),
        "water_level_review_row_count": int((frame["qc_status"] == "REVIEW").sum()) if not frame.empty else 0,
        "normalization_computed_from_split": provenance,
        "normalization_water_level_range": [normalization_min, normalization_max],
        "reference_shift_rule": (
            "TRAIN event median Tukey outer fences (Q1-3*IQR, Q3+3*IQR); "
            "FAIL only when the entire event min/max range lies outside a fence."
        ),
        "jump_rule": (
            "Hourly absolute changes are compared with the station TRAIN "
            "Tukey upper outer fence; non-consecutive timestamps are not differenced."
        ),
    }
    return frame, summary


def build_dataset_quality_audit(root: str | Path) -> DatasetQualityAudit:
    dataset_root = Path(root).expanduser().resolve()
    events = _read_csv(dataset_root / "events" / "flood_events_final.csv")
    splits = _read_csv(dataset_root / "events" / "data_split.csv")
    samples = _read_csv(dataset_root / "events" / "sample_index.csv")
    target_mapping = _read_csv(
        dataset_root / "events" / "target_variable_by_graph.csv"
    )
    _parse_times(
        events,
        (
            "RAIN_START",
            "RAIN_END",
            "HYDRO_START",
            "PEAK_TIME",
            "HYDRO_END",
            "SAMPLE_START",
            "SAMPLE_END",
        ),
    )
    _parse_times(samples, ("INPUT_START", "FORECAST_TIME", "TARGET_START", "TARGET_END"))
    if "TARGET_START" not in samples:
        # The loader contract derives the first target as FORECAST_TIME + 1 h;
        # newer builders may persist TARGET_START explicitly as a convenience.
        samples["TARGET_START"] = samples["FORECAST_TIME"] + pd.Timedelta(
            1, unit="h"
        )
    events = _validate_event_references(events, splits, samples)
    event_frame, event_summary = _event_overlap_audit(
        dataset_root, events, samples, target_mapping
    )
    water_frame, water_summary = _water_level_audit(
        dataset_root, events, samples, target_mapping
    )
    summary = {
        "dataset_root": dataset_root.name,
        "event": event_summary,
        "water_level": water_summary,
        "strict_failure_counts": {
            "must_merge": int((event_frame["status"] == "MUST_MERGE").sum()),
            "cross_split_leakage": int(
                (event_frame["status"] == "CROSS_SPLIT_LEAKAGE").sum()
            ),
            "water_level_fail_stations": water_summary[
                "water_level_fail_station_count"
            ],
        },
    }
    return DatasetQualityAudit(event_frame, water_frame, summary)


def enforce_strict_quality(audit: DatasetQualityAudit) -> None:
    event_failures = audit.event_hydrograph_overlap[
        audit.event_hydrograph_overlap["status"].isin(
            {"MUST_MERGE", "CROSS_SPLIT_LEAKAGE"}
        )
    ]
    water_failures = audit.water_level_station_audit[
        audit.water_level_station_audit["qc_status"].eq("FAIL")
    ]
    if event_failures.empty and water_failures.empty:
        return
    event_examples = [
        f"{row.GRAPH_ID}:{row.EVENT_ID_A}+{row.EVENT_ID_B}:{row.status}"
        for row in event_failures.head(10).itertuples(index=False)
    ]
    water_examples = sorted(set(water_failures["station_id"].astype(str)))[:10]
    raise ValueError(
        "strict_validation拒绝当前数据集："
        f"重复洪水关系={len(event_failures)} {event_examples}; "
        f"水位FAIL站={len(set(water_failures['station_id']))} {water_examples}。"
        "请先修复上游事件/水位基准并重建全部依赖文件。"
    )
