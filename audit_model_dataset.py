"""Read-only structural and statistical audit for Hunan model datasets."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    try:
        # Station IDs such as 611E2950 resemble scientific notation.  Preserve
        # identifier columns as text while allowing measurements/masks to keep
        # their numeric dtype for independent statistical checks.
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={
                "STATION_ID": str,
                "OUTLET_ID": str,
                "FROM_STATION": str,
                "TO_STATION": str,
            },
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stats_record(values: Iterable[float]) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    std = float(series.std(ddof=0))
    if std == 0:
        std = 1.0
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": std,
        "min": float(series.min()),
        "max": float(series.max()),
    }


def merge_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ordered = sorted((start, end) for start, end in intervals if start <= end)
    merged: list[list[pd.Timestamp]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + pd.Timedelta(hours=1):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def normalization_sections(raw: dict[str, Any]) -> tuple[dict, dict, dict]:
    dynamic = raw.get("features") or raw.get("dynamic") or {
        key: raw[key] for key in ("RAIN_MM", "FLOW", "WATER_LEVEL") if key in raw
    }
    return dynamic, raw.get("node_static", {}), raw.get("edge_static", {})


def compare_stats(expected: dict, actual: dict, tolerance: float = 1e-8) -> dict[str, Any]:
    mismatches: list[str] = []
    for section, records in expected.items():
        actual_records = actual.get(section, {})
        for feature, record in records.items():
            found = actual_records.get(feature)
            if not isinstance(found, dict):
                mismatches.append(f"{section}.{feature}: missing")
                continue
            for field in ("count", "mean", "std", "min", "max"):
                left, right = record.get(field), found.get(field)
                if left is None or right is None:
                    equal = left is None and right is None
                elif field == "count":
                    equal = int(left) == int(right)
                else:
                    equal = math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
                if not equal:
                    mismatches.append(f"{section}.{feature}.{field}: recomputed={left}, file={right}")
    return {"matches": not mismatches, "mismatches": mismatches[:50]}


def recompute_normalization(root: Path, events: pd.DataFrame, samples: pd.DataFrame) -> dict[str, Any]:
    schema = json.loads(
        (root / "metadata" / "feature_schema.json").read_text(encoding="utf-8-sig")
    )
    train_samples = samples[samples["SPLIT"] == "TRAIN"].copy()
    train_samples["INPUT_START"] = pd.to_datetime(train_samples["INPUT_START"])
    train_samples["FORECAST_TIME"] = pd.to_datetime(train_samples["FORECAST_TIME"])
    dynamic_values: dict[str, list[pd.Series]] = {
        "RAIN_MM": [], "FLOW": [], "WATER_LEVEL": [],
    }
    for graph_id, group in train_samples.groupby("GRAPH_ID"):
        path = root / "dynamic" / f"graph_{graph_id}_hourly.csv"
        frame = read_csv(path)
        frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"])
        intervals = merge_intervals(list(zip(group["INPUT_START"], group["FORECAST_TIME"])))
        selected = pd.Series(False, index=frame.index)
        for start, end in intervals:
            selected |= (frame["TIMESTAMP"] >= start) & (frame["TIMESTAMP"] <= end)
        part = frame[selected]
        for feature, mask in (
            ("RAIN_MM", "RAIN_MASK"),
            ("FLOW", "FLOW_MASK"),
            ("WATER_LEVEL", "WATER_LEVEL_MASK"),
        ):
            dynamic_values[feature].append(part.loc[part[mask] == 1, feature])

    dynamic_stats = {
        feature: stats_record(pd.concat(parts, ignore_index=True) if parts else [])
        for feature, parts in dynamic_values.items()
    }
    train_graphs = set(events.loc[events["SPLIT"] == "TRAIN", "GRAPH_ID"])
    node = read_csv(root / "graph" / "node_static_attributes.csv")
    edge = read_csv(root / "graph" / "edge_static_attributes.csv")
    node = node[node["GRAPH_ID"].isin(train_graphs)]
    edge = edge[edge["GRAPH_ID"].isin(train_graphs)]
    node_features = list(schema["node_static_features"])
    edge_features = list(schema["edge_static_features"])
    return {
        "features": dynamic_stats,
        "node_static": {feature: stats_record(node[feature]) for feature in node_features},
        "edge_static": {feature: stats_record(edge[feature]) for feature in edge_features},
    }


def audit(root: Path) -> dict[str, Any]:
    graph = root / "graph"
    events_dir = root / "events"
    metadata = root / "metadata"
    qc = root / "qc"
    catalog = read_csv(graph / "node_catalog.csv")
    topology = read_csv(graph / "edge_topology.csv")
    events_all = read_csv(events_dir / "flood_events_all.csv")
    events = read_csv(events_dir / "flood_events_final.csv")
    split = read_csv(events_dir / "data_split.csv")
    samples = read_csv(events_dir / "sample_index.csv")
    rejections = read_csv(qc / "sample_rejection.csv")
    schema = json.loads((metadata / "feature_schema.json").read_text(encoding="utf-8-sig"))
    normalization = json.loads((metadata / "normalization_stats.json").read_text(encoding="utf-8-sig"))

    split_lookup = split.set_index("EVENT_ID")["SPLIT"]
    if "SPLIT" in events.columns:
        mapped = events["EVENT_ID"].map(split_lookup)
        if not events["SPLIT"].astype(str).equals(mapped.astype(str)):
            raise ValueError(f"flood_events_final与data_split的SPLIT不一致: {root}")
    else:
        events["SPLIT"] = events["EVENT_ID"].map(split_lookup)
    for column in (
        "RAIN_START", "RAIN_END", "HYDRO_START", "PEAK_TIME", "HYDRO_END",
        "SAMPLE_START", "SAMPLE_END",
    ):
        events[column] = pd.to_datetime(events[column], errors="coerce")
    for column in ("INPUT_START", "FORECAST_TIME", "TARGET_START", "TARGET_END"):
        samples[column] = pd.to_datetime(samples[column], errors="coerce")

    valid_negative_rows = 0
    negative_by_station: dict[str, int] = {}
    dynamic_rows = 0
    for path in sorted((root / "dynamic").glob("graph_*_hourly.csv")):
        frame = read_csv(path)
        dynamic_rows += len(frame)
        flow = pd.to_numeric(frame["FLOW"], errors="coerce")
        invalid = (pd.to_numeric(frame["FLOW_MASK"], errors="coerce") == 1) & (flow < 0)
        valid_negative_rows += int(invalid.sum())
        if invalid.any():
            counts = frame.loc[invalid, "STATION_ID"].astype(str).value_counts()
            for station, count in counts.items():
                negative_by_station[station] = negative_by_station.get(station, 0) + int(count)

    sample_event_ids = set(samples["EVENT_ID"].astype(str))
    no_sample = events[~events["EVENT_ID"].astype(str).isin(sample_event_ids)]
    no_sample_details: list[dict[str, Any]] = []
    for row in no_sample.itertuples(index=False):
        related = (
            rejections[rejections["EVENT_ID"].astype(str) == str(row.EVENT_ID)]
            if "EVENT_ID" in rejections.columns else pd.DataFrame()
        )
        no_sample_details.append({
            "event_id": str(row.EVENT_ID),
            "graph_id": str(row.GRAPH_ID),
            "split": str(row.SPLIT),
            "rejection_rows": int(len(related)),
            "reasons": sorted(related["REASON"].dropna().astype(str).unique().tolist())
            if "REASON" in related.columns else [],
        })

    compound = events["COMPOUND_EVENT"].astype(str).str.lower().isin({"1", "true", "yes"})
    compound_mismatch = 0
    if "SOURCE_RAIN_EVENT_COUNT" in events.columns:
        source_count = pd.to_numeric(events["SOURCE_RAIN_EVENT_COUNT"], errors="coerce")
        compound_mismatch = int((compound != (source_count > 1)).sum())

    history = pd.to_numeric(samples["HISTORY_HOURS"], errors="coerce")
    forecast = pd.to_numeric(samples["FORECAST_HOURS"], errors="coerce")
    history_delta = (samples["FORECAST_TIME"] - samples["INPUT_START"]).dt.total_seconds() / 3600
    forecast_delta = (samples["TARGET_END"] - samples["FORECAST_TIME"]).dt.total_seconds() / 3600

    recomputed = recompute_normalization(root, events, samples)
    dynamic_stats, node_stats, edge_stats = normalization_sections(normalization)
    normalization_check = compare_stats(
        recomputed,
        {"features": dynamic_stats, "node_static": node_stats, "edge_static": edge_stats},
    )
    normalization_check["computed_from_split"] = normalization.get("computed_from_split")
    normalization_check["recomputed_counts"] = {
        section: {feature: record["count"] for feature, record in values.items()}
        for section, values in recomputed.items()
    }

    return {
        "root": str(root),
        "graph_count": int(catalog["GRAPH_ID"].nunique()),
        "node_count": int(len(catalog)),
        "edge_count": int(len(topology)),
        "event_graph_count": int(events["GRAPH_ID"].nunique()),
        "event_all_count": int(len(events_all)),
        "event_final_count": int(len(events)),
        "event_split_counts": {str(k): int(v) for k, v in split["SPLIT"].value_counts().items()},
        "sample_count": int(len(samples)),
        "sample_split_counts": {str(k): int(v) for k, v in samples["SPLIT"].value_counts().items()},
        "dynamic_row_count": dynamic_rows,
        "valid_negative_flow_count": valid_negative_rows,
        "valid_negative_flow_by_station": negative_by_station,
        "missing_hydro_start_count": int(events["HYDRO_START"].isna().sum()),
        "missing_hydro_end_count": int(events["HYDRO_END"].isna().sum()),
        "hydro_start_after_peak_count": int((events["HYDRO_START"] > events["PEAK_TIME"]).sum()),
        "hydro_end_before_peak_count": int((events["HYDRO_END"] < events["PEAK_TIME"]).sum()),
        "compound_event_count": int(compound.sum()),
        "compound_field_mismatch_count": compound_mismatch,
        "no_sample_event_count": int(len(no_sample)),
        "no_sample_events": no_sample_details,
        "sample_rejection_row_count": int(len(rejections)),
        "sample_rejection_reason_counts": {
            str(k): int(v) for k, v in rejections.get("REASON", pd.Series(dtype=str)).value_counts().items()
        },
        "history_values": sorted(history.dropna().astype(int).unique().tolist()),
        "forecast_values": sorted(forecast.dropna().astype(int).unique().tolist()),
        "history_window_violation_count": int((history_delta != history - 1).sum()),
        "forecast_window_violation_count": int((forecast_delta != forecast).sum()),
        "node_static_dim": len(schema["node_static_features"]),
        "edge_static_dim": len(schema["edge_static_features"]),
        "dynamic_dim": len(schema["dynamic_features"]),
        "normalization": normalization_check,
    }


def comparable_frame(root: Path, relative: str, keys: list[str]) -> pd.DataFrame:
    frame = read_csv(root / relative)
    return frame.sort_values(keys).reset_index(drop=True)


def compare_frame(before: Path, candidate: Path, relative: str, keys: list[str]) -> dict[str, Any]:
    left = comparable_frame(before, relative, keys)
    right = comparable_frame(candidate, relative, keys)
    candidate_columns_present = all(column in left.columns for column in right.columns)
    if not candidate_columns_present:
        return {"equal": False, "reason": "candidate has columns absent from before"}
    left = left[right.columns]
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, rtol=1e-9, atol=1e-9)
        return {"equal": True, "rows": int(len(right))}
    except AssertionError as exc:
        return {"equal": False, "reason": str(exc).splitlines()[0]}


def compare(before: Path, candidate: Path) -> dict[str, Any]:
    before_split = read_csv(before / "events" / "data_split.csv")
    candidate_split = read_csv(candidate / "events" / "data_split.csv")
    split_equal = before_split[["EVENT_ID", "SPLIT"]].sort_values("EVENT_ID").reset_index(drop=True).equals(
        candidate_split[["EVENT_ID", "SPLIT"]].sort_values("EVENT_ID").reset_index(drop=True)
    )
    before_target = read_csv(before / "events" / "target_variable_by_graph.csv")
    candidate_target = read_csv(candidate / "events" / "target_variable_by_graph.csv")
    target_equal = before_target[["GRAPH_ID", "TARGET_VARIABLE"]].sort_values("GRAPH_ID").reset_index(drop=True).equals(
        candidate_target[["GRAPH_ID", "TARGET_VARIABLE"]].sort_values("GRAPH_ID").reset_index(drop=True)
    )
    return {
        "split_assignment_equal": bool(split_equal),
        "target_variable_mapping_equal": bool(target_equal),
        "static_and_topology": {
            "node_catalog": compare_frame(before, candidate, "graph/node_catalog.csv", ["GRAPH_ID", "NODE_INDEX"]),
            "edge_topology": compare_frame(before, candidate, "graph/edge_topology.csv", ["GRAPH_ID", "FROM_NODE", "TO_NODE"]),
            "node_static": compare_frame(before, candidate, "graph/node_static_attributes.csv", ["GRAPH_ID", "NODE_INDEX"]),
            "edge_static": compare_frame(before, candidate, "graph/edge_static_attributes.csv", ["GRAPH_ID", "FROM_NODE", "TO_NODE"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    before = Path(args.before).expanduser().resolve()
    candidate = Path(args.candidate).expanduser().resolve()
    result = {
        "before": audit(before),
        "candidate": audit(candidate),
        "comparison": compare(before, candidate),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
