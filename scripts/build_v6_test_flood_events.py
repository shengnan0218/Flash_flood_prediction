"""Rebuild flood-event TEST references from frozen Step16 continuous data.

Events are evaluation metadata only.  This script never edits the dataset,
never imports Step13 IDs, and never changes TRAIN sampling or target scales.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


@dataclass(frozen=True)
class Event:
    event_id: str
    graph_id: str
    outlet_id: str
    event_start: datetime
    event_end: datetime
    rain_start: datetime
    rain_end: datetime
    hydro_start: datetime
    peak_time: datetime
    hydro_end: datetime
    rainfall_total_mm: float
    baseline_q_m3s: float
    peak_q_m3s: float
    event_grade: str


def parse_time(value: str) -> datetime:
    value = value.strip()
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None or parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f"时间必须是无时区整点: {value!r}")
    return parsed


def format_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("不能对空序列计算quantile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_contract(root: Path) -> tuple[datetime, datetime]:
    schema = json.loads(
        (root / "metadata" / "feature_schema.json").read_text(encoding="utf-8-sig")
    )
    if schema.get("contract") != "continuous-hourly-dual-target-v1":
        raise ValueError("只接受冻结Step16 continuous-hourly-dual-target-v1")
    test = schema["time_splits"]["TEST"]
    return parse_time(test["start"]), parse_time(test["end"])


def load_graphs(root: Path) -> dict[str, dict]:
    static: dict[tuple[str, str], float] = {}
    with (root / "graph" / "node_static_attributes.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            static[(row["GRAPH_ID"], row["STATION_ID"])] = math.exp(
                float(row["log_incremental_area"])
            )
    graphs: dict[str, dict] = {}
    with (root / "graph" / "node_catalog.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            graph = graphs.setdefault(
                row["GRAPH_ID"],
                {"basin_id": row["BASIN_ID"], "outlet_id": row["OUTLET_ID"], "nodes": {}},
            )
            station = row["STATION_ID"]
            graph["nodes"][station] = static[(row["GRAPH_ID"], station)]
    return graphs


def rainfall_episodes(
    times: list[datetime], rain: list[float], *, wet_mm: float = 0.1, dry_gap_hours: int = 6
) -> Iterable[tuple[int, int]]:
    wet = [index for index, value in enumerate(rain) if value >= wet_mm]
    if not wet:
        return
    start = previous = wet[0]
    for index in wet[1:]:
        gap = int((times[index] - times[previous]).total_seconds() // 3600) - 1
        if gap > dry_gap_hours:
            yield start, previous
            start = index
        previous = index
    yield start, previous


def read_graph_dynamic(root: Path, graph_id: str, graph: dict) -> tuple:
    path = root / "dynamic" / f"graph_{graph['basin_id']}_hourly.csv"
    by_time: dict[datetime, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = parse_time(row["TIMESTAMP"])
            record = by_time.setdefault(timestamp, {"rain_sum": 0.0, "rain_area": 0.0, "q": None})
            station = row["STATION_ID"]
            if row["RAIN_MASK"].strip() in {"1", "true", "TRUE"}:
                area = graph["nodes"][station]
                record["rain_sum"] += float(row["RAIN_MM"] or 0.0) * area
                record["rain_area"] += area
            if station == graph["outlet_id"] and row["FLOW_MASK"].strip() in {"1", "true", "TRUE"}:
                record["q"] = float(row["FLOW"])
    times = sorted(by_time)
    rain = [
        by_time[time]["rain_sum"] / by_time[time]["rain_area"]
        if by_time[time]["rain_area"] > 0
        else 0.0
        for time in times
    ]
    q = [by_time[time]["q"] for time in times]
    return times, rain, q


def build_events(root: Path, graphs: dict, test_start: datetime, test_end: datetime) -> tuple[list[Event], int]:
    events: list[Event] = []
    cross_split = 0
    for graph_id in sorted(graphs):
        graph = graphs[graph_id]
        times, rain, q = read_graph_dynamic(root, graph_id, graph)
        index_by_time = {time: index for index, time in enumerate(times)}
        valid_q = [value for value in q if value is not None]
        if len(valid_q) < 24:
            continue
        q90, q95 = quantile(valid_q, 0.90), quantile(valid_q, 0.95)
        changes = [
            abs(float(q[index]) - float(q[index - 1]))
            for index in range(1, len(q))
            if q[index] is not None and q[index - 1] is not None
        ]
        change_scale = median(changes) if changes else 0.0
        sequence = 0
        for rain_first, rain_last in rainfall_episodes(times, rain):
            rain_total = sum(rain[rain_first : rain_last + 1])
            if rain_total < 5.0:
                continue
            response_last_time = min(times[rain_last] + timedelta(hours=48), times[-1])
            response_last = index_by_time[response_last_time]
            response_indices = [
                index
                for index in range(rain_first, response_last + 1)
                if q[index] is not None
            ]
            if len(response_indices) < 6:
                continue
            baseline_values = [
                float(q[index])
                for index in range(max(0, rain_first - 12), rain_first + 1)
                if q[index] is not None
            ]
            if not baseline_values:
                continue
            baseline = median(baseline_values)
            peak_index = max(response_indices, key=lambda index: float(q[index]))
            peak = float(q[peak_index])
            rise = peak - baseline
            response_threshold = max(1.0, 3.0 * change_scale)
            if peak < q90 or rise < response_threshold:
                continue
            hydro_start = next(
                (
                    index
                    for index in response_indices
                    if index <= peak_index and float(q[index]) >= baseline + 0.1 * rise
                ),
                rain_first,
            )
            hydro_end = next(
                (
                    index
                    for index in response_indices
                    if index > peak_index and float(q[index]) <= baseline + 0.2 * rise
                ),
                response_indices[-1],
            )
            event_start = min(times[rain_first], times[hydro_start])
            event_end = times[hydro_end]
            if event_start < test_start or event_end > test_end:
                cross_split += 1
                continue
            sequence += 1
            event_id = f"P2_{graph_id}_{times[peak_index]:%Y%m%d%H}_{sequence:03d}"
            events.append(
                Event(
                    event_id,
                    graph_id,
                    graph["outlet_id"],
                    event_start,
                    event_end,
                    times[rain_first],
                    times[rain_last],
                    times[hydro_start],
                    times[peak_index],
                    times[hydro_end],
                    rain_total,
                    baseline,
                    peak,
                    "A" if peak >= q95 and rain_total >= 10.0 else "B",
                )
            )
    return events, cross_split


EVENT_FIELDS = [
    "EVENT_ID", "GRAPH_ID", "OUTLET_ID", "EVENT_START", "EVENT_END",
    "RAIN_START", "RAIN_END", "HYDRO_START", "PEAK_TIME", "HYDRO_END",
    "RAINFALL_TOTAL_MM", "BASELINE_Q_M3S", "PEAK_Q_M3S", "EVENT_GRADE", "SPLIT",
]


def event_row(event: Event) -> dict[str, str | float]:
    return {
        "EVENT_ID": event.event_id,
        "GRAPH_ID": event.graph_id,
        "OUTLET_ID": event.outlet_id,
        "EVENT_START": format_time(event.event_start),
        "EVENT_END": format_time(event.event_end),
        "RAIN_START": format_time(event.rain_start),
        "RAIN_END": format_time(event.rain_end),
        "HYDRO_START": format_time(event.hydro_start),
        "PEAK_TIME": format_time(event.peak_time),
        "HYDRO_END": format_time(event.hydro_end),
        "RAINFALL_TOTAL_MM": event.rainfall_total_mm,
        "BASELINE_Q_M3S": event.baseline_q_m3s,
        "PEAK_Q_M3S": event.peak_q_m3s,
        "EVENT_GRADE": event.event_grade,
        "SPLIT": "TEST",
    }


def build_sample_rows(root: Path, events: list[Event], test_start: datetime, test_end: datetime) -> list[dict]:
    by_graph: dict[str, list[Event]] = {}
    for event in events:
        by_graph.setdefault(event.graph_id, []).append(event)
    output: list[dict] = []
    source = root / "samples" / "sample_index.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["SPLIT"].strip().upper() != "TEST" or row["GRAPH_ID"] not in by_graph:
                continue
            input_start = parse_time(row["INPUT_START"])
            target_start = parse_time(row["TARGET_START"])
            target_end = parse_time(row["TARGET_END"])
            if input_start < test_start or target_end > test_end:
                raise ValueError(f"Step16 TEST sample跨绝对边界: {row['SAMPLE_ID']}")
            for event in by_graph[row["GRAPH_ID"]]:
                if target_end < event.event_start or target_start > event.event_end:
                    continue
                record = dict(row)
                record.update(event_row(event))
                record["CONTINUOUS_SAMPLE_ID"] = row["SAMPLE_ID"]
                record["SAMPLE_ID"] = f"{event.event_id}__{row['SAMPLE_ID']}"
                record["FORECAST_HORIZONS"] = "h1;h2;h3;h4;h5;h6"
                output.append(record)
    return output


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="重建P2 TEST-only洪水事件及连续窗口引用")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", default="outputs/p2_continuous_flood_events")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    test_start, test_end = load_contract(root)
    graphs = load_graphs(root)
    events, cross_split = build_events(root, graphs, test_start, test_end)
    samples = build_sample_rows(root, events, test_start, test_end)
    write_csv(output / "test_flood_events.csv", EVENT_FIELDS, [event_row(event) for event in events])
    source_fields = []
    with (root / "samples" / "sample_index.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        source_fields = list(csv.DictReader(handle).fieldnames or ())
    sample_fields = [
        "SAMPLE_ID", "CONTINUOUS_SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "OUTLET_ID",
        "EVENT_START", "EVENT_END", "RAIN_START", "RAIN_END", "HYDRO_START",
        "PEAK_TIME", "HYDRO_END", "EVENT_GRADE", "INPUT_START", "FORECAST_TIME",
        "TARGET_START", "TARGET_END", "HISTORY_HOURS", "FORECAST_HOURS",
        "FORECAST_HORIZONS", "Q_VALID_COUNT", "Z_VALID_COUNT", "Q_COVERAGE",
        "Z_COVERAGE", "SPLIT",
    ]
    write_csv(output / "test_flood_event_samples.csv", sample_fields, samples)
    report = (
        "# P2 continuous flood-event TEST build\n\n"
        f"- dataset: `{root}`\n"
        f"- TEST boundary: `{format_time(test_start)}` through `{format_time(test_end)}`\n"
        f"- current graphs scanned: {len(graphs)}\n"
        f"- TEST-only events retained: {len(events)}\n"
        f"- cross-split events excluded: {cross_split}\n"
        f"- event-window references: {len(samples)}\n\n"
        "Events are evaluation-only and were rebuilt from current graph outlets, "
        "area-weighted current node rainfall and current outlet Q. No Step13 EVENT_ID "
        "or dynamic-data copy is used.\n"
    )
    (output / "FLOOD_EVENT_BUILD_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"events": len(events), "samples": len(samples), "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
