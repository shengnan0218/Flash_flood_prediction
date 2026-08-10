from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = Path(
    os.environ.get(
        "UPSTREAM_BUILDER_PATH",
        REPO_ROOT / "scripts" / "16_build_model_dataset_v3.py",
    )
)
SPEC = importlib.util.spec_from_file_location("upstream_builder_event_zqc", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def _event(event_id: str, start: str, end: str, split: str = "TRAIN") -> dict:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    return {
        "EVENT_ID": event_id,
        "GRAPH_ID": "G1",
        "BASIN_ID": "G1",
        "OUTLET_ID": "S1",
        "RAIN_START": start_time - pd.Timedelta(2, unit="h"),
        "RAIN_END": start_time,
        "HYDRO_START": start_time,
        "PEAK_TIME": start_time + pd.Timedelta(1, unit="h"),
        "HYDRO_END": end_time,
        "SAMPLE_START": start_time - pd.Timedelta(24, unit="h"),
        "SAMPLE_END": end_time + pd.Timedelta(6, unit="h"),
        "EVENT_TYPE": "HYDRO_FLOOD",
        "EVENT_GRADE": "A",
        "COMPOUND_EVENT": False,
        "PEAK_COUNT": 1,
        "SOURCE_RAIN_EVENT_IDS": event_id.replace("F", "R"),
        "SOURCE_RAIN_EVENT_COUNT": 1,
        "SPLIT": split,
    }


def _sample(sample_id: str, event_id: str, start: str, end: str, split: str) -> dict:
    target_start = pd.Timestamp(start)
    target_end = pd.Timestamp(end)
    return {
        "SAMPLE_ID": sample_id,
        "EVENT_ID": event_id,
        "GRAPH_ID": "G1",
        "OUTLET_ID": "S1",
        "INPUT_START": target_start - pd.Timedelta(24, unit="h"),
        "FORECAST_TIME": target_start - pd.Timedelta(1, unit="h"),
        "TARGET_START": target_start,
        "TARGET_END": target_end,
        "HISTORY_HOURS": 24,
        "FORECAST_HOURS": 6,
        "TARGET_VARIABLE": "FLOW",
        "TARGET_COVERAGE": 1.0,
        "SPLIT": split,
    }


def _dynamic(values: dict[str, float], variable: str) -> pd.DataFrame:
    rows = []
    for timestamp, value in sorted(values.items()):
        rows.append({
            "GRAPH_ID": "G1",
            "TIMESTAMP": pd.Timestamp(timestamp),
            "NODE_INDEX": 0,
            "STATION_ID": "S1",
            "RAIN_MM": 0.0,
            "FLOW": value if variable == "FLOW" else float("nan"),
            "WATER_LEVEL": value if variable == "WATER_LEVEL" else float("nan"),
            "RAIN_MASK": 1,
            "FLOW_MASK": 1 if variable == "FLOW" else 0,
            "WATER_LEVEL_MASK": 1 if variable == "WATER_LEVEL" else 0,
        })
    return pd.DataFrame(rows)


def test_transitive_duplicate_processes_merge_before_final_split() -> None:
    events = pd.DataFrame([
        _event("G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00"),
        _event("G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00"),
        _event("G1_F0003", "2024-01-01 04:00", "2024-01-01 08:00"),
    ])
    samples = pd.DataFrame([
        _sample("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN"),
        _sample("S2", "G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", "TRAIN"),
        _sample("S3", "G1_F0003", "2024-01-01 04:00", "2024-01-01 08:00", "TRAIN"),
    ])
    observations = {
        f"2024-01-01 {hour:02d}:00": (10.0 if hour == 4 else 1.0)
        for hour in range(1, 9)
    }
    target = pd.DataFrame([
        {"GRAPH_ID": "G1", "OUTLET_ID": "S1", "TARGET_VARIABLE": "FLOW"}
    ])
    overlap, info = BUILDER.build_event_overlap_audit(
        events, samples, {"G1": _dynamic(observations, "FLOW")}, target, 6
    )
    merged, merge_audit, summary = BUILDER.merge_event_components(events, overlap, info)
    assert summary["merged_component_count"] == 1
    assert summary["event_reduction_count"] == 2
    assert len(merged) == 1
    merged_event = merged.iloc[0]
    assert merged_event["SOURCE_EVENT_IDS"] == "G1_F0001;G1_F0002;G1_F0003"
    assert merged_event["SOURCE_RAIN_EVENT_IDS"] == "G1_R0001;G1_R0002;G1_R0003"
    assert merged_event["SOURCE_RAIN_EVENT_COUNT"] == 3
    assert bool(merged_event["COMPOUND_EVENT"])
    assert merged_event["RAIN_START"] == events["RAIN_START"].min()
    assert merged_event["RAIN_END"] == events["RAIN_END"].max()
    assert merged_event["SAMPLE_START"] == events["SAMPLE_START"].min()
    assert merged_event["SAMPLE_END"] == events["SAMPLE_END"].max()
    assert set(merge_audit.loc[merge_audit["MERGE_ACTION"] == "MERGED", "MERGED_EVENT_ID"]) == {"G1_F0001"}


def test_water_level_rule_excludes_train_shift_but_never_test_ood() -> None:
    dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
    splits = ["TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TEST"]
    levels = [100.0, 100.5, 101.0, 99.5, 2.0, 120.0]
    events = []
    samples = []
    observations: dict[str, float] = {}
    for index, (date, split, level) in enumerate(zip(dates, splits, levels), 1):
        event_id = f"G1_F{index:04d}"
        start = f"{date} 01:00"
        end = f"{date} 03:00"
        events.append(_event(event_id, start, end, split))
        sample = _sample(f"S{index}", event_id, start, end, split)
        sample["TARGET_VARIABLE"] = "WATER_LEVEL"
        samples.append(sample)
        for hour in range(1, 4):
            observations[f"{date} {hour:02d}:00"] = level + hour / 10
    events_frame = pd.DataFrame(events)
    samples_frame = pd.DataFrame(samples)
    dynamic = {"G1": _dynamic(observations, "WATER_LEVEL")}
    target = pd.DataFrame([
        {"GRAPH_ID": "G1", "OUTLET_ID": "S1", "TARGET_VARIABLE": "WATER_LEVEL"}
    ])
    audit = BUILDER.build_water_level_reference_audit(
        events_frame, samples_frame, dynamic, target
    )
    train_shift = audit[audit["EVENT_ID"] == "G1_F0005"].iloc[0]
    test_ood = audit[audit["EVENT_ID"] == "G1_F0006"].iloc[0]
    assert train_shift["QC_STATUS"] == "FAIL"
    assert train_shift["ACTION"] == "EXCLUDE_EVENT_AND_MASK_SOURCE_WINDOW"
    assert test_ood["QC_STATUS"] == "REVIEW"
    assert test_ood["ACTION"] == "NONE_DO_NOT_FILTER_VALIDATION_OR_TEST"


def test_real_candidate_regression_matches_audited_counts() -> None:
    root = Path(
        os.environ.get(
            "MODEL_DATASET_ROOT",
            REPO_ROOT / "_model_dataset_v4_candidate",
        )
    )
    if not root.is_dir():
        return
    events = pd.read_csv(root / "events" / "flood_events_final.csv", dtype=str)
    for column in [
        "RAIN_START", "RAIN_END", "HYDRO_START", "PEAK_TIME",
        "HYDRO_END", "SAMPLE_START", "SAMPLE_END",
    ]:
        events[column] = pd.to_datetime(events[column], errors="coerce")
    samples = pd.read_csv(root / "events" / "sample_index.csv", dtype=str)
    for column in ["INPUT_START", "FORECAST_TIME", "TARGET_START", "TARGET_END"]:
        samples[column] = pd.to_datetime(samples[column], errors="coerce")
    target = pd.read_csv(root / "events" / "target_variable_by_graph.csv", dtype=str)
    dynamic = {}
    for graph_id in sorted(samples["GRAPH_ID"].unique()):
        frame = pd.read_csv(
            root / "dynamic" / f"graph_{graph_id}_hourly.csv",
            dtype={"GRAPH_ID": str, "STATION_ID": str},
            low_memory=False,
        )
        frame["TIMESTAMP"] = pd.to_datetime(frame["TIMESTAMP"])
        dynamic[graph_id] = frame
    overlap, info = BUILDER.build_event_overlap_audit(events, samples, dynamic, target, 6)
    _merged, _audit, summary = BUILDER.merge_event_components(events, overlap, info)
    water = BUILDER.build_water_level_reference_audit(events, samples, dynamic, target)
    assert summary["must_merge_pair_count"] == 15
    assert summary["merged_component_count"] == 13
    assert summary["event_reduction_count"] == 14
    assert set(water.loc[water["QC_STATUS"] == "FAIL", "EVENT_ID"]) == {
        "B016_F0013", "B016_F0014"
    }
