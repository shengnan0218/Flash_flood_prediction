from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dataset_quality import build_dataset_quality_audit, enforce_strict_quality


EVENT_COLUMNS = [
    "EVENT_ID", "GRAPH_ID", "BASIN_ID", "OUTLET_ID", "RAIN_START",
    "RAIN_END", "HYDRO_START", "PEAK_TIME", "HYDRO_END", "SAMPLE_START",
    "SAMPLE_END", "EVENT_TYPE", "EVENT_GRADE", "COMPOUND_EVENT",
    "PEAK_COUNT", "SOURCE_RAIN_EVENT_IDS", "SOURCE_RAIN_EVENT_COUNT", "SPLIT",
]


def event_row(
    event_id: str,
    start: str,
    end: str,
    split: str = "TRAIN",
    *,
    hydro_start: str | None = None,
    hydro_end: str | None = None,
) -> dict[str, object]:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    return {
        "EVENT_ID": event_id,
        "GRAPH_ID": "G1",
        "BASIN_ID": "G1",
        "OUTLET_ID": "S1",
        "RAIN_START": start_time - pd.Timedelta(2, unit="h"),
        "RAIN_END": start_time,
        "HYDRO_START": hydro_start or start,
        "PEAK_TIME": start_time + pd.Timedelta(1, unit="h"),
        "HYDRO_END": hydro_end or end,
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


def sample_row(
    sample_id: str,
    event_id: str,
    start: str,
    end: str,
    split: str,
    target: str,
) -> dict[str, object]:
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
        "FORECAST_HOURS": max(1, int((target_end - target_start).total_seconds() / 3600) + 1),
        "TARGET_VARIABLE": target,
        "TARGET_COVERAGE": 1.0,
        "SPLIT": split,
    }


def make_dataset(
    root: Path,
    events: list[dict[str, object]],
    samples: list[dict[str, object]],
    observations: dict[str, float],
    *,
    target: str = "FLOW",
    normalization_split: str = "TRAIN",
) -> None:
    (root / "events").mkdir(parents=True)
    (root / "dynamic").mkdir(parents=True)
    (root / "metadata").mkdir(parents=True)
    event_frame = pd.DataFrame(events, columns=EVENT_COLUMNS)
    event_frame.to_csv(root / "events" / "flood_events_final.csv", index=False)
    pd.DataFrame(
        [
            {
                "EVENT_ID": row["EVENT_ID"],
                "GRAPH_ID": row["GRAPH_ID"],
                "EVENT_YEAR": pd.Timestamp(row["PEAK_TIME"]).year,
                "EVENT_GRADE": row["EVENT_GRADE"],
                "SPLIT": row["SPLIT"],
                "SPLIT_REASON": "FIXTURE",
            }
            for row in events
        ]
    ).to_csv(root / "events" / "data_split.csv", index=False)
    pd.DataFrame(samples).to_csv(root / "events" / "sample_index.csv", index=False)
    pd.DataFrame(
        [{"GRAPH_ID": "G1", "OUTLET_ID": "S1", "TARGET_VARIABLE": target}]
    ).to_csv(root / "events" / "target_variable_by_graph.csv", index=False)
    dynamic_rows = []
    for timestamp, value in sorted(observations.items()):
        dynamic_rows.append(
            {
                "GRAPH_ID": "G1",
                "TIMESTAMP": timestamp,
                "STATION_ID": "S1",
                "RAIN_MM": 0.0,
                "FLOW": value if target == "FLOW" else "",
                "WATER_LEVEL": value if target == "WATER_LEVEL" else "",
                "RAIN_MASK": 1,
                "FLOW_MASK": 1 if target == "FLOW" else 0,
                "WATER_LEVEL_MASK": 1 if target == "WATER_LEVEL" else 0,
            }
        )
    pd.DataFrame(dynamic_rows).to_csv(root / "dynamic" / "graph_G1_hourly.csv", index=False)
    (root / "metadata" / "normalization_stats.json").write_text(
        pd.Series(
            {
                "computed_from_split": normalization_split,
                "features": {
                    "WATER_LEVEL": {"mean": 100.0, "std": 2.0, "min": 90.0, "max": 110.0}
                },
            }
        ).to_json(),
        encoding="utf-8",
    )


class TestDatasetQualityAudit(unittest.TestCase):
    def test_non_overlapping_events_are_ok(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 03:00"),
                event_row("G1_F0002", "2024-01-03 01:00", "2024-01-03 03:00"),
            ]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 03:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0002", "2024-01-03 01:00", "2024-01-03 03:00", "TRAIN", "FLOW"),
            ]
            observations = {
                "2024-01-01 01:00": 1.0,
                "2024-01-01 02:00": 3.0,
                "2024-01-01 03:00": 1.0,
                "2024-01-03 01:00": 1.0,
                "2024-01-03 02:00": 4.0,
                "2024-01-03 03:00": 1.0,
            }
            make_dataset(root, events, samples, observations)
            audit = build_dataset_quality_audit(root)
            self.assertEqual(audit.event_hydrograph_overlap.iloc[0]["status"], "OK")

    def test_overlapping_official_hydro_windows_must_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 04:00", hydro_end="2024-01-01 05:00"),
                event_row("G1_F0002", "2024-01-01 04:00", "2024-01-01 07:00", hydro_start="2024-01-01 04:00"),
            ]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 04:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0002", "2024-01-01 04:00", "2024-01-01 07:00", "TRAIN", "FLOW"),
            ]
            observations = {f"2024-01-01 {hour:02d}:00": float(hour) for hour in range(1, 8)}
            make_dataset(root, events, samples, observations)
            audit = build_dataset_quality_audit(root)
            self.assertEqual(audit.event_hydrograph_overlap.iloc[0]["status"], "MUST_MERGE")

    def test_shared_observed_peak_across_event_ids_must_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", hydro_end=None),
                event_row("G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", hydro_end=None),
            ]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", "TRAIN", "FLOW"),
            ]
            observations = {f"2024-01-01 {hour:02d}:00": (10.0 if hour == 4 else 1.0) for hour in range(1, 8)}
            make_dataset(root, events, samples, observations)
            audit = build_dataset_quality_audit(root)
            row = audit.event_hydrograph_overlap.iloc[0]
            self.assertTrue(bool(row["same_observed_peak_time"]))
            self.assertEqual(row["status"], "MUST_MERGE")

    def test_cross_split_duplicate_fails_strict_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN"),
                event_row("G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", "TEST"),
            ]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", "TEST", "FLOW"),
            ]
            observations = {f"2024-01-01 {hour:02d}:00": (10.0 if hour == 4 else 1.0) for hour in range(1, 8)}
            make_dataset(root, events, samples, observations)
            audit = build_dataset_quality_audit(root)
            self.assertEqual(audit.event_hydrograph_overlap.iloc[0]["status"], "CROSS_SPLIT_LEAKAGE")
            with self.assertRaisesRegex(ValueError, "strict_validation"):
                enforce_strict_quality(audit)

    def test_sliding_windows_within_one_event_are_not_cross_event_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 06:00")]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0001", "2024-01-01 02:00", "2024-01-01 06:00", "TRAIN", "FLOW"),
            ]
            observations = {f"2024-01-01 {hour:02d}:00": float(hour) for hour in range(1, 7)}
            make_dataset(root, events, samples, observations)
            audit = build_dataset_quality_audit(root)
            self.assertTrue(audit.event_hydrograph_overlap.empty)

    def test_stale_sample_event_after_merge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 03:00")]
            samples = [sample_row("S1", "G1_F_OLD", "2024-01-01 01:00", "2024-01-01 03:00", "TRAIN", "FLOW")]
            observations = {f"2024-01-01 {hour:02d}:00": float(hour) for hour in range(1, 4)}
            make_dataset(root, events, samples, observations)
            with self.assertRaisesRegex(ValueError, "stale EVENT_ID"):
                build_dataset_quality_audit(root)

    def test_audit_is_deterministic_under_input_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [
                event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00"),
                event_row("G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00"),
            ]
            samples = [
                sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 05:00", "TRAIN", "FLOW"),
                sample_row("S2", "G1_F0002", "2024-01-01 03:00", "2024-01-01 07:00", "TRAIN", "FLOW"),
            ]
            observations = {f"2024-01-01 {hour:02d}:00": (10.0 if hour == 4 else 1.0) for hour in range(1, 8)}
            make_dataset(root, events, samples, observations)
            first = build_dataset_quality_audit(root).event_hydrograph_overlap
            pd.read_csv(root / "events" / "sample_index.csv").sample(frac=1, random_state=7).to_csv(
                root / "events" / "sample_index.csv", index=False
            )
            second = build_dataset_quality_audit(root).event_hydrograph_overlap
            pd.testing.assert_frame_equal(first, second)

    def test_water_level_ood_and_train_reference_shift_are_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]
            splits = ["TRAIN", "TRAIN", "TRAIN", "TRAIN", "TRAIN", "TEST"]
            levels = [100.0, 100.5, 101.0, 1.0, 100.2, 120.0]
            events = []
            samples = []
            observations: dict[str, float] = {}
            for index, (date, split, level) in enumerate(zip(dates, splits, levels), 1):
                start = f"{date} 01:00"
                end = f"{date} 03:00"
                event_id = f"G1_F{index:04d}"
                events.append(event_row(event_id, start, end, split))
                samples.append(sample_row(f"S{index}", event_id, start, end, split, "WATER_LEVEL"))
                for hour in range(1, 4):
                    observations[f"{date} {hour:02d}:00"] = level + 0.1 * hour
            make_dataset(root, events, samples, observations, target="WATER_LEVEL")
            audit = build_dataset_quality_audit(root)
            rows = audit.water_level_station_audit
            train = rows[rows["split"].eq("TRAIN")].iloc[0]
            test = rows[rows["split"].eq("TEST")].iloc[0]
            self.assertEqual(train["qc_status"], "FAIL")
            self.assertIn("G1_F0004", train["train_reference_shift_event_ids"])
            self.assertGreater(int(test["out_of_train_range_count"]), 0)
            with self.assertRaisesRegex(ValueError, "水位FAIL站"):
                enforce_strict_quality(audit)

    def test_normalization_provenance_must_be_train(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = [event_row("G1_F0001", "2024-01-01 01:00", "2024-01-01 03:00")]
            samples = [sample_row("S1", "G1_F0001", "2024-01-01 01:00", "2024-01-01 03:00", "TRAIN", "WATER_LEVEL")]
            observations = {f"2024-01-01 {hour:02d}:00": 100.0 for hour in range(1, 4)}
            make_dataset(
                root,
                events,
                samples,
                observations,
                target="WATER_LEVEL",
                normalization_split="ALL",
            )
            audit = build_dataset_quality_audit(root)
            self.assertTrue((audit.water_level_station_audit["qc_status"] == "FAIL").all())


if __name__ == "__main__":
    unittest.main()
