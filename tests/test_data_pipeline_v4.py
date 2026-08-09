from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


HUNAN_ROOT = Path(__file__).parents[2]
WORKFLOW = HUNAN_ROOT / "Arcgis" / "MERIT_workflow"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, WORKFLOW / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STEP13 = _load_script("hunan_step13", "13_match_rain_hydro_flood_events.py")
STEP16 = _load_script("hunan_step16", "16_build_model_dataset_v3.py")


def _candidate(
    rain_id: str,
    *,
    rain_start: str,
    rain_end: str,
    response_start: str,
    peak_time: str,
    peak_value: float,
    event_end: str | None,
    provisional_end: str,
    complete: bool,
) -> dict:
    start = pd.Timestamp(rain_start)
    end = pd.Timestamp(rain_end)
    sample_start = start - pd.Timedelta(hours=48)
    sample_end = end + pd.Timedelta(hours=48)
    return {
        "MATCH_STATUS": "HYDRO_FLOOD",
        "basin_id": "BTEST",
        "outlet_id": "OTEST",
        "rain_event_id": rain_id,
        "rain_start": start,
        "rain_end": end,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "_RAIN_START": start,
        "_RAIN_END": end,
        "_SAMPLE_START": sample_start,
        "_SAMPLE_END": sample_end,
        "RESPONSE_START": pd.Timestamp(response_start),
        "PEAK_TIME": pd.Timestamp(peak_time),
        "PEAK_VALUE": peak_value,
        "RISE_VALUE": peak_value - 1.0,
        "EVENT_END": pd.Timestamp(event_end) if event_end else pd.NaT,
        "PROVISIONAL_SEARCH_END": pd.Timestamp(provisional_end),
        "RECESSION_COMPLETE": complete,
        "HYDRO_MISSING_PCT": 0.0,
        "LAG_PEAK_FROM_RAIN_CENTROID_H": 1.0,
        "RAIN_CENTROID": start,
        "RAIN_CENTROID_SOURCE": "TEST",
        "RAIN_CENTROID_HOUR_COUNT": 1,
        "total_basin_rain_mm": 10.0,
        "wet_hour_count": 2,
        "peak_basin_hourly_rain_mm": 5.0,
        "max_node_hourly_rain_mm": 6.0,
    }


class TestEventMergeContract(unittest.TestCase):
    def test_weighted_rain_centroid_preserves_datetime_unit(self) -> None:
        frame = pd.DataFrame({
            "_START": pd.to_datetime([
                "2021-06-02 01:00:00", "2021-06-02 02:00:00"
            ]),
            "_END": pd.to_datetime([
                "2021-06-02 02:00:00", "2021-06-02 03:00:00"
            ]),
            "_RAIN": [1.0, 3.0],
        })

        centroid, source, count = STEP13.weighted_rain_centroid(
            frame,
            pd.Timestamp("2021-06-02 01:00:00"),
            pd.Timestamp("2021-06-02 03:00:00"),
        )

        self.assertEqual(centroid, pd.Timestamp("2021-06-02 02:15:00"))
        self.assertEqual(source, "BASIN_HOURLY_WEIGHTED")
        self.assertEqual(count, 2)

    def test_later_incomplete_peak_clears_obsolete_end_and_expands_windows(self) -> None:
        first = _candidate(
            "R1",
            rain_start="2021-05-20 07:00:00",
            rain_end="2021-05-20 20:00:00",
            response_start="2021-05-20 08:00:00",
            peak_time="2021-05-20 08:00:00",
            peak_value=119.0,
            event_end="2021-05-21 10:00:00",
            provisional_end="2021-05-22 08:00:00",
            complete=True,
        )
        second = _candidate(
            "R2",
            rain_start="2021-05-21 07:00:00",
            rain_end="2021-05-23 20:00:00",
            response_start="2021-05-21 18:00:00",
            peak_time="2021-05-22 08:00:00",
            peak_value=246.0,
            event_end=None,
            provisional_end="2021-05-24 08:00:00",
            complete=False,
        )

        merged, _review = STEP13.merge_accepted_events(pd.DataFrame([first, second]), 2)

        self.assertEqual(len(merged), 1)
        row = merged.iloc[0]
        self.assertTrue(bool(row["COMPOUND_EVENT"]))
        self.assertEqual(int(row["SOURCE_RAIN_EVENT_COUNT"]), 2)
        self.assertEqual(int(row["PEAK_COUNT"]), 2)
        self.assertEqual(row["SOURCE_RAIN_EVENT_IDS"], "R1;R2")
        self.assertEqual(pd.Timestamp(row["PEAK_TIME"]), pd.Timestamp("2021-05-22 08:00:00"))
        self.assertTrue(pd.isna(row["EVENT_END"]))
        self.assertFalse(bool(row["RECESSION_COMPLETE"]))
        self.assertEqual(pd.Timestamp(row["rain_end"]), pd.Timestamp("2021-05-23 20:00:00"))
        self.assertEqual(pd.Timestamp(row["sample_end"]), pd.Timestamp("2021-05-25 20:00:00"))

    def test_complete_components_use_latest_end(self) -> None:
        first = _candidate(
            "R1",
            rain_start="2024-01-01 00:00:00",
            rain_end="2024-01-01 03:00:00",
            response_start="2024-01-01 01:00:00",
            peak_time="2024-01-01 02:00:00",
            peak_value=10.0,
            event_end="2024-01-01 05:00:00",
            provisional_end="2024-01-03 02:00:00",
            complete=True,
        )
        second = _candidate(
            "R2",
            rain_start="2024-01-01 04:00:00",
            rain_end="2024-01-01 07:00:00",
            response_start="2024-01-01 05:00:00",
            peak_time="2024-01-01 06:00:00",
            peak_value=12.0,
            event_end="2024-01-01 09:00:00",
            provisional_end="2024-01-03 06:00:00",
            complete=True,
        )

        merged, _review = STEP13.merge_accepted_events(pd.DataFrame([first, second]), 2)

        row = merged.iloc[0]
        self.assertEqual(pd.Timestamp(row["EVENT_END"]), pd.Timestamp("2024-01-01 09:00:00"))
        self.assertLessEqual(pd.Timestamp(row["PEAK_TIME"]), pd.Timestamp(row["EVENT_END"]))


class TestModelDatasetV4Rules(unittest.TestCase):
    def test_station_id_with_e_is_never_parsed_as_scientific_notation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rain.csv"
            path.write_text(
                "node_id,area_rain_mm\n611E2950,1.25\n",
                encoding="utf-8",
            )

            frame = STEP16.read_table(path)

            self.assertEqual(frame.loc[0, "node_id"], "611E2950")
            self.assertEqual(STEP16.normalize_id(frame.loc[0, "node_id"]), "611E2950")

    def test_611e2950_negative_flow_becomes_missing_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station.csv"
            pd.DataFrame({
                "STCD": ["611E2950", "611E2950"],
                "TM": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
                "Q": [-2.5, 3.0],
                "Z": [1.0, 1.1],
            }).to_csv(path, index=False, encoding="utf-8-sig")

            hydro, audit = STEP16.load_hydro(
                [(path, "611E2950")],
                {"611E2950"},
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
            )

            first = hydro.sort_values("TIMESTAMP").iloc[0]
            self.assertTrue(np.isnan(first["FLOW"]))
            self.assertNotEqual(first["FLOW"], 0.0)
            self.assertEqual(int(audit.iloc[0]["NEGATIVE_FLOW_RAW_AS_MISSING"]), 1)

    def test_low_target_coverage_is_traced_and_closes_event(self) -> None:
        event_start = pd.Timestamp("2024-01-01 00:00:00")
        events = pd.DataFrame([{
            "EVENT_ID": "E1",
            "GRAPH_ID": "B015",
            "OUTLET_ID": "611E2950",
            "SAMPLE_START": event_start,
            "RAIN_START": event_start,
            "HYDRO_END": event_start + pd.Timedelta(hours=4),
            "SAMPLE_END": event_start + pd.Timedelta(hours=4),
            "SPLIT": "TEST",
        }])
        dynamic = {"B015": pd.DataFrame({
            "STATION_ID": ["611E2950"] * 5,
            "TIMESTAMP": pd.date_range(event_start, periods=5, freq="h"),
            "FLOW_MASK": [1, 1, 0, 0, 0],
            "WATER_LEVEL_MASK": [1, 1, 1, 1, 1],
        })}
        target_map = pd.DataFrame([{
            "GRAPH_ID": "B015",
            "OUTLET_ID": "611E2950",
            "TARGET_VARIABLE": "FLOW",
        }])

        samples, rejected = STEP16.build_sample_index(
            events, dynamic, target_map,
            history_hours=2, forecast_hours=2, step_hours=1,
            min_target_coverage=0.8,
        )

        self.assertTrue(samples.empty)
        self.assertFalse(rejected.empty)
        self.assertEqual(set(rejected["EVENT_ID"]), {"E1"})
        self.assertEqual(set(rejected["REASON"]), {"TARGET_COVERAGE_BELOW_THRESHOLD"})
        self.assertTrue((rejected["TARGET_COVERAGE"] < rejected["MIN_TARGET_COVERAGE"]).all())


if __name__ == "__main__":
    unittest.main()
