from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from datasets.hunan import HunanGraphEventDataset


class TestHunanTargetMapScope(unittest.TestCase):
    @staticmethod
    def _dataset(root: Path) -> HunanGraphEventDataset:
        dataset = HunanGraphEventDataset.__new__(HunanGraphEventDataset)
        dataset.root = root
        # B009 represents a valid static graph which was excluded before the
        # formal event/sample stage and therefore has no target or dynamic file.
        dataset._graphs = {"B001": object(), "B009": object()}
        return dataset

    def test_target_map_only_requires_graphs_used_by_formal_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            events.mkdir(parents=True)
            (events / "target_variable_by_graph.csv").write_text(
                "GRAPH_ID,TARGET_VARIABLE\nB001,FLOW\n", encoding="utf-8"
            )

            result = self._dataset(root)._load_target_variables_by_graph({"B001"})

            self.assertEqual(result, {"B001": frozenset({"FLOW"})})

    def test_target_map_still_rejects_missing_formal_event_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            events.mkdir(parents=True)
            (events / "target_variable_by_graph.csv").write_text(
                "GRAPH_ID,TARGET_VARIABLE\nB001,FLOW\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "B009"):
                self._dataset(root)._load_target_variables_by_graph(
                    {"B001", "B009"}
                )

    def test_event_loader_allows_unfinished_hydro_end_but_requires_hydro_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            events.mkdir(parents=True)
            (events / "flood_events_final.csv").write_text(
                "EVENT_ID,GRAPH_ID,BASIN_ID,OUTLET_ID,RAIN_START,RAIN_END,"
                "HYDRO_START,PEAK_TIME,HYDRO_END,SAMPLE_START,SAMPLE_END,"
                "EVENT_TYPE,EVENT_GRADE,COMPOUND_EVENT,PEAK_COUNT,"
                "SOURCE_RAIN_EVENT_IDS,SOURCE_RAIN_EVENT_COUNT\n"
                "E1,B001,B001,O1,2022-01-01 00:00:00,2022-01-01 02:00:00,"
                "2022-01-01 01:00:00,"
                "2022-01-01 03:00:00,,2021-12-31 00:00:00,"
                "2022-01-03 00:00:00,HYDRO_FLOOD,A,false,1,R1,1\n",
                encoding="utf-8",
            )
            dataset = HunanGraphEventDataset.__new__(HunanGraphEventDataset)
            dataset.root = root
            dataset._graphs = {
                "B001": SimpleNamespace(basin_id="B001", outlet_id="O1")
            }

            loaded = dataset._load_events()

            self.assertEqual(loaded["E1"].event_year, 2022)
            self.assertEqual(loaded["E1"].split_time.hour, 3)

            text = (events / "flood_events_final.csv").read_text(encoding="utf-8")
            (events / "flood_events_final.csv").write_text(
                text.replace("2022-01-01 01:00:00,2022-01-01 03:00:00", ",2022-01-01 03:00:00"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "HYDRO_START不能为空"):
                dataset._load_events()


if __name__ == "__main__":
    unittest.main()
