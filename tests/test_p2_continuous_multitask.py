from __future__ import annotations

import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from config import load_config
from datasets.hunan import HunanContinuousDataset, build_hunan_loader
from losses import FloodMultitaskLoss
from trainers import Trainer
from scripts.build_v6_test_flood_events import Event, build_sample_rows
from metrics.p2_event_evaluation import evaluate_p2_flood_events


ROOT = Path(__file__).parents[1]
STATIC = (
    "log_incremental_area", "log_upstream_area", "mean_hillslope_flow_distance_m",
    "mean_slope_deg", "elevation_std_m", "drainage_density_km_per_km2",
    "soil_log_ksat_0_30cm", "soil_profile_depth_cm", "forest_fraction",
    "impervious_fraction",
)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture(root: Path) -> None:
    write_csv(
        root / "graph/node_catalog.csv",
        ["GRAPH_ID", "BASIN_ID", "NODE_INDEX", "STATION_ID", "OUTLET_ID", "ROLE", "IS_OUTLET", "STATIC_QC"],
        [{"GRAPH_ID": "G1", "BASIN_ID": "G1", "NODE_INDEX": 0, "STATION_ID": "S1", "OUTLET_ID": "S1", "ROLE": "OUTLET", "IS_OUTLET": 1, "STATIC_QC": "REVIEW"}],
    )
    write_csv(
        root / "graph/node_static_attributes.csv",
        ["GRAPH_ID", "STATION_ID", "STATIC_QC", *STATIC],
        [{"GRAPH_ID": "G1", "STATION_ID": "S1", "STATIC_QC": "REVIEW", **{name: (2.0 if name.startswith("log_") else 1.0) for name in STATIC}}],
    )
    write_csv(root / "graph/edge_topology.csv", ["GRAPH_ID", "FROM_NODE", "TO_NODE", "FROM_STATION", "TO_STATION"], [])
    write_csv(root / "graph/edge_static_attributes.csv", ["GRAPH_ID", "FROM_STATION", "TO_STATION", "reach_length_km", "reach_slope_m_per_m"], [])
    starts = {
        "TRAIN": datetime(2020, 1, 1),
        "VALIDATION": datetime(2020, 2, 1),
        "TEST": datetime(2020, 3, 1),
    }
    schema = {
        "contract": "continuous-hourly-dual-target-v1",
        "history_hours": 24,
        "forecast_hours": 6,
        "dynamic_features": ["RAIN_MM", "FLOW", "WATER_LEVEL"],
        "node_static_features": list(STATIC),
        "edge_static_features": ["reach_length_km", "reach_slope_m_per_m"],
        "physical_features": {"incremental_area_km2": {"source": "log_incremental_area", "transform": "ln", "unit": "km2"}},
        "time_splits": {name: {"start": str(start), "end": str(start + timedelta(hours=29))} for name, start in starts.items()},
    }
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "metadata/feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
    stats = {"computed_from_split": "TRAIN", "features": {name: {"count": 30, "mean": 1.0, "std": 1.0, "min": 0.0, "max": 30.0} for name in ("RAIN_MM", "FLOW", "WATER_LEVEL")}}
    (root / "metadata/normalization_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    dynamic = []
    samples = []
    for split, start in starts.items():
        for hour in range(30):
            q_offset = {"TRAIN": 0.0, "VALIDATION": 100.0, "TEST": 1000.0}[split]
            z_mask = not (split == "TEST" and hour == 23)
            dynamic.append({
                "GRAPH_ID": "G1", "TIMESTAMP": start + timedelta(hours=hour),
                "NODE_INDEX": 0, "STATION_ID": "S1", "RAIN_MM": 0.0,
                "FLOW": q_offset + hour, "WATER_LEVEL": 10.0 + hour / 10 if z_mask else "",
                "RAIN_MASK": 1, "FLOW_MASK": 1, "WATER_LEVEL_MASK": int(z_mask),
            })
        samples.append({
            "SAMPLE_ID": f"S_{split}", "GRAPH_ID": "G1", "OUTLET_ID": "S1",
            "INPUT_START": start, "FORECAST_TIME": start + timedelta(hours=23),
            "TARGET_START": start + timedelta(hours=24), "TARGET_END": start + timedelta(hours=29),
            "HISTORY_HOURS": 24, "FORECAST_HOURS": 6, "Q_VALID_COUNT": 6,
            "Z_VALID_COUNT": 6, "Q_COVERAGE": 1, "Z_COVERAGE": 1, "SPLIT": split,
        })
    write_csv(
        root / "dynamic/graph_G1_hourly.csv",
        ["GRAPH_ID", "TIMESTAMP", "NODE_INDEX", "STATION_ID", "RAIN_MM", "FLOW", "WATER_LEVEL", "RAIN_MASK", "FLOW_MASK", "WATER_LEVEL_MASK"],
        dynamic,
    )
    write_csv(
        root / "samples/sample_index.csv",
        ["SAMPLE_ID", "GRAPH_ID", "OUTLET_ID", "INPUT_START", "FORECAST_TIME", "TARGET_START", "TARGET_END", "HISTORY_HOURS", "FORECAST_HOURS", "Q_VALID_COUNT", "Z_VALID_COUNT", "Q_COVERAGE", "Z_COVERAGE", "SPLIT"],
        samples,
    )


class TestP2ContinuousMultitask(unittest.TestCase):
    def test_loader_dual_masks_t0_and_train_only_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            train = HunanContinuousDataset(root, "TRAIN", 24, 6)
            test = HunanContinuousDataset(root, "TEST", 24, 6)
            train_item = train[0]
            test_item = test[0]
            self.assertEqual(int(train_item.q_target_mask.sum()), 6)
            self.assertEqual(int(train_item.z_target_mask.sum()), 6)
            torch.testing.assert_close(train_item.z_target[:, 0], torch.arange(1, 7) / 10)
            self.assertEqual(int(test_item.z_target_mask.sum()), 0)
            self.assertFalse(bool(test_item.z_reference_mask[0]))
            statistics = train.train_target_statistics()
            self.assertLess(statistics["q_by_graph"]["G1"]["mean_m3s"], 30.0)
            self.assertAlmostEqual(
                statistics["delta_z_by_station"]["S1"]["mean_m"], 0.35, places=6
            )

    def test_weighted_sampler_is_train_only_and_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            train = HunanContinuousDataset(root, "TRAIN", 24, 6)
            weights = train.hydrologic_sampling_weights(
                q_scales={"G1": 1.0}, delta_z_scales={"S1": 0.1},
                response_strength=1.0, response_cap=4.0,
                minimum_weight=0.25, maximum_weight=4.0,
            )
            weighted = build_hunan_loader(train, 1, True, sampling_weights=weights)
            ordinary = build_hunan_loader(train, 1, True, sampling_weights=None)
            self.assertNotEqual(type(weighted.batch_sampler), type(ordinary.batch_sampler))
            validation = HunanContinuousDataset(root, "VALIDATION", 24, 6)
            with self.assertRaisesRegex(ValueError, "只允许TRAIN"):
                build_hunan_loader(validation, 1, False, sampling_weights=torch.ones(1))

    def test_delta_loss_independent_denominators_and_missing_task(self) -> None:
        cfg = load_config(ROOT / "configs/hunan_p2_continuous_multitask.yaml")
        cfg["_runtime"] = {"loss_scales": {"discharge": 999.0, "water_level": 999.0, "discharge_by_graph": {"G1": 2.0}, "delta_z_by_station": {"S1": 0.5}}}
        engine = FloodMultitaskLoss(cfg)
        class Batch: pass
        batch = Batch()
        batch.graph_id = ("G1",)
        batch.target_station_id = ("S1",)
        batch.q_target = torch.tensor([[[2.0], [0.0]]])
        batch.z_target = torch.tensor([[[0.0], [1.0]]])
        batch.q_target_mask = torch.tensor([[[True], [False]]])
        batch.z_target_mask = torch.tensor([[[False], [True]]])
        batch.z_history = torch.zeros(1, 24, 1)
        batch.z_mask = torch.zeros(1, 24, 1, dtype=torch.bool)
        batch.sample_weight = torch.ones(1)
        statistics = engine.batch_statistics({"q": batch.q_target.clone(), "z": batch.z_target.clone()}, batch)
        self.assertEqual(statistics["q_point"].denominator, 1)
        self.assertEqual(statistics["z_level"].denominator, 1)
        self.assertTrue(torch.isfinite(engine.combine(statistics)))
        batch.q_target_mask[:] = False
        statistics = engine.batch_statistics({"q": batch.q_target, "z": batch.z_target}, batch)
        self.assertEqual(statistics["q_point"].denominator, 0)
        self.assertTrue(torch.isfinite(engine.combine(statistics)))

    def test_p2_config_has_no_actual_early_stopping(self) -> None:
        cfg = load_config(ROOT / "configs/hunan_p2_continuous_multitask.yaml")
        self.assertEqual(cfg["training"]["epochs"], 100)
        self.assertFalse(cfg["training"]["early_stopping"])
        self.assertTrue(cfg["training"]["final_checkpoint"].endswith("epoch100.pt"))

    def test_event_index_references_only_strict_test_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            start = datetime(2020, 3, 1)
            event = Event(
                "P2_G1_TEST", "G1", "S1", start + timedelta(hours=24),
                start + timedelta(hours=29), start + timedelta(hours=20),
                start + timedelta(hours=22), start + timedelta(hours=24),
                start + timedelta(hours=27), start + timedelta(hours=29),
                20.0, 1.0, 5.0, "A",
            )
            rows = build_sample_rows(
                root, [event], start, start + timedelta(hours=29)
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["SPLIT"], "TEST")
            self.assertEqual(rows[0]["EVENT_ID"], "P2_G1_TEST")
            self.assertEqual(rows[0]["FORECAST_HORIZONS"], "h1;h2;h3;h4;h5;h6")

    def test_disabled_early_stopping_runs_epoch_0_through_99(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = load_config(ROOT / "configs/hunan_p2_continuous_multitask.yaml")
            cfg["device"] = "cpu"
            cfg["amp"] = False
            cfg["training"]["patience"] = 1
            cfg["training"]["checkpoint"] = str(Path(directory) / "best.pt")
            cfg["training"]["final_checkpoint"] = str(Path(directory) / "epoch100.pt")
            cfg["training"]["log_csv"] = str(Path(directory) / "train.csv")
            cfg["_runtime"] = {"loss_scales": {"discharge": 1.0, "water_level": 1.0}}
            trainer = Trainer(torch.nn.Linear(1, 1), cfg, torch.device("cpu"))
            epochs: list[int] = []

            def constant_epoch(_loader, epoch: int):
                epochs.append(epoch)
                return {"loss": 1.0}

            with (
                mock.patch.object(trainer, "train_epoch", side_effect=constant_epoch),
                mock.patch.object(trainer, "save_checkpoint") as save,
                mock.patch("builtins.print"),
            ):
                history = trainer.fit([object()], val_loader=None)
            self.assertEqual(epochs, list(range(100)))
            self.assertEqual(len(history), 100)
            self.assertTrue(
                any(
                    call.args[1] == 99 and call.kwargs.get("kind") == "final"
                    for call in save.call_args_list
                )
            )

    def test_evaluation_recovers_absolute_z_from_t0_plus_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            fixture(root)
            dataset = HunanContinuousDataset(root, "TRAIN", 24, 6)
            loader = build_hunan_loader(dataset, 1, False)

            class PerfectModel(torch.nn.Module):
                def forward(self, batch):
                    return {"q": batch.q_target.clone(), "z": batch.z_target.clone()}

            output = Path(directory) / "evaluation"
            summary = evaluate_p2_flood_events(
                PerfectModel(), loader, torch.device("cpu"), output
            )
            self.assertEqual(summary["q_valid_points"], 6)
            self.assertEqual(summary["delta_z_valid_points"], 6)
            with (output / "test_predictions_q_z_delta_z.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["DELTA_Z_OBS"], rows[0]["DELTA_Z_PRED"])
            self.assertEqual(rows[0]["Z_OBS"], rows[0]["Z_PRED"])


if __name__ == "__main__":
    unittest.main()
