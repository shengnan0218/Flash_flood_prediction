from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
import yaml

from audit_model_dataset import recompute_normalization
from config import load_config, validate_config
from models import HybridFloodModel
from scripts.common import (
    setup_evaluation,
    setup_training,
    validate_checkpoint_config,
)
from trainers import Trainer
from validate_dataset import validate_dataset


ROOT = Path(__file__).parents[1]
NODE_FEATURES = (
    "log_incremental_area",
    "log_upstream_area",
    "mean_hillslope_flow_distance_m",
    "mean_slope_deg",
    "elevation_std_m",
    "drainage_density_km_per_km2",
    "soil_log_ksat_0_30cm",
    "soil_profile_depth_cm",
    "forest_fraction",
    "impervious_fraction",
)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def build_formal_fixture(root: Path) -> None:
    graphs = {
        "G_FLOW": {
            "basin": "B_FLOW",
            "outlet": "F_OUT",
            "target": "FLOW",
            "stations": ("F_UP", "F_OUT"),
            "edges": ((0, 1),),
        },
        "G_LEVEL": {
            "basin": "B_LEVEL",
            "outlet": "Z_OUT",
            "target": "WATER_LEVEL",
            "stations": ("Z_UP1", "Z_UP2", "Z_OUT"),
            "edges": ((0, 2), (1, 2)),
        },
    }

    node_rows: list[dict] = []
    topology_rows: list[dict] = []
    node_static_rows: list[dict] = []
    edge_static_rows: list[dict] = []
    for graph_number, (graph_id, graph) in enumerate(graphs.items(), start=1):
        stations = graph["stations"]
        for node_index, station in enumerate(stations):
            node_rows.append(
                {
                    "GRAPH_ID": graph_id,
                    "BASIN_ID": graph["basin"],
                    "NODE_INDEX": node_index,
                    "STATION_ID": station,
                    "OUTLET_ID": graph["outlet"],
                    "ROLE": "OUTLET" if station == graph["outlet"] else "UPSTREAM",
                    "IS_OUTLET": int(station == graph["outlet"]),
                }
            )
            incremental_area = 10.0 + graph_number * 5.0 + node_index
            upstream_area = incremental_area * (node_index + 1)
            values = {
                "log_incremental_area": math.log1p(incremental_area),
                "log_upstream_area": math.log1p(upstream_area),
                "mean_hillslope_flow_distance_m": 600.0 + node_index * 100.0,
                "mean_slope_deg": 3.0 + node_index,
                "elevation_std_m": 40.0 + node_index,
                "drainage_density_km_per_km2": 0.8 + node_index * 0.1,
                "soil_log_ksat_0_30cm": 1.2 + node_index * 0.1,
                "soil_profile_depth_cm": 80.0 + node_index,
                "forest_fraction": 0.5,
                "impervious_fraction": 0.05,
            }
            node_static_rows.append(
                {"GRAPH_ID": graph_id, "STATION_ID": station, **values}
            )
        for edge_number, (source, destination) in enumerate(graph["edges"]):
            source_station = stations[source]
            destination_station = stations[destination]
            topology_rows.append(
                {
                    "GRAPH_ID": graph_id,
                    "FROM_NODE": source,
                    "TO_NODE": destination,
                    "FROM_STATION": source_station,
                    "TO_STATION": destination_station,
                }
            )
            edge_static_rows.append(
                {
                    "GRAPH_ID": graph_id,
                    "FROM_STATION": source_station,
                    "TO_STATION": destination_station,
                    "reach_length_km": 5.0 + edge_number,
                    "reach_slope_m_per_m": 0.001 + edge_number * 0.0002,
                }
            )

    _write_csv(
        root / "graph" / "node_catalog.csv",
        [
            "GRAPH_ID",
            "BASIN_ID",
            "NODE_INDEX",
            "STATION_ID",
            "OUTLET_ID",
            "ROLE",
            "IS_OUTLET",
        ],
        node_rows,
    )
    _write_csv(
        root / "graph" / "edge_topology.csv",
        ["GRAPH_ID", "FROM_NODE", "TO_NODE", "FROM_STATION", "TO_STATION"],
        topology_rows,
    )
    _write_csv(
        root / "graph" / "node_static_attributes.csv",
        ["GRAPH_ID", "STATION_ID", *NODE_FEATURES],
        node_static_rows,
    )
    _write_csv(
        root / "graph" / "edge_static_attributes.csv",
        [
            "GRAPH_ID",
            "FROM_STATION",
            "TO_STATION",
            "reach_length_km",
            "reach_slope_m_per_m",
        ],
        edge_static_rows,
    )

    event_rows: list[dict] = []
    split_rows: list[dict] = []
    sample_rows: list[dict] = []
    all_event_rows: list[dict] = []
    split_days = {"TRAIN": 1, "VALIDATION": 2, "TEST": 3}
    for graph_id, graph in graphs.items():
        dynamic_rows: list[dict] = []
        for split, day in split_days.items():
            start = datetime(2020, 1, day, 0)
            # A two-hour inclusive history is [start, start + 1h].
            forecast = start + timedelta(hours=1)
            end = forecast + timedelta(hours=2)
            event_id = f"{graph_id}_{split}"
            sample_id = f"S_{event_id}"
            event_rows.append(
                {
                    "EVENT_ID": event_id,
                    "GRAPH_ID": graph_id,
                    "BASIN_ID": graph["basin"],
                    "OUTLET_ID": graph["outlet"],
                    "RAIN_START": _time(start),
                    "RAIN_END": _time(forecast),
                    "HYDRO_START": _time(forecast),
                    "PEAK_TIME": _time(forecast + timedelta(hours=1)),
                    "HYDRO_END": _time(end),
                    "SAMPLE_START": _time(start),
                    "SAMPLE_END": _time(end),
                    "EVENT_TYPE": "HYDRO_FLOOD",
                    "EVENT_GRADE": "A",
                    "COMPOUND_EVENT": 0,
                    "PEAK_COUNT": 1,
                    "SOURCE_RAIN_EVENT_IDS": event_id,
                    "SOURCE_RAIN_EVENT_COUNT": 1,
                }
            )
            all_event_rows.append({"EVENT_ID": event_id, "GRAPH_ID": graph_id})
            split_rows.append(
                {
                    "EVENT_ID": event_id,
                    "GRAPH_ID": graph_id,
                    "EVENT_YEAR": 2020,
                    "EVENT_GRADE": "A",
                    "SPLIT": split,
                    "SPLIT_REASON": "chronological fixture",
                }
            )
            sample_rows.append(
                {
                    "SAMPLE_ID": sample_id,
                    "EVENT_ID": event_id,
                    "GRAPH_ID": graph_id,
                    "OUTLET_ID": graph["outlet"],
                    "INPUT_START": _time(start),
                    "FORECAST_TIME": _time(forecast),
                    "TARGET_END": _time(end),
                    "HISTORY_HOURS": 2,
                    "FORECAST_HOURS": 2,
                    "TARGET_VARIABLE": graph["target"],
                    "SPLIT": split,
                }
            )
            for hour in range(5):
                timestamp = start + timedelta(hours=hour)
                for node_index, station in enumerate(graph["stations"]):
                    flow_mask = not (
                        graph_id == "G_FLOW"
                        and split == "TRAIN"
                        and hour == 1
                        and node_index == 0
                    )
                    flow = 8.0 + day * 2.0 + hour + node_index
                    dynamic_rows.append(
                        {
                            "GRAPH_ID": graph_id,
                            "TIMESTAMP": _time(timestamp),
                            "STATION_ID": station,
                            "RAIN_MM": max(0.0, 3.0 - abs(hour - 2))
                            + node_index * 0.1,
                            "FLOW": flow if flow_mask else "",
                            "WATER_LEVEL": 0.8 + 0.08 * flow,
                            "RAIN_MASK": 1,
                            "FLOW_MASK": int(flow_mask),
                            "WATER_LEVEL_MASK": 1,
                        }
                    )
        _write_csv(
            root / "dynamic" / f"graph_{graph['basin']}_hourly.csv",
            [
                "GRAPH_ID",
                "TIMESTAMP",
                "STATION_ID",
                "RAIN_MM",
                "FLOW",
                "WATER_LEVEL",
                "RAIN_MASK",
                "FLOW_MASK",
                "WATER_LEVEL_MASK",
            ],
            dynamic_rows,
        )

    event_columns = [
        "EVENT_ID",
        "GRAPH_ID",
        "BASIN_ID",
        "OUTLET_ID",
        "RAIN_START",
        "RAIN_END",
        "HYDRO_START",
        "PEAK_TIME",
        "HYDRO_END",
        "SAMPLE_START",
        "SAMPLE_END",
        "EVENT_TYPE",
        "EVENT_GRADE",
        "COMPOUND_EVENT",
        "PEAK_COUNT",
        "SOURCE_RAIN_EVENT_IDS",
        "SOURCE_RAIN_EVENT_COUNT",
    ]
    _write_csv(root / "events" / "flood_events_all.csv", ["EVENT_ID", "GRAPH_ID"], all_event_rows)
    _write_csv(root / "events" / "flood_events_final.csv", event_columns, event_rows)
    _write_csv(
        root / "events" / "data_split.csv",
        ["EVENT_ID", "GRAPH_ID", "EVENT_YEAR", "EVENT_GRADE", "SPLIT", "SPLIT_REASON"],
        split_rows,
    )
    _write_csv(
        root / "events" / "sample_index.csv",
        [
            "SAMPLE_ID",
            "EVENT_ID",
            "GRAPH_ID",
            "OUTLET_ID",
            "INPUT_START",
            "FORECAST_TIME",
            "TARGET_END",
            "HISTORY_HOURS",
            "FORECAST_HOURS",
            "TARGET_VARIABLE",
            "SPLIT",
        ],
        sample_rows,
    )
    _write_csv(
        root / "events" / "target_variable_by_graph.csv",
        ["GRAPH_ID", "BASIN_ID", "OUTLET_ID", "TARGET_VARIABLE"],
        [
            {
                "GRAPH_ID": graph_id,
                "BASIN_ID": graph["basin"],
                "OUTLET_ID": graph["outlet"],
                "TARGET_VARIABLE": graph["target"],
            }
            for graph_id, graph in graphs.items()
        ],
    )

    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    feature_schema = {
        "dynamic_features": ["FLOW", "WATER_LEVEL"],
        "node_static_features": list(NODE_FEATURES),
        "edge_static_features": [
            "reach_length_km",
            "reach_slope_m_per_m",
        ],
        "physical_features": {
            "incremental_area_km2": {
                "source": "log_incremental_area",
                "transform": "log1p",
                "unit": "km2",
            }
        },
    }
    (metadata / "feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    events_for_stats = pd.DataFrame(event_rows)
    split_lookup = {row["EVENT_ID"]: row["SPLIT"] for row in split_rows}
    events_for_stats["SPLIT"] = events_for_stats["EVENT_ID"].map(split_lookup)
    normalization = {
        "computed_from_split": "TRAIN",
        **recompute_normalization(
            root, events_for_stats, pd.DataFrame(sample_rows)
        ),
    }
    (metadata / "normalization_stats.json").write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(
        metadata / "dataset_summary.csv",
        ["KEY", "VALUE"],
        [{"KEY": "graphs", "VALUE": 2}],
    )
    (metadata / "source_manifest.json").write_text(
        json.dumps({"name": "formal integration fixture"}), encoding="utf-8"
    )
    (metadata / "build_log.txt").write_text("fixture built\n", encoding="utf-8")

    _write_csv(root / "qc" / "event_exclusion.csv", ["EVENT_ID", "REASON"], [])
    _write_csv(
        root / "qc" / "sample_rejection.csv",
        [
            "REJECTION_ID", "SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "OUTLET_ID",
            "FORECAST_TIME", "TARGET_START", "TARGET_END", "TARGET_VARIABLE",
            "TARGET_COVERAGE", "MIN_TARGET_COVERAGE", "REASON", "SPLIT",
        ],
        [],
    )
    for name in (
        "dynamic_coverage",
        "hydro_file_selection",
        "hydro_load_audit",
        "rain_source_coverage",
    ):
        _write_csv(root / "qc" / f"{name}.csv", ["STATUS"], [{"STATUS": "OK"}])


def write_test_config(path: Path, dataset_root: Path, output_root: Path) -> dict:
    cfg = deepcopy(load_config(ROOT / "configs" / "hunan_e1_pure_ai.yaml"))
    cfg.update(
        {
            "history_length": 2,
            "forecast_horizon": 2,
            "hidden_dim": 8,
            "batch_size": 2,
            "device": "cpu",
            "amp": False,
        }
    )
    cfg["data"]["dataset_root"] = str(dataset_root)
    cfg["training"] = {
        "epochs": 1,
        "patience": 1,
        "gradient_clip": 1.0,
        "checkpoint": str(output_root / "best.pt"),
        "log_csv": str(output_root / "train.csv"),
    }
    validate_config(cfg)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return cfg


def write_qnorm_test_config(
    path: Path, dataset_root: Path, output_root: Path
) -> dict:
    cfg = write_test_config(path, dataset_root, output_root)
    cfg["loss"]["mode"] = "multitask"
    cfg["loss"]["q_scale_mode"] = "per_graph"
    cfg["loss"]["q_scale_floor_m3s"] = 1.0
    cfg["validation_selection"]["mode"] = "composite"
    validate_config(cfg)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return cfg


class TestHunanFormalIntegration(unittest.TestCase):
    def test_per_graph_q_scale_is_train_only_unique_and_floor_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dataset_root = temporary / "_model_dataset"
            build_formal_fixture(dataset_root)

            # Duplicate one TRAIN sliding window under a new SAMPLE_ID.  Its two
            # target timestamps must still contribute exactly once to std.
            sample_path = dataset_root / "events" / "sample_index.csv"
            with sample_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or ())
                sample_rows = list(reader)
            duplicate = next(
                dict(row)
                for row in sample_rows
                if row["EVENT_ID"] == "G_FLOW_TRAIN"
            )
            duplicate["SAMPLE_ID"] = "S_G_FLOW_TRAIN_DUPLICATE_WINDOW"
            sample_rows.append(duplicate)
            _write_csv(sample_path, fieldnames, sample_rows)

            # Make non-TRAIN outlet FLOW deliberately extreme.  Per-graph scale
            # must remain the TRAIN values 13 and 14 m3/s.
            dynamic_path = dataset_root / "dynamic" / "graph_B_FLOW_hourly.csv"
            with dynamic_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                dynamic_fields = list(reader.fieldnames or ())
                dynamic_rows = list(reader)
            for row in dynamic_rows:
                if (
                    row["STATION_ID"] == "F_OUT"
                    and not row["TIMESTAMP"].startswith("2020-01-01")
                ):
                    row["FLOW"] = "10000.0"
            _write_csv(dynamic_path, dynamic_fields, dynamic_rows)

            config_path = temporary / "qnorm.yaml"
            write_qnorm_test_config(
                config_path, dataset_root, temporary / "out"
            )
            cfg, _model, train_loader, _validation_loader, _device = setup_training(
                config_path
            )
            audit = cfg["_runtime"]["q_scale_audit"]["graphs"]["G_FLOW"]
            self.assertEqual(audit["valid_unique_point_count"], 2)
            self.assertAlmostEqual(audit["mean_m3s"], 13.5)
            self.assertAlmostEqual(audit["std_m3s"], 0.5)
            self.assertAlmostEqual(audit["q_loss_scale_m3s"], 1.0)
            self.assertTrue(audit["floor_applied"])
            self.assertEqual(len(train_loader.dataset), 3)

            eval_cfg, _eval_model, _test_loader, _eval_device = setup_evaluation(
                config_path, split="TEST"
            )
            self.assertEqual(
                eval_cfg["_runtime"]["loss_scales"]["discharge_by_graph"],
                cfg["_runtime"]["loss_scales"]["discharge_by_graph"],
            )
            level_cfg, _level_model, _level_train, _level_val, _level_device = (
                setup_training(config_path, graph_id="G_LEVEL")
            )
            self.assertEqual(
                level_cfg["_runtime"]["loss_scales"]["discharge_by_graph"],
                {},
            )
            self.assertEqual(
                level_cfg["_runtime"]["q_scale_audit"]["graphs"]["G_LEVEL"][
                    "status"
                ],
                "NOT_APPLICABLE_NO_FLOW_SUPERVISION",
            )
            checkpoint = {"config": deepcopy(cfg)}
            validate_checkpoint_config(checkpoint, eval_cfg)
            changed_scale_cfg = deepcopy(eval_cfg)
            changed_scale_cfg["_runtime"]["loss_scales"][
                "discharge_by_graph"
            ]["G_FLOW"] += 0.25
            with self.assertRaisesRegex(
                ValueError, "loss_scales.discharge_by_graph.G_FLOW"
            ):
                validate_checkpoint_config(checkpoint, changed_scale_cfg)

    def test_per_graph_q_scale_fails_with_one_unique_train_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dataset_root = temporary / "_model_dataset"
            build_formal_fixture(dataset_root)
            dynamic_path = dataset_root / "dynamic" / "graph_B_FLOW_hourly.csv"
            with dynamic_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or ())
                rows = list(reader)
            for row in rows:
                if (
                    row["STATION_ID"] == "F_OUT"
                    and row["TIMESTAMP"] == "2020-01-01 03:00:00"
                ):
                    row["FLOW"] = ""
                    row["FLOW_MASK"] = "0"
            _write_csv(dynamic_path, fieldnames, rows)
            config_path = temporary / "qnorm.yaml"
            write_qnorm_test_config(
                config_path, dataset_root, temporary / "out"
            )
            with self.assertRaisesRegex(
                ValueError, "GRAPH_ID=G_FLOW.*仅1个"
            ):
                setup_training(config_path)

    def test_two_graph_auto_target_train_validate_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            dataset_root = temporary / "_model_dataset"
            build_formal_fixture(dataset_root)
            config_path = temporary / "formal.yaml"
            raw_cfg = write_test_config(config_path, dataset_root, temporary / "out")

            preflight = validate_dataset(config_path)
            self.assertEqual(preflight["status"], "VALID")
            self.assertEqual(preflight["train"]["samples"], 2)
            self.assertFalse(any(preflight["event_overlap"].values()))

            cfg, model, train_loader, validation_loader, device = setup_training(
                config_path
            )
            self.assertEqual(device, torch.device("cpu"))
            self.assertEqual(train_loader.dataset.num_stations, 5)
            self.assertEqual(
                train_loader.dataset.graph_node_counts,
                {"G_FLOW": 2, "G_LEVEL": 3},
            )
            self.assertTrue(
                train_loader.dataset.event_ids.isdisjoint(
                    validation_loader.dataset.event_ids
                )
            )
            self.assertIs(
                train_loader.dataset._dynamic["G_FLOW"],
                validation_loader.dataset._dynamic["G_FLOW"],
            )

            batches = list(train_loader)
            self.assertEqual({batch.graph_id[0] for batch in batches}, {"G_FLOW", "G_LEVEL"})
            for batch in batches:
                self.assertEqual(len(set(batch.graph_id)), 1)
                self.assertTrue(torch.isfinite(batch.dynamic_node_features).all())
                self.assertFalse(batch.rainfall_mask[:, -2:].any())
                self.assertTrue((batch.node_area_km2 > 0).all())
                self.assertGreaterEqual(batch.edge_static[:, 0].min().item(), 5000.0)
                output = model(batch)
                self.assertEqual(output["q"].shape, batch.q_target.shape)
                self.assertTrue(torch.isfinite(output["q"]).all())
                self.assertTrue(torch.isfinite(output["z"]).all())
                if batch.graph_id[0] == "G_FLOW":
                    self.assertEqual(batch.q_target_mask.sum().item(), 2)
                    self.assertEqual(batch.z_target_mask.sum().item(), 0)
                else:
                    self.assertEqual(batch.q_target_mask.sum().item(), 0)
                    self.assertEqual(batch.z_target_mask.sum().item(), 2)

            trainer = Trainer(model, cfg, device)
            history = trainer.fit(train_loader, validation_loader)
            self.assertEqual(len(history), 1)
            self.assertTrue(Path(cfg["training"]["checkpoint"]).is_file())

            eval_cfg, eval_model, test_loader, eval_device = setup_evaluation(
                config_path, split="TEST"
            )
            eval_trainer = Trainer(eval_model, eval_cfg, eval_device)
            checkpoint = eval_trainer.load_weights(cfg["training"]["checkpoint"])
            validate_checkpoint_config(checkpoint, eval_cfg)
            unseen_cfg = deepcopy(eval_cfg)
            unseen_cfg["_runtime"]["data_contract"]["graph_ids"].append(
                "G_NEVER_TRAINED"
            )
            with self.assertRaisesRegex(ValueError, "data_contract.graph_ids"):
                validate_checkpoint_config(checkpoint, unseen_cfg)
            remapped_cfg = deepcopy(eval_cfg)
            remapped_cfg["_runtime"]["data_contract"]["station_ids"].reverse()
            with self.assertRaisesRegex(ValueError, "data_contract.station_ids"):
                validate_checkpoint_config(checkpoint, remapped_cfg)
            changed_solver_cfg = deepcopy(eval_cfg)
            changed_solver_cfg["solver"]["implicit_iterations"] += 1
            with self.assertRaisesRegex(ValueError, "solver"):
                validate_checkpoint_config(checkpoint, changed_solver_cfg)
            changed_bounds_cfg = deepcopy(eval_cfg)
            changed_bounds_cfg["physical_bounds"]["width"][1] += 1.0
            with self.assertRaisesRegex(ValueError, "physical_bounds"):
                validate_checkpoint_config(checkpoint, changed_bounds_cfg)
            metrics = eval_trainer.evaluate(test_loader)
            self.assertEqual(metrics["q_valid_count"], 2)
            self.assertEqual(metrics["z_valid_count"], 2)
            self.assertTrue(math.isfinite(float(metrics["loss"])))
            self.assertTrue(
                train_loader.dataset.event_ids.isdisjoint(test_loader.dataset.event_ids)
            )

            # The same formal batch also executes the complete physical E4 path.
            physical_cfg = deepcopy(raw_cfg)
            physical_cfg["runoff_mode"] = "water_balance_lstm"
            physical_cfg["routing_mode"] = "kinematic_wave_gnn"
            self.assertEqual(
                physical_cfg["solver"]["integration_scheme"], "backward_euler"
            )
            self.assertEqual(physical_cfg["solver"]["implicit_iterations"], 8)
            physical = HybridFloodModel(physical_cfg, train_loader.dataset.num_stations)
            physical_batch = batches[0]
            physical_output = physical(physical_batch)
            self.assertTrue(torch.isfinite(physical_output["q"]).all())
            diagnostics = physical_output["diagnostics"]
            self.assertIn("node_channel_storage", diagnostics)
            self.assertIn("explicit_equivalent_substeps", diagnostics)
            self.assertIn("implicit_relative_residual", diagnostics)
            self.assertLessEqual(
                diagnostics["implicit_relative_residual"].max().item(),
                physical_cfg["solver"]["implicit_residual_tolerance"],
            )
            self.assertLess(
                diagnostics["routing_mass_balance_residual"].abs().max().item(),
                0.1,
            )


if __name__ == "__main__":
    unittest.main()
