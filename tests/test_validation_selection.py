from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch

from metrics.validation_selection import (
    bounded_efficiency,
    bounded_error_skill,
    validation_selection_score,
)
from metrics.validation_diagnostics import _quantile
from trainers import Trainer


CONFIG = {
    "q_nse_weight": 0.35,
    "q_kge_weight": 0.15,
    "q_peak_weight": 0.20,
    "q_volume_weight": 0.10,
    "z_level_weight": 0.10,
    "z_slope_weight": 0.10,
    "efficiency_clip_min": -1.0,
    "efficiency_clip_max": 1.0,
}


def metrics(value: float) -> dict[str, float]:
    return {
        "q_graph_nse_median": value,
        "q_graph_kge_median": value,
        "q_event_absolute_relative_peak_error_median": 1.0 - value,
        "q_event_absolute_relative_volume_error_median": 1.0 - value,
        "z_station_mae_median": 1.0 - value,
        "z_slope_station_mae_median": 1.0 - value,
    }


class TestValidationSelection(unittest.TestCase):
    def test_component_directions_and_total_are_higher_is_better(self) -> None:
        poor = validation_selection_score(
            metrics(0.0), {"water_level": 1.0}, CONFIG
        )
        good = validation_selection_score(
            metrics(1.0), {"water_level": 1.0}, CONFIG
        )
        self.assertGreater(
            good["validation_selection_score"],
            poor["validation_selection_score"],
        )
        self.assertEqual(good["validation_selection_score"], 1.0)

    def test_extreme_negative_efficiency_is_bounded(self) -> None:
        self.assertEqual(bounded_efficiency(-10_000.0, -1.0, 1.0), 0.0)
        self.assertEqual(bounded_efficiency(10_000.0, -1.0, 1.0), 1.0)

    def test_error_skill_is_bounded_and_monotone(self) -> None:
        self.assertEqual(bounded_error_skill(0.0), 1.0)
        self.assertGreater(bounded_error_skill(1.0), bounded_error_skill(10.0))
        self.assertGreaterEqual(bounded_error_skill(10_000.0), 0.0)

    def test_undefined_component_fails_instead_of_reweighting(self) -> None:
        values = metrics(0.5)
        values["z_slope_station_mae_median"] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "不会静默重分配"):
            validation_selection_score(values, {"water_level": 1.0}, CONFIG)

    def test_undefined_efficiency_keeps_weight_and_gets_explicit_worst_skill(self) -> None:
        values = metrics(0.5)
        values["q_graph_kge_median"] = float("nan")
        result = validation_selection_score(
            values, {"water_level": 1.0}, CONFIG
        )
        self.assertEqual(result["validation_selection_q_kge_skill"], 0.0)
        self.assertEqual(result["validation_selection_q_kge_defined"], 0.0)

    def test_median_is_not_destroyed_by_one_extreme_group(self) -> None:
        self.assertEqual(_quantile([-10_000.0, 0.5, 0.6], 0.5), 0.5)

    def test_best_checkpoint_uses_composite_not_lower_val_loss(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

        class Diagnostics:
            def __init__(self, summary_metrics):
                self.summary_metrics = summary_metrics

            def write(self, *_args, **_kwargs):
                return {}

        class SequenceTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.index = 0

            def train_epoch(self, _loader, epoch=0):
                return {"loss": 1.0, "q_valid_count": 1, "z_valid_count": 1}

            def evaluate(self, _loader, **_kwargs):
                # Epoch 1 has a lower val_loss but much worse scientific score.
                value = (1.0, 0.0)[self.index]
                val_loss = (1.0, 0.1)[self.index]
                self.index += 1
                summary = metrics(value)
                return {
                    "loss": val_loss,
                    **summary,
                    "_validation_diagnostics": Diagnostics(summary),
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {
                "optimizer": {"lr": 0.01, "weight_decay": 0.0},
                "amp": False,
                "loss_weights": {"discharge": 2.0, "water_level": 1.0},
                "loss": {
                    "mode": "multitask",
                    "discharge_weight": 2.0,
                    "water_level_weight": 1.0,
                    "q_point_weight": 1.0,
                    "q_peak_weight": 0.25,
                    "q_volume_weight": 0.25,
                    "z_level_weight": 1.0,
                    "z_slope_weight": 0.25,
                },
                "validation_selection": {"mode": "composite", **CONFIG},
                "_runtime": {
                    "loss_scales": {"discharge": 1.0, "water_level": 1.0}
                },
                "gradient_accumulation_steps": 1,
                "batch_size": 1,
                "debug_mode": False,
                "data": {"mode": "hunan"},
                "training": {
                    "epochs": 2,
                    "patience": 2,
                    "gradient_clip": 1.0,
                    "checkpoint": str(root / "best.pt"),
                    "log_csv": str(root / "train.csv"),
                },
            }
            trainer = SequenceTrainer(Model(), cfg, torch.device("cpu"))
            trainer.fit([SimpleNamespace()], [SimpleNamespace()])
            best = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
            last = torch.load(
                root / "best.last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(last["epoch"], 1)
            self.assertEqual(best["selection_metric"], "validation_selection_score")
            self.assertEqual(best["selection_direction"], "maximize")


if __name__ == "__main__":
    unittest.main()
