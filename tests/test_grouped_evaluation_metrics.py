from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

import torch

from evaluate import _json_safe
from trainers import Trainer


@dataclass
class GroupedBatch:
    prediction: torch.Tensor
    q_target: torch.Tensor
    z_target: torch.Tensor
    q_target_mask: torch.Tensor
    z_target_mask: torch.Tensor
    event_id: tuple[str, ...]
    graph_id: tuple[str, ...]

    def to(self, device: torch.device) -> "GroupedBatch":
        return GroupedBatch(
            self.prediction.to(device),
            self.q_target.to(device),
            self.z_target.to(device),
            self.q_target_mask.to(device),
            self.z_target_mask.to(device),
            self.event_id,
            self.graph_id,
        )


class IdentityForecast(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, batch: GroupedBatch) -> dict[str, torch.Tensor | dict]:
        prediction = self.scale * batch.prediction
        return {"q": prediction, "z": prediction, "diagnostics": {}}


def config(root: Path) -> dict:
    return {
        "optimizer": {"lr": 0.01, "weight_decay": 0.0},
        "amp": False,
        "loss_weights": {"discharge": 1.0, "water_level": 1.0},
        "gradient_accumulation_steps": 1,
        "training": {
            "gradient_clip": 1.0,
            "checkpoint": str(root / "best.pt"),
            "log_csv": str(root / "train.csv"),
        },
    }


class TestGroupedEvaluationMetrics(unittest.TestCase):
    def test_event_and_graph_window_macros_and_details(self) -> None:
        prediction = torch.tensor(
            [
                [[2.0], [4.0]],
                [[3.0], [0.0]],
                [[19.0], [0.0]],
            ]
        )
        target = torch.tensor(
            [
                [[1.0], [3.0]],
                [[2.0], [0.0]],
                [[10.0], [20.0]],
            ]
        )
        mask = torch.tensor(
            [
                [[True], [True]],
                [[True], [False]],
                [[True], [False]],
            ]
        )
        batch = GroupedBatch(
            prediction,
            target,
            target.clone(),
            mask,
            mask.clone(),
            ("E1", "E1", "E2"),
            ("G1", "G1", "G2"),
        )

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                IdentityForecast(), config(Path(directory)), torch.device("cpu")
            )
            metrics = trainer.evaluate([batch], include_group_details=True)

        # Four valid points have absolute errors 1, 1, 1 and 9.
        self.assertAlmostEqual(float(metrics["q_mae"]), 3.0)
        # E1/G1 MAE is 1 and E2/G2 MAE is 9; macro averaging weights both equally.
        self.assertAlmostEqual(float(metrics["q_event_window_macro_mae"]), 5.0)
        self.assertAlmostEqual(float(metrics["q_graph_window_macro_mae"]), 5.0)
        self.assertEqual(metrics["q_event_window_group_count"], 2)
        self.assertEqual(metrics["q_event_window_macro_mae_defined_count"], 2)

        self.assertAlmostEqual(float(metrics["q_sample_peak_bias"]), 11.0 / 3.0)
        self.assertAlmostEqual(
            float(metrics["q_event_window_macro_sample_peak_bias"]), 5.0
        )
        self.assertAlmostEqual(float(metrics["q_sample_peak_timing_bias_hours"]), 0.0)

        details = metrics["window_group_metrics"]
        self.assertAlmostEqual(float(details["event"]["E1"]["q"]["mae"]), 1.0)
        self.assertAlmostEqual(float(details["event"]["E2"]["q"]["mae"]), 9.0)
        self.assertAlmostEqual(float(details["graph"]["G1"]["z"]["mae"]), 1.0)

        serialised = _json_safe(metrics)
        self.assertIsNone(serialised["window_group_metrics"]["event"]["E2"]["q"]["nse"])
        # The real evaluation entry point uses allow_nan=False, so nested
        # undefined group metrics must already have become JSON null.
        json.dumps(serialised, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
