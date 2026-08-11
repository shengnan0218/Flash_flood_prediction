from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from config import load_config


ROOT = Path(__file__).parents[1]


class TestQNormExperimentConfig(unittest.TestCase):
    def test_qnorm_is_a_strict_two_change_experiment(self) -> None:
        baseline = load_config(ROOT / "configs" / "hunan_e4_multitask.yaml")
        expected = deepcopy(baseline)
        expected["loss"]["q_scale_mode"] = "per_graph"
        expected["training"].update(
            {
                "epochs": 100,
                "patience": 100,
                "checkpoint": "outputs/hunan_e4_multitask_qnorm_v1_best.pt",
                "log_csv": "outputs/hunan_e4_multitask_qnorm_v1_train.csv",
            }
        )
        actual = load_config(
            ROOT / "configs" / "hunan_e4_multitask_qnorm_v1.yaml"
        )
        self.assertEqual(actual, expected)
        self.assertEqual(actual["loss"]["q_scale_floor_m3s"], 1.0)
        self.assertEqual(actual["training"]["epochs"], 100)
        self.assertEqual(actual["training"]["patience"], 100)


if __name__ == "__main__":
    unittest.main()
