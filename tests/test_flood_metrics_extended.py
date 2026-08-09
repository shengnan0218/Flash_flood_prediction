from __future__ import annotations

import math
import unittest

import torch

from metrics.flood_metrics import hydrograph_sample_sums


class TestHydrographSampleSumsExtended(unittest.TestCase):
    def test_signed_peak_magnitude_and_timing_conventions(self) -> None:
        # Node 0 overpredicts its peak by 2 and peaks one hour late.
        # Node 1 underpredicts its peak by 2 and peaks two hours early.
        prediction = torch.tensor(
            [[[1.0, 1.0], [3.0, 4.0], [7.0, 2.0], [2.0, 1.0]]]
        )
        observation = torch.tensor(
            [[[1.0, 1.0], [5.0, 2.0], [3.0, 3.0], [2.0, 6.0]]]
        )
        mask = torch.ones_like(observation, dtype=torch.bool)

        sums = hydrograph_sample_sums(prediction, observation, mask)

        self.assertEqual(sums["peak_count"], 2)
        self.assertAlmostEqual(float(sums["peak_absolute_error"]), 4.0)
        self.assertAlmostEqual(float(sums["peak_signed_error"]), 0.0)
        self.assertAlmostEqual(
            float(sums["peak_relative_error"]), 2.0 / 5.0 - 2.0 / 6.0
        )
        self.assertEqual(sums["peak_relative_count"], 2)
        self.assertAlmostEqual(float(sums["peak_timing_absolute_error"]), 3.0)
        self.assertAlmostEqual(float(sums["peak_timing_signed_error"]), -1.0)

    def test_zero_observed_peak_is_excluded_only_from_relative_peak(self) -> None:
        prediction = torch.tensor([[[0.0], [2.0], [1.0]]])
        observation = torch.zeros_like(prediction)
        mask = torch.ones_like(observation, dtype=torch.bool)

        sums = hydrograph_sample_sums(prediction, observation, mask)

        self.assertEqual(sums["peak_count"], 1)
        self.assertAlmostEqual(float(sums["peak_signed_error"]), 2.0)
        self.assertAlmostEqual(float(sums["peak_absolute_error"]), 2.0)
        self.assertEqual(sums["peak_relative_count"], 0)
        self.assertAlmostEqual(float(sums["peak_relative_error"]), 0.0)
        # torch.argmax selects the first occurrence of the tied observed peak.
        self.assertAlmostEqual(float(sums["peak_timing_signed_error"]), 1.0)

    def test_mask_controls_peak_values_and_preserves_original_time_indices(self) -> None:
        prediction = torch.tensor([[[1.0], [1000.0], [3.0], [9.0]]])
        observation = torch.tensor([[[2.0], [float("nan")], [8.0], [4.0]]])
        mask = torch.tensor([[[True], [False], [True], [True]]])

        sums = hydrograph_sample_sums(prediction, observation, mask)

        self.assertAlmostEqual(float(sums["peak_signed_error"]), 1.0)
        self.assertAlmostEqual(float(sums["peak_relative_error"]), 1.0 / 8.0)
        # Prediction peaks at original index 3; observation peaks at index 2.
        self.assertAlmostEqual(float(sums["peak_timing_signed_error"]), 1.0)

    def test_empty_mask_returns_zero_sums_and_counts(self) -> None:
        prediction = torch.full((2, 3, 2), float("inf"))
        observation = torch.full((2, 3, 2), float("nan"))
        mask = torch.zeros_like(observation, dtype=torch.bool)

        sums = hydrograph_sample_sums(prediction, observation, mask)

        expected_keys = {
            "peak_absolute_error",
            "peak_signed_error",
            "peak_relative_error",
            "peak_timing_absolute_error",
            "peak_timing_signed_error",
            "peak_count",
            "peak_relative_count",
            "relative_volume_error",
            "volume_count",
        }
        self.assertEqual(set(sums), expected_keys)
        for value in sums.values():
            self.assertEqual(float(value), 0.0)

    def test_nonfinite_value_inside_mask_is_rejected(self) -> None:
        target = torch.tensor([[[1.0], [2.0]]])
        mask = torch.ones_like(target, dtype=torch.bool)

        with self.assertRaisesRegex(FloatingPointError, "预测包含NaN/Inf"):
            hydrograph_sample_sums(
                torch.tensor([[[1.0], [float("inf")]]]), target, mask
            )
        with self.assertRaisesRegex(ValueError, "target包含NaN/Inf"):
            hydrograph_sample_sums(
                target, torch.tensor([[[1.0], [float("nan")]]]), mask
            )

    def test_relative_peak_is_a_ratio_not_a_percentage(self) -> None:
        prediction = torch.tensor([[[1.0], [12.0]]])
        observation = torch.tensor([[[2.0], [10.0]]])
        mask = torch.ones_like(observation, dtype=torch.bool)

        sums = hydrograph_sample_sums(prediction, observation, mask)

        self.assertTrue(math.isclose(float(sums["peak_relative_error"]), 0.2))


if __name__ == "__main__":
    unittest.main()
