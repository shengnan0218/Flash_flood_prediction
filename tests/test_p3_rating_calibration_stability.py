from __future__ import annotations

import unittest

import torch

from models.observation import TrainFittedMonotoneRating


class TestStableCalibratedRating(unittest.TestCase):
    def _rating(self) -> TrainFittedMonotoneRating:
        rating = TrainFittedMonotoneRating(1)
        rating.configure(
            {
                "stations": {
                    "S1": {
                        "usable_calibrated": True,
                        "calibrated_q_knots_m3s": [10.0, 20.0, 40.0],
                        "calibrated_z_knots_m": [100.0, 100.2, 100.3],
                        # Deliberately differs from both local edge slopes
                        # (0.02 and 0.005 m per m3/s).
                        "extrapolation_slope_m_per_m3s": 0.01,
                    }
                }
            },
            {"S1": 0},
        )
        return rating

    def test_outside_train_range_uses_global_train_slope(self) -> None:
        rating = self._rating()
        q = torch.tensor([[[5.0], [50.0]]], requires_grad=True)
        z, available = rating(q)
        self.assertTrue(bool(available[0]))
        self.assertAlmostEqual(float(z[0, 0, 0]), 99.95, places=5)
        self.assertAlmostEqual(float(z[0, 1, 0]), 100.4, places=5)
        z.sum().backward()
        self.assertAlmostEqual(float(q.grad[0, 0, 0]), 0.01, places=6)
        self.assertAlmostEqual(float(q.grad[0, 1, 0]), 0.01, places=6)

    def test_inside_train_range_keeps_data_calibrated_local_slope(self) -> None:
        rating = self._rating()
        q = torch.tensor([[[15.0], [30.0]]], requires_grad=True)
        z, _ = rating(q)
        z.sum().backward()
        self.assertAlmostEqual(float(q.grad[0, 0, 0]), 0.02, places=6)
        self.assertAlmostEqual(float(q.grad[0, 1, 0]), 0.005, places=6)

    def test_inverse_uses_reciprocal_global_slope_outside_range(self) -> None:
        rating = self._rating()
        z = torch.tensor([[99.9]])
        q, available = rating.inverse_from_z(z)
        self.assertTrue(bool(available[0, 0]))
        self.assertAlmostEqual(float(q[0, 0]), 0.0, places=6)

        z_high = torch.tensor([[100.4]])
        q_high, _ = rating.inverse_from_z(z_high)
        self.assertAlmostEqual(float(q_high[0, 0]), 50.0, places=5)


if __name__ == "__main__":
    unittest.main()
