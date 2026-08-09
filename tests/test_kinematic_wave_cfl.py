from __future__ import annotations

import math
import unittest

import torch

from models.routing import KinematicWaveGNN


class TestKinematicWaveCflEstimate(unittest.TestCase):
    """Regression coverage for the B013 high-flow CFL overestimate."""

    FLOW = 1611.417
    LENGTH_M = 957.5702976
    SLOPE = 0.006191295121

    def setUp(self) -> None:
        torch.manual_seed(20260809)
        self.solver = {
            "dx": 1000.0,
            "cfl": 0.8,
            "maximum_substeps": 64,
            "minimum_slope": 1.0e-6,
            "minimum_length": 10.0,
            "seconds_per_step": 3600.0,
        }
        self.router = KinematicWaveGNN(
            node_static_dim=2,
            edge_static_dim=2,
            hidden_dim=8,
            bounds={"width": [2.0, 80.0], "manning_n": [0.015, 0.12]},
            solver=self.solver,
        )
        self.node_static = torch.zeros(2, 2, dtype=torch.float32)
        self.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        self.edge_static = torch.tensor(
            [[self.LENGTH_M, self.SLOPE]], dtype=torch.float32
        )

    def _alpha(self, diagnostics: dict[str, torch.Tensor]) -> float:
        width = diagnostics["learned_effective_width"][0].detach().item()
        manning = diagnostics["learned_effective_manning_n"][0].detach().item()
        return math.sqrt(self.SLOPE) / (manning * width ** (2.0 / 3.0))

    def _required_substeps(self, alpha: float, area: float) -> int:
        celerity = (5.0 / 3.0) * alpha * area ** (2.0 / 3.0)
        cell_length = min(self.LENGTH_M, self.solver["dx"])
        ratio = (
            celerity
            * self.solver["seconds_per_step"]
            / cell_length
            / self.solver["cfl"]
        )
        return max(1, math.ceil(ratio))

    def test_high_flow_uses_equilibrium_area_not_full_hour_storage(self) -> None:
        q_lateral = torch.tensor(
            [[[self.FLOW, 0.0], [self.FLOW, 0.0]]], dtype=torch.float32
        )

        _routed, diagnostics = self.router(
            q_lateral, self.node_static, self.edge_index, self.edge_static
        )

        alpha = self._alpha(diagnostics)
        equilibrium_area = (self.FLOW / alpha) ** (3.0 / 5.0)
        expected = self._required_substeps(alpha, equilibrium_area)
        actual = diagnostics["substeps"].to(torch.int64).tolist()

        # The old estimate treated Qin * 3600 as storage with no outflow.  At
        # the second hour it also added existing storage and required ~265
        # substeps, so the formal maximum_substeps=64 raised an exception.
        stored_after_first = (
            diagnostics["node_channel_storage"][0, 0, 1].detach().item()
        )
        legacy_area = (
            stored_after_first + self.FLOW * self.solver["seconds_per_step"]
        ) / self.LENGTH_M
        legacy_required = self._required_substeps(alpha, legacy_area)

        self.assertEqual(actual, [expected, expected])
        self.assertEqual(expected, 38)
        self.assertLessEqual(max(actual), self.solver["maximum_substeps"])
        self.assertGreater(legacy_required, self.solver["maximum_substeps"])
        self.assertGreater(legacy_required, 6 * expected)

    def test_storage_branch_is_conservative_nonnegative_and_differentiable(
        self,
    ) -> None:
        q_lateral = torch.tensor(
            [[[self.FLOW, 0.0], [100.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )

        routed, diagnostics = self.router(
            q_lateral, self.node_static, self.edge_index, self.edge_static
        )

        alpha = self._alpha(diagnostics)
        stored_after_first = (
            diagnostics["node_channel_storage"][0, 0, 1].detach().item()
        )
        current_area = stored_after_first / self.LENGTH_M
        second_equilibrium_area = (100.0 / alpha) ** (3.0 / 5.0)
        expected_storage_steps = self._required_substeps(alpha, current_area)
        actual = diagnostics["substeps"].to(torch.int64).tolist()

        self.assertGreater(current_area, second_equilibrium_area)
        self.assertEqual(actual[1], expected_storage_steps)
        self.assertGreater(actual[1], 1)
        residual = diagnostics["routing_mass_balance_residual"]
        volume_scale = self.FLOW * self.solver["seconds_per_step"]
        self.assertLess(residual.abs().max().item(), volume_scale * 1.0e-6)

        storage_after = diagnostics["node_channel_storage"].sum(dim=-1)
        storage_before = torch.cat(
            [torch.zeros_like(storage_after[:, :1]), storage_after[:, :-1]], dim=1
        )
        external_inflow = q_lateral.sum(dim=-1)
        sink_outflow = routed[:, :, 1]
        reconstructed_residual = (
            storage_before
            + external_inflow * self.solver["seconds_per_step"]
            - storage_after
            - sink_outflow * self.solver["seconds_per_step"]
        )
        torch.testing.assert_close(reconstructed_residual, residual)
        relative_residual = reconstructed_residual.abs() / (
            storage_before
            + external_inflow * self.solver["seconds_per_step"]
        ).clamp_min(1.0)
        self.assertLess(relative_residual.max().item(), 2.0e-6)

        for tensor in (
            routed,
            diagnostics["edge_storage"],
            diagnostics["node_channel_storage"],
        ):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertTrue((tensor >= 0).all())

        routed[:, :, 1].sum().backward()
        self.assertIsNotNone(q_lateral.grad)
        self.assertTrue(torch.isfinite(q_lateral.grad).all())
        self.assertGreater(q_lateral.grad[:, :, 0].abs().sum().item(), 0.0)

        output_gradient = self.router.edge_net.net[-1].weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertTrue(torch.isfinite(output_gradient).all())
        self.assertGreater(output_gradient.abs().sum().item(), 0.0)

    def test_maximum_substeps_remains_anomaly_guard(self) -> None:
        extreme_lateral_flow = torch.tensor(
            [[[1_000_000.0, 0.0]]], dtype=torch.float32
        )

        with self.assertRaisesRegex(
            RuntimeError, "超过maximum_substeps=64.*上限仅用于异常保护"
        ):
            self.router(
                extreme_lateral_flow,
                self.node_static,
                self.edge_index,
                self.edge_static,
            )


if __name__ == "__main__":
    unittest.main()
