from __future__ import annotations

import math
import unittest

import torch

from models.routing import KinematicWaveGNN


class TestKinematicWaveImplicitSolver(unittest.TestCase):
    """Regression coverage for stiff short reaches in the formal Hunan data."""

    def setUp(self) -> None:
        torch.manual_seed(20260809)
        self.solver = {
            "dx": 1000.0,
            "cfl": 0.8,
            "integration_scheme": "backward_euler",
            "implicit_iterations": 8,
            "implicit_residual_tolerance": 1.0e-5,
            "minimum_slope": 1.0e-6,
            "minimum_length": 10.0,
            "seconds_per_step": 3600.0,
        }
        self.bounds = {
            "width": [2.0, 80.0],
            "manning_n": [0.015, 0.12],
        }
        self.node_static = torch.zeros(2, 2, dtype=torch.float32)
        self.edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    def _router(self, solver: dict | None = None) -> KinematicWaveGNN:
        return KinematicWaveGNN(
            node_static_dim=2,
            edge_static_dim=2,
            hidden_dim=8,
            bounds=self.bounds,
            solver=self.solver if solver is None else solver,
        )

    def _route_single_reach(
        self,
        *,
        flow: float,
        length: float,
        slope: float,
        solver: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_lateral = torch.tensor([[[flow, 0.0]]], dtype=torch.float32)
        edge_static = torch.tensor([[length, slope]], dtype=torch.float32)
        return self._router(solver)(
            q_lateral, self.node_static, self.edge_index, edge_static
        )

    def _assert_stiff_case_succeeds(
        self, *, flow: float, length: float, slope: float
    ) -> None:
        routed, diagnostics = self._route_single_reach(
            flow=flow, length=length, slope=slope
        )

        # This value describes what the removed explicit method would need.  It
        # is deliberately not a loop count or a failure threshold anymore.
        self.assertGreater(
            diagnostics["explicit_equivalent_substeps"].item(), 256
        )
        self.assertGreater(diagnostics["maximum_celerity_m_per_s"].item(), 0)
        self.assertLessEqual(
            diagnostics["implicit_relative_residual"].item(),
            self.solver["implicit_residual_tolerance"],
        )
        self.assertEqual(diagnostics["implicit_iterations"].item(), 8)
        for tensor in (
            routed,
            diagnostics["edge_storage"],
            diagnostics["node_channel_storage"],
        ):
            self.assertTrue(torch.isfinite(tensor).all())
            self.assertTrue((tensor >= 0).all())

    def test_b008_short_reach_routes_without_cfl_abort(self) -> None:
        # Formal v4 reach 611H4513 -> 611E1460, which produced 291 > 256
        # after the first 17 optimiser updates on the server.
        self._assert_stiff_case_succeeds(
            flow=641.47644,
            length=124.630432,
            slope=0.025416698,
        )

    def test_b002_short_reach_routes_without_cfl_abort(self) -> None:
        # Formal v4 short reach 611H0396 -> 611H0399.
        self._assert_stiff_case_succeeds(
            flow=1700.0,
            length=185.325134,
            slope=0.045068333,
        )

    def test_mass_nonnegative_storage_and_gradients(self) -> None:
        flow = 1611.417
        length = 957.5702976
        slope = 0.006191295121
        q_lateral = torch.tensor(
            [[[flow, 0.0], [100.0, 0.0]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        edge_static = torch.tensor([[length, slope]], dtype=torch.float32)
        router = self._router()

        routed, diagnostics = router(
            q_lateral, self.node_static, self.edge_index, edge_static
        )

        residual = diagnostics["routing_mass_balance_residual"]
        storage_after = diagnostics["node_channel_storage"].sum(dim=-1)
        storage_before = torch.cat(
            [torch.zeros_like(storage_after[:, :1]), storage_after[:, :-1]], dim=1
        )
        reconstructed = (
            storage_before
            + q_lateral.sum(dim=-1) * self.solver["seconds_per_step"]
            - storage_after
            - routed[:, :, 1] * self.solver["seconds_per_step"]
        )
        torch.testing.assert_close(reconstructed, residual)
        relative_residual = reconstructed.abs() / (
            storage_before
            + q_lateral.sum(dim=-1) * self.solver["seconds_per_step"]
        ).clamp_min(1.0)
        self.assertLess(relative_residual.max().item(), 2.0e-6)
        self.assertLessEqual(
            diagnostics["implicit_relative_residual"].max().item(),
            self.solver["implicit_residual_tolerance"],
        )

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

        output_gradient = router.edge_net.net[-1].weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertTrue(torch.isfinite(output_gradient).all())
        self.assertGreater(output_gradient[0].abs().sum().item(), 0.0)
        self.assertGreater(output_gradient[1].abs().sum().item(), 0.0)

    def test_backward_euler_matches_fine_explicit_reference_when_nonstiff(
        self,
    ) -> None:
        solver = dict(self.solver)
        solver["seconds_per_step"] = 60.0
        flow = 10.0
        length = 20_000.0
        slope = 0.001
        routed, diagnostics = self._route_single_reach(
            flow=flow, length=length, slope=slope, solver=solver
        )
        self.assertEqual(
            diagnostics["explicit_equivalent_substeps"].item(), 1.0
        )

        width = diagnostics["learned_effective_width"][0].detach().item()
        manning = diagnostics["learned_effective_manning_n"][0].detach().item()
        alpha = math.sqrt(slope) / (manning * width ** (2.0 / 3.0))
        reference_volume = 0.0
        reference_steps = 20_000
        reference_dt = solver["seconds_per_step"] / reference_steps
        for _ in range(reference_steps):
            capacity = alpha * (reference_volume / length) ** (5.0 / 3.0)
            reference_volume += (flow - capacity) * reference_dt
        reference_mean_outflow = (
            flow * solver["seconds_per_step"] - reference_volume
        ) / solver["seconds_per_step"]

        actual_volume = diagnostics["edge_storage"][0, 0].item()
        actual_mean_outflow = routed[0, 0, 1].item()
        self.assertAlmostEqual(actual_volume, reference_volume, delta=0.02)
        self.assertAlmostEqual(
            actual_mean_outflow, reference_mean_outflow, delta=5.0e-4
        )

    def test_exactly_dry_reach_has_zero_state_and_finite_gradient(self) -> None:
        q_lateral = torch.zeros(
            1, 2, 2, dtype=torch.float32, requires_grad=True
        )
        edge_static = torch.tensor(
            [[124.630432, 0.025416698]], dtype=torch.float32
        )
        router = self._router()

        routed, diagnostics = router(
            q_lateral, self.node_static, self.edge_index, edge_static
        )

        self.assertTrue(torch.equal(routed, q_lateral.detach()))
        self.assertTrue(
            torch.equal(
                diagnostics["edge_storage"],
                torch.zeros_like(diagnostics["edge_storage"]),
            )
        )
        (routed.sum() + diagnostics["edge_storage"].sum()).backward()
        self.assertIsNotNone(q_lateral.grad)
        self.assertTrue(torch.isfinite(q_lateral.grad).all())
        for parameter in router.edge_net.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_nonconvergence_raises_numerical_not_cfl_error(self) -> None:
        solver = dict(self.solver)
        solver["implicit_iterations"] = 1
        solver["implicit_residual_tolerance"] = 1.0e-12

        with self.assertRaisesRegex(RuntimeError, "隐式后向Euler求解未收敛"):
            self._route_single_reach(
                flow=641.47644,
                length=124.630432,
                slope=0.025416698,
                solver=solver,
            )


if __name__ == "__main__":
    unittest.main()
