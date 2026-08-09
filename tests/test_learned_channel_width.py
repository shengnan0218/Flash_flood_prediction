from __future__ import annotations

import unittest

import torch

from models.routing import EdgeParameterNetwork, KinematicWaveGNN


class TestLearnedChannelWidth(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260807)
        self.bounds = {"width": [2.0, 80.0], "manning_n": [0.015, 0.12]}
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

    def test_edge_parameter_network_is_bounded_and_differentiable(self) -> None:
        network = EdgeParameterNetwork(
            input_dim=6,
            hidden_dim=8,
            width_bounds=(2.0, 80.0),
            n_bounds=(0.015, 0.12),
        )
        attributes = torch.randn(4, 6)

        width, manning = network(attributes)
        self.assertTrue(torch.all((width > 2.0) & (width < 80.0)))
        self.assertTrue(torch.all((manning > 0.015) & (manning < 0.12)))

        (width.sum() + manning.sum()).backward()
        output_gradient = network.net[-1].weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertTrue(torch.isfinite(output_gradient).all())
        self.assertGreater(output_gradient.abs().sum().item(), 0.0)

    def test_routing_learns_width_from_two_edge_attributes(self) -> None:
        router = KinematicWaveGNN(
            node_static_dim=2,
            edge_static_dim=2,
            hidden_dim=8,
            bounds=self.bounds,
            solver=self.solver,
        )
        node_static = torch.tensor(
            [[20.0, 0.1], [35.0, 0.2], [50.0, 0.3]], dtype=torch.float32
        )
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        edge_static = torch.tensor(
            [[5000.0, 0.001], [8000.0, 0.002]], dtype=torch.float32
        )
        q_lateral = torch.full((2, 3, 3), 1.0, dtype=torch.float32)

        routed, diagnostics = router(
            q_lateral, node_static, edge_index, edge_static
        )
        width = diagnostics["learned_effective_width"]
        self.assertEqual(tuple(width.shape), (2,))
        self.assertTrue(torch.all((width > 2.0) & (width < 80.0)))
        self.assertNotIn("observed_channel_width", diagnostics)

        routed.sum().backward()
        output_gradient = router.edge_net.net[-1].weight.grad
        self.assertIsNotNone(output_gradient)
        self.assertTrue(torch.isfinite(output_gradient).all())
        self.assertGreater(output_gradient[0].abs().sum().item(), 0.0)

    def test_routing_rejects_a_width_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "恰好两个边静态特征"):
            KinematicWaveGNN(
                node_static_dim=2,
                edge_static_dim=3,
                hidden_dim=8,
                bounds=self.bounds,
                solver=self.solver,
            )


if __name__ == "__main__":
    unittest.main()
