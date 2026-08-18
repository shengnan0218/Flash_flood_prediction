from __future__ import annotations

import torch

from models.routing import KinematicWaveGNN as OptimizedKinematicWaveGNN
from models.routing.kinematic_wave import KinematicWaveGNN as ReferenceKinematicWaveGNN


BOUNDS = {
    "width": [2.0, 80.0],
    "manning_n": [0.015, 0.12],
}
SOLVER = {
    "dx": 1000.0,
    "cfl": 0.8,
    "integration_scheme": "backward_euler",
    "implicit_iterations": 8,
    "implicit_residual_tolerance": 1.0e-5,
    "minimum_slope": 1.0e-6,
    "minimum_length": 10.0,
    "seconds_per_step": 3600.0,
}


def _routers() -> tuple[ReferenceKinematicWaveGNN, OptimizedKinematicWaveGNN]:
    torch.manual_seed(20260818)
    reference = ReferenceKinematicWaveGNN(3, 2, 12, BOUNDS, SOLVER)
    optimized = OptimizedKinematicWaveGNN(3, 2, 12, BOUNDS, SOLVER)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    return reference, optimized


def _case():
    # Converging DAG, no divergence: 0->2 <-1, then 2->3->4.
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [2, 2, 3, 4]], dtype=torch.long
    )
    edge_static = torch.tensor(
        [
            [1350.0, 0.0070],
            [920.0, 0.0110],
            [1710.0, 0.0045],
            [640.0, 0.0180],
        ],
        dtype=torch.float32,
    )
    torch.manual_seed(17)
    node_static = torch.randn(5, 3, dtype=torch.float32)
    q_lat = torch.rand(3, 6, 5, dtype=torch.float32) * 35.0
    initial_storage = torch.rand(3, 4, dtype=torch.float32) * 12_000.0
    return q_lat, node_static, edge_index, edge_static, initial_storage


def test_optimized_matches_reference_outputs_and_diagnostics() -> None:
    reference, optimized = _routers()
    q_lat, node_static, edge_index, edge_static, initial_storage = _case()

    ref_q, ref_diag = reference(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=initial_storage,
    )
    opt_q, opt_diag = optimized(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=initial_storage,
    )

    torch.testing.assert_close(opt_q, ref_q, rtol=2.0e-6, atol=2.0e-5)
    for key in (
        "routing_mass_balance_residual",
        "explicit_equivalent_substeps",
        "maximum_celerity_m_per_s",
        "implicit_relative_residual",
        "learned_effective_width",
        "learned_effective_manning_n",
        "initial_edge_discharge_m3s",
        "initial_edge_storage_m3",
        "edge_storage",
        "node_channel_storage",
    ):
        torch.testing.assert_close(
            opt_diag[key], ref_diag[key], rtol=3.0e-6, atol=3.0e-5
        )
    assert int(opt_diag["implicit_iterations"].item()) == int(
        ref_diag["implicit_iterations"].item()
    )
    assert int(opt_diag["explicit_cfl_exceedance_count"].item()) == int(
        ref_diag["explicit_cfl_exceedance_count"].item()
    )


def test_optimized_matches_reference_gradients() -> None:
    reference, optimized = _routers()
    q_lat, node_static, edge_index, edge_static, initial_storage = _case()
    q_ref = q_lat.clone().requires_grad_(True)
    q_opt = q_lat.clone().requires_grad_(True)
    storage_ref = initial_storage.clone().requires_grad_(True)
    storage_opt = initial_storage.clone().requires_grad_(True)

    ref_q, ref_diag = reference(
        q_ref,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=storage_ref,
    )
    opt_q, opt_diag = optimized(
        q_opt,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=storage_opt,
    )
    ref_loss = ref_q.square().mean() + 1.0e-6 * ref_diag["edge_storage"].sum()
    opt_loss = opt_q.square().mean() + 1.0e-6 * opt_diag["edge_storage"].sum()
    ref_loss.backward()
    opt_loss.backward()

    torch.testing.assert_close(q_opt.grad, q_ref.grad, rtol=2.0e-5, atol=2.0e-6)
    torch.testing.assert_close(
        storage_opt.grad, storage_ref.grad, rtol=2.0e-5, atol=2.0e-7
    )
    for ref_parameter, opt_parameter in zip(
        reference.edge_net.parameters(), optimized.edge_net.parameters()
    ):
        assert ref_parameter.grad is not None
        assert opt_parameter.grad is not None
        torch.testing.assert_close(
            opt_parameter.grad,
            ref_parameter.grad,
            rtol=3.0e-5,
            atol=3.0e-6,
        )


def test_topology_cache_keeps_multiple_shuffled_graphs() -> None:
    _, optimized = _routers()
    q_lat, node_static, edge_index, edge_static, initial_storage = _case()
    optimized(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=initial_storage,
    )

    edge_index_second = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_static_second = torch.tensor(
        [[900.0, 0.01], [1200.0, 0.008]], dtype=torch.float32
    )
    q_second = torch.rand(2, 4, 3, dtype=torch.float32) * 10.0
    node_second = torch.randn(3, 3, dtype=torch.float32)
    optimized(q_second, node_second, edge_index_second, edge_static_second)

    # Returning to the first graph must not evict/rebuild its CPU DAG record.
    optimized(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=initial_storage,
    )
    assert len(optimized._topology_cache) == 2


def test_optimized_initial_discharge_path_matches_reference() -> None:
    reference, optimized = _routers()
    q_lat, node_static, edge_index, edge_static, _ = _case()
    initial_q = torch.rand(q_lat.shape[0], edge_index.shape[1]) * 15.0
    ref_q, ref_diag = reference(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_discharge=initial_q,
    )
    opt_q, opt_diag = optimized(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_discharge=initial_q,
    )
    torch.testing.assert_close(opt_q, ref_q, rtol=2.0e-6, atol=2.0e-5)
    torch.testing.assert_close(
        opt_diag["edge_storage"], ref_diag["edge_storage"], rtol=3.0e-6, atol=3.0e-5
    )
