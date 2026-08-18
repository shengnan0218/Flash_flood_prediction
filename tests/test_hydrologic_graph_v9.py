from pathlib import Path

import pytest
import torch

from models.hydrologic_graph_v9 import ExplicitStateDeltaZV9Head
from models.routing import KinematicWaveGNN
from models.runoff.water_balance_v9 import continuous_release_fraction
from scripts.v8_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_continuous_release_rate_is_resolution_invariant() -> None:
    rate = torch.tensor([0.2], dtype=torch.float64)
    hourly = continuous_release_fraction(rate, 3600.0)
    minute = continuous_release_fraction(rate, 60.0)
    hourly_retention = 1.0 - hourly
    minute_retention_over_hour = (1.0 - minute).pow(60)
    assert torch.allclose(hourly_retention, minute_retention_over_hour, atol=1e-12, rtol=1e-12)


def test_kinematic_wave_accepts_exact_edge_storage_state() -> None:
    routing = KinematicWaveGNN(
        node_static_dim=1,
        edge_static_dim=2,
        hidden_dim=4,
        bounds={"width": [2.0, 20.0], "manning_n": [0.02, 0.08]},
        solver={
            "dx": 1000.0,
            "cfl": 0.8,
            "integration_scheme": "backward_euler",
            "implicit_iterations": 16,
            "implicit_residual_tolerance": 1.0e-4,
            "minimum_slope": 1.0e-6,
            "minimum_length": 10.0,
            "seconds_per_step": 3600.0,
        },
    )
    q_lat = torch.zeros(1, 1, 2)
    node_static = torch.zeros(2, 1)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_static = torch.tensor([[1000.0, 0.01]])
    initial_storage = torch.tensor([[1000.0]])
    routed, diagnostics = routing(
        q_lat,
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=initial_storage,
    )
    assert routed.shape == (1, 1, 2)
    assert torch.equal(diagnostics["initial_edge_storage_m3"], initial_storage)
    assert diagnostics["initial_edge_discharge_m3s"].item() >= 0


def test_v9_z_head_detaches_hydraulic_features() -> None:
    head = ExplicitStateDeltaZV9Head(hidden_dim=4, horizon=2, stations=1)
    node_context = torch.randn(1, 1, 4, requires_grad=True)
    observation_context = torch.randn(1, 1, 4, requires_grad=True)
    q0_model = torch.randn(1, 1, requires_grad=True)
    q0_observed = torch.randn(1, 1, requires_grad=True)
    q_future = torch.randn(1, 2, 1, requires_grad=True)
    q_delta = torch.randn(1, 2, 1, requires_grad=True)
    channel0 = torch.randn(1, 1, requires_grad=True)
    channel_future = torch.randn(1, 2, 1, requires_grad=True)
    channel_delta = torch.randn(1, 2, 1, requires_grad=True)
    delta, _ = head(
        node_context=node_context,
        observation_context=observation_context,
        obs_node_index=torch.tensor([0]),
        obs_station_index=torch.tensor([0]),
        z_state_features=torch.zeros(1, 1, 8),
        q0_model_norm=q0_model,
        q0_observed_norm=q0_observed,
        q0_observed_available=torch.ones(1, 1, dtype=torch.bool),
        q_future_norm=q_future,
        q_delta_norm=q_delta,
        channel0_log=channel0,
        channel_future_log=channel_future,
        channel_delta_log=channel_delta,
        channel_available=torch.tensor(True),
        dz_scale=torch.ones(1, 1, 1),
    )
    delta.sum().backward()
    assert node_context.grad is not None
    assert observation_context.grad is not None
    for tensor in (
        q0_model,
        q0_observed,
        q_future,
        q_delta,
        channel0,
        channel_future,
        channel_delta,
    ):
        assert tensor.grad is None


@pytest.mark.parametrize(
    ("filename", "runoff_mode", "routing_mode"),
    [
        ("hunan_e1_v9.yaml", "pure_lstm", "pure_gnn"),
        ("hunan_e2_v9.yaml", "water_balance_lstm", "pure_gnn"),
        ("hunan_e3_v9.yaml", "pure_lstm", "kinematic_wave_gnn"),
        ("hunan_e4_v9.yaml", "water_balance_lstm", "kinematic_wave_gnn"),
    ],
)
def test_v9_four_experiment_yaml_matrix(
    filename: str, runoff_mode: str, routing_mode: str
) -> None:
    cfg = _load_yaml(ROOT / "configs" / filename)
    assert cfg["model_version"] == "v9"
    assert cfg["runoff_mode"] == runoff_mode
    assert cfg["routing_mode"] == routing_mode
    assert cfg["warmup"] == {"enabled": True, "initial_state": "static_prior"}
    assert cfg["loss"]["qz_consistency_weight"] == 0.0
    assert cfg["training"]["early_stopping"] is False
    assert cfg["temporal"]["forcing_step_seconds"] == 3600
