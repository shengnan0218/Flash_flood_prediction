from pathlib import Path

import pytest
import torch

from data.v8_schema import HydrologicGraphBatch
from losses.hydrologic_graph_v9_loss import HydrologicGraphV9Loss
from models.hydrologic_graph_v9 import (
    ExplicitStateDeltaZV9Head,
    HydrologicGraphV9Model,
    ObservationStateCorrectorV9,
)
from models.routing import KinematicWaveGNN
from models.runoff.water_balance_v9 import (
    ContinuousTimeWaterBalanceLSTM,
    continuous_release_fraction,
)
from scripts.v8_training import _load_yaml
from scripts.v9_training import (
    V9_TIME_SEMANTICS,
    extract_v9_transferable_state_dict,
)


ROOT = Path(__file__).resolve().parents[1]


def _runtime_cfg(filename: str) -> dict:
    cfg = _load_yaml(ROOT / "configs" / filename)
    cfg["hidden_dim"] = 8
    cfg["solver"]["implicit_iterations"] = 16
    cfg["solver"]["implicit_residual_tolerance"] = 1.0e-4
    cfg["_runtime"] = {
        "v8_station_count": 1,
        "v8_station_ids": ["S1"],
        "v8_normalization": {
            "rain_mean": 0.0,
            "rain_scale": 1.0,
            "node_static_mean": [0.0] * 10,
            "node_static_scale": [1.0] * 10,
            "q_history_mean": [0.0],
            "q_history_scale": [1.0],
            "z_history_mean": [100.0],
            "z_history_scale": [1.0],
            "q_target_mean": [0.0],
            "q_target_scale": [1.0],
            "dz_target_scale": [0.1],
        },
    }
    return cfg


def _synthetic_batch() -> HydrologicGraphBatch:
    batch = 1
    history = 24
    horizon = 6
    nodes = 2
    obs = 1
    history_rain = torch.full((batch, history, nodes, 1), 0.10)
    future_rain = torch.full((batch, horizon, nodes, 1), 0.05)
    node_static = torch.zeros(nodes, 10)
    area = torch.tensor([5.0, 10.0])
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_static = torch.tensor([[1000.0, 0.01]])
    obs_node_index = torch.tensor([1], dtype=torch.long)
    obs_station_index = torch.tensor([0], dtype=torch.long)
    q_history = torch.linspace(0.5, 1.0, history).view(1, history, 1)
    z_history = torch.linspace(99.9, 100.1, history).view(1, history, 1)
    q_target = torch.ones(batch, horizon, obs)
    z_target = torch.linspace(0.01, 0.06, horizon).view(1, horizon, 1)
    q_mask = torch.ones_like(q_history, dtype=torch.bool)
    z_mask = torch.ones_like(z_history, dtype=torch.bool)
    q_target_mask = torch.ones_like(q_target, dtype=torch.bool)
    z_target_mask = torch.ones_like(z_target, dtype=torch.bool)
    return HydrologicGraphBatch(
        history_rain=history_rain,
        future_rain=future_rain,
        node_static=node_static,
        incremental_area_km2=area,
        edge_index=edge_index,
        edge_static=edge_static,
        obs_node_index=obs_node_index,
        obs_station_index=obs_station_index,
        q_history=q_history,
        z_history=z_history,
        q_mask=q_mask,
        z_mask=z_mask,
        q_target=q_target,
        z_target=z_target,
        q_target_mask=q_target_mask,
        z_target_mask=z_target_mask,
        obs_station_ids=("S1",),
        sample_id=("sample",),
        event_id=("event",),
        graph_id=("G",),
        forecast_time=("2020-01-01 00:00:00",),
        sample_weight=torch.ones(batch),
    )


def test_continuous_release_rate_is_resolution_invariant() -> None:
    rate = torch.tensor([0.2], dtype=torch.float64)
    hourly = continuous_release_fraction(rate, 3600.0)
    minute = continuous_release_fraction(rate, 60.0)
    hourly_retention = 1.0 - hourly
    minute_retention_over_hour = (1.0 - minute).pow(60)
    assert torch.allclose(
        hourly_retention,
        minute_retention_over_hour,
        atol=1e-12,
        rtol=1e-12,
    )


def test_water_balance_split_run_is_exact_continuation() -> None:
    torch.manual_seed(3)
    runoff = ContinuousTimeWaterBalanceLSTM(input_dim=3, hidden_dim=4)
    features = torch.randn(1, 5, 2, 3)
    rain = torch.rand(1, 5, 2, 1)
    area = torch.tensor([2.0, 3.0])
    q_full, d_full = runoff(features, rain, area, seconds=3600.0)
    q_a, d_a = runoff(features[:, :3], rain[:, :3], area, seconds=3600.0)
    state = (
        d_a["final_h"],
        d_a["final_c"],
        d_a["final_storage_fast_mm"],
        d_a["final_storage_slow_mm"],
    )
    q_b, d_b = runoff(
        features[:, 3:],
        rain[:, 3:],
        area,
        seconds=3600.0,
        initial_state=state,
    )
    assert torch.allclose(q_full, torch.cat([q_a, q_b], dim=1), atol=1e-6)
    assert torch.allclose(d_full["final_h"], d_b["final_h"], atol=1e-6)
    assert torch.allclose(
        d_full["final_storage_fast_mm"], d_b["final_storage_fast_mm"], atol=1e-6
    )


def _routing() -> KinematicWaveGNN:
    return KinematicWaveGNN(
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


def test_kinematic_wave_accepts_exact_edge_storage_state() -> None:
    routing = _routing()
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


def test_kinematic_wave_split_run_matches_single_run() -> None:
    torch.manual_seed(4)
    routing = _routing()
    q_lat = torch.rand(1, 5, 2) * 0.05
    node_static = torch.zeros(2, 1)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_static = torch.tensor([[1000.0, 0.01]])
    full, full_diag = routing(q_lat, node_static, edge_index, edge_static)
    first, first_diag = routing(
        q_lat[:, :3], node_static, edge_index, edge_static
    )
    second, second_diag = routing(
        q_lat[:, 3:],
        node_static,
        edge_index,
        edge_static,
        initial_edge_storage=first_diag["edge_storage"],
    )
    assert torch.allclose(full, torch.cat([first, second], dim=1), atol=1e-6)
    assert torch.allclose(
        full_diag["edge_storage"], second_diag["edge_storage"], atol=1e-6
    )


def test_state_correction_is_exact_noop_without_observations() -> None:
    corrector = ObservationStateCorrectorV9(
        node_static_dim=2,
        hidden_dim=4,
        hidden_residual_scale=0.25,
        storage_log_scale=0.35,
    )
    with torch.no_grad():
        corrector.h_head.bias.fill_(1.0)
        corrector.c_head.bias.fill_(1.0)
        corrector.storage_head.bias.fill_(1.0)
        corrector.edge_storage_head.bias.fill_(1.0)
    state = {
        "h": torch.randn(1, 2, 4),
        "c": torch.randn(1, 2, 4),
        "storage_fast_mm": torch.rand(1, 2),
        "storage_slow_mm": torch.rand(1, 2),
    }
    corrected, diag = corrector(
        state=state,
        node_observation_context=torch.randn(1, 2, 4),
        node_observation_available=torch.zeros(1, 2, 1, dtype=torch.bool),
        node_q0_residual_norm=torch.zeros(1, 2, 1),
        node_q0_residual_available=torch.zeros(1, 2, 1, dtype=torch.bool),
        node_static_norm=torch.zeros(2, 2),
    )
    for key in state:
        assert torch.equal(corrected[key], state[key])
    assert torch.count_nonzero(diag["edge_node_log_factor"]) == 0


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
    delta, increment = head(
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
    assert torch.allclose(delta[:, 0], increment[:, 0])
    assert torch.allclose(delta[:, 1], increment[:, 0] + increment[:, 1])
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
    assert cfg["state_correction"]["mode"] == "v8_history_residual_after_warmup"
    assert cfg["state_correction"]["use_qz_history"] is True
    assert cfg["loss"]["qz_consistency_weight"] == 0.0
    assert cfg["training"]["early_stopping"] is False
    assert cfg["temporal"]["forcing_step_seconds"] == 3600
    for key, expected in V9_TIME_SEMANTICS.items():
        assert cfg["temporal"][key] == expected


@pytest.mark.parametrize(
    "filename",
    ["hunan_e1_v9.yaml", "hunan_e2_v9.yaml", "hunan_e3_v9.yaml", "hunan_e4_v9.yaml"],
)
def test_v9_all_four_models_full_forward_loss_backward(filename: str) -> None:
    torch.manual_seed(7)
    cfg = _runtime_cfg(filename)
    model = HydrologicGraphV9Model(cfg)
    batch = _synthetic_batch()
    output = model(batch)
    assert output["q"].shape == batch.q_target.shape
    assert output["z"].shape == batch.z_target.shape
    assert torch.isfinite(output["q"]).all()
    assert torch.isfinite(output["z"]).all()
    if cfg["runoff_mode"] == "water_balance_lstm":
        assert model.runoff.cell.cell.input_size == 11
    if cfg["routing_mode"] == "kinematic_wave_gnn":
        diagnostics = output["diagnostics"]
        assert torch.allclose(
            diagnostics["warmup_edge_storage_t0_m3"],
            diagnostics["forecast_initial_edge_storage_m3"],
            atol=0.0,
            rtol=0.0,
        )
        assert "explicit_equivalent_substeps" in diagnostics
    loss_engine = HydrologicGraphV9Loss(cfg)
    statistics = loss_engine.batch_statistics(output, batch)
    loss = loss_engine.combine(statistics)
    assert torch.isfinite(loss)
    loss.backward()
    trainable_grads = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert trainable_grads
    assert all(torch.isfinite(grad).all() for grad in trainable_grads)


def test_transfer_state_excludes_hunan_station_specific_parameters() -> None:
    checkpoint = {
        "model": {
            "runoff.cell.weight": torch.ones(1),
            "q_history_mean": torch.ones(2),
            "observation_encoder.station_embedding.weight": torch.ones(2, 3),
            "z_head.station_embedding.weight": torch.ones(2, 3),
            "state_corrector.h_head.weight": torch.ones(1),
        }
    }
    state = extract_v9_transferable_state_dict(checkpoint)
    assert "runoff.cell.weight" in state
    assert "state_corrector.h_head.weight" in state
    assert "q_history_mean" not in state
    assert "observation_encoder.station_embedding.weight" not in state
    assert "z_head.station_embedding.weight" not in state
