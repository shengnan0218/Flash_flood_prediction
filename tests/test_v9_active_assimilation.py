from pathlib import Path

import torch

from data.v8_schema import HydrologicGraphBatch
from losses.hydrologic_graph_v9_loss import HydrologicGraphV9Loss
from models.hydrologic_graph_v9_assimilated import (
    HydrologicGraphV9Model,
    MassAwareObservationStateCorrectorV9,
)
from scripts.v8_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _runtime_cfg() -> dict:
    cfg = _load_yaml(ROOT / "configs" / "hunan_e4_v9.yaml")
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
    history = 24
    horizon = 6
    nodes = 3
    history_rain = torch.full((1, history, nodes, 1), 0.02)
    future_rain = torch.full((1, horizon, nodes, 1), 0.03)
    q_history = torch.linspace(0.5, 2.0, history).view(1, history, 1)
    z_history = torch.linspace(99.9, 100.1, history).view(1, history, 1)
    q_target = torch.ones(1, horizon, 1)
    z_target = torch.linspace(0.01, 0.06, horizon).view(1, horizon, 1)
    return HydrologicGraphBatch(
        history_rain=history_rain,
        future_rain=future_rain,
        node_static=torch.zeros(nodes, 10),
        incremental_area_km2=torch.tensor([1.0, 2.0, 3.0]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_static=torch.tensor([[1000.0, 0.01], [1200.0, 0.008]]),
        obs_node_index=torch.tensor([2], dtype=torch.long),
        obs_station_index=torch.tensor([0], dtype=torch.long),
        q_history=q_history,
        z_history=z_history,
        q_mask=torch.ones_like(q_history, dtype=torch.bool),
        z_mask=torch.ones_like(z_history, dtype=torch.bool),
        q_target=q_target,
        z_target=z_target,
        q_target_mask=torch.ones_like(q_target, dtype=torch.bool),
        z_target_mask=torch.ones_like(z_target, dtype=torch.bool),
        obs_station_ids=("S1",),
        sample_id=("sample",),
        event_id=("event",),
        graph_id=("G",),
        forecast_time=("2020-01-01 00:00:00",),
        sample_weight=torch.ones(1),
    )


def test_v9_config_uses_observed_future_rainfall() -> None:
    cfg = _load_yaml(ROOT / "configs" / "hunan_e4_v9.yaml")
    assert cfg["data"]["future_rainfall_mode"] == "observed_hindcast"
    assert cfg["state_correction"]["propagate_upstream"] is True
    assert cfg["state_correction"]["additive_storage_from_q_residual"] is True


def test_additive_state_analysis_can_restore_zero_storage() -> None:
    corrector = MassAwareObservationStateCorrectorV9(
        node_static_dim=2,
        hidden_dim=4,
        hidden_residual_scale=0.25,
        storage_log_scale=0.35,
        max_additive_storage_hours=6.0,
    )
    state = {
        "h": torch.zeros(1, 1, 4),
        "c": torch.zeros(1, 1, 4),
        "storage_fast_mm": torch.zeros(1, 1),
        "storage_slow_mm": torch.zeros(1, 1),
    }
    corrected, diag = corrector(
        state=state,
        node_observation_context=torch.zeros(1, 1, 4),
        node_observation_available=torch.ones(1, 1, 1, dtype=torch.bool),
        node_q0_residual_norm=torch.ones(1, 1, 1),
        node_q0_residual_m3s=torch.ones(1, 1, 1),
        node_q0_residual_available=torch.ones(1, 1, 1, dtype=torch.bool),
        node_static_norm=torch.zeros(1, 2),
        incremental_area_km2=torch.tensor([1.0]),
    )
    assert corrected["storage_fast_mm"].item() > 0
    assert corrected["storage_slow_mm"].item() > 0
    assert diag["storage_additive_total_mm"].item() > 0


def test_outlet_q_residual_is_propagated_upstream_by_area() -> None:
    model = HydrologicGraphV9Model(_runtime_cfg())
    context = torch.ones(1, 1, model.hidden_dim)
    fields = model._upstream_analysis_fields(
        observation_context=context,
        observation_available=torch.ones(1, 1, dtype=torch.bool),
        q0_residual_norm=torch.tensor([[3.0]]),
        q0_residual_m3s=torch.tensor([[6.0]]),
        q0_available=torch.ones(1, 1, dtype=torch.bool),
        obs_node_index=torch.tensor([2]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        incremental_area_km2=torch.tensor([1.0, 2.0, 3.0]),
        nodes=3,
    )
    node_context, node_available, _, q_residual, q_available = fields
    assert node_context.shape == (1, 3, model.hidden_dim)
    assert node_available.all()
    assert q_available.all()
    assert torch.allclose(
        q_residual[0, :, 0], torch.tensor([1.0, 2.0, 3.0]), atol=1.0e-6
    )


def test_active_v9_forward_uses_observed_origin_for_z_features() -> None:
    torch.manual_seed(11)
    cfg = _runtime_cfg()
    model = HydrologicGraphV9Model(cfg)
    batch = _synthetic_batch()
    output = model(batch)
    diagnostics = output["diagnostics"]
    assert torch.allclose(
        diagnostics["q_origin_analysis_m3s"], batch.q_history[:, -1], atol=0, rtol=0
    )
    assert torch.allclose(
        diagnostics["warmup_edge_storage_t0_m3"],
        diagnostics["forecast_initial_edge_storage_m3"],
        atol=0,
        rtol=0,
    )
    assert torch.isfinite(output["q"]).all()
    assert torch.isfinite(output["z"]).all()
    loss_engine = HydrologicGraphV9Loss(cfg)
    loss = loss_engine.combine(loss_engine.batch_statistics(output, batch))
    assert torch.isfinite(loss)
    loss.backward()
