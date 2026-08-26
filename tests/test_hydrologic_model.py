from pathlib import Path

import pytest
import torch

from data.hydrologic_schema import HydrologicGraphBatch
from losses.hydrologic_loss import HydrologicLoss
from models.hydrologic_model import HydrologicModel, PureRunoffLSTM
from scripts.training import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def config(runoff: str, routing: str) -> dict:
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    cfg.update(runoff_mode=runoff, routing_mode=routing, hidden_dim=8)
    cfg["output_head"]["hidden_dim"] = 8
    cfg["solver"]["implicit_iterations"] = 16
    cfg["solver"]["implicit_residual_tolerance"] = 1.0e-4
    cfg["_runtime"] = {
        "station_ids": ["S1"],
        "normalization": {
            "log_rain_mean": 0.0,
            "log_rain_scale": 1.0,
            "node_static_mean": [0.0] * 10,
            "node_static_scale": [1.0] * 10,
            "edge_static_mean": [1000.0, 0.01],
            "edge_static_scale": [500.0, 0.005],
            "q_target_mean": [0.0],
            "q_target_scale": [2.0],
        },
        "high_flow_quantiles": {
            "stations": {"S1": {"available": True, "q80_m3s": 2.0, "q99_m3s": 4.0}}
        },
        "rating_curves": {
            "stations": {
                "S1": {
                    "available": True,
                    "slope_m_per_m3s": 0.5,
                    "intercept_m": 100.0,
                }
            }
        },
    }
    return cfg


def batch() -> HydrologicGraphBatch:
    q_history = torch.linspace(0.5, 2.0, 24).view(1, 24, 1)
    return HydrologicGraphBatch(
        history_rain=torch.full((1, 72, 3, 1), 0.02),
        future_rain=torch.full((1, 6, 3, 1), 0.03),
        node_static=torch.zeros(3, 10),
        incremental_area_km2=torch.tensor([1.0, 2.0, 3.0]),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_static=torch.tensor([[1000.0, 0.01], [1200.0, 0.008]]),
        obs_node_index=torch.tensor([2]),
        obs_station_index=torch.tensor([0]),
        q_history=q_history,
        z_history=100.0 + 0.5 * q_history,
        q_mask=torch.ones(1, 24, 1, dtype=torch.bool),
        z_mask=torch.ones(1, 24, 1, dtype=torch.bool),
        q_target=torch.linspace(2.0, 4.5, 6).view(1, 6, 1),
        z_target=torch.linspace(0.01, 0.06, 6).view(1, 6, 1),
        q_target_mask=torch.ones(1, 6, 1, dtype=torch.bool),
        z_target_mask=torch.ones(1, 6, 1, dtype=torch.bool),
        obs_station_ids=("S1",),
        sample_id=("sample",), event_id=("event",), graph_id=("G",),
        forecast_time=("2020-01-01 00:00:00",), sample_weight=torch.ones(1),
    )


@pytest.mark.parametrize(
    ("runoff", "routing"),
    [
        ("pure_lstm", "pure_gnn"),
        ("water_balance_lstm", "pure_gnn"),
        ("pure_lstm", "kinematic_wave_gnn"),
        ("water_balance_lstm", "kinematic_wave_gnn"),
    ],
)
def test_four_ablation_models_forward_and_backward(runoff: str, routing: str) -> None:
    model = HydrologicModel(config(runoff, routing))
    sample = batch()
    output = model(sample)
    assert output["q"].shape == (1, 6, 1)
    assert torch.isfinite(output["q"]).all() and (output["q"] >= 0).all()
    loss = HydrologicLoss(model.cfg).combine(
        HydrologicLoss(model.cfg).batch_statistics(output, sample)
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_no_state_correction_and_q0_only_anchors_output() -> None:
    model = HydrologicModel(config("water_balance_lstm", "kinematic_wave_gnn"))
    forbidden = ("state_correct", "observation_encoder", "storage_correction", "upstream_analysis")
    names = [name.lower() for name, _ in model.named_modules()] + [name.lower() for name in model.state_dict()]
    assert not any(token in name for name in names for token in forbidden)
    output = model(batch())
    torch.testing.assert_close(output["q"], torch.relu(output["diagnostics"]["q_residual_base_m3s"]))


def test_future_stage_truth_is_never_read() -> None:
    model = HydrologicModel(config("pure_lstm", "pure_gnn")).eval()
    sample = batch()
    before = model(sample)
    sample.z_target.fill_(9999.0)
    sample.z_target_mask.logical_not_()
    after = model(sample)
    torch.testing.assert_close(after["q"], before["q"], rtol=0, atol=0)
    torch.testing.assert_close(after["z_delta"], before["z_delta"], rtol=0, atol=0)


def test_pure_runoff_uses_incremental_area_conversion() -> None:
    torch.manual_seed(31)
    runoff = PureRunoffLSTM(input_dim=3, hidden_dim=4)
    features = torch.zeros(1, 2, 2, 3)
    q, diagnostics = runoff(
        features, torch.tensor([1.0, 2.0]), seconds=3600.0
    )
    torch.testing.assert_close(q[:, :, 1], 2.0 * q[:, :, 0])
    assert diagnostics["unit_runoff_mm"].shape == (1, 2, 2)


def test_latest_available_q_and_trends_feed_output_head() -> None:
    model = HydrologicModel(config("pure_lstm", "pure_gnn")).eval()
    sample = batch()
    sample.q_mask[:, -1] = False
    sample.q_history[:, -1] = 0.0
    output = model(sample)
    torch.testing.assert_close(
        output["diagnostics"]["q_origin_observation_age_hours"],
        torch.ones(1, 1),
    )
    torch.testing.assert_close(
        output["q0_analysis"], sample.q_history[:, -2]
    )
    assert output["diagnostics"]["q_delta_1h_available"].all()
    assert output["diagnostics"]["q_delta_3h_available"].all()
