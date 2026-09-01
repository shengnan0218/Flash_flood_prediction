from pathlib import Path

import pytest
import torch

from data.hydrologic_schema import HydrologicGraphBatch
from losses.hydrologic_loss import HydrologicLoss
from models.hydrologic_model import HydrologicModel, PureRunoffLSTM
from models.routing.muskingum import MuskingumGraphRouter
from models.runoff import MassConservingRunoffLSTM
from scripts.training import load_yaml


ROOT = Path(__file__).resolve().parents[1]


def config(runoff: str, routing: str) -> dict:
    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    cfg.update(runoff_mode=runoff, routing_mode=routing, hidden_dim=8)
    cfg["output_head"]["hidden_dim"] = 8
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
                    "q_min_m3s": 0.2,
                    "q_max_m3s": 20.0,
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
        ("pure_lstm", "muskingum_gnn"),
        ("water_balance_lstm", "muskingum_gnn"),
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


def test_q0_is_only_a_gate_anchor_without_additive_state_or_output_correction() -> None:
    model = HydrologicModel(config("water_balance_lstm", "muskingum_gnn"))
    forbidden = ("state_correct", "observation_encoder", "storage_correction", "upstream_analysis")
    names = [name.lower() for name, _ in model.named_modules()] + [name.lower() for name in model.state_dict()]
    assert not any(token in name for name in names for token in forbidden)
    output = model(batch())
    diagnostics = output["diagnostics"]
    expected = torch.relu(
        output["q0_analysis"].unsqueeze(1)
        + diagnostics["q_route_gate"] * diagnostics["q_route_delta_m3s"]
    )
    torch.testing.assert_close(output["q"], expected)
    assert "q_output_correction_m3s" not in diagnostics


def test_rating_curve_retains_train_domain_for_reporting_only() -> None:
    model = HydrologicModel(config("pure_lstm", "pure_gnn"))
    torch.testing.assert_close(model.rating.q_min_m3s, torch.tensor([0.2]))
    torch.testing.assert_close(model.rating.q_max_m3s, torch.tensor([20.0]))
    assert model.rating.q_min_m3s.requires_grad is False
    assert model.rating.q_max_m3s.requires_grad is False


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
    q, diagnostics = runoff(features, torch.tensor([1.0, 2.0]), seconds=3600.0)
    torch.testing.assert_close(q[:, :, 1], 2.0 * q[:, :, 0])
    assert diagnostics["unit_runoff_mm"].shape == (1, 2, 2)


def test_water_balance_receives_rain_and_store_context_and_closes_mass() -> None:
    runoff = MassConservingRunoffLSTM(2, 4)
    assert runoff.cell.cell.input_size == 5  # two static + rain + two stores
    static = torch.zeros(1, 3, 1, 2)
    rain = torch.tensor([[[[0.0]], [[4.0]], [[2.0]]]])
    rain_feature = torch.log1p(rain)
    q, diagnostics = runoff(static, rain, rain_feature, torch.tensor([1.0]))
    assert torch.isfinite(q).all() and (q >= 0).all()
    assert diagnostics["unobserved_loss_mm"].sum() > 0
    torch.testing.assert_close(
        diagnostics["runoff_water_balance_residual"],
        torch.zeros_like(diagnostics["runoff_water_balance_residual"]),
        atol=1.0e-6,
        rtol=0,
    )
    zero_q, _ = runoff(static, torch.zeros_like(rain), torch.zeros_like(rain), torch.tensor([1.0]))
    torch.testing.assert_close(zero_q, torch.zeros_like(zero_q), atol=1.0e-7, rtol=0)


def test_muskingum_route_has_travel_time_prior_and_mass_closure() -> None:
    cfg = config("water_balance_lstm", "muskingum_gnn")
    router = MuskingumGraphRouter(10, 2, 8, cfg["muskingum_routing"], seconds_per_step=3600.0)
    q_lat = torch.zeros(1, 8, 2)
    q_lat[:, 0, 0] = 5.0
    node_static = torch.zeros(2, 10)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    short_edge = torch.tensor([[1000.0, 0.01]])
    long_edge = torch.tensor([[4000.0, 0.01]])
    routed_short, diagnostics = router(
        q_lat, node_static, edge_index, short_edge, neural_edge_static=torch.zeros_like(short_edge)
    )
    _, diagnostics_long = router(
        q_lat, node_static, edge_index, long_edge, neural_edge_static=torch.zeros_like(long_edge)
    )
    assert routed_short[:, :, 1].sum() > 0
    assert diagnostics_long["routing_travel_time_prior_hours"].item() > diagnostics["routing_travel_time_prior_hours"].item()
    torch.testing.assert_close(
        diagnostics["routing_mass_balance_residual_m3"],
        torch.zeros_like(diagnostics["routing_mass_balance_residual_m3"]),
        atol=1.0e-3,
        rtol=0,
    )


def test_latest_available_q_and_trends_feed_output_head() -> None:
    model = HydrologicModel(config("pure_lstm", "pure_gnn")).eval()
    sample = batch()
    sample.q_mask[:, -1] = False
    sample.q_history[:, -1] = 0.0
    output = model(sample)
    torch.testing.assert_close(output["diagnostics"]["q_origin_observation_age_hours"], torch.ones(1, 1))
    torch.testing.assert_close(output["q0_analysis"], sample.q_history[:, -2])
    assert output["diagnostics"]["q_delta_1h_available"].all()
    assert output["diagnostics"]["q_delta_3h_available"].all()
