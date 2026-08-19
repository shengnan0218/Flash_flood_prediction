from pathlib import Path

import torch

from data.v8_schema import HydrologicGraphBatch
from losses.hydrologic_graph_v10_loss import HydrologicGraphV10Loss
from models.hydrologic_graph_v10 import HydrologicGraphV10Model
from scripts.v8_training import _load_yaml


ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    cfg = _load_yaml(ROOT / "configs" / "hunan_e4_v10.yaml")
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
            "q_target_scale": [2.0],
            "dz_target_mean": [0.0],
            "dz_target_scale": [0.1],
        },
        "v10_rating_curves": {
            "stations": {
                "S1": {
                    "available": True,
                    "unique_train_pair_count": 100,
                    "slope_m_per_m3s": 0.5,
                    "intercept_m": 100.0,
                    "q_min_m3s": 0.0,
                    "q_max_m3s": 100.0,
                }
            }
        },
    }
    return cfg


def _batch() -> HydrologicGraphBatch:
    history, horizon, nodes = 24, 6, 3
    q_history = torch.linspace(0.5, 2.0, history).view(1, history, 1)
    z_history = (100.0 + 0.5 * q_history).clone()
    return HydrologicGraphBatch(
        history_rain=torch.full((1, history, nodes, 1), 0.02),
        future_rain=torch.full((1, horizon, nodes, 1), 0.03),
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
        q_target=torch.full((1, horizon, 1), 2.5),
        z_target=torch.linspace(0.01, 0.06, horizon).view(1, horizon, 1),
        q_target_mask=torch.ones(1, horizon, 1, dtype=torch.bool),
        z_target_mask=torch.ones(1, horizon, 1, dtype=torch.bool),
        obs_station_ids=("S1",),
        sample_id=("sample",),
        event_id=("event",),
        graph_id=("G",),
        forecast_time=("2020-01-01 00:00:00",),
        sample_weight=torch.ones(1),
    )


def test_v10_has_no_trainable_future_z_path() -> None:
    model = HydrologicGraphV10Model(_cfg())
    state_keys = tuple(model.state_dict())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert not any(key.startswith("z_head.") for key in state_keys)
    assert not any(key.startswith("node_context_projection.") for key in state_keys)
    assert "dz_target_scale" not in state_keys
    assert not any(name.startswith("rating.") for name in parameter_names)
    assert "rating.slope" in state_keys
    assert "rating.intercept" in state_keys


def test_v10_final_history_bin_q0_z0_give_rating_aligned_stage() -> None:
    torch.manual_seed(11)
    model = HydrologicGraphV10Model(_cfg())
    batch = _batch()
    output = model(batch)
    expected_q0 = batch.q_history[:, -1]
    assert torch.equal(output["q0_analysis"], expected_q0)
    expected_dz = 0.5 * (output["q"].detach() - expected_q0.unsqueeze(1))
    expected_abs = batch.z_history[:, -1].unsqueeze(1) + expected_dz
    torch.testing.assert_close(output["z_delta"], expected_dz)
    torch.testing.assert_close(output["z_abs"], expected_abs)
    assert output["z_available_mask"].all()
    assert output["q"].requires_grad
    assert not output["z_delta"].requires_grad
    assert not output["z_abs"].requires_grad
    assert not output["z_rating_raw_abs"].requires_grad


def test_v10_stage_correction_is_invariant_to_rating_intercept() -> None:
    torch.manual_seed(12)
    model = HydrologicGraphV10Model(_cfg())
    batch = _batch()
    before = model(batch)
    with torch.no_grad():
        model.rating.intercept.add_(10.0)
    after = model(batch)
    torch.testing.assert_close(after["q"], before["q"], rtol=0, atol=0)
    torch.testing.assert_close(after["z_delta"], before["z_delta"], rtol=0, atol=0)
    torch.testing.assert_close(after["z_abs"], before["z_abs"], rtol=0, atol=0)
    torch.testing.assert_close(
        after["z_rating_raw_abs"], before["z_rating_raw_abs"] + 10.0
    )


def test_v10_missing_final_history_bin_z_masks_stage_without_backward_search() -> None:
    model = HydrologicGraphV10Model(_cfg())
    batch = _batch()
    batch.z_mask[:, -1] = False
    batch.z_target_mask[:] = False
    output = model(batch)
    assert not output["z_available_mask"].any()
    assert torch.equal(output["z_delta"], torch.zeros_like(output["z_delta"]))
    assert torch.equal(output["z_abs"], torch.zeros_like(output["z_abs"]))


def test_v10_missing_observed_q0_uses_model_origin_but_can_still_output_stage() -> None:
    torch.manual_seed(13)
    model = HydrologicGraphV10Model(_cfg())
    batch = _batch()
    batch.q_mask[:, -1] = False
    output = model(batch)
    assert not output["diagnostics"]["q_origin_observed_available"].any()
    torch.testing.assert_close(
        output["q0_analysis"],
        output["diagnostics"]["q_origin_model_corrected_m3s"],
    )
    expected = 0.5 * (
        output["q"].detach() - output["q0_analysis"].detach().unsqueeze(1)
    )
    torch.testing.assert_close(output["z_delta"], expected)
    assert output["z_available_mask"].all()


def test_v10_loss_is_q_only_and_rating_buffers_receive_no_gradient() -> None:
    torch.manual_seed(14)
    cfg = _cfg()
    model = HydrologicGraphV10Model(cfg)
    batch = _batch()
    output = model(batch)
    engine = HydrologicGraphV10Loss(cfg)
    baseline = engine.batch_statistics(output, batch)
    modified = dict(output)
    modified["z"] = torch.full_like(output["z"], 1.0e9)
    modified["z_delta"] = torch.full_like(output["z_delta"], -1.0e9)
    comparison = engine.batch_statistics(modified, batch)
    for key in engine.coefficients():
        torch.testing.assert_close(baseline[key].numerator, comparison[key].numerator)
        assert baseline[key].denominator == comparison[key].denominator
    loss = engine.combine(baseline)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.rating.slope.grad is None
    assert model.rating.intercept.grad is None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_v10_rejects_negative_observed_q0_instead_of_clamping() -> None:
    model = HydrologicGraphV10Model(_cfg())
    batch = _batch()
    batch.q_history[:, -1] = -1.0
    try:
        model(batch)
    except ValueError as exc:
        assert "负流量" in str(exc)
    else:
        raise AssertionError("v10 must fail fast on negative observed Q0")
