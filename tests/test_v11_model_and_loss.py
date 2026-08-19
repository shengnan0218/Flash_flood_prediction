from pathlib import Path

import torch

from data.v8_schema import HydrologicGraphBatch
from losses.hydrologic_graph_v11_loss import HydrologicGraphV11Loss
from models.hydrologic_graph_v11 import HydrologicGraphV11Model
from scripts.v8_training import _load_yaml
from trainers.v11_trainer import V11Trainer

ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> dict:
    cfg = _load_yaml(ROOT / "configs" / "hunan_e4_v11.yaml")
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
        "v11_high_flow_quantiles": {
            "stations": {
                "S1": {
                    "available": True,
                    "q80_m3s": 2.0,
                    "q99_m3s": 4.0,
                }
            }
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
        "v11_rating_curves": {
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
    rain_history, obs_history, horizon, nodes = 72, 24, 6, 3
    q_history = torch.linspace(0.5, 2.0, obs_history).view(1, obs_history, 1)
    z_history = (100.0 + 0.5 * q_history).clone()
    q_target = torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0, 4.5]).view(1, horizon, 1)
    return HydrologicGraphBatch(
        history_rain=torch.full((1, rain_history, nodes, 1), 0.02),
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
        q_target=q_target,
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


def test_v11_uses_72h_rain_but_only_24h_qz_and_has_no_z_head() -> None:
    torch.manual_seed(21)
    model = HydrologicGraphV11Model(_cfg())
    batch = _batch()
    output = model(batch)
    assert output["q"].shape == (1, 6, 1)
    assert batch.history_rain.shape[1] == 72
    assert batch.q_history.shape[1] == 24
    assert float(output["diagnostics"]["rain_history_hours"].item()) == 72.0
    assert float(output["diagnostics"]["observation_history_hours"].item()) == 24.0
    state_keys = tuple(model.state_dict())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert not any(key.startswith("z_head.") for key in state_keys)
    assert not any(key.startswith("node_context_projection.") for key in state_keys)
    assert "dz_target_scale" not in state_keys
    assert not any(name.startswith("rating.") for name in parameter_names)
    assert not output["z_delta"].requires_grad


def test_v11_stage_remains_q_derived_and_future_z_cannot_change_forecast() -> None:
    torch.manual_seed(22)
    model = HydrologicGraphV11Model(_cfg())
    batch = _batch()
    before = model(batch)
    batch.z_target[:] = 9999.0
    batch.z_target_mask[:] = ~batch.z_target_mask
    after = model(batch)
    torch.testing.assert_close(after["q"], before["q"], rtol=0, atol=0)
    torch.testing.assert_close(after["z_delta"], before["z_delta"], rtol=0, atol=0)
    torch.testing.assert_close(after["z_abs"], before["z_abs"], rtol=0, atol=0)
    expected = 0.5 * (
        before["q"].detach() - before["q0_analysis"].detach().unsqueeze(1)
    )
    torch.testing.assert_close(before["z_delta"], expected)


def test_v11_loss_has_high_flow_term_and_no_window_peak_term() -> None:
    cfg = _cfg()
    engine = HydrologicGraphV11Loss(cfg)
    batch = _batch()
    prediction = batch.q_target.clone().requires_grad_(True)
    prediction.data[:, 4:] += 1.0
    stats = engine.batch_statistics({"q": prediction}, batch)
    assert set(stats) == {"q_point", "q_high_flow", "q_volume"}
    assert "q_peak" not in engine.coefficients()
    assert stats["q_high_flow"].denominator == 6
    assert stats["q_high_flow"].numerator > 0
    loss = engine.combine(stats)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None


def test_v11_trainer_runs_q_only_high_flow_objective_and_generalization_metrics() -> None:
    torch.manual_seed(23)
    cfg = _cfg()
    trainer = V11Trainer(HydrologicGraphV11Model(cfg), cfg, torch.device("cpu"))
    batch = _batch()
    train = trainer.train_epoch([batch], epoch=0)
    validation = trainer.evaluate([batch])
    for result in (train, validation):
        assert result["q_valid_count"] == 6
        assert "q_high_flow_loss" in result
        assert "q_peak_loss" not in result
        assert "z_loss" not in result
        assert torch.isfinite(torch.tensor(float(result["loss"])))
    for key in (
        "q0_observed_valid_count",
        "q0_subset_model_nse",
        "q0_persistence_nse",
        "q_skill_over_persistence",
        "delta_q_rmse",
        "delta_q_nse",
    ):
        assert key in validation
        assert key not in train
