from copy import deepcopy
from pathlib import Path

import torch

from data.v8_schema import HydrologicGraphBatch
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
    z_history = 100.0 + 0.5 * q_history
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


def test_v10_forward_does_not_read_future_z_truth_or_mask() -> None:
    torch.manual_seed(20260819)
    model = HydrologicGraphV10Model(_cfg()).eval()
    original = _batch()
    altered = deepcopy(original)
    altered.z_target = torch.full_like(altered.z_target, 9999.0)
    # Keep the mask structurally valid but deliberately change which future Z
    # points are declared evaluable.  Forward outputs must still be identical.
    altered.z_target_mask[:, ::2] = False

    first = model(original)
    second = model(altered)
    for key in ("q", "q0_analysis", "z_delta", "z_abs", "z_rating_raw_abs"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.equal(first["z_available_mask"], second["z_available_mask"])
