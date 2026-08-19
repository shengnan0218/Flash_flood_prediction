from pathlib import Path

import pytest
import torch

from scripts.v10_training import (
    _load_yaml,
    _validate_v10_config,
    extract_v10_transferable_state_dict,
)


ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return _load_yaml(ROOT / "configs" / "hunan_e4_v10.yaml")


def test_formal_v10_config_is_q_only_without_z_head_or_z_loss() -> None:
    cfg = _cfg()
    _validate_v10_config(cfg)
    assert cfg["data"]["target_variable"] == "Q"
    assert cfg["loss"]["mode"] == "q_only"
    assert "z_head" not in cfg
    forbidden = {
        "water_level_weight",
        "z_level_weight",
        "z_slope_weight",
        "z_target_mode",
        "delta_z_scale_mode",
        "delta_z_scale_floor_m",
        "qz_consistency_weight",
    }
    assert not (forbidden & set(cfg["loss"]))
    assert cfg["validation_selection"] == {"mode": "val_loss"}
    assert cfg["training"]["epochs"] == 100
    assert cfg["training"]["early_stopping"] is False


def test_v10_stage_contract_forbids_backward_z_search() -> None:
    cfg = _cfg()
    cfg["stage_output"]["allow_backward_z_search"] = True
    with pytest.raises(ValueError, match="stage_output"):
        _validate_v10_config(cfg)


def test_v10_rejects_reintroduction_of_future_z_loss() -> None:
    cfg = _cfg()
    cfg["loss"]["z_level_weight"] = 1.0
    with pytest.raises(ValueError, match="Z任务项"):
        _validate_v10_config(cfg)


def test_v10_transfer_excludes_hunan_station_specific_state() -> None:
    checkpoint = {
        "model": {
            "runoff.weight": torch.tensor([1.0]),
            "q_history_mean": torch.tensor([1.0]),
            "z_history_scale": torch.tensor([1.0]),
            "rating.slope": torch.tensor([0.1]),
            "rating.intercept": torch.tensor([100.0]),
            "observation_encoder.station_embedding.weight": torch.ones(2, 3),
        }
    }
    state = extract_v10_transferable_state_dict(checkpoint)
    assert set(state) == {"runoff.weight"}
