from types import SimpleNamespace

import pytest

from scripts.v8_training import _load_yaml
from scripts.v9_training import (
    V9_TIME_SEMANTICS,
    _apply_v9_scale_floors,
    _validate_dataset_time_contract,
    _validate_v9_config,
)


def test_v9_config_declares_exact_hour_bin_contract() -> None:
    cfg = _load_yaml("configs/hunan_e4_v9.yaml")
    _validate_v9_config(cfg)
    for key, expected in V9_TIME_SEMANTICS.items():
        assert cfg["temporal"][key] == expected


def test_v9_runtime_enforces_configured_target_scale_floors() -> None:
    cfg = {
        "loss": {
            "q_scale_floor_m3s": 1.0,
            "delta_z_scale_floor_m": 0.01,
        },
        "_runtime": {
            "v8_station_ids": ["A", "B"],
            "v8_normalization": {
                "q_target_scale": [0.2, 2.0],
                "dz_target_scale": [0.005, 0.02],
            },
        },
    }
    _apply_v9_scale_floors(cfg)
    normal = cfg["_runtime"]["v8_normalization"]
    assert normal["q_target_scale"] == [1.0, 2.0]
    assert normal["dz_target_scale"] == [0.01, 0.02]
    audit = cfg["_runtime"]["target_scale_audit"]["stations"]
    assert audit["A"]["q_floor_applied"] is True
    assert audit["A"]["delta_z_floor_applied"] is True
    assert audit["B"]["q_floor_applied"] is False
    assert audit["B"]["delta_z_floor_applied"] is False


def test_dataset_time_contract_rejects_explicit_mismatch() -> None:
    cfg = {"_runtime": {}}
    bad = dict(V9_TIME_SEMANTICS)
    bad["forecast_origin_anchor"] = "instantaneous_label"
    dataset = SimpleNamespace(contract={"timestamp_semantics": bad})
    with pytest.raises(ValueError, match="timestamp semantics"):
        _validate_dataset_time_contract(cfg, dataset)


def test_legacy_frozen_contract_is_accepted_only_with_runtime_interpretation() -> None:
    cfg = {"_runtime": {}}
    dataset = SimpleNamespace(contract={})
    _validate_dataset_time_contract(cfg, dataset)
    semantics = cfg["_runtime"]["timestamp_semantics"]
    assert semantics["dataset_declared"] is False
    for key, expected in V9_TIME_SEMANTICS.items():
        assert semantics[key] == expected
