from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.v10_rating import fit_train_only_linear_ratings


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    (root / "samples" / "tensors").mkdir(parents=True)
    (root / "graph").mkdir(parents=True)
    pd.DataFrame(
        [
            {"SAMPLE_ID": "T0", "GRAPH_ID": "G1", "SPLIT": "TRAIN", "TENSOR_FILE": "samples/tensors/graph_G1.npz", "TENSOR_ROW": "0"},
            {"SAMPLE_ID": "T1", "GRAPH_ID": "G1", "SPLIT": "TRAIN", "TENSOR_FILE": "samples/tensors/graph_G1.npz", "TENSOR_ROW": "1"},
            {"SAMPLE_ID": "V0", "GRAPH_ID": "G1", "SPLIT": "VALIDATION", "TENSOR_FILE": "samples/tensors/graph_G1.npz", "TENSOR_ROW": "2"},
        ]
    ).to_csv(root / "samples" / "sample_index.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"GRAPH_ID": "G1", "STATION_ID": "S1", "IS_OUTLET_STATION": "1"}]
    ).to_csv(
        root / "graph" / "station_observation_mapping.csv", index=False, encoding="utf-8-sig"
    )

    # TRAIN physical relation: Z = 0.5 Q + 100.
    # T0 targets unix hours 11,12; T1 targets 12,13. Hour 12 is deliberately
    # duplicated by overlapping forecast windows and must count only once.
    q_target = np.array([[[4.0, 6.0]], [[6.0, 8.0]], [[100.0, 120.0]]], dtype=np.float32)
    q_target_mask = np.ones_like(q_target, dtype=bool)
    z_history = np.zeros((3, 1, 24), dtype=np.float32)
    z_history[:, :, -1] = np.array([[101.0], [102.0], [999.0]], dtype=np.float32)
    z_history_mask = np.ones_like(z_history, dtype=bool)
    z_target = np.array([[[1.0, 2.0]], [[1.0, 2.0]], [[50.0, 60.0]]], dtype=np.float32)
    z_target_mask = np.ones_like(z_target, dtype=bool)
    np.savez_compressed(
        root / "samples" / "tensors" / "graph_G1.npz",
        obs_station_id=np.array(["S1"]),
        forecast_time_unix_hour=np.array([10, 11, 20], dtype=np.int64),
        q_target=q_target,
        q_target_mask=q_target_mask,
        z_history=z_history,
        z_history_mask=z_history_mask,
        z_target=z_target,
        z_target_mask=z_target_mask,
    )
    return root


def test_v10_rating_is_train_only_deduplicated_and_does_not_require_q0(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    result = fit_train_only_linear_ratings(
        root,
        ("S1",),
        min_unique_pairs=3,
        require_all_outlet_stations=True,
    )
    station = result["stations"]["S1"]
    assert result["fit_split"] == "TRAIN"
    assert result["candidate_pair_occurrences"] == 4
    assert result["unique_pair_count"] == 3
    assert result["duplicate_value_conflict_count"] == 0
    assert result["outlet_missing_curve"] == []
    assert station["available"] is True
    assert station["unique_train_pair_count"] == 3
    assert station["slope_m_per_m3s"] == pytest.approx(0.5)
    assert station["intercept_m"] == pytest.approx(100.0)
    # The NPZ deliberately contains no q_history/q_history_mask.  Successful fit
    # proves curve calibration is not incorrectly conditioned on forecast Q0.


def test_v10_rating_fails_when_required_outlet_has_no_usable_train_curve(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    with pytest.raises(ValueError, match="所有outlet"):
        fit_train_only_linear_ratings(
            root,
            ("S1",),
            min_unique_pairs=4,
            require_all_outlet_stations=True,
        )


def test_v10_rating_ignores_validation_values_when_fitting(tmp_path: Path) -> None:
    root = _dataset(tmp_path)
    first = fit_train_only_linear_ratings(
        root, ("S1",), min_unique_pairs=3, require_all_outlet_stations=True
    )
    path = root / "samples" / "tensors" / "graph_G1.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["q_target"][2, 0] = np.array([10000.0, 20000.0], dtype=np.float32)
    arrays["z_target"][2, 0] = np.array([-500.0, 800.0], dtype=np.float32)
    np.savez_compressed(path, **arrays)
    second = fit_train_only_linear_ratings(
        root, ("S1",), min_unique_pairs=3, require_all_outlet_stations=True
    )
    for key in ("slope_m_per_m3s", "intercept_m", "train_rmse_m", "train_nse"):
        assert second["stations"]["S1"][key] == pytest.approx(first["stations"]["S1"][key])
    assert second["artifact_sha256"] == first["artifact_sha256"]
