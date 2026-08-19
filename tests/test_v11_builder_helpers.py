import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "20_build_hydrologic_graph_model_dataset_v11.py"
spec = importlib.util.spec_from_file_location("v11_builder_test_module", BUILDER_PATH)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_v11_rainfall_loader_requires_real_72h_coverage_and_preserves_zero_semantics(tmp_path: Path) -> None:
    rainfall = tmp_path / "rainfall"
    sparse_dir = rainfall / "node_hourly_rain_sparse"
    sparse_dir.mkdir(parents=True)
    start_hour = int(pd.Timestamp("2020-01-01 00:00:00").value // 3_600_000_000_000)
    end_hour = start_hour + 79
    pd.DataFrame(
        [
            {
                "GRAPH_ID": "G1",
                "NODE_ID": node,
                "VALID_START": "2020-01-01 00:00:00",
                "VALID_END": "2020-01-04 07:00:00",
                "ZERO_SEMANTICS": "ABSENT_SPARSE_ROW_WITHIN_VALID_PERIOD_IS_0_MM",
            }
            for node in ("N1", "N2")
        ]
    ).to_csv(rainfall / "node_rainfall_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "GRAPH_ID": "G1",
                "NODE_ID": "N2",
                "START_TIME": "2020-01-02 00:00:00",
                "END_TIME": "2020-01-02 01:00:00",
                "RAIN_MM": "3.5",
            }
        ]
    ).to_csv(
        sparse_dir / "graph_G1_hourly_sparse.csv",
        index=False,
        encoding="utf-8-sig",
    )
    matrix = builder._load_rainfall_matrix(
        "G1",
        ("N1", "N2"),
        tmp_path,
        required_start_hour=start_hour,
        required_end_hour=end_hour,
    )
    assert matrix.shape == (80, 2)
    assert matrix[24, 1] == 3.5
    assert float(matrix.sum()) == 3.5

    try:
        builder._load_rainfall_matrix(
            "G1",
            ("N1", "N2"),
            tmp_path,
            required_start_hour=start_hour - 1,
            required_end_hour=end_hour,
        )
    except ValueError as exc:
        assert "禁止zero-pad" in str(exc)
    else:
        raise AssertionError("V11 must fail when 72h antecedent exceeds valid rainfall coverage")


def test_v11_phase_labels_use_outlet_event_response_and_high_flow_is_train_only_unique_hour() -> None:
    samples = pd.DataFrame(
        {
            "EVENT_ID": ["E1"] * 4,
            "SAMPLE_ID": ["S0", "S1", "S2", "S3"],
            "TENSOR_ROW": [0, 1, 2, 3],
            "Q_TARGET_VALID_COUNT": [6, 6, 6, 6],
        }
    )
    q = np.array(
        [
            [[1, 2, 3, 4, 5, 6]],
            [[4, 6, 8, 10, 12, 14]],
            [[20, 18, 16, 14, 12, 10]],
            [[4, 3, 2, 1, 1, 1]],
        ],
        dtype=np.float32,
    )
    arrays = {
        "q_target": q,
        "q_target_mask": np.ones_like(q, dtype=bool),
        "forecast_time_unix_hour": np.asarray([100, 106, 112, 118], dtype=np.int64),
    }
    labels = builder._event_phase_labels(samples, arrays, outlet_obs=0)
    assert len(labels) == 4
    assert set(labels).issubset(set(builder.EVENT_PHASES))
    assert "PEAK" in labels
    assert "LOW" in labels

    paired = {
        ("OUT", hour): float(hour - 99)
        for hour in range(100, 140)
    }
    payload = builder._high_flow_payload(paired, ("OUT",), {"OUT"})
    assert payload["fit_split"] == "TRAIN"
    assert payload["deduplication_key"] == "STATION_ID+PHYSICAL_TARGET_UNIX_HOUR"
    assert payload["outlet_missing_threshold"] == []
    assert payload["stations"]["OUT"]["q99_m3s"] > payload["stations"]["OUT"]["q80_m3s"]
