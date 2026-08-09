from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from datasets.hunan import (
    EDGE_STATIC_MODEL_FEATURES,
    EDGE_STATIC_SOURCE_FEATURES,
    HunanGraphEventDataset,
)
from tests.test_hunan_integration import build_formal_fixture


def _remove_channel_width(root: Path) -> None:
    edge_path = root / "graph" / "edge_static_attributes.csv"
    with edge_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = [
        "GRAPH_ID",
        "FROM_STATION",
        "TO_STATION",
        *EDGE_STATIC_SOURCE_FEATURES,
    ]
    with edge_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    schema_path = root / "metadata" / "feature_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["edge_static_features"] = list(EDGE_STATIC_SOURCE_FEATURES)
    schema_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class TestHunanTwoDimensionalEdgeStatic(unittest.TestCase):
    def test_formal_dataset_does_not_require_channel_width(self) -> None:
        self.assertEqual(
            EDGE_STATIC_SOURCE_FEATURES,
            ("reach_length_km", "reach_slope_m_per_m"),
        )
        self.assertEqual(
            EDGE_STATIC_MODEL_FEATURES,
            ("reach_length_m", "reach_slope_m_per_m"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "_model_dataset"
            build_formal_fixture(root)
            _remove_channel_width(root)

            dataset = HunanGraphEventDataset(
                root,
                "TRAIN",
                history_hours=2,
                forecast_hours=2,
            )

            self.assertEqual(dataset.edge_static_dim, 2)
            edge_static = dataset[0].edge_static
            self.assertEqual(tuple(edge_static.shape), (1, 2))
            torch.testing.assert_close(
                edge_static[0], torch.tensor([5000.0, 0.001])
            )


if __name__ == "__main__":
    unittest.main()
