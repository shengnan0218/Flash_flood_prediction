from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from datasets.hunan import (
    EDGE_STATIC_MODEL_FEATURES,
    EDGE_STATIC_SOURCE_FEATURES,
    HunanGraphEventDataset,
    NODE_STATIC_FEATURES,
)


class TestHunanFeatureSchemaContract(unittest.TestCase):
    @staticmethod
    def _write_schema(root: Path, edge_features: tuple[str, ...]) -> None:
        metadata = root / "metadata"
        metadata.mkdir(parents=True)
        schema = {
            "dynamic_features": ["FLOW", "WATER_LEVEL"],
            "node_static_features": list(NODE_STATIC_FEATURES),
            "edge_static_features": list(edge_features),
            "physical_features": {
                "incremental_area_km2": {
                    "source": "log_incremental_area",
                    "transform": "log1p",
                    "unit": "km2",
                }
            },
        }
        (metadata / "feature_schema.json").write_text(
            json.dumps(schema), encoding="utf-8"
        )

    @staticmethod
    def _schema_only_dataset(root: Path) -> HunanGraphEventDataset:
        dataset = HunanGraphEventDataset.__new__(HunanGraphEventDataset)
        dataset.root = root
        return dataset

    def test_formal_source_schema_accepts_reach_length_km(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_schema(root, EDGE_STATIC_SOURCE_FEATURES)

            dynamic = self._schema_only_dataset(root)._load_feature_schema()

            self.assertEqual(dynamic, ("FLOW", "WATER_LEVEL"))

    def test_model_internal_reach_length_m_is_rejected_as_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_schema(root, EDGE_STATIC_MODEL_FEATURES)

            with self.assertRaisesRegex(ValueError, "reach_length_km"):
                self._schema_only_dataset(root)._load_feature_schema()


if __name__ == "__main__":
    unittest.main()
