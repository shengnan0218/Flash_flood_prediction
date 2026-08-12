from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from datasets.continuous_sampling import HunanContinuousDataset


class TestEventDomainSampling(unittest.TestCase):
    def _dataset(self, domain: str, samples: list[tuple[str, str]]) -> HunanContinuousDataset:
        dataset = object.__new__(HunanContinuousDataset)
        dataset._continuous_schema = {"sampling_domain": domain}
        dataset.split = "TRAIN"
        dataset._samples = [
            SimpleNamespace(graph_id=graph_id, event_id=event_id)
            for graph_id, event_id in samples
        ]
        return dataset

    def test_hydrologic_event_domain_uses_graph_event_window_balance(self) -> None:
        dataset = self._dataset(
            "hydrologic_events_v1",
            [("G1", "E1"), ("G1", "E1"), ("G1", "E2"), ("G2", "E3")],
        )
        self.assertEqual(dataset.train_sampling_mode, "event_balanced")
        weights = dataset.hydrologic_sampling_weights(
            q_scales={},
            delta_z_scales={},
            response_strength=1.0,
            response_cap=4.0,
            minimum_weight=0.25,
            maximum_weight=4.0,
        )
        torch.testing.assert_close(
            weights,
            torch.tensor([0.5, 0.5, 1.0, 2.0], dtype=torch.float32),
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), 2.0)
        self.assertAlmostEqual(float(weights[3:].sum()), 2.0)
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))

    def test_event_domain_requires_event_id_for_every_train_sample(self) -> None:
        dataset = self._dataset("hydrologic_events_v1", [("G1", "")])
        with self.assertRaisesRegex(ValueError, "EVENT_ID"):
            dataset.hydrologic_sampling_weights(
                q_scales={},
                delta_z_scales={},
                response_strength=1.0,
                response_cap=4.0,
                minimum_weight=0.25,
                maximum_weight=4.0,
            )

    def test_full_record_continuous_schema_keeps_response_weighting_mode(self) -> None:
        dataset = self._dataset("", [("G1", "")])
        self.assertEqual(dataset.train_sampling_mode, "response_weighted")


if __name__ == "__main__":
    unittest.main()
