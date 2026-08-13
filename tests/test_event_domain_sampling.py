from __future__ import annotations

from types import SimpleNamespace
import unittest

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

    def test_hydrologic_event_domain_is_full_pass_not_weighted(self) -> None:
        dataset = self._dataset(
            "hydrologic_events_v1",
            [("G1", "E1"), ("G1", "E1"), ("G1", "E2"), ("G2", "E3")],
        )
        self.assertEqual(dataset.train_sampling_mode, "event_full_pass")
        with self.assertRaisesRegex(RuntimeError, "禁止weighted/replacement sampling"):
            dataset.hydrologic_sampling_weights(
                q_scales={},
                delta_z_scales={},
                response_strength=1.0,
                response_cap=4.0,
                minimum_weight=0.25,
                maximum_weight=4.0,
            )

    def test_event_domain_rejects_weighting_even_when_event_ids_exist(self) -> None:
        dataset = self._dataset("hydrologic_events_v1", [("G1", "E1")])
        with self.assertRaisesRegex(RuntimeError, "完整遍历一次"):
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
