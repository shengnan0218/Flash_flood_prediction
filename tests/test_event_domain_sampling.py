from __future__ import annotations

from types import SimpleNamespace
import unittest

from datasets.continuous_sampling import HunanContinuousDataset
from datasets.hunan import GraphGroupedBatchSampler


class _GroupedIndexDataset:
    def __init__(self, graph_ids: list[str]) -> None:
        self.graph_ids = graph_ids

    def __len__(self) -> int:
        return len(self.graph_ids)

    def graph_id_for_index(self, index: int) -> str:
        return self.graph_ids[index]


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

    def test_shuffled_group_sampler_is_one_full_pass_without_replacement(self) -> None:
        dataset = _GroupedIndexDataset(["G1", "G1", "G1", "G2", "G2", "G3"])
        sampler = GraphGroupedBatchSampler(
            dataset, batch_size=2, shuffle=True, drop_last=False, seed=17
        )
        sampler.set_epoch(3)
        batches = list(iter(sampler))
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(dataset))))
        self.assertEqual(len(flattened), len(set(flattened)))
        for batch in batches:
            self.assertEqual(
                len({dataset.graph_id_for_index(index) for index in batch}), 1
            )

    def test_full_record_continuous_schema_keeps_response_weighting_mode(self) -> None:
        dataset = self._dataset("", [("G1", "")])
        self.assertEqual(dataset.train_sampling_mode, "response_weighted")


if __name__ == "__main__":
    unittest.main()
