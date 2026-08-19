import pandas as pd

from datasets.hydrologic_graph_v11 import EventBalancedV11BatchSampler


class _DummyDataset:
    split = "TRAIN"
    require_q_supervision = True

    def __init__(self) -> None:
        rows = []
        for event, graph, count in (("E1", "G1", 12), ("E2", "G1", 5), ("E3", "G2", 9)):
            phases = ["LOW", "RISING", "PEAK", "RECESSION"]
            for index in range(count):
                rows.append(
                    {
                        "EVENT_ID": event,
                        "GRAPH_ID": graph,
                        "EVENT_PHASE": phases[index % 4],
                    }
                )
        self.samples = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.samples)

    def graph_id_for_index(self, index: int) -> str:
        return str(self.samples.iloc[index]["GRAPH_ID"])


def _plan(sampler: EventBalancedV11BatchSampler) -> list[int]:
    selected = []
    for batch in sampler:
        graphs = {sampler.dataset.graph_id_for_index(index) for index in batch}
        assert len(graphs) == 1
        selected.extend(batch)
    return selected


def test_v11_sampler_balances_events_without_repeating_origins() -> None:
    dataset = _DummyDataset()
    sampler = EventBalancedV11BatchSampler(
        dataset,
        batch_size=4,
        origins_per_event=8,
        phase_quota=2,
        seed=42,
    )
    selected = _plan(sampler)
    assert len(selected) == 8 + 5 + 8
    assert len(selected) == len(set(selected))
    counts = dataset.samples.iloc[selected].groupby("EVENT_ID").size().to_dict()
    assert counts == {"E1": 8, "E2": 5, "E3": 8}
    audit = sampler.audit()
    assert audit["event_count"] == 3
    assert audit["selected_samples_per_epoch"] == 21
    assert audit["events_with_fewer_than_target_origins"] == 1


def test_v11_sampler_is_epoch_deterministic_but_can_rotate_origins() -> None:
    dataset = _DummyDataset()
    first = EventBalancedV11BatchSampler(dataset, 4, seed=7)
    second = EventBalancedV11BatchSampler(dataset, 4, seed=7)
    assert _plan(first) == _plan(second)
    first.set_epoch(1)
    epoch1 = _plan(first)
    first.set_epoch(1)
    assert epoch1 == _plan(first)
    # E1/E3 have more candidates than the 8-origin budget, so changing epoch
    # should change at least one selected physical forecast origin.
    first.set_epoch(0)
    assert set(epoch1) != set(_plan(first))


def test_v11_sampler_rejects_non_8_equal_phase_design() -> None:
    dataset = _DummyDataset()
    try:
        EventBalancedV11BatchSampler(
            dataset,
            batch_size=4,
            origins_per_event=7,
            phase_quota=2,
        )
    except ValueError as exc:
        assert "phase_quota" in str(exc)
    else:
        raise AssertionError("formal V11 must enforce 2x4=8 phase budget")
