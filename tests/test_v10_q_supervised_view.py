import pandas as pd
import pytest

from datasets.hydrologic_graph_v10 import filter_q_supervised_samples


def test_v10_q_view_keeps_only_positive_q_target_count_and_preserves_order() -> None:
    frame = pd.DataFrame(
        {
            "SAMPLE_ID": ["a", "b", "c", "d"],
            "Q_TARGET_VALID_COUNT": [0, 3, 0, 1],
            "EVENT_ID": ["e0", "e1", "e2", "e3"],
        }
    )
    filtered = filter_q_supervised_samples(frame, split="TRAIN")
    assert filtered["SAMPLE_ID"].tolist() == ["b", "d"]
    assert filtered["Q_TARGET_VALID_COUNT"].tolist() == [3, 1]
    # The source frozen frame is never mutated.
    assert frame["SAMPLE_ID"].tolist() == ["a", "b", "c", "d"]


def test_v10_q_view_rejects_missing_audit_field() -> None:
    with pytest.raises(ValueError, match="Q_TARGET_VALID_COUNT"):
        filter_q_supervised_samples(
            pd.DataFrame({"SAMPLE_ID": ["a"]}), split="VALIDATION"
        )


def test_v10_q_view_rejects_negative_audit_count() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        filter_q_supervised_samples(
            pd.DataFrame({"Q_TARGET_VALID_COUNT": [-1, 2]}), split="TRAIN"
        )


def test_v10_q_view_rejects_empty_supervised_domain() -> None:
    with pytest.raises(ValueError, match="没有任何Q监督窗口"):
        filter_q_supervised_samples(
            pd.DataFrame({"Q_TARGET_VALID_COUNT": [0, 0]}), split="TRAIN"
        )
