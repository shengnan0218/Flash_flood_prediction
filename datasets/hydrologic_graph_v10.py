"""Read-only V10 views over the frozen V8 hydrologic-graph dataset."""
from __future__ import annotations

import pandas as pd

from datasets.hydrologic_graph_v8 import HydrologicGraphV8Dataset


def filter_q_supervised_samples(frame: pd.DataFrame, *, split: str) -> pd.DataFrame:
    """Return an order-preserving read-only sample-frame view with Q supervision."""
    if "Q_TARGET_VALID_COUNT" not in frame.columns:
        raise ValueError(
            "v10 Q-only TRAIN/VALIDATION要求冻结sample_index含Q_TARGET_VALID_COUNT"
        )
    count = pd.to_numeric(frame["Q_TARGET_VALID_COUNT"], errors="raise").astype("int64")
    if (count < 0).any():
        raise ValueError("Q_TARGET_VALID_COUNT不能为负")
    result = frame.loc[count.gt(0)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError(f"{split}没有任何Q监督窗口，不能用于v10 Q-only学习/选择")
    return result


class HydrologicGraphV10Dataset(HydrologicGraphV8Dataset):
    """Reuse frozen V8 tensors, optionally retaining only Q-supervised windows.

    No tensor, event, split, timestamp, graph, or target value is rebuilt.  The
    TRAIN/VALIDATION view removes windows whose frozen sample-index audit says
    Q_TARGET_VALID_COUNT == 0 because V10 has no future-Z learning objective.
    Final evaluation can keep the full frozen split so derived stage remains
    evaluable wherever Z truth exists even if Q truth is absent.
    """

    def __init__(self, *args, require_q_supervision: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.require_q_supervision = bool(require_q_supervision)
        self.frozen_sample_count_before_q_filter = len(self.samples)
        if not self.require_q_supervision:
            raw = self.samples.get("Q_TARGET_VALID_COUNT")
            self.q_supervised_sample_count = (
                int(pd.to_numeric(raw, errors="coerce").fillna(0).gt(0).sum())
                if raw is not None
                else 0
            )
            return
        self.samples = filter_q_supervised_samples(self.samples, split=self.split)
        self.q_supervised_sample_count = len(self.samples)
        self.graph_ids = tuple(sorted(self.samples["GRAPH_ID"].unique().tolist()))
        self.event_ids = tuple(sorted(self.samples["EVENT_ID"].unique().tolist()))
        self.train_sampling_mode = "v10_q_supervised_full_pass_same_graph_batches"

    @property
    def q_filter_removed_count(self) -> int:
        return self.frozen_sample_count_before_q_filter - len(self.samples)
