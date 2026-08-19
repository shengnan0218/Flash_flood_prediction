"""Read-only V10 views over the frozen V8 hydrologic-graph dataset."""
from __future__ import annotations

import pandas as pd

from datasets.hydrologic_graph_v8 import HydrologicGraphV8Dataset


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
            self.q_supervised_sample_count = int(
                pd.to_numeric(
                    self.samples.get("Q_TARGET_VALID_COUNT", pd.Series(index=self.samples.index, data=0)),
                    errors="coerce",
                ).fillna(0).gt(0).sum()
            )
            return
        if "Q_TARGET_VALID_COUNT" not in self.samples.columns:
            raise ValueError(
                "v10 Q-only TRAIN/VALIDATION要求冻结sample_index含Q_TARGET_VALID_COUNT"
            )
        count = pd.to_numeric(
            self.samples["Q_TARGET_VALID_COUNT"], errors="raise"
        ).astype("int64")
        if (count < 0).any():
            raise ValueError("Q_TARGET_VALID_COUNT不能为负")
        keep = count.gt(0)
        self.samples = self.samples.loc[keep].reset_index(drop=True)
        if self.samples.empty:
            raise ValueError(f"{self.split}没有任何Q监督窗口，不能用于v10 Q-only学习/选择")
        self.q_supervised_sample_count = len(self.samples)
        self.graph_ids = tuple(sorted(self.samples["GRAPH_ID"].unique().tolist()))
        self.event_ids = tuple(sorted(self.samples["EVENT_ID"].unique().tolist()))
        self.train_sampling_mode = "v10_q_supervised_full_pass_same_graph_batches"

    @property
    def q_filter_removed_count(self) -> int:
        return self.frozen_sample_count_before_q_filter - len(self.samples)
