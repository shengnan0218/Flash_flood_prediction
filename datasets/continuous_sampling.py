"""Sampling policy adapter for continuous-format Hunan datasets.

The tensor/storage contract stays ``continuous-hourly-dual-target-v1`` for both
full-record and event-domain datasets.  The TRAIN sampling policy, however,
depends on the declared sampling domain:

- ordinary continuous records keep the existing hydrologic response weighting;
- ``hydrologic_events_v1`` uses graph/event/window balancing only, because the
  dataset builder has already restricted admissible windows to hydrologic
  response/recession periods.
"""
from __future__ import annotations

from typing import Mapping

import torch

from .hunan import (
    HunanContinuousDataset as _BaseHunanContinuousDataset,
    event_graph_balancing_weights,
)


class HunanContinuousDataset(_BaseHunanContinuousDataset):
    """Continuous-format dataset with sampling-domain-aware TRAIN weighting."""

    @property
    def train_sampling_mode(self) -> str:
        domain = str(self._continuous_schema.get("sampling_domain", "")).strip()
        return "event_balanced" if domain == "hydrologic_events_v1" else "response_weighted"

    def hydrologic_sampling_weights(
        self,
        *,
        q_scales: Mapping[str, float],
        delta_z_scales: Mapping[str, float],
        response_strength: float,
        response_cap: float,
        minimum_weight: float,
        maximum_weight: float,
    ) -> torch.Tensor:
        """Return TRAIN weights appropriate for the dataset sampling domain.

        Event-domain datasets must not receive a second hydrologic-response
        emphasis after event admission.  Instead, every graph gets equal total
        mass, every event within a graph gets equal total mass, and extra
        sliding windows only divide that event's fixed mass.
        """

        if self.train_sampling_mode != "event_balanced":
            return super().hydrologic_sampling_weights(
                q_scales=q_scales,
                delta_z_scales=delta_z_scales,
                response_strength=response_strength,
                response_cap=response_cap,
                minimum_weight=minimum_weight,
                maximum_weight=maximum_weight,
            )

        if self.split != "TRAIN":
            raise ValueError("event-balanced sampling只能用于TRAIN")
        graph_ids = [sample.graph_id for sample in self._samples]
        event_ids = [sample.event_id for sample in self._samples]
        if any(not event_id.strip() for event_id in event_ids):
            raise ValueError(
                "sampling_domain=hydrologic_events_v1要求每个TRAIN sample都有非空EVENT_ID"
            )
        return torch.tensor(
            event_graph_balancing_weights(graph_ids, event_ids),
            dtype=torch.float32,
        )
