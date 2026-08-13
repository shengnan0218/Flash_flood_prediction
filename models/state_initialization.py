"""Learn antecedent hydrological states from pre-forecast observations.

The initializer is deliberately separated from the physical transition models.
It uses only information available at forecast origin, then produces
non-negative catchment storages and non-negative reach discharge states. The
subsequent rainfall-runoff and routing evolution remains governed by the
existing mass-conserving WaterBalanceCell and kinematic-wave solver.
"""
from __future__ import annotations

import torch
from torch import nn


class HydrologicalStateInitializer(nn.Module):
    """Encode observed history plus an explicit forecast-origin Q anchor.

    The recurrent history encoder remains the existing LSTM family.  P3's
    forecast-origin anchor is fused *after* that encoder as two extra physical
    descriptors per node: log(1+Q0) and an availability flag.  Q0 is supplied
    by the hybrid model from the exact t0 discharge observation when available,
    otherwise from a TRAIN-only rating-curve inversion of observed Z0.  If
    neither exists, the availability flag is zero and the learned history state
    remains solely responsible for initialization.

    A dedicated feed-forward stage-history context still preserves all hourly
    inputs in order for the small Z residual path; no GRU or temporal pooling is
    introduced.
    """

    def __init__(
        self,
        temporal_input_dim: int,
        node_static_dim: int,
        edge_static_dim: int,
        hidden_dim: int,
        history_length: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_static_dim = int(node_static_dim)
        self.edge_static_dim = int(edge_static_dim)
        self.temporal_input_dim = int(temporal_input_dim)
        self.history_length = int(history_length)
        if self.history_length <= 0:
            raise ValueError("history_length必须大于0")

        self.history_encoder = nn.LSTM(
            temporal_input_dim,
            hidden_dim,
            batch_first=True,
        )
        # +2 = explicit physical Q0 descriptor + availability flag.
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim + node_static_dim + 2, hidden_dim),
            nn.SiLU(),
        )
        self.h_head = nn.Linear(hidden_dim, hidden_dim)
        self.c_head = nn.Linear(hidden_dim, hidden_dim)
        self.storage_head = nn.Linear(hidden_dim, 2)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_static_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.positive = nn.Softplus()

        stage_input_dim = self.history_length * self.temporal_input_dim
        self.stage_history_head = nn.Sequential(
            nn.Linear(stage_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    @staticmethod
    def _compressed_static(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(value.abs())

    def _stage_history_context(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("stage history sequence必须为[B*N,H,D]")
        if sequence.shape[1] != self.history_length:
            raise ValueError(
                f"stage history长度应为{self.history_length}，实际={sequence.shape[1]}"
            )
        if sequence.shape[2] != self.temporal_input_dim:
            raise ValueError(
                "stage history特征维应为"
                f"{self.temporal_input_dim}，实际={sequence.shape[2]}"
            )
        return self.stage_history_head(sequence.reshape(sequence.shape[0], -1))

    def forward(
        self,
        history_features: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
        q_origin_anchor: torch.Tensor | None = None,
        q_origin_anchor_available: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if history_features.ndim != 4:
            raise ValueError("history_features必须为[B,H,N,D]")
        batch, history, nodes, features = history_features.shape
        if history != self.history_length:
            raise ValueError(
                f"state initializer history应为{self.history_length}小时，实际={history}"
            )
        if features != self.temporal_input_dim:
            raise ValueError(
                f"history_features特征维应为{self.temporal_input_dim}，实际={features}"
            )
        if node_static.ndim != 2 or node_static.shape != (
            nodes,
            self.node_static_dim,
        ):
            raise ValueError(
                "node_static必须为[N,node_static_dim]并与history节点数一致"
            )
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index必须为[2,E]")
        edges = int(edge_index.shape[1])
        if edge_static.ndim != 2 or edge_static.shape != (
            edges,
            self.edge_static_dim,
        ):
            raise ValueError("edge_static必须为[E,edge_static_dim]")

        if q_origin_anchor is None:
            q_origin_anchor = history_features.new_zeros((batch, nodes)).float()
        if q_origin_anchor_available is None:
            q_origin_anchor_available = torch.zeros(
                (batch, nodes), dtype=torch.bool, device=history_features.device
            )
        if q_origin_anchor.shape != (batch, nodes):
            raise ValueError("q_origin_anchor必须为[B,N]")
        if q_origin_anchor_available.shape != (batch, nodes):
            raise ValueError("q_origin_anchor_available必须为[B,N]")
        if not torch.isfinite(q_origin_anchor).all() or (q_origin_anchor < 0).any():
            raise ValueError("q_origin_anchor必须为有限非负流量")

        sequence = (
            history_features.float()
            .permute(0, 2, 1, 3)
            .reshape(batch * nodes, history, -1)
        )
        stage_context = self._stage_history_context(sequence).reshape(
            batch, nodes, self.hidden_dim
        )

        _, (encoded_h, encoded_c) = self.history_encoder(sequence)
        encoded_h = encoded_h[-1].reshape(batch, nodes, self.hidden_dim)
        encoded_c = encoded_c[-1].reshape(batch, nodes, self.hidden_dim)

        static_node = self._compressed_static(node_static)
        static_node = static_node.unsqueeze(0).expand(batch, -1, -1)
        q_descriptor = torch.stack(
            (
                torch.log1p(q_origin_anchor.float()),
                q_origin_anchor_available.to(history_features.dtype),
            ),
            dim=-1,
        )
        context = self.node_fusion(
            torch.cat([encoded_h, static_node, q_descriptor], dim=-1)
        )

        h0 = torch.tanh(self.h_head(context))
        c0 = torch.tanh(self.c_head(context + encoded_c))
        storage = self.positive(self.storage_head(context))
        storage_fast = storage[..., 0]
        storage_slow = storage[..., 1]

        if edges:
            source, destination = edge_index.long()
            static_edge = self._compressed_static(edge_static)
            static_edge = static_edge.unsqueeze(0).expand(batch, -1, -1)
            edge_context = torch.cat(
                [context[:, source], context[:, destination], static_edge],
                dim=-1,
            )
            edge_discharge = self.edge_head(edge_context).squeeze(-1)
        else:
            edge_discharge = history_features.new_zeros((batch, 0)).float()

        return {
            "h0": h0,
            "c0": c0,
            "storage_fast_mm": storage_fast,
            "storage_slow_mm": storage_slow,
            "edge_discharge_m3s": edge_discharge,
            "history_context": stage_context,
            "q_origin_anchor_m3s": q_origin_anchor,
            "q_origin_anchor_available": q_origin_anchor_available,
        }
