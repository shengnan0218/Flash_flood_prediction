"""Learn antecedent hydrological states from pre-forecast observations.

The initializer is deliberately separated from the physical transition models.
It uses only the history window available at forecast origin, then produces
non-negative catchment storages and non-negative reach discharge states. The
subsequent rainfall-runoff and routing evolution remains governed by the
existing mass-conserving WaterBalanceCell and kinematic-wave solver.
"""
from __future__ import annotations

import torch
from torch import nn


class HydrologicalStateInitializer(nn.Module):
    """Encode observed history into WaterBalanceLSTM/routing initial states.

    Node history is encoded independently for each graph node with a shared
    LSTM, matching the recurrent family already used by the runoff module.
    Static attributes are fused after the temporal encoder. Fast/slow storages
    are expressed in mm and constrained non-negative with Softplus. For each
    directed edge the initializer predicts a non-negative discharge in m3/s;
    the routing module converts that discharge to a physically consistent
    initial reach volume using its own kinematic-wave relation.

    ``history_context`` is also exposed for the P3 observation path.  It is the
    already-computed forecast-origin representation of the same 24 h history;
    exposing it restores access to antecedent stage information without adding
    another recurrent encoder or changing the Q transition equations.
    """

    def __init__(
        self,
        temporal_input_dim: int,
        node_static_dim: int,
        edge_static_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_static_dim = int(node_static_dim)
        self.edge_static_dim = int(edge_static_dim)

        self.history_encoder = nn.LSTM(
            temporal_input_dim,
            hidden_dim,
            batch_first=True,
        )
        self.node_fusion = nn.Sequential(
            nn.Linear(hidden_dim + node_static_dim, hidden_dim),
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

    @staticmethod
    def _compressed_static(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        return torch.sign(value) * torch.log1p(value.abs())

    def forward(
        self,
        history_features: torch.Tensor,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_features.ndim != 4:
            raise ValueError("history_features必须为[B,H,N,D]")
        batch, history, nodes, _ = history_features.shape
        if history < 1:
            raise ValueError("state initializer至少需要1个历史时步")
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

        sequence = (
            history_features.float()
            .permute(0, 2, 1, 3)
            .reshape(batch * nodes, history, -1)
        )
        _, (encoded_h, encoded_c) = self.history_encoder(sequence)
        encoded_h = encoded_h[-1].reshape(batch, nodes, self.hidden_dim)
        encoded_c = encoded_c[-1].reshape(batch, nodes, self.hidden_dim)

        static_node = self._compressed_static(node_static)
        static_node = static_node.unsqueeze(0).expand(batch, -1, -1)
        context = self.node_fusion(torch.cat([encoded_h, static_node], dim=-1))

        # Initialize the downstream WaterBalanceLSTM recurrent state from the
        # same LSTM family. encoded_c is retained explicitly so the initializer
        # does not discard the history encoder's memory-cell information.
        h0 = torch.tanh(self.h_head(context))
        c_context = context + encoded_c
        c0 = torch.tanh(self.c_head(c_context))
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
            "history_context": context,
        }
