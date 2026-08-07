"""Station-conditioned monotone discharge-to-water-level observation head."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MonotonicQZObservation(nn.Module):
    """Map routed discharge to water level with positive station-wise weights.

    ``station_index`` maps the nodes of the current graph into a province-wide
    station catalogue.  This lets batches contain different (unpadded) graphs
    while retaining one set of observation parameters per physical station.
    The three inputs -- Q, sqrt(Q), and log(1 + channel storage) -- are all
    monotone, so the learned response is structurally non-decreasing in Q and
    storage.
    """

    def __init__(
        self, stations: int, embedding_dim: int = 8, hidden_dim: int = 16
    ) -> None:
        super().__init__()
        if stations <= 0:
            raise ValueError("stations必须大于0")
        self.num_stations = int(stations)
        self.station_embedding = nn.Embedding(stations, embedding_dim)
        self.raw_w1 = nn.Parameter(torch.full((stations, 3, hidden_dim), -4.0))
        self.raw_w2 = nn.Parameter(torch.full((stations, hidden_dim), -3.0))
        self.bias = nn.Parameter(torch.zeros(stations))
        self.embed_bias = nn.Linear(embedding_dim, 1)
        nn.init.normal_(self.station_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.embed_bias.weight)
        nn.init.zeros_(self.embed_bias.bias)

    def _indices(
        self, nodes: int, station_index: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        if station_index is None:
            if nodes > self.num_stations:
                raise ValueError(
                    f"当前图有{nodes}个节点，但观测头仅配置{self.num_stations}个站点"
                )
            return torch.arange(nodes, device=device)
        if station_index.dtype != torch.long or tuple(station_index.shape) != (nodes,):
            raise ValueError("station_index必须是形状[N]的LongTensor")
        indices = station_index.to(device=device)
        if (indices < 0).any() or (indices >= self.num_stations).any():
            raise ValueError(
                "station_index超出观测头的全局站点范围"
                f"[0,{self.num_stations - 1}]"
            )
        if indices.unique().numel() != nodes:
            raise ValueError("同一河网内station_index必须唯一")
        return indices

    def forward(
        self,
        q: torch.Tensor,
        channel_state: torch.Tensor | None = None,
        station_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim != 3:
            raise ValueError(f"q必须为[B,T,N]，实际形状={tuple(q.shape)}")
        if not torch.isfinite(q).all():
            raise FloatingPointError("观测头输入q包含NaN/Inf")
        batch, steps, nodes = q.shape
        if channel_state is None:
            state = torch.zeros_like(q)
        else:
            if channel_state.shape != q.shape:
                raise ValueError("channel_state必须与q形状一致")
            if not torch.isfinite(channel_state).all():
                raise FloatingPointError("channel_state包含NaN/Inf")
            state = channel_state

        ids = self._indices(nodes, station_index, q.device)
        nonnegative_q = q.clamp_min(0)
        x = torch.stack(
            (
                nonnegative_q,
                torch.sqrt(nonnegative_q + 1.0e-4) - 1.0e-2,
                torch.log1p(state.clamp_min(0)),
            ),
            dim=-1,
        )
        w1 = F.softplus(self.raw_w1[ids])
        w2 = F.softplus(self.raw_w2[ids])
        hidden = F.softplus(torch.einsum("btni,nih->btnh", x, w1))
        # Remove the softplus activation at zero input.  The response is then
        # exactly zero at Q=storage=0 and remains non-negative/monotone.
        response = torch.einsum("btnh,nh->btn", hidden - math.log(2.0), w2)
        station_offset = self.bias[ids] + self.embed_bias(
            self.station_embedding(ids)
        ).squeeze(-1)
        return response + station_offset.view(1, 1, nodes)
