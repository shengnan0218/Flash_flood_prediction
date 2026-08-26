"""Tensor contract for hydrologic computational graphs with sparse Q/Z observations."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

from data.schema import topological_levels

Tensor = torch.Tensor


@dataclass
class HydrologicGraphBatch:
    """One same-graph mini-batch under the sparse-observation contract.

    Computational-node fields use Nnode. Observation fields use Nobs and are
    linked only through ``obs_node_index``; Nobs is never broadcast to Nnode.
    """

    history_rain: Tensor
    future_rain: Tensor
    node_static: Tensor
    incremental_area_km2: Tensor
    edge_index: Tensor
    edge_static: Tensor
    obs_node_index: Tensor
    obs_station_index: Tensor
    q_history: Tensor
    z_history: Tensor
    q_mask: Tensor
    z_mask: Tensor
    q_target: Tensor
    z_target: Tensor
    q_target_mask: Tensor
    z_target_mask: Tensor
    obs_station_ids: tuple[str, ...]
    sample_id: str | tuple[str, ...] | None = None
    event_id: str | tuple[str, ...] | None = None
    graph_id: str | tuple[str, ...] | None = None
    forecast_time: str | tuple[str, ...] | None = None
    sample_weight: Tensor | None = None

    def to(self, device: torch.device) -> "HydrologicGraphBatch":
        return HydrologicGraphBatch(
            **{
                item.name: (
                    getattr(self, item.name).to(device)
                    if isinstance(getattr(self, item.name), Tensor)
                    else getattr(self, item.name)
                )
                for item in fields(self)
            }
        )


def _shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name}形状应为{expected}，实际={tuple(value.shape)}")


def _bool(name: str, value: Tensor) -> None:
    if value.dtype != torch.bool:
        raise ValueError(f"{name}必须为BoolTensor，实际={value.dtype}")


def validate_hydrologic_base_batch(
    batch: HydrologicGraphBatch,
    *,
    history_hours: int = 24,
    forecast_hours: int = 6,
    node_static_dim: int = 10,
    edge_static_dim: int = 2,
) -> None:
    if batch.history_rain.ndim != 4:
        raise ValueError("history_rain应为[B,H,Nnode,1]")
    b, h, n, rain_dim = batch.history_rain.shape
    if h != history_hours or rain_dim != 1 or b <= 0 or n <= 0:
        raise ValueError(
            f"history_rain契约错误，期望H={history_hours}, feature=1，"
            f"实际={tuple(batch.history_rain.shape)}"
        )
    _shape("future_rain", batch.future_rain, (b, forecast_hours, n, 1))
    if batch.node_static.ndim != 2 or tuple(batch.node_static.shape) != (
        n,
        node_static_dim,
    ):
        raise ValueError(
            f"node_static应为[Nnode,{node_static_dim}]，"
            f"实际={tuple(batch.node_static.shape)}"
        )
    _shape("incremental_area_km2", batch.incremental_area_km2, (n,))
    if batch.edge_index.dtype != torch.long or batch.edge_index.ndim != 2:
        raise ValueError("edge_index必须是LongTensor [2,Nedge]")
    if batch.edge_index.shape[0] != 2:
        raise ValueError("edge_index第一维必须为2")
    e = int(batch.edge_index.shape[1])
    if tuple(batch.edge_static.shape) != (e, edge_static_dim):
        raise ValueError(
            f"edge_static应为[Nedge,{edge_static_dim}]，"
            f"实际={tuple(batch.edge_static.shape)}"
        )
    if batch.edge_index.numel() and (
        batch.edge_index.min() < 0 or batch.edge_index.max() >= n
    ):
        raise ValueError("edge_index包含超出computational-node范围的索引")

    if batch.obs_node_index.dtype != torch.long or batch.obs_node_index.ndim != 1:
        raise ValueError("obs_node_index必须为LongTensor [Nobs]")
    o = int(batch.obs_node_index.numel())
    if o <= 0:
        raise ValueError("每个正式graph至少需要一个观测站")
    if (batch.obs_node_index < 0).any() or (batch.obs_node_index >= n).any():
        raise ValueError("obs_node_index超出computational-node范围")
    if batch.obs_station_index.dtype != torch.long:
        raise ValueError("obs_station_index必须为LongTensor")
    _shape("obs_station_index", batch.obs_station_index, (o,))
    if (batch.obs_station_index < 0).any():
        raise ValueError("obs_station_index必须非负")
    if batch.obs_station_index.unique().numel() != o:
        raise ValueError("同一graph内观测站全局索引必须唯一")
    if len(batch.obs_station_ids) != o:
        raise ValueError("obs_station_ids数量必须等于Nobs")

    for name in ("q_history", "z_history", "q_mask", "z_mask"):
        _shape(name, getattr(batch, name), (b, history_hours, o))
    for name in ("q_target", "z_target", "q_target_mask", "z_target_mask"):
        _shape(name, getattr(batch, name), (b, forecast_hours, o))
    for name in ("q_mask", "z_mask", "q_target_mask", "z_target_mask"):
        _bool(name, getattr(batch, name))

    finite = {
        "history_rain": batch.history_rain,
        "future_rain": batch.future_rain,
        "node_static": batch.node_static,
        "incremental_area_km2": batch.incremental_area_km2,
        "edge_static": batch.edge_static,
        "q_history": batch.q_history,
        "z_history": batch.z_history,
        "q_target": batch.q_target,
        "z_target": batch.z_target,
    }
    for name, value in finite.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}含NaN/Inf；loader必须先用mask安全占位")
    if (batch.history_rain < 0).any() or (batch.future_rain < 0).any():
        raise ValueError("rain forcing必须非负")
    if (batch.incremental_area_km2 <= 0).any():
        raise ValueError("incremental_area_km2必须为有限正数")
    if e:
        if (batch.edge_static[:, 0] <= 0).any():
            raise ValueError("reach_length_m必须>0")
        if (batch.edge_static[:, 1] <= 0).any():
            raise ValueError("reach_slope_m_per_m必须>0")

    topological_levels(batch.edge_index, n)
    if batch.sample_weight is not None:
        _shape("sample_weight", batch.sample_weight, (b,))
        if (
            not torch.is_floating_point(batch.sample_weight)
            or not torch.isfinite(batch.sample_weight).all()
            or (batch.sample_weight <= 0).any()
        ):
            raise ValueError("sample_weight必须为[B]有限正浮点数")
