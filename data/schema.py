"""Tensor contract shared by synthetic and formal hydrological datasets.

Physical units at this boundary are rainfall mm/hour, discharge m3/s and water
level metre.  Formal datasets may standardise ``dynamic_node_features`` but the
explicit rainfall, history and target tensors remain in physical units.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch

Tensor = torch.Tensor


@dataclass
class GraphEventBatch:
    dynamic_node_features: Tensor
    rainfall: Tensor
    node_static: Tensor
    edge_index: Tensor
    edge_static: Tensor
    q_history: Tensor
    z_history: Tensor
    q_mask: Tensor
    z_mask: Tensor
    q_target: Tensor
    z_target: Tensor
    q_target_mask: Tensor
    z_target_mask: Tensor
    node_mask: Tensor | None = None
    event_mask: Tensor | None = None
    # Optional formal-data fields are appended to preserve the positional API
    # used by SyntheticEventDataset.
    rainfall_mask: Tensor | None = None
    station_index: Tensor | None = None
    node_area_km2: Tensor | None = None
    station_ids: tuple[str, ...] | None = None
    sample_id: str | tuple[str, ...] | None = None
    event_id: str | tuple[str, ...] | None = None
    graph_id: str | tuple[str, ...] | None = None

    def to(self, device: torch.device) -> "GraphEventBatch":
        return GraphEventBatch(
            **{
                f.name: (
                    getattr(self, f.name).to(device)
                    if isinstance(getattr(self, f.name), Tensor)
                    else getattr(self, f.name)
                )
                for f in fields(self)
            }
        )


def topological_levels(edge_index: Tensor, nodes: int) -> tuple[list[list[int]], list[int]]:
    """Return topological levels and order, raising on invalid/cyclic edges."""
    if nodes <= 0:
        raise ValueError("河网节点数必须大于0")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index必须为[2,E]")
    src, dst = edge_index.detach().cpu().tolist()
    indeg = [0] * nodes
    children: list[list[int]] = [[] for _ in range(nodes)]
    for edge_no, (s, d) in enumerate(zip(src, dst)):
        if not (0 <= s < nodes and 0 <= d < nodes):
            raise ValueError(f"第{edge_no}条边包含未知节点索引: {s}->{d}")
        indeg[d] += 1
        children[s].append(d)
    frontier = [i for i, value in enumerate(indeg) if value == 0]
    levels: list[list[int]] = []
    order: list[int] = []
    while frontier:
        levels.append(frontier)
        order.extend(frontier)
        next_frontier: list[int] = []
        for source in frontier:
            for destination in children[source]:
                indeg[destination] -= 1
                if indeg[destination] == 0:
                    next_frontier.append(destination)
        frontier = next_frontier
    if len(order) != nodes:
        raise ValueError("河网有环：edge_index必须是有向无环图(DAG)")
    return levels, order


def _require_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise ValueError(f"{name}形状应为{expected}，实际为{actual}")


def _require_bool(name: str, value: Tensor) -> None:
    if value.dtype != torch.bool:
        raise ValueError(f"{name}必须为BoolTensor，实际dtype={value.dtype}")


def validate_batch(x: GraphEventBatch, expected: dict[str, int] | None = None) -> None:
    """Strictly validate one collated batch before model execution.

    Missing observations must already be imputed to a finite placeholder and be
    identified by their corresponding boolean mask.  This prevents NaNs from
    silently entering recurrent and routing states.
    """
    if x.dynamic_node_features.ndim != 4:
        raise ValueError("dynamic_node_features应为[B,H,N,D]")
    batch_size, history, nodes, dynamic_dim = x.dynamic_node_features.shape
    if batch_size <= 0 or history <= 0 or nodes <= 0 or dynamic_dim <= 0:
        raise ValueError("batch、历史长度、节点数和动态特征维度都必须大于0")
    if x.q_target.ndim != 3:
        raise ValueError("q_target应为[B,F,N]")
    forecast = x.q_target.shape[1]
    if forecast <= 0:
        raise ValueError("预测时长必须大于0")

    if x.rainfall.ndim != 4 or x.rainfall.shape[0] != batch_size or x.rainfall.shape[2:] != (nodes, 1):
        raise ValueError("rainfall应为[B,T,N,1]且batch/节点数一致")
    if x.rainfall.shape[1] < history + forecast:
        raise ValueError(
            f"rainfall至少应覆盖history+forecast={history + forecast}小时，实际{x.rainfall.shape[1]}"
        )
    for name in ("q_history", "z_history", "q_mask", "z_mask"):
        _require_shape(name, tuple(getattr(x, name).shape), (batch_size, history, nodes))
    for name in ("q_target", "z_target", "q_target_mask", "z_target_mask"):
        _require_shape(name, tuple(getattr(x, name).shape), (batch_size, forecast, nodes))
    for name in ("q_mask", "z_mask", "q_target_mask", "z_target_mask"):
        _require_bool(name, getattr(x, name))

    if x.node_static.ndim != 2 or x.node_static.shape[0] != nodes:
        raise ValueError("node_static应为[N,D_node]且节点数量一致")
    if x.edge_index.dtype != torch.long or x.edge_index.ndim != 2 or x.edge_index.shape[0] != 2:
        raise ValueError("edge_index必须是[2,E]的LongTensor，第一行为src、第二行为dst")
    if x.edge_static.ndim != 2 or x.edge_static.shape[0] != x.edge_index.shape[1]:
        raise ValueError("edge_static应为[E,D_edge]且边数量与edge_index一致")
    if x.edge_index.numel() and (x.edge_index.min() < 0 or x.edge_index.max() >= nodes):
        raise ValueError("edge_index包含未知节点ID")

    finite_fields = {
        "dynamic_node_features": x.dynamic_node_features,
        "rainfall": x.rainfall,
        "node_static": x.node_static,
        "edge_static": x.edge_static,
        "q_history": x.q_history,
        "z_history": x.z_history,
        "q_target": x.q_target,
        "z_target": x.z_target,
    }
    for name, value in finite_fields.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}含NaN/Inf；缺失值必须先有限值插补并由mask标记")
    if (x.rainfall < 0).any():
        raise ValueError("rainfall含负值，疑似单位或质量问题（期望mm/h）")

    if x.rainfall_mask is not None:
        _require_shape("rainfall_mask", tuple(x.rainfall_mask.shape), tuple(x.rainfall.shape))
        _require_bool("rainfall_mask", x.rainfall_mask)
    if x.station_index is not None:
        if x.station_index.dtype != torch.long or tuple(x.station_index.shape) != (nodes,):
            raise ValueError("station_index必须为[N]的LongTensor")
        if (x.station_index < 0).any() or x.station_index.unique().numel() != nodes:
            raise ValueError("station_index必须是非负且图内唯一的全局站点索引")
    if x.node_area_km2 is not None:
        if tuple(x.node_area_km2.shape) != (nodes,):
            raise ValueError("node_area_km2必须为[N]")
        if not torch.isfinite(x.node_area_km2).all() or (x.node_area_km2 <= 0).any():
            raise ValueError("node_area_km2必须全部为有限正数（单位km²）")
    if x.station_ids is not None and len(x.station_ids) != nodes:
        raise ValueError("station_ids数量必须与节点数一致")

    if x.node_mask is not None:
        _require_bool("node_mask", x.node_mask)
        if tuple(x.node_mask.shape) not in ((nodes,), (batch_size, nodes)):
            raise ValueError("node_mask应为[N]或[B,N]")
    if x.event_mask is not None:
        _require_bool("event_mask", x.event_mask)
        if tuple(x.event_mask.shape) not in ((), (batch_size,)):
            raise ValueError("event_mask应为标量或[B]")

    if x.edge_static.shape[1] >= 2:
        if (x.edge_static[:, 0] <= 0).any() or (x.edge_static[:, 1] < 0).any():
            raise ValueError("边长应>0、坡降应>=0；请检查单位和src/dst方向")
    topological_levels(x.edge_index, nodes)

    if expected:
        actual_values = {
            "history_length": history,
            "forecast_horizon": forecast,
            "dynamic_dim": dynamic_dim,
            "node_static_dim": x.node_static.shape[1],
            "edge_static_dim": x.edge_static.shape[1],
        }
        unknown = set(expected) - set(actual_values)
        if unknown:
            raise ValueError(f"未知的批次校验项: {sorted(unknown)}")
        for key, value in expected.items():
            actual = actual_values[key]
            if actual != value:
                raise ValueError(f"{key}期望{value}，实际{actual}")
