"""Formal Hunan event dataset backed by graph/static/dynamic/index files.

The adapter intentionally keeps graph structure and static attributes separate
from hourly observations.  It materialises only the time windows named by
``sample_index.csv`` and guarantees that every DataLoader batch contains one
graph, so static tensors never get silently mixed across catchments.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data.schema import GraphEventBatch, topological_levels, validate_batch
from .normalization import NormalizationStats


NODE_STATIC_FEATURES = (
    "log_incremental_area",
    "log_upstream_area",
    "mean_hillslope_flow_distance_m",
    "mean_slope_deg",
    "elevation_std_m",
    "drainage_density_km_per_km2",
    "soil_log_ksat_0_30cm",
    "soil_profile_depth_cm",
    "forest_fraction",
    "impervious_fraction",
)
EDGE_STATIC_SOURCE_FEATURES = (
    "reach_length_km",
    "reach_slope_m_per_m",
)
# The routing module consumes metres, not the kilometre unit used by the CSV.
EDGE_STATIC_MODEL_FEATURES = (
    "reach_length_m",
    "reach_slope_m_per_m",
)
DEFAULT_DYNAMIC_FEATURES = ("FLOW", "WATER_LEVEL")
SUPPORTED_DYNAMIC_FEATURES = {
    "RAIN_MM",
    "FLOW",
    "WATER_LEVEL",
    "RAIN_MASK",
    "FLOW_MASK",
    "WATER_LEVEL_MASK",
}

NODE_CATALOG_COLUMNS = (
    "GRAPH_ID",
    "BASIN_ID",
    "NODE_INDEX",
    "STATION_ID",
    "OUTLET_ID",
    "ROLE",
    "IS_OUTLET",
)
EDGE_TOPOLOGY_COLUMNS = ("GRAPH_ID", "FROM_NODE", "TO_NODE", "FROM_STATION", "TO_STATION")
DYNAMIC_COLUMNS = (
    "GRAPH_ID",
    "TIMESTAMP",
    "STATION_ID",
    "RAIN_MM",
    "FLOW",
    "WATER_LEVEL",
    "RAIN_MASK",
    "FLOW_MASK",
    "WATER_LEVEL_MASK",
)
EVENT_COLUMNS = (
    "EVENT_ID",
    "GRAPH_ID",
    "BASIN_ID",
    "OUTLET_ID",
    "RAIN_START",
    "RAIN_END",
    "HYDRO_START",
    "PEAK_TIME",
    "HYDRO_END",
    "SAMPLE_START",
    "SAMPLE_END",
    "EVENT_TYPE",
    "EVENT_GRADE",
    "COMPOUND_EVENT",
    "PEAK_COUNT",
    "SOURCE_RAIN_EVENT_IDS",
    "SOURCE_RAIN_EVENT_COUNT",
)
SAMPLE_COLUMNS = (
    "SAMPLE_ID",
    "EVENT_ID",
    "GRAPH_ID",
    "OUTLET_ID",
    "INPUT_START",
    "FORECAST_TIME",
    "TARGET_END",
    "HISTORY_HOURS",
    "FORECAST_HOURS",
    "TARGET_VARIABLE",
    "SPLIT",
)
SPLIT_COLUMNS = ("EVENT_ID", "GRAPH_ID", "EVENT_YEAR", "EVENT_GRADE", "SPLIT", "SPLIT_REASON")
SAMPLE_REJECTION_COLUMNS = (
    "REJECTION_ID",
    "SAMPLE_ID",
    "EVENT_ID",
    "GRAPH_ID",
    "OUTLET_ID",
    "FORECAST_TIME",
    "TARGET_START",
    "TARGET_END",
    "TARGET_VARIABLE",
    "TARGET_COVERAGE",
    "MIN_TARGET_COVERAGE",
    "REASON",
    "SPLIT",
)


@dataclass(frozen=True)
class _Node:
    graph_id: str
    basin_id: str
    node_index: int
    station_id: str
    outlet_id: str
    role: str
    is_outlet: bool


@dataclass(frozen=True)
class _Graph:
    graph_id: str
    basin_id: str
    outlet_id: str
    nodes: tuple[_Node, ...]
    node_static: torch.Tensor
    node_area_km2: torch.Tensor
    edge_index: torch.Tensor
    edge_static: torch.Tensor


@dataclass(frozen=True)
class _Event:
    event_id: str
    graph_id: str
    basin_id: str
    outlet_id: str
    sample_start: datetime
    sample_end: datetime
    event_type: str
    event_grade: str
    split_time: datetime
    event_year: int


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    event_id: str
    graph_id: str
    outlet_id: str
    input_start: datetime
    forecast_time: datetime
    target_end: datetime
    history_hours: int
    forecast_hours: int
    target_variables: frozenset[str]
    split: str


@dataclass(frozen=True)
class _DynamicGraph:
    timestamps: tuple[datetime, ...]
    time_to_index: Mapping[datetime, int]
    rainfall: torch.Tensor
    flow: torch.Tensor
    water_level: torch.Tensor
    rainfall_mask: torch.Tensor
    flow_mask: torch.Tensor
    water_level_mask: torch.Tensor


def _read_csv(path: Path, required: Sequence[str]) -> list[dict[str, str]]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"缺少正式数据文件: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV缺少表头: {path}")
        duplicate_headers = sorted({name for name in reader.fieldnames if reader.fieldnames.count(name) > 1})
        if duplicate_headers:
            raise ValueError(f"CSV表头有重复列{duplicate_headers}: {path}")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV缺少必需列{missing}: {path}")
        rows: list[dict[str, str]] = []
        for line_number, raw in enumerate(reader, start=2):
            if raw.get(None):
                raise ValueError(f"CSV第{line_number}行字段数量超过表头: {path}")
            row = {name: (raw.get(name) or "").strip() for name in reader.fieldnames}
            if not any(row.values()):
                continue
            row["__line__"] = str(line_number)
            rows.append(row)
        return rows


def _context(path: Path, row: Mapping[str, str]) -> str:
    return f"{path}第{row.get('__line__', '?')}行"


def _required_text(value: str, name: str, context: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{context}: {name}不能为空")
    return value


def _parse_bool(value: str, name: str, context: str) -> bool:
    normal = value.strip().upper()
    if normal in {"1", "TRUE", "T", "YES", "Y"}:
        return True
    if normal in {"0", "FALSE", "F", "NO", "N"}:
        return False
    raise ValueError(f"{context}: {name}必须是0/1或true/false，实际为{value!r}")


def _parse_int(value: str, name: str, context: str, minimum: int | None = None) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{context}: {name}必须是整数，实际为{value!r}") from exc
    if minimum is not None and number < minimum:
        raise ValueError(f"{context}: {name}必须>={minimum}，实际为{number}")
    return number


def _parse_float(value: str, name: str, context: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: {name}必须是数值，实际为{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: {name}必须为有限数，实际为{value!r}")
    return number


def _parse_datetime(value: str, name: str, context: str) -> datetime:
    text = _required_text(value, name, context)
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
    except ValueError as exc:
        raise ValueError(f"{context}: {name}不是有效ISO日期时间: {value!r}") from exc
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ValueError(f"{context}: {name}必须对齐整点，实际为{value!r}")
    # Canonicalise timezone-aware inputs to naive UTC so files with different
    # but equivalent offsets compare reliably. Naive inputs remain local time.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _normalise_split(value: str, context: str = "split") -> str:
    aliases = {
        "TRAIN": "TRAIN",
        "TRAINING": "TRAIN",
        "VALID": "VALIDATION",
        "VAL": "VALIDATION",
        "VALIDATION": "VALIDATION",
        "TEST": "TEST",
        "TESTING": "TEST",
    }
    key = value.strip().upper()
    if key not in aliases:
        raise ValueError(f"{context}只能是TRAIN/VALIDATION/TEST，实际为{value!r}")
    return aliases[key]


def _parse_target_variables(value: str | Sequence[str], context: str) -> frozenset[str]:
    if isinstance(value, str):
        normal = value.upper().replace("+", ",").replace("/", ",").replace("|", ",")
        values = [part.strip() for part in normal.split(",") if part.strip()]
    else:
        values = [str(part).strip().upper() for part in value]
    aliases = {"Q": "FLOW", "DISCHARGE": "FLOW", "Z": "WATER_LEVEL", "LEVEL": "WATER_LEVEL"}
    expanded: set[str] = set()
    for value_name in values:
        if value_name in {"BOTH", "ALL"}:
            expanded.update(("FLOW", "WATER_LEVEL"))
        else:
            expanded.add(aliases.get(value_name, value_name))
    if not expanded or not expanded.issubset({"FLOW", "WATER_LEVEL"}):
        raise ValueError(f"{context}: TARGET_VARIABLE仅支持FLOW/WATER_LEVEL/BOTH，实际为{value!r}")
    return frozenset(expanded)


def _feature_list(raw: Any, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = raw.get(key) if isinstance(raw, dict) else None
    if value is None:
        return default
    if not isinstance(value, list) or not value:
        raise ValueError(f"feature_schema.json中的{key}必须是非空数组")
    names: list[str] = []
    for item in value:
        name = item.get("name") if isinstance(item, dict) else item
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"feature_schema.json中的{key}存在无效特征: {item!r}")
        names.append(name.strip())
    if len(set(names)) != len(names):
        raise ValueError(f"feature_schema.json中的{key}包含重复特征")
    return tuple(names)


class HunanGraphEventDataset(Dataset[GraphEventBatch]):
    """Strict event-window dataset for the formal Hunan directory contract."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        history_hours: int | None = None,
        forecast_hours: int | None = None,
        *,
        graph_id: str | None = None,
        target_variables: str | Sequence[str] | None = None,
        normalize_dynamic: bool = True,
        future_rainfall_mode: str = "persistence",
        use_observation_masks: bool = True,
        strict: bool = True,
        allow_divergence: bool = False,
        dynamic_cache: dict[tuple[str, str], _DynamicGraph] | None = None,
        eligible_event_types: Sequence[str] = ("HYDRO_FLOOD",),
        eligible_event_grades: Sequence[str] = ("A", "B"),
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"湖南正式数据根目录不存在或不是目录: {self.root}")
        self.split = _normalise_split(split, "数据集split")
        self.strict = strict
        self.allow_divergence = allow_divergence
        self._dynamic_cache = dynamic_cache if dynamic_cache is not None else {}
        self.normalize_dynamic = normalize_dynamic
        self.future_rainfall_mode = future_rainfall_mode.strip().lower()
        if self.future_rainfall_mode not in {"observed_hindcast", "zero", "persistence"}:
            raise ValueError(
                "future_rainfall_mode只能是observed_hindcast/zero/persistence，"
                f"实际为{future_rainfall_mode!r}"
            )
        if not use_observation_masks:
            raise ValueError("湖南正式数据必须设置use_observation_masks=true，禁止把缺测值当作真实0")
        self.use_observation_masks = True
        self._requested_graph_id = graph_id.strip() if graph_id is not None else None
        auto_targets = isinstance(target_variables, str) and target_variables.strip().upper() == "AUTO"
        self._requested_targets = (
            None
            if target_variables is None or auto_targets
            else _parse_target_variables(target_variables, "target_variables")
        )
        self._eligible_event_types = {value.strip().upper() for value in eligible_event_types}
        self._eligible_event_grades = {value.strip().upper() for value in eligible_event_grades}
        if not self._eligible_event_types or not self._eligible_event_grades:
            raise ValueError("eligible_event_types和eligible_event_grades不能为空")

        self.dynamic_features = self._load_feature_schema()
        self.node_static_features = NODE_STATIC_FEATURES
        self.edge_static_features = EDGE_STATIC_MODEL_FEATURES
        self.dynamic_dim = len(self.dynamic_features)
        self.node_static_dim = len(self.node_static_features)
        self.edge_static_dim = len(self.edge_static_features)
        required_stats = tuple(
            feature for feature in self.dynamic_features if feature in {"RAIN_MM", "FLOW", "WATER_LEVEL"}
        )
        # Always require the three documented training statistics. They are
        # loaded for every split; validation/test never calculate their own.
        required_stats = tuple(dict.fromkeys((*required_stats, "RAIN_MM", "FLOW", "WATER_LEVEL")))
        self.normalization = NormalizationStats.load(
            self.root / "metadata" / "normalization_stats.json",
            required=required_stats,
            require_train_provenance=self.strict,
        )

        nodes_by_graph, graph_metadata = self._load_node_catalog()
        self.station_ids = tuple(sorted({node.station_id for nodes in nodes_by_graph.values() for node in nodes}))
        self.station_to_index = {station: index for index, station in enumerate(self.station_ids)}
        self.num_stations = len(self.station_ids)
        self._graphs = self._load_graphs(nodes_by_graph, graph_metadata)
        if self._requested_graph_id is not None and self._requested_graph_id not in self._graphs:
            raise ValueError(
                f"指定GRAPH_ID={self._requested_graph_id!r}不在node_catalog.csv中；"
                f"可用值={sorted(self._graphs)}"
            )
        self._events = self._load_events()
        event_graph_ids = {event.graph_id for event in self._events.values()}
        self.target_variables_by_graph = self._load_target_variables_by_graph(
            event_graph_ids
        )
        self._validate_final_events_against_all()
        event_splits = self._load_splits()
        self.event_split_by_id = dict(event_splits)
        self._samples = self._load_samples(event_splits, history_hours, forecast_hours)
        used_graphs = tuple(dict.fromkeys(sample.graph_id for sample in self._samples))
        self.graph_ids = used_graphs
        self.graph_node_counts = {gid: len(self._graphs[gid].nodes) for gid in used_graphs}
        self.event_ids = frozenset(sample.event_id for sample in self._samples)
        self.history_hours = self._samples[0].history_hours
        self.forecast_hours = self._samples[0].forecast_hours
        self.artifact_status, self.qc_status = self._inspect_metadata_and_qc()
        self._dynamic = self._load_dynamic_data(set(used_graphs))
        self._validate_all_windows()

    @property
    def num_nodes(self) -> int:
        if len(self.graph_ids) != 1:
            raise ValueError(
                "数据集包含多个GRAPH_ID，节点数不唯一；请使用graph_node_counts或按graph_id筛选"
            )
        return self.graph_node_counts[self.graph_ids[0]]

    def num_nodes_for_graph(self, graph_id: str) -> int:
        try:
            return self.graph_node_counts[graph_id]
        except KeyError as exc:
            raise KeyError(f"当前split没有GRAPH_ID={graph_id!r}") from exc

    def graph_id_for_index(self, index: int) -> str:
        return self._samples[index].graph_id

    def _load_feature_schema(self) -> tuple[str, ...]:
        path = self.root / "metadata" / "feature_schema.json"
        if not path.exists():
            raise FileNotFoundError(
                f"缺少feature_schema.json，无法安全反变换log_incremental_area得到km²物理面积: {path}"
            )
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"feature_schema.json不是有效JSON: {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"feature_schema.json根节点必须是对象: {path}")
        dynamic = tuple(name.upper() for name in _feature_list(raw, "dynamic_features", DEFAULT_DYNAMIC_FEATURES))
        unsupported = sorted(set(dynamic) - SUPPORTED_DYNAMIC_FEATURES)
        if unsupported:
            raise ValueError(f"feature_schema.json包含不支持的动态特征{unsupported}")
        node = _feature_list(raw, "node_static_features", NODE_STATIC_FEATURES)
        edge = _feature_list(raw, "edge_static_features", EDGE_STATIC_SOURCE_FEATURES)
        if node != NODE_STATIC_FEATURES:
            raise ValueError(
                "node_static_features必须严格按正式10项顺序排列: " + ", ".join(NODE_STATIC_FEATURES)
            )
        if edge != EDGE_STATIC_SOURCE_FEATURES:
            raise ValueError(
                "edge_static_features必须是reach_length_km/reach_slope_m_per_m"
            )
        self._area_source, self._area_inverse = self._parse_area_contract(raw, path)
        return dynamic

    @staticmethod
    def _parse_area_contract(raw: Mapping[str, Any], path: Path) -> tuple[str, str]:
        """Find an explicit physical-area mapping; never guess a logarithm base."""
        contract: Mapping[str, Any] | None = None
        for container_name in ("physical_features", "derived_features"):
            container = raw.get(container_name)
            if isinstance(container, dict) and isinstance(container.get("incremental_area_km2"), dict):
                contract = container["incremental_area_km2"]
                break
        if contract is None:
            metadata = raw.get("node_static_feature_metadata", raw.get("node_static_metadata"))
            if isinstance(metadata, dict) and isinstance(metadata.get("log_incremental_area"), dict):
                contract = metadata["log_incremental_area"]
        if contract is None:
            items = raw.get("node_static_features")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("name") in {
                        "log_incremental_area",
                        "incremental_area_km2",
                    }:
                        contract = item
                        break
        example = (
            '示例: "physical_features": {"incremental_area_km2": '
            '{"source": "log_incremental_area", "transform": "log1p", "unit": "km2"}}'
        )
        if contract is None:
            raise ValueError(f"{path}: 未声明增量面积反变换；不能猜测log底数。{example}")
        source = str(contract.get("source", contract.get("from", contract.get("name", "")))).strip()
        if source not in {"log_incremental_area", "incremental_area_km2"}:
            raise ValueError(
                f"{path}: incremental_area_km2.source必须是log_incremental_area或incremental_area_km2。{example}"
            )
        unit = str(
            contract.get(
                "unit",
                contract.get("physical_unit", contract.get("output_unit", contract.get("source_unit", ""))),
            )
        ).strip().lower()
        if unit not in {"km2", "km^2", "km²", "square_kilometres", "square_kilometers"}:
            raise ValueError(f"{path}: 增量面积物理单位必须明确为km2，实际为{unit!r}。{example}")
        if source == "incremental_area_km2":
            return source, "identity"
        transform = str(contract.get("transform", "")).strip().lower()
        inverse = str(contract.get("inverse", contract.get("inverse_transform", ""))).strip().lower()
        if transform in {"ln", "loge", "natural_log", "log"} or inverse in {"exp", "exponential"}:
            return source, "exp"
        if transform in {"log1p", "ln1p"} or inverse in {"expm1"}:
            return source, "expm1"
        if transform in {"log10", "lg"} or inverse in {"pow10", "10**x", "10^x"}:
            return source, "pow10"
        raise ValueError(
            f"{path}: log_incremental_area必须明确transform=ln/log1p/log10或对应inverse，"
            f"实际transform={transform!r}, inverse={inverse!r}。{example}"
        )

    def _load_node_catalog(self) -> tuple[dict[str, tuple[_Node, ...]], dict[str, tuple[str, str]]]:
        path = self.root / "graph" / "node_catalog.csv"
        rows = _read_csv(path, NODE_CATALOG_COLUMNS)
        if not rows:
            raise ValueError(f"node_catalog.csv没有节点记录: {path}")
        grouped: dict[str, list[_Node]] = {}
        station_keys: set[tuple[str, str]] = set()
        for row in rows:
            context = _context(path, row)
            graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            basin_id = _required_text(row["BASIN_ID"], "BASIN_ID", context)
            station_id = _required_text(row["STATION_ID"], "STATION_ID", context)
            outlet_id = _required_text(row["OUTLET_ID"], "OUTLET_ID", context)
            key = (graph_id, station_id)
            if key in station_keys:
                raise ValueError(f"{context}: GRAPH_ID/STATION_ID重复: {key}")
            station_keys.add(key)
            grouped.setdefault(graph_id, []).append(
                _Node(
                    graph_id,
                    basin_id,
                    _parse_int(row["NODE_INDEX"], "NODE_INDEX", context, 0),
                    station_id,
                    outlet_id,
                    row["ROLE"],
                    _parse_bool(row["IS_OUTLET"], "IS_OUTLET", context),
                )
            )
        result: dict[str, tuple[_Node, ...]] = {}
        metadata: dict[str, tuple[str, str]] = {}
        for gid, unordered in grouped.items():
            nodes = tuple(sorted(unordered, key=lambda node: node.node_index))
            actual = [node.node_index for node in nodes]
            if actual != list(range(len(nodes))):
                raise ValueError(f"GRAPH_ID={gid}: NODE_INDEX必须从0连续递增，实际为{actual}")
            basins = {node.basin_id for node in nodes}
            outlets = {node.outlet_id for node in nodes}
            flagged = [node.station_id for node in nodes if node.is_outlet]
            if len(basins) != 1 or len(outlets) != 1:
                raise ValueError(f"GRAPH_ID={gid}: BASIN_ID和OUTLET_ID必须图内一致")
            outlet = next(iter(outlets))
            if flagged != [outlet]:
                raise ValueError(
                    f"GRAPH_ID={gid}: 必须且只能把OUTLET_ID={outlet!r}对应节点标为IS_OUTLET=1，实际={flagged}"
                )
            result[gid] = nodes
            metadata[gid] = (next(iter(basins)), outlet)
        return result, metadata

    def _load_graphs(
        self,
        nodes_by_graph: Mapping[str, tuple[_Node, ...]],
        metadata: Mapping[str, tuple[str, str]],
    ) -> dict[str, _Graph]:
        node_path = self.root / "graph" / "node_static_attributes.csv"
        node_rows = _read_csv(node_path, ("GRAPH_ID", "STATION_ID", *NODE_STATIC_FEATURES))
        node_values: dict[tuple[str, str], list[float]] = {}
        node_areas: dict[tuple[str, str], float] = {}
        for row in node_rows:
            context = _context(node_path, row)
            key = (
                _required_text(row["GRAPH_ID"], "GRAPH_ID", context),
                _required_text(row["STATION_ID"], "STATION_ID", context),
            )
            if key in node_values:
                raise ValueError(f"{context}: 节点静态属性重复: {key}")
            node_values[key] = [_parse_float(row[name], name, context) for name in NODE_STATIC_FEATURES]
            if self._area_source not in row:
                raise ValueError(
                    f"{context}: feature_schema声明面积来源{self._area_source!r}，"
                    "但node_static_attributes.csv没有该列"
                )
            encoded_area = _parse_float(row[self._area_source], self._area_source, context)
            try:
                if self._area_inverse == "identity":
                    area = encoded_area
                elif self._area_inverse == "exp":
                    area = math.exp(encoded_area)
                elif self._area_inverse == "expm1":
                    area = math.expm1(encoded_area)
                elif self._area_inverse == "pow10":
                    area = 10.0**encoded_area
                else:
                    raise AssertionError(f"未知面积反变换{self._area_inverse}")
            except OverflowError as exc:
                raise ValueError(f"{context}: 增量面积反变换溢出，编码值={encoded_area}") from exc
            if not math.isfinite(area) or area <= 0:
                raise ValueError(
                    f"{context}: 反变换后的incremental_area_km2必须为有限正数，实际={area}"
                )
            node_areas[key] = area

        edge_path = self.root / "graph" / "edge_topology.csv"
        edge_rows = _read_csv(edge_path, EDGE_TOPOLOGY_COLUMNS)
        edges_by_graph: dict[str, list[tuple[int, int, str, str]]] = {gid: [] for gid in nodes_by_graph}
        edge_keys: set[tuple[str, str, str]] = set()
        for row in edge_rows:
            context = _context(edge_path, row)
            gid = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            if gid not in nodes_by_graph:
                raise ValueError(f"{context}: edge_topology引用未知GRAPH_ID={gid!r}")
            source_station = _required_text(row["FROM_STATION"], "FROM_STATION", context)
            destination_station = _required_text(row["TO_STATION"], "TO_STATION", context)
            station_to_local = {node.station_id: node.node_index for node in nodes_by_graph[gid]}
            if source_station not in station_to_local or destination_station not in station_to_local:
                raise ValueError(f"{context}: 边引用未知站点{source_station!r}->{destination_station!r}")
            source = _parse_int(row["FROM_NODE"], "FROM_NODE", context, 0)
            destination = _parse_int(row["TO_NODE"], "TO_NODE", context, 0)
            if source != station_to_local[source_station] or destination != station_to_local[destination_station]:
                raise ValueError(f"{context}: FROM_NODE/TO_NODE与站点NODE_INDEX不一致")
            if source == destination:
                raise ValueError(f"{context}: 不允许自环边{source_station!r}->{destination_station!r}")
            key = (gid, source_station, destination_station)
            if key in edge_keys:
                raise ValueError(f"{context}: 重复边{key}")
            edge_keys.add(key)
            edges_by_graph[gid].append((source, destination, source_station, destination_station))

        edge_static_path = self.root / "graph" / "edge_static_attributes.csv"
        edge_static_rows = _read_csv(
            edge_static_path,
            ("GRAPH_ID", "FROM_STATION", "TO_STATION", *EDGE_STATIC_SOURCE_FEATURES),
        )
        edge_values: dict[tuple[str, str, str], list[float]] = {}
        for row in edge_static_rows:
            context = _context(edge_static_path, row)
            key = (
                _required_text(row["GRAPH_ID"], "GRAPH_ID", context),
                _required_text(row["FROM_STATION"], "FROM_STATION", context),
                _required_text(row["TO_STATION"], "TO_STATION", context),
            )
            if key in edge_values:
                raise ValueError(f"{context}: 边静态属性重复: {key}")
            values = [_parse_float(row[name], name, context) for name in EDGE_STATIC_SOURCE_FEATURES]
            if values[0] <= 0 or values[1] < 0:
                raise ValueError(f"{context}: 河长必须>0，坡降必须>=0")
            # CSV kilometres -> routing metres.
            values[0] *= 1000.0
            edge_values[key] = values

        expected_nodes = {(gid, node.station_id) for gid, nodes in nodes_by_graph.items() for node in nodes}
        if set(node_values) != expected_nodes:
            missing = sorted(expected_nodes - set(node_values))[:10]
            extra = sorted(set(node_values) - expected_nodes)[:10]
            raise ValueError(f"node_static_attributes与node_catalog不一一对应；缺少={missing}，多余={extra}")
        if set(edge_values) != edge_keys:
            missing = sorted(edge_keys - set(edge_values))[:10]
            extra = sorted(set(edge_values) - edge_keys)[:10]
            raise ValueError(f"edge_static_attributes与edge_topology不一一对应；缺少={missing}，多余={extra}")

        graphs: dict[str, _Graph] = {}
        for gid, nodes in nodes_by_graph.items():
            edge_records = edges_by_graph[gid]
            if edge_records:
                edge_index = torch.tensor([[edge[0] for edge in edge_records], [edge[1] for edge in edge_records]], dtype=torch.long)
                edge_static = torch.tensor(
                    [edge_values[(gid, edge[2], edge[3])] for edge in edge_records], dtype=torch.float32
                )
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_static = torch.empty((0, len(EDGE_STATIC_MODEL_FEATURES)), dtype=torch.float32)
            topological_levels(edge_index, len(nodes))
            self._validate_hydrological_topology(gid, nodes, edge_index)
            graph_node_static = torch.tensor(
                [node_values[(gid, node.station_id)] for node in nodes], dtype=torch.float32
            )
            graph_node_area = torch.tensor(
                [node_areas[(gid, node.station_id)] for node in nodes], dtype=torch.float32
            )
            basin_id, outlet_id = metadata[gid]
            graphs[gid] = _Graph(
                gid,
                basin_id,
                outlet_id,
                nodes,
                graph_node_static,
                graph_node_area,
                edge_index,
                edge_static,
            )
        return graphs

    def _validate_hydrological_topology(
        self, graph_id: str, nodes: tuple[_Node, ...], edge_index: torch.Tensor
    ) -> None:
        outlet = next(node.node_index for node in nodes if node.is_outlet)
        sources, destinations = edge_index.tolist()
        if outlet in sources:
            raise ValueError(f"GRAPH_ID={graph_id}: 出口节点不能再有下游边")
        outdegree = [0] * len(nodes)
        reverse: list[list[int]] = [[] for _ in nodes]
        for source, destination in zip(sources, destinations):
            outdegree[source] += 1
            reverse[destination].append(source)
        divergent = [nodes[index].station_id for index, degree in enumerate(outdegree) if degree > 1]
        if divergent and not self.allow_divergence:
            raise ValueError(
                f"GRAPH_ID={graph_id}: 当前物理路由会在分汊处重复水量，检测到出度>1的站点{divergent}；"
                "若使用已处理分流权重的非物理路由，请显式设置allow_divergence=True"
            )
        reachable = {outlet}
        frontier = [outlet]
        while frontier:
            current = frontier.pop()
            for upstream in reverse[current]:
                if upstream not in reachable:
                    reachable.add(upstream)
                    frontier.append(upstream)
        if len(reachable) != len(nodes):
            missing = [nodes[index].station_id for index in range(len(nodes)) if index not in reachable]
            raise ValueError(f"GRAPH_ID={graph_id}: 以下节点不能沿河网到达出口: {missing}")

    def _load_target_variables_by_graph(
        self, required_graph_ids: set[str]
    ) -> dict[str, frozenset[str]]:
        path = self.root / "events" / "target_variable_by_graph.csv"
        rows = _read_csv(path, ("GRAPH_ID", "TARGET_VARIABLE"))
        result: dict[str, frozenset[str]] = {}
        for row in rows:
            context = _context(path, row)
            graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            if graph_id not in self._graphs:
                raise ValueError(f"{context}: 目标变量表引用未知GRAPH_ID={graph_id!r}")
            if graph_id in result:
                raise ValueError(f"{context}: 目标变量表GRAPH_ID重复: {graph_id!r}")
            graph = self._graphs[graph_id]
            if "BASIN_ID" in row and row["BASIN_ID"] and row["BASIN_ID"] != graph.basin_id:
                raise ValueError(f"{context}: BASIN_ID与node_catalog不一致")
            if "OUTLET_ID" in row and row["OUTLET_ID"] and row["OUTLET_ID"] != graph.outlet_id:
                raise ValueError(f"{context}: OUTLET_ID与node_catalog不一致")
            result[graph_id] = _parse_target_variables(row["TARGET_VARIABLE"], context)
        unknown_required = sorted(required_graph_ids - set(self._graphs))
        if unknown_required:
            raise ValueError(
                "flood_events_final.csv引用未知GRAPH_ID=" + str(unknown_required)
            )
        missing = sorted(required_graph_ids - set(result))
        if missing:
            raise ValueError(
                "target_variable_by_graph.csv未覆盖全部正式事件GRAPH_ID，"
                f"缺少={missing}"
            )
        return result

    def _validate_final_events_against_all(self) -> None:
        path = self.root / "events" / "flood_events_all.csv"
        rows = _read_csv(path, ("EVENT_ID", "GRAPH_ID"))
        all_events: dict[str, str] = {}
        for row in rows:
            context = _context(path, row)
            event_id = _required_text(row["EVENT_ID"], "EVENT_ID", context)
            graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            if event_id in all_events:
                raise ValueError(f"{context}: flood_events_all.csv中EVENT_ID重复: {event_id!r}")
            all_events[event_id] = graph_id
        missing = sorted(set(self._events) - set(all_events))
        if missing:
            raise ValueError(f"flood_events_final.csv中的事件不在flood_events_all.csv中: {missing[:10]}")
        mismatched = [
            event_id
            for event_id, event in self._events.items()
            if all_events[event_id] != event.graph_id
        ]
        if mismatched:
            raise ValueError(f"final/all事件表GRAPH_ID不一致: {mismatched[:10]}")

    def _inspect_metadata_and_qc(self) -> tuple[dict[str, bool], dict[str, Any]]:
        """Discover provenance artifacts and gate explicitly rejected records."""
        artifact_paths = {
            "dataset_summary": self.root / "metadata" / "dataset_summary.csv",
            "source_manifest": self.root / "metadata" / "source_manifest.json",
            "build_log": self.root / "metadata" / "build_log.txt",
            "dynamic_coverage": self.root / "qc" / "dynamic_coverage.csv",
            "event_exclusion": self.root / "qc" / "event_exclusion.csv",
            "hydro_file_selection": self.root / "qc" / "hydro_file_selection.csv",
            "hydro_load_audit": self.root / "qc" / "hydro_load_audit.csv",
            "rain_source_coverage": self.root / "qc" / "rain_source_coverage.csv",
            "sample_rejection": self.root / "qc" / "sample_rejection.csv",
        }
        status = {name: path.is_file() for name, path in artifact_paths.items()}
        missing = [str(artifact_paths[name]) for name, exists in status.items() if not exists]
        if missing and self.strict:
            raise FileNotFoundError("正式数据契约缺少metadata/qc文件: " + "; ".join(missing))

        row_counts: dict[str, int] = {}
        for name, path in artifact_paths.items():
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            rows = _read_csv(path, ())
            row_counts[name] = len(rows)
        manifest_path = artifact_paths["source_manifest"]
        self.source_manifest: Any = None
        if manifest_path.is_file():
            try:
                self.source_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"source_manifest.json不是有效JSON: {manifest_path}: {exc}") from exc
        log_path = artifact_paths["build_log"]
        if log_path.is_file():
            try:
                log_path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError as exc:
                raise ValueError(f"build_log.txt必须是UTF-8文本: {log_path}") from exc

        excluded_events: set[str] = set()
        event_path = artifact_paths["event_exclusion"]
        if event_path.is_file():
            for row in _read_csv(event_path, ("EVENT_ID",)):
                if row["EVENT_ID"]:
                    excluded_events.add(row["EVENT_ID"])
        rejected_samples: set[str] = set()
        rejected_event_ids: set[str] = set()
        rejection_ids: set[str] = set()
        sample_path = artifact_paths["sample_rejection"]
        if sample_path.is_file():
            for row in _read_csv(sample_path, SAMPLE_REJECTION_COLUMNS):
                context = _context(sample_path, row)
                rejection_id = _required_text(row["REJECTION_ID"], "REJECTION_ID", context)
                if rejection_id in rejection_ids:
                    raise ValueError(f"{context}: REJECTION_ID重复: {rejection_id!r}")
                rejection_ids.add(rejection_id)
                event_id = _required_text(row["EVENT_ID"], "EVENT_ID", context)
                if event_id not in self._events:
                    raise ValueError(f"{context}: 拒绝记录引用未知final EVENT_ID={event_id!r}")
                graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
                event = self._events[event_id]
                if graph_id != event.graph_id:
                    raise ValueError(f"{context}: 拒绝记录GRAPH_ID与事件表不一致")
                outlet_id = row["OUTLET_ID"].strip()
                if outlet_id and outlet_id != event.outlet_id:
                    raise ValueError(f"{context}: 拒绝记录OUTLET_ID与事件表不一致")
                split = _normalise_split(row["SPLIT"], f"{context}: SPLIT")
                if split != self.event_split_by_id[event_id]:
                    raise ValueError(f"{context}: 拒绝记录SPLIT与data_split不一致")
                reason = _required_text(row["REASON"], "REASON", context)
                if reason == "TARGET_COVERAGE_BELOW_THRESHOLD":
                    forecast_time = _parse_datetime(row["FORECAST_TIME"], "FORECAST_TIME", context)
                    target_start = _parse_datetime(row["TARGET_START"], "TARGET_START", context)
                    target_end = _parse_datetime(row["TARGET_END"], "TARGET_END", context)
                    if not forecast_time < target_start <= target_end:
                        raise ValueError(f"{context}: 目标覆盖率拒绝记录的时间窗无效")
                    coverage = _parse_float(row["TARGET_COVERAGE"], "TARGET_COVERAGE", context)
                    threshold = _parse_float(
                        row["MIN_TARGET_COVERAGE"], "MIN_TARGET_COVERAGE", context
                    )
                    if not 0.0 <= coverage < threshold <= 1.0:
                        raise ValueError(f"{context}: 目标覆盖率拒绝条件无效")
                if row.get("SAMPLE_ID", ""):
                    rejected_samples.add(row["SAMPLE_ID"])
                rejected_event_ids.add(event_id)

        final_event_ids = set(self._events)
        excluded_final_events = sorted(final_event_ids & excluded_events)
        if excluded_final_events:
            raise ValueError(
                "event_exclusion.csv中的事件仍出现在flood_events_final.csv: "
                f"{excluded_final_events[:10]}"
            )
        unaccounted_events = sorted(
            final_event_ids - set(self._all_sample_event_ids) - rejected_event_ids
        )
        if unaccounted_events:
            raise ValueError(
                "final事件必须至少有一个sample或明确拒绝记录；未闭环EVENT_ID="
                f"{unaccounted_events[:20]}"
            )
        event_conflicts = sorted(set(self._all_sample_event_ids) & excluded_events)
        sample_conflicts = sorted(set(self._all_sample_ids) & rejected_samples)
        if event_conflicts or sample_conflicts:
            raise ValueError(
                "QC门禁失败：被排除/拒绝的记录仍进入当前数据集；"
                f"EVENT_ID冲突={event_conflicts[:10]}，SAMPLE_ID冲突={sample_conflicts[:10]}"
            )
        qc_status = {
            "row_counts": row_counts,
            "excluded_event_ids": frozenset(excluded_events),
            "rejected_event_ids": frozenset(rejected_event_ids),
            "rejection_ids": frozenset(rejection_ids),
            "rejected_sample_ids": frozenset(rejected_samples),
            "loaded_conflicts": {"event_ids": (), "sample_ids": ()},
        }
        return status, qc_status

    def _load_events(self) -> dict[str, _Event]:
        path = self.root / "events" / "flood_events_final.csv"
        rows = _read_csv(path, EVENT_COLUMNS)
        events: dict[str, _Event] = {}
        for row in rows:
            context = _context(path, row)
            event_id = _required_text(row["EVENT_ID"], "EVENT_ID", context)
            if event_id in events:
                raise ValueError(f"{context}: EVENT_ID重复: {event_id!r}")
            graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            if graph_id not in self._graphs:
                raise ValueError(f"{context}: 洪水事件引用未知GRAPH_ID={graph_id!r}")
            graph = self._graphs[graph_id]
            basin_id = _required_text(row["BASIN_ID"], "BASIN_ID", context)
            outlet_id = _required_text(row["OUTLET_ID"], "OUTLET_ID", context)
            if basin_id != graph.basin_id or outlet_id != graph.outlet_id:
                raise ValueError(f"{context}: 事件BASIN_ID/OUTLET_ID与node_catalog不一致")
            rain_start = _parse_datetime(row["RAIN_START"], "RAIN_START", context)
            rain_end = _parse_datetime(row["RAIN_END"], "RAIN_END", context)
            peak_time = _parse_datetime(row["PEAK_TIME"], "PEAK_TIME", context)
            hydro_start = (
                _parse_datetime(row["HYDRO_START"], "HYDRO_START", context)
                if row["HYDRO_START"]
                else None
            )
            hydro_end = (
                _parse_datetime(row["HYDRO_END"], "HYDRO_END", context)
                if row["HYDRO_END"]
                else None
            )
            sample_start = _parse_datetime(row["SAMPLE_START"], "SAMPLE_START", context)
            sample_end = _parse_datetime(row["SAMPLE_END"], "SAMPLE_END", context)
            if rain_start > rain_end or sample_start > sample_end:
                raise ValueError(f"{context}: 洪水事件时间顺序无效")
            if hydro_start is None:
                raise ValueError(f"{context}: HYDRO_START不能为空，必须由RESPONSE_START映射")
            if hydro_start > peak_time:
                raise ValueError(f"{context}: HYDRO_START不能晚于PEAK_TIME")
            if hydro_end is not None and peak_time > hydro_end:
                raise ValueError(f"{context}: HYDRO_END不能早于PEAK_TIME")
            if not sample_start <= peak_time <= sample_end:
                raise ValueError(f"{context}: PEAK_TIME必须位于事件样本时段内")
            compound = _parse_bool(row["COMPOUND_EVENT"], "COMPOUND_EVENT", context)
            peak_count = _parse_int(row["PEAK_COUNT"], "PEAK_COUNT", context, 1)
            source_count = _parse_int(
                row["SOURCE_RAIN_EVENT_COUNT"], "SOURCE_RAIN_EVENT_COUNT", context, 1
            )
            source_ids = [
                value.strip() for value in row["SOURCE_RAIN_EVENT_IDS"].split(";") if value.strip()
            ]
            if len(source_ids) != source_count:
                raise ValueError(
                    f"{context}: SOURCE_RAIN_EVENT_IDS数量={len(source_ids)}与"
                    f"SOURCE_RAIN_EVENT_COUNT={source_count}不一致"
                )
            if peak_count != source_count:
                raise ValueError(
                    f"{context}: PEAK_COUNT={peak_count}与SOURCE_RAIN_EVENT_COUNT={source_count}不一致"
                )
            if compound != (source_count > 1):
                raise ValueError(
                    f"{context}: COMPOUND_EVENT必须等于(SOURCE_RAIN_EVENT_COUNT>1)"
                )
            events[event_id] = _Event(
                event_id,
                graph_id,
                basin_id,
                outlet_id,
                sample_start,
                sample_end,
                _required_text(row["EVENT_TYPE"], "EVENT_TYPE", context).upper(),
                _required_text(row["EVENT_GRADE"], "EVENT_GRADE", context).upper(),
                peak_time,
                peak_time.year,
            )
        if not events:
            raise ValueError(f"flood_events_final.csv没有事件记录: {path}")
        return events

    def _load_splits(self) -> dict[str, str]:
        path = self.root / "events" / "data_split.csv"
        rows = _read_csv(path, SPLIT_COLUMNS)
        result: dict[str, str] = {}
        for row in rows:
            context = _context(path, row)
            event_id = _required_text(row["EVENT_ID"], "EVENT_ID", context)
            if event_id in result:
                raise ValueError(f"{context}: data_split中EVENT_ID重复: {event_id!r}")
            if event_id not in self._events:
                raise ValueError(f"{context}: data_split引用未知EVENT_ID={event_id!r}")
            event = self._events[event_id]
            if row["GRAPH_ID"] != event.graph_id:
                raise ValueError(f"{context}: data_split的GRAPH_ID与事件表不一致")
            if row["EVENT_GRADE"].upper() != event.event_grade:
                raise ValueError(f"{context}: data_split的EVENT_GRADE与事件表不一致")
            event_year = _parse_int(row["EVENT_YEAR"], "EVENT_YEAR", context)
            if event_year != event.event_year:
                raise ValueError(
                    f"{context}: EVENT_YEAR={event_year}与事件PEAK_TIME年份{event.event_year}不一致"
                )
            result[event_id] = _normalise_split(row["SPLIT"], f"{context}: SPLIT")
        missing = sorted(set(self._events) - set(result))
        if missing:
            raise ValueError(f"data_split.csv未覆盖全部EVENT_ID，缺少={missing[:10]}")
        self._validate_split_chronology(result)
        return result

    def _validate_split_chronology(self, splits: Mapping[str, str]) -> None:
        by_graph: dict[str, dict[str, list[datetime]]] = {}
        for event_id, split in splits.items():
            event = self._events[event_id]
            by_graph.setdefault(event.graph_id, {}).setdefault(split, []).append(event.split_time)
        for graph_id, groups in by_graph.items():
            train = groups.get("TRAIN", [])
            validation = groups.get("VALIDATION", [])
            test = groups.get("TEST", [])
            if train and validation and max(train) > min(validation):
                raise ValueError(
                    f"GRAPH_ID={graph_id}: data_split不是事件级时间顺序划分，TRAIN晚于VALIDATION"
                )
            if train and test and max(train) > min(test):
                raise ValueError(
                    f"GRAPH_ID={graph_id}: data_split不是事件级时间顺序划分，TRAIN晚于TEST"
                )
            if validation and test and max(validation) > min(test):
                raise ValueError(
                    f"GRAPH_ID={graph_id}: data_split不是事件级时间顺序划分，VALIDATION晚于TEST"
                )

    def _load_samples(
        self,
        event_splits: Mapping[str, str],
        requested_history: int | None,
        requested_forecast: int | None,
    ) -> list[_Sample]:
        path = self.root / "events" / "sample_index.csv"
        rows = _read_csv(path, SAMPLE_COLUMNS)
        samples: list[_Sample] = []
        seen_ids: set[str] = set()
        all_sample_event_ids: set[str] = set()
        inferred_history = requested_history
        inferred_forecast = requested_forecast
        for row in rows:
            context = _context(path, row)
            sample_id = _required_text(row["SAMPLE_ID"], "SAMPLE_ID", context)
            if sample_id in seen_ids:
                raise ValueError(f"{context}: SAMPLE_ID重复: {sample_id!r}")
            seen_ids.add(sample_id)
            event_id = _required_text(row["EVENT_ID"], "EVENT_ID", context)
            if event_id not in self._events:
                raise ValueError(f"{context}: sample_index引用未知EVENT_ID={event_id!r}")
            all_sample_event_ids.add(event_id)
            event = self._events[event_id]
            graph_id = _required_text(row["GRAPH_ID"], "GRAPH_ID", context)
            outlet_id = _required_text(row["OUTLET_ID"], "OUTLET_ID", context)
            if graph_id != event.graph_id or outlet_id != event.outlet_id:
                raise ValueError(f"{context}: 样本GRAPH_ID/OUTLET_ID与事件表不一致")
            split = _normalise_split(row["SPLIT"], f"{context}: SPLIT")
            if split != event_splits[event_id]:
                raise ValueError(f"{context}: sample_index与data_split的SPLIT不一致")
            history = _parse_int(row["HISTORY_HOURS"], "HISTORY_HOURS", context, 1)
            forecast = _parse_int(row["FORECAST_HOURS"], "FORECAST_HOURS", context, 1)
            inferred_history = history if inferred_history is None else inferred_history
            inferred_forecast = forecast if inferred_forecast is None else inferred_forecast
            if history != inferred_history or forecast != inferred_forecast:
                raise ValueError(
                    f"{context}: 当前DataLoader要求统一窗口，期望H/F={inferred_history}/{inferred_forecast}，"
                    f"实际={history}/{forecast}"
                )
            input_start = _parse_datetime(row["INPUT_START"], "INPUT_START", context)
            forecast_time = _parse_datetime(row["FORECAST_TIME"], "FORECAST_TIME", context)
            target_end = _parse_datetime(row["TARGET_END"], "TARGET_END", context)
            if forecast_time - input_start != timedelta(hours=history - 1):
                raise ValueError(
                    f"{context}: 历史窗口含FORECAST_TIME时刻，"
                    "FORECAST_TIME-INPUT_START必须等于HISTORY_HOURS-1"
                )
            if target_end - forecast_time != timedelta(hours=forecast):
                raise ValueError(f"{context}: TARGET_END-FORECAST_TIME必须等于FORECAST_HOURS")
            if input_start < event.sample_start or target_end > event.sample_end:
                raise ValueError(f"{context}: 样本窗口超出事件SAMPLE_START/SAMPLE_END")
            targets = _parse_target_variables(row["TARGET_VARIABLE"], context)
            authoritative_targets = self.target_variables_by_graph[graph_id]
            if targets != authoritative_targets:
                raise ValueError(
                    f"{context}: TARGET_VARIABLE={sorted(targets)}与target_variable_by_graph.csv权威值"
                    f"{sorted(authoritative_targets)}不一致"
                )
            graph_selected = (
                self._requested_graph_id is None
                or graph_id == self._requested_graph_id
            )
            if (
                self._requested_targets is not None
                and graph_selected
                and targets != self._requested_targets
            ):
                raise ValueError(
                    f"{context}: TARGET_VARIABLE={sorted(targets)}与配置target_variables="
                    f"{sorted(self._requested_targets)}不一致"
                )
            sample = _Sample(
                sample_id,
                event_id,
                graph_id,
                outlet_id,
                input_start,
                forecast_time,
                target_end,
                history,
                forecast,
                targets,
                split,
            )
            eligible = (
                event.event_type in self._eligible_event_types and event.event_grade in self._eligible_event_grades
            )
            if split == self.split and eligible and graph_selected:
                samples.append(sample)
        self._all_sample_ids = frozenset(seen_ids)
        self._all_sample_event_ids = frozenset(all_sample_event_ids)
        if not samples:
            filters = (
                f"split={self.split}, graph_id={self._requested_graph_id!r}, "
                f"event_types={sorted(self._eligible_event_types)}, grades={sorted(self._eligible_event_grades)}"
            )
            raise ValueError(f"sample_index.csv在当前筛选条件下没有可用样本: {filters}")
        return samples

    def _load_dynamic_data(self, required_graphs: set[str]) -> dict[str, _DynamicGraph]:
        dynamic_dir = self.root / "dynamic"
        if not dynamic_dir.is_dir():
            raise FileNotFoundError(f"缺少dynamic目录: {dynamic_dir}")
        result: dict[str, _DynamicGraph] = {}
        seen_paths: dict[Path, str] = {}
        for expected_graph_id in sorted(required_graphs):
            basin_id = self._graphs[expected_graph_id].basin_id
            if not basin_id or Path(basin_id).name != basin_id:
                raise ValueError(f"BASIN_ID不能包含路径字符，实际={basin_id!r}")
            path = dynamic_dir / f"graph_{basin_id}_hourly.csv"
            if path in seen_paths:
                raise ValueError(
                    f"GRAPH_ID={expected_graph_id!r}与{seen_paths[path]!r}共享BASIN_ID={basin_id!r}，"
                    "但权威契约要求每个graph_<BASIN_ID>_hourly.csv只含一个GRAPH_ID"
                )
            seen_paths[path] = expected_graph_id
            cache_key = (str(self.root), expected_graph_id)
            if cache_key in self._dynamic_cache:
                result[expected_graph_id] = self._dynamic_cache[cache_key]
                continue
            rows = _read_csv(path, DYNAMIC_COLUMNS)
            if not rows:
                raise ValueError(f"逐时动态文件为空: {path}")
            graph_ids = {row["GRAPH_ID"] for row in rows}
            if "" in graph_ids or len(graph_ids) != 1:
                raise ValueError(
                    f"每个graph_<BASIN_ID>_hourly.csv必须且只能包含一个非空GRAPH_ID: {path}, 实际={graph_ids}"
                )
            graph_id = next(iter(graph_ids))
            if graph_id != expected_graph_id:
                raise ValueError(
                    f"动态文件名按BASIN_ID映射到GRAPH_ID={expected_graph_id!r}，"
                    f"但文件内GRAPH_ID={graph_id!r}: {path}"
                )
            dynamic = self._build_dynamic_graph(graph_id, path, rows)
            self._dynamic_cache[cache_key] = dynamic
            result[graph_id] = dynamic
        return result

    def _build_dynamic_graph(
        self, graph_id: str, path: Path, rows: list[dict[str, str]]
    ) -> _DynamicGraph:
        graph = self._graphs[graph_id]
        station_to_local = {node.station_id: node.node_index for node in graph.nodes}
        parsed: dict[tuple[datetime, int], tuple[float, float, float, bool, bool, bool]] = {}
        timestamps: set[datetime] = set()
        for row in rows:
            context = _context(path, row)
            station = _required_text(row["STATION_ID"], "STATION_ID", context)
            if station not in station_to_local:
                raise ValueError(f"{context}: 动态数据引用图{graph_id!r}中的未知STATION_ID={station!r}")
            timestamp = _parse_datetime(row["TIMESTAMP"], "TIMESTAMP", context)
            local = station_to_local[station]
            key = (timestamp, local)
            if key in parsed:
                raise ValueError(f"{context}: GRAPH_ID/TIMESTAMP/STATION_ID重复: {graph_id}/{timestamp}/{station}")
            rain_mask = _parse_bool(row["RAIN_MASK"], "RAIN_MASK", context)
            flow_mask = _parse_bool(row["FLOW_MASK"], "FLOW_MASK", context)
            level_mask = _parse_bool(row["WATER_LEVEL_MASK"], "WATER_LEVEL_MASK", context)
            rain = _parse_float(row["RAIN_MM"], "RAIN_MM", context) if rain_mask else 0.0
            flow = _parse_float(row["FLOW"], "FLOW", context) if flow_mask else 0.0
            level = _parse_float(row["WATER_LEVEL"], "WATER_LEVEL", context) if level_mask else 0.0
            if rain_mask and rain < 0:
                raise ValueError(f"{context}: 有效RAIN_MM不能为负，实际={rain}")
            if flow_mask and flow < 0:
                raise ValueError(f"{context}: 有效FLOW不能为负，实际={flow}")
            parsed[key] = (rain, flow, level, rain_mask, flow_mask, level_mask)
            timestamps.add(timestamp)
        ordered_times = tuple(sorted(timestamps))
        expected_rows = len(ordered_times) * len(graph.nodes)
        if len(parsed) != expected_rows:
            missing_examples: list[str] = []
            for timestamp in ordered_times:
                for node in graph.nodes:
                    if (timestamp, node.node_index) not in parsed:
                        missing_examples.append(f"{timestamp.isoformat()} / {node.station_id}")
                        if len(missing_examples) == 10:
                            break
                if len(missing_examples) == 10:
                    break
            raise ValueError(
                f"动态文件必须为每个时间提供全部节点行（缺失观测用MASK=0）；"
                f"期望{expected_rows}行，实际{len(parsed)}，缺失示例={missing_examples}: {path}"
            )
        shape = (len(ordered_times), len(graph.nodes))
        rainfall = torch.zeros(shape, dtype=torch.float32)
        flow = torch.zeros(shape, dtype=torch.float32)
        level = torch.zeros(shape, dtype=torch.float32)
        rain_mask_tensor = torch.zeros(shape, dtype=torch.bool)
        flow_mask_tensor = torch.zeros(shape, dtype=torch.bool)
        level_mask_tensor = torch.zeros(shape, dtype=torch.bool)
        for time_index, timestamp in enumerate(ordered_times):
            for node in graph.nodes:
                rain, q, z, rm, qm, zm = parsed[(timestamp, node.node_index)]
                rainfall[time_index, node.node_index] = rain
                flow[time_index, node.node_index] = q
                level[time_index, node.node_index] = z
                rain_mask_tensor[time_index, node.node_index] = rm
                flow_mask_tensor[time_index, node.node_index] = qm
                level_mask_tensor[time_index, node.node_index] = zm
        return _DynamicGraph(
            ordered_times,
            {timestamp: index for index, timestamp in enumerate(ordered_times)},
            rainfall,
            flow,
            level,
            rain_mask_tensor,
            flow_mask_tensor,
            level_mask_tensor,
        )

    def _window_indices(self, sample: _Sample) -> tuple[list[int], list[int]]:
        dynamic = self._dynamic[sample.graph_id]
        history_times = [sample.input_start + timedelta(hours=offset) for offset in range(sample.history_hours)]
        # Forecast horizon 1..F: FORECAST_TIME itself is the forecast origin.
        future_times = [
            sample.forecast_time + timedelta(hours=offset)
            for offset in range(1, sample.forecast_hours + 1)
        ]
        missing = [time for time in (*history_times, *future_times) if time not in dynamic.time_to_index]
        if missing:
            rendered = [time.isoformat(sep=" ") for time in missing[:10]]
            raise ValueError(
                f"SAMPLE_ID={sample.sample_id}: 动态文件缺少样本所需整点{rendered}"
                + ("（仅显示前10项）" if len(missing) > 10 else "")
            )
        return (
            [dynamic.time_to_index[time] for time in history_times],
            [dynamic.time_to_index[time] for time in future_times],
        )

    def _validate_all_windows(self) -> None:
        for sample in self._samples:
            history_indices, future_indices = self._window_indices(sample)
            dynamic = self._dynamic[sample.graph_id]
            outlet = next(
                node.node_index for node in self._graphs[sample.graph_id].nodes if node.station_id == sample.outlet_id
            )
            if "FLOW" in sample.target_variables and not dynamic.flow_mask[future_indices, outlet].any():
                raise ValueError(f"SAMPLE_ID={sample.sample_id}: 未来窗口出口FLOW全部缺失，不能作为FLOW训练样本")
            if "WATER_LEVEL" in sample.target_variables and not dynamic.water_level_mask[future_indices, outlet].any():
                raise ValueError(
                    f"SAMPLE_ID={sample.sample_id}: 未来窗口出口WATER_LEVEL全部缺失，不能作为水位训练样本"
                )
            if len(history_indices) != self.history_hours or len(future_indices) != self.forecast_hours:
                raise AssertionError("内部窗口长度校验失败")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> GraphEventBatch:
        sample = self._samples[index]
        graph = self._graphs[sample.graph_id]
        dynamic = self._dynamic[sample.graph_id]
        history_indices, future_indices = self._window_indices(sample)
        history_flow = dynamic.flow[history_indices].clone()
        history_level = dynamic.water_level[history_indices].clone()
        q_mask = dynamic.flow_mask[history_indices].clone()
        z_mask = dynamic.water_level_mask[history_indices].clone()
        feature_values: list[torch.Tensor] = []
        for feature in self.dynamic_features:
            if feature == "FLOW":
                value = history_flow
                mask = q_mask
            elif feature == "WATER_LEVEL":
                value = history_level
                mask = z_mask
            elif feature == "RAIN_MM":
                value = dynamic.rainfall[history_indices]
                mask = dynamic.rainfall_mask[history_indices]
            elif feature == "FLOW_MASK":
                feature_values.append(q_mask.float())
                continue
            elif feature == "WATER_LEVEL_MASK":
                feature_values.append(z_mask.float())
                continue
            elif feature == "RAIN_MASK":
                feature_values.append(dynamic.rainfall_mask[history_indices].float())
                continue
            else:  # Guarded by feature-schema validation.
                raise AssertionError(f"未实现动态特征{feature}")
            if self.normalize_dynamic:
                value, _ = self.normalization.transform(feature, value, mask)
            else:
                value = torch.where(mask, value.float(), torch.zeros_like(value, dtype=torch.float32))
            feature_values.append(value)
        dynamic_features = torch.stack(feature_values, dim=-1)

        history_rain = dynamic.rainfall[history_indices].clone()
        history_rain_mask = dynamic.rainfall_mask[history_indices].clone()
        history_rain = torch.where(history_rain_mask, history_rain, torch.zeros_like(history_rain))
        if self.future_rainfall_mode == "observed_hindcast":
            future_rain = dynamic.rainfall[future_indices].clone()
            future_rain_mask = dynamic.rainfall_mask[future_indices].clone()
            future_rain = torch.where(future_rain_mask, future_rain, torch.zeros_like(future_rain))
        elif self.future_rainfall_mode == "zero":
            future_rain = torch.zeros(
                (sample.forecast_hours, len(graph.nodes)), dtype=torch.float32
            )
            future_rain_mask = torch.zeros_like(future_rain, dtype=torch.bool)
        elif self.future_rainfall_mode == "persistence":
            persisted = torch.zeros(len(graph.nodes), dtype=torch.float32)
            for node_index in range(len(graph.nodes)):
                valid_positions = history_rain_mask[:, node_index].nonzero(as_tuple=False).flatten()
                if valid_positions.numel():
                    persisted[node_index] = history_rain[valid_positions[-1], node_index]
            future_rain = persisted.unsqueeze(0).expand(sample.forecast_hours, -1).clone()
            # False means this forcing is neither an observation nor a supplied
            # forecast product; the value is an explicit operational baseline.
            future_rain_mask = torch.zeros_like(future_rain, dtype=torch.bool)
        else:
            raise AssertionError(f"未知future_rainfall_mode={self.future_rainfall_mode}")
        rainfall = torch.cat((history_rain, future_rain), dim=0).unsqueeze(-1)
        rainfall_mask = torch.cat((history_rain_mask, future_rain_mask), dim=0).unsqueeze(-1)

        nodes = len(graph.nodes)
        q_target = torch.zeros((sample.forecast_hours, nodes), dtype=torch.float32)
        z_target = torch.zeros_like(q_target)
        q_target_mask = torch.zeros((sample.forecast_hours, nodes), dtype=torch.bool)
        z_target_mask = torch.zeros_like(q_target_mask)
        outlet = next(node.node_index for node in graph.nodes if node.station_id == sample.outlet_id)
        if "FLOW" in sample.target_variables:
            q_values = dynamic.flow[future_indices, outlet]
            q_valid = dynamic.flow_mask[future_indices, outlet]
            q_target[:, outlet] = torch.where(q_valid, q_values, torch.zeros_like(q_values))
            q_target_mask[:, outlet] = q_valid
        if "WATER_LEVEL" in sample.target_variables:
            z_values = dynamic.water_level[future_indices, outlet]
            z_valid = dynamic.water_level_mask[future_indices, outlet]
            z_target[:, outlet] = torch.where(z_valid, z_values, torch.zeros_like(z_values))
            z_target_mask[:, outlet] = z_valid

        station_ids = tuple(node.station_id for node in graph.nodes)
        station_index = torch.tensor(
            [self.station_to_index[station] for station in station_ids], dtype=torch.long
        )
        return GraphEventBatch(
            dynamic_node_features=dynamic_features,
            rainfall=rainfall,
            node_static=graph.node_static,
            edge_index=graph.edge_index,
            edge_static=graph.edge_static,
            q_history=history_flow,
            z_history=history_level,
            q_mask=q_mask,
            z_mask=z_mask,
            q_target=q_target,
            z_target=z_target,
            q_target_mask=q_target_mask,
            z_target_mask=z_target_mask,
            node_mask=torch.ones(nodes, dtype=torch.bool),
            event_mask=torch.tensor(True),
            rainfall_mask=rainfall_mask,
            station_index=station_index,
            node_area_km2=graph.node_area_km2,
            station_ids=station_ids,
            sample_id=sample.sample_id,
            event_id=sample.event_id,
            graph_id=sample.graph_id,
        )


def collate_hunan_graph_events(items: list[GraphEventBatch]) -> GraphEventBatch:
    """Collate samples from one graph and reject accidental graph mixing."""
    if not items:
        raise ValueError("不能collate空样本列表")
    graph_ids = [item.graph_id for item in items]
    if any(not isinstance(graph_id, str) for graph_id in graph_ids):
        raise ValueError("正式样本必须携带字符串graph_id")
    if len(set(graph_ids)) != 1:
        raise ValueError(f"一个batch只能包含同一GRAPH_ID，实际={graph_ids}")
    for name in ("node_static", "edge_index", "edge_static", "station_index", "node_area_km2"):
        first = getattr(items[0], name)
        if first is None or any(value is None or not torch.equal(first, value) for value in (getattr(item, name) for item in items[1:])):
            raise ValueError(f"同图batch内{name}不一致")
    if any(item.station_ids != items[0].station_ids for item in items[1:]):
        raise ValueError("同图batch内station_ids不一致")

    static_names = {
        "node_static",
        "edge_index",
        "edge_static",
        "station_index",
        "node_area_km2",
        "station_ids",
    }
    metadata_names = {"sample_id", "event_id", "graph_id"}
    kwargs: dict[str, Any] = {}
    for name in GraphEventBatch.__dataclass_fields__:
        values = [getattr(item, name) for item in items]
        if name in static_names:
            kwargs[name] = values[0]
        elif name in metadata_names:
            kwargs[name] = tuple(values)
        elif values[0] is None:
            if any(value is not None for value in values):
                raise ValueError(f"batch内{name}有的样本为None、有的不是")
            kwargs[name] = None
        else:
            kwargs[name] = torch.stack(values)
    batch = GraphEventBatch(**kwargs)
    validate_batch(batch)
    return batch


class GraphGroupedBatchSampler(Sampler[list[int]]):
    """Yield deterministic same-graph mini-batches for variable-size graphs."""

    def __init__(
        self,
        dataset: HunanGraphEventDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)
        groups: dict[str, list[int]] = {}
        for index in range(len(dataset)):
            groups.setdefault(dataset.graph_id_for_index(index), []).append(index)
        self._groups = groups

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.generator.manual_seed(self.seed + self.epoch)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for indices in self._groups.values():
            ordered = list(indices)
            if self.shuffle:
                permutation = torch.randperm(len(ordered), generator=self.generator).tolist()
                ordered = [ordered[position] for position in permutation]
            for start in range(0, len(ordered), self.batch_size):
                batch = ordered[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[position] for position in permutation]
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self._groups.values())
        return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self._groups.values())


def build_hunan_loader(
    dataset: HunanGraphEventDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader[GraphEventBatch]:
    """Build a DataLoader whose batches are safe for shared graph tensors."""
    sampler = GraphGroupedBatchSampler(dataset, batch_size, shuffle, drop_last, seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_hunan_graph_events,
        generator=sampler.generator,
    )
