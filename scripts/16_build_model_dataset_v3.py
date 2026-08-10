#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第16步：整理山洪预报模型数据集
================================

将 MERIT_workflow 已有结果整理为模型可直接读取的数据集：

1. 图结构：节点目录、边拓扑；
2. 静态属性：节点10项核心属性、边静态属性；
3. 动态数据：逐图、逐时、逐节点的降雨/流量/水位及缺失掩膜；
4. 洪水事件：按实测水文过程合并重复候选事件；
5. 水位质控：只用暂定TRAIN识别不可恢复的站内基准断裂，事件级排除并留痕；
6. 数据划分：在真实独立事件层重新执行时间顺序训练/验证/测试划分；
7. 样本索引：历史24小时预测未来1—6小时，不提前复制滑动窗口；
8. 标准化参数：仅使用最终训练集时间窗计算；
9. 质控和来源清单。

运行环境：Python 3，依赖 pandas、numpy。
不依赖 ArcPy，不修改第06—15步源文件；所有修复仅写入新的第16步输出目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import shutil
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("build_model_dataset")
CSV_ENCODING = "utf-8-sig"
NEGATIVE_FLOW_AS_MISSING_STATIONS = frozenset({"611E2950"})
TUKEY_OUTER_FENCE_MULTIPLIER = 3.0

EVENT_OVERLAP_COLUMNS = [
    "GRAPH_ID",
    "EVENT_ID_A",
    "EVENT_ID_B",
    "target_station_id",
    "target_variable",
    "hydro_start_A",
    "hydro_end_A",
    "hydro_start_B",
    "hydro_end_B",
    "overlap_hours",
    "overlap_fraction",
    "official_hydro_overlap_hours",
    "hydro_gap_hours",
    "same_observed_peak_time",
    "peak_time_A",
    "peak_time_B",
    "observed_peak_A",
    "observed_peak_B",
    "split_A",
    "split_B",
    "cross_split",
    "status",
    "reason",
    "suggested_action",
]

WATER_LEVEL_REFERENCE_COLUMNS = [
    "GRAPH_ID",
    "STATION_ID",
    "EVENT_ID",
    "SPLIT",
    "TARGET_HOUR_COUNT",
    "MIN_Z",
    "MAX_Z",
    "MEDIAN_Z",
    "TRAIN_EVENT_COUNT",
    "TRAIN_MEDIAN_Q1",
    "TRAIN_MEDIAN_Q3",
    "TRAIN_MEDIAN_IQR",
    "TRAIN_OUTER_FENCE_LOW",
    "TRAIN_OUTER_FENCE_HIGH",
    "OUTSIDE_TRAIN_OUTER_FENCE",
    "QC_STATUS",
    "QC_REASON",
    "ACTION",
    "MASKED_SOURCE_ROW_COUNT",
]

SAMPLE_REJECTION_COLUMNS = [
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
]


# -----------------------------------------------------------------------------
# 通用工具
# -----------------------------------------------------------------------------

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers[:] = []
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(sh)
    LOG.addHandler(fh)


def clean_column_name(value: object) -> str:
    return str(value).replace("\ufeff", "").replace("\u3000", " ").strip()


def column_key(value: object) -> str:
    return re.sub(r"[\s_\-./\\()（）]+", "", clean_column_name(value)).lower()


def normalize_id(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if re.fullmatch(r"[0-9]+\.0+", text):
        text = text.split(".", 1)[0]
    return text


def safe_filename(value: object) -> str:
    text = normalize_id(value) or "unknown"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)


def parse_bool_value(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "是", "复合", "compound"}




def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, nrows=nrows)
    else:
        last: Optional[Exception] = None
        df = None
        for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=None,
                    engine="python",
                    nrows=nrows,
                    # IDs such as 611E2950 are station codes, not scientific
                    # notation.  Keep raw CSV tokens textual and convert each
                    # numeric field explicitly at its canonicalization point.
                    dtype=str,
                )
                break
            except UnicodeDecodeError as exc:
                last = exc
            except pd.errors.ParserError as exc:
                last = exc
        if df is None:
            raise RuntimeError("无法读取文件 {}：{}".format(path, last))
    df.columns = [clean_column_name(c) for c in df.columns]
    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING, float_format="%.10g")


def resolve_column(
    df: pd.DataFrame,
    aliases: Sequence[str],
    label: str,
    required: bool = True,
) -> Optional[str]:
    lookup: Dict[str, str] = {}
    for col in df.columns:
        lookup.setdefault(column_key(col), col)
    for alias in aliases:
        key = column_key(alias)
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(
            "{}未找到字段。候选字段={}；实际字段={}".format(
                label, list(aliases), list(df.columns)
            )
        )
    return None


def find_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path and path.exists():
            return path
    return None


def recursive_find_first(base: Path, names: Sequence[str]) -> Optional[Path]:
    if not base.exists():
        return None
    wanted = {n.lower() for n in names}
    for path in sorted(base.rglob("*")):
        if path.is_file() and path.name.lower() in wanted:
            return path
    return None


def require_path(path: Optional[Path], label: str) -> Path:
    if path is None or not path.exists():
        raise FileNotFoundError("未找到{}：{}".format(label, path))
    return path


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_local_time_series(series: pd.Series) -> pd.Series:
    """
    解析水文/降雨时间：
    - 带 Z 或时区偏移的时间视为UTC/显式时区，转换到北京时间后去时区；
    - 无时区时间按北京时间本地时间处理。
    """
    raw = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    tz_mask = raw.str.contains(r"(?:Z|[+-][0-9]{2}:?[0-9]{2})$", regex=True, na=False)
    if tz_mask.any():
        aware = pd.to_datetime(raw[tz_mask], errors="coerce", utc=True)
        result.loc[tz_mask] = aware.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    if (~tz_mask).any():
        result.loc[~tz_mask] = pd.to_datetime(raw[~tz_mask], errors="coerce")
    return result


def floor_hour(series: pd.Series) -> pd.Series:
    return series.dt.floor("h")


def merge_intervals(intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    valid = sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in intervals if pd.notna(a) and pd.notna(b) and a <= b)
    if not valid:
        return []
    merged: List[List[pd.Timestamp]] = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        if start <= merged[-1][1] + pd.Timedelta(hours=1):
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def timestamps_from_intervals(intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DatetimeIndex:
    pieces = [pd.date_range(a, b, freq="h") for a, b in merge_intervals(intervals)]
    if not pieces:
        return pd.DatetimeIndex([])
    values = pieces[0]
    for part in pieces[1:]:
        values = values.union(part)
    return values.sort_values()


def file_fingerprint(path: Path) -> Dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_time": pd.Timestamp(stat.st_mtime, unit="s").isoformat(),
        "sha256": digest.hexdigest(),
    }


# -----------------------------------------------------------------------------
# 路径发现
# -----------------------------------------------------------------------------

@dataclass
class SourcePaths:
    project_root: Path
    workflow_dir: Path
    hydro_dir: Path
    rain_dir: Path
    events_csv: Path
    edges_csv: Path
    node_static_csv: Path
    edge_static_csv: Optional[Path]
    output_dir: Path


def discover_sources(args: argparse.Namespace) -> SourcePaths:
    project_root = Path(args.project_root).resolve()
    workflow = Path(args.workflow_dir).resolve() if args.workflow_dir else project_root / "Arcgis" / "MERIT_workflow"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else workflow / "16_model_dataset"

    hydro_dir = Path(args.hydro_data).resolve() if args.hydro_data else project_root / "归档_正确解压" / "河道"

    if args.node_rain_dir:
        rain_dir = Path(args.node_rain_dir).resolve()
    else:
        rain_candidates = [
            workflow / "11_node_rain",
            workflow / "11_node_basin_hourly_rain",
            workflow / "11_build_node_basin_hourly_rain_sparse",
        ]
        rain_dir = next((p for p in rain_candidates if p.exists()), workflow / "11_node_rain")
        if not rain_dir.exists():
            idx = recursive_find_first(workflow, ["node_hourly_rain_index.csv", "node_rain_index.csv"])
            if idx:
                rain_dir = idx.parent

    if args.events_csv:
        events = Path(args.events_csv).resolve()
    else:
        events = find_existing([
            workflow / "13_final_flood_events" / "final_flood_events.csv",
            workflow / "10_final_flood_events" / "final_flood_events.csv",
            workflow / "13_match_rain_hydro_flood_events" / "final_flood_events.csv",
        ]) or recursive_find_first(workflow, ["final_flood_events.csv"])

    if args.edges_csv:
        edges = Path(args.edges_csv).resolve()
    else:
        edges = find_existing([
            workflow / "06_topology" / "edges.csv",
            workflow / "06_topology" / "topology_edges.csv",
        ]) or recursive_find_first(workflow, ["edges.csv", "topology_edges.csv"])

    if args.node_static_csv:
        node_static = Path(args.node_static_csv).resolve()
    else:
        # 优先使用新版 Python 3 第15步最终合并表；只有不存在时才退回旧目录。
        node_static = find_existing([
            workflow / "15_static_attributes_ccam_py3" / "node_static_attributes_final.csv",
            workflow / "15_static_attributes_ccam_py3" / "node_static_attributes_ccam.csv",
            workflow / "15_static_attributes_ccam" / "node_static_attributes_final.csv",
            workflow / "15_static_attributes_ccam" / "node_static_attributes_ccam.csv",
            workflow / "14_static_attributes_dem_cisc_py3" / "node_static_attributes_dem_cisc.csv",
            workflow / "14_static_attributes_dem_cisc" / "node_static_attributes_dem_cisc.csv",
        ]) or recursive_find_first(workflow, [
            "node_static_attributes_final.csv",
            "node_static_attributes_ccam.csv",
            "node_static_attributes_dem_cisc.csv",
        ])

    if args.edge_static_csv:
        edge_static = Path(args.edge_static_csv).resolve()
    else:
        # 边属性来自第14步；同样优先新版 Python 3 输出。
        edge_static = find_existing([
            workflow / "14_static_attributes_dem_cisc_py3" / "edge_static_attributes_dem.csv",
            workflow / "15_static_attributes_ccam_py3" / "edge_static_attributes_final.csv",
            workflow / "14_static_attributes_dem_cisc" / "edge_static_attributes_dem.csv",
            workflow / "15_static_attributes_ccam" / "edge_static_attributes_final.csv",
        ]) or recursive_find_first(workflow, [
            "edge_static_attributes_dem.csv",
            "edge_static_attributes_final.csv",
        ])

    return SourcePaths(
        project_root=project_root,
        workflow_dir=require_path(workflow, "MERIT_workflow目录"),
        hydro_dir=require_path(hydro_dir, "河道站数据目录"),
        rain_dir=require_path(rain_dir, "第11步节点逐时降雨目录"),
        events_csv=require_path(events, "第13步最终洪水事件表"),
        edges_csv=require_path(edges, "第06步边拓扑表"),
        node_static_csv=require_path(node_static, "节点静态属性表"),
        edge_static_csv=edge_static if edge_static and edge_static.exists() else None,
        output_dir=output_dir,
    )


# -----------------------------------------------------------------------------
# 静态节点和图结构
# -----------------------------------------------------------------------------

NODE_FEATURE_ALIASES: Dict[str, Sequence[str]] = {
    "log_incremental_area": ["log_incremental_area", "log_incr_area", "log_incremental_area_km2"],
    "log_upstream_area": ["log_upstream_area", "log_upa", "log_upstream_area_km2"],
    "mean_hillslope_flow_distance_m": [
        "mean_hillslope_flow_distance_m", "mean_hillslope_flow_distance", "hillslope_flow_distance_m"
    ],
    "mean_slope_deg": ["mean_slope_deg", "mean_slope", "slope_mean_deg"],
    "elevation_std_m": ["elevation_std_m", "elevation_std", "dem_std_m"],
    "drainage_density_km_per_km2": [
        "drainage_density_km_per_km2", "drainage_density", "drainage_density_km_km2"
    ],
    "soil_log_ksat_0_30cm": [
        "soil_log_ksat_0_30cm", "soil_log_ksat", "log_ksat_0_30cm"
    ],
    "soil_profile_depth_cm": [
        "soil_profile_depth_cm", "soil_profile_depth", "soil_storage", "pdep_mean", "pdep"
    ],
    "forest_fraction": ["forest_fraction", "forest_frac"],
    "impervious_fraction": ["impervious_fraction", "impervious_frac", "impervious_rate"],
}

EDGE_FEATURE_ALIASES: Dict[str, Sequence[str]] = {
    "reach_length_km": ["reach_length_km", "river_distance_km", "reach_length", "edge_length_km"],
    "reach_slope_m_per_m": ["reach_slope_m_per_m", "reach_slope", "slope_m_per_m"],
}


def canonicalize_node_static(path: Path, allow_missing: bool) -> Tuple[pd.DataFrame, List[str]]:
    df = read_table(path)
    basin_col = resolve_column(df, ["BASIN_ID", "basin_id", "GRAPH_ID"], "节点静态属性-BASIN_ID")
    station_col = resolve_column(df, ["STATION_ID", "station_id", "NODE_ID", "node_id", "STCD"], "节点静态属性-STATION_ID")
    outlet_col = resolve_column(df, ["OUTLET_ID", "outlet_id", "OUTLET_STATION_ID"], "节点静态属性-OUTLET_ID", required=False)
    role_col = resolve_column(df, ["ROLE", "role", "NODE_ROLE"], "节点静态属性-ROLE", required=False)
    upa_col = resolve_column(df, ["UPA_KM2", "upstream_area_km2", "upa_km2"], "节点静态属性-UPA_KM2", required=False)
    qc_col = resolve_column(df, ["attribute_qc", "ATTRIBUTE_QC", "QC_STAT", "qc_stat", "CCAM_QC"], "节点静态属性-QC", required=False)

    out = pd.DataFrame({
        "GRAPH_ID": df[basin_col].map(normalize_id),
        "BASIN_ID": df[basin_col].map(normalize_id),
        "STATION_ID": df[station_col].map(normalize_id),
        "OUTLET_ID": df[outlet_col].map(normalize_id) if outlet_col else "",
        "ROLE": df[role_col].astype(str).str.strip() if role_col else "NODE",
        "UPSTREAM_AREA_KM2": to_numeric(df[upa_col]) if upa_col else np.nan,
        "STATIC_QC": df[qc_col].astype(str).str.strip() if qc_col else "",
    })

    missing: List[str] = []
    for canonical, aliases in NODE_FEATURE_ALIASES.items():
        col = resolve_column(df, aliases, "节点属性-{}".format(canonical), required=False)
        if col:
            out[canonical] = to_numeric(df[col])
        else:
            out[canonical] = np.nan
            missing.append(canonical)

    out = out[(out["GRAPH_ID"] != "") & (out["STATION_ID"] != "")].copy()
    out = out.drop_duplicates(["GRAPH_ID", "STATION_ID"], keep="first")

    # 如果上一步未写对数面积，但有原始面积，自动补算。
    if out["log_incremental_area"].isna().all():
        area_col = resolve_column(df, ["incremental_area_km2", "INCR_KM2", "incr_km2"], "增量面积", required=False)
        if area_col:
            area = to_numeric(df.loc[out.index, area_col]).clip(lower=0)
            out["log_incremental_area"] = np.log1p(area)
            if "log_incremental_area" in missing:
                missing.remove("log_incremental_area")
    if out["log_upstream_area"].isna().all() and out["UPSTREAM_AREA_KM2"].notna().any():
        out["log_upstream_area"] = np.log1p(out["UPSTREAM_AREA_KM2"].clip(lower=0))
        if "log_upstream_area" in missing:
            missing.remove("log_upstream_area")

    if missing and not allow_missing:
        raise ValueError(
            "节点静态属性缺少核心字段：{}。请先完成第14、15步，或使用 --allow-missing-static。实际文件：{}"
            .format(", ".join(missing), path)
        )
    return out.reset_index(drop=True), missing


def canonicalize_edges(path: Path) -> pd.DataFrame:
    df = read_table(path)
    basin_col = resolve_column(df, ["BASIN_ID", "basin_id", "GRAPH_ID"], "边表-BASIN_ID", required=False)
    from_col = resolve_column(
        df,
        ["FROM_STATION", "FROM_STATION_ID", "UPSTREAM_STATION_ID", "FROM_NODE", "UP_NODE", "SOURCE"],
        "边表-上游节点",
    )
    to_col = resolve_column(
        df,
        ["TO_STATION", "TO_STATION_ID", "DOWNSTREAM_STATION_ID", "TO_NODE", "DOWN_NODE", "TARGET"],
        "边表-下游节点",
    )
    out = pd.DataFrame({
        "GRAPH_ID": df[basin_col].map(normalize_id) if basin_col else "",
        "FROM_STATION": df[from_col].map(normalize_id),
        "TO_STATION": df[to_col].map(normalize_id),
    })
    for canonical, aliases in EDGE_FEATURE_ALIASES.items():
        col = resolve_column(df, aliases, "边属性-{}".format(canonical), required=False)
        out[canonical] = to_numeric(df[col]) if col else np.nan
    out = out[(out["FROM_STATION"] != "") & (out["TO_STATION"] != "")].copy()
    return out.drop_duplicates(["GRAPH_ID", "FROM_STATION", "TO_STATION"], keep="first")


def merge_edge_static(edges: pd.DataFrame, edge_static_path: Optional[Path]) -> pd.DataFrame:
    if edge_static_path is None:
        return edges
    stat = canonicalize_edges(edge_static_path)
    keys = ["FROM_STATION", "TO_STATION"]
    if edges["GRAPH_ID"].ne("").any() and stat["GRAPH_ID"].ne("").any():
        keys = ["GRAPH_ID"] + keys
    feature_cols = list(EDGE_FEATURE_ALIASES)
    stat = stat[keys + feature_cols].copy()
    merged = edges.merge(stat, on=keys, how="left", suffixes=("", "_STATIC"))
    for col in feature_cols:
        sc = col + "_STATIC"
        if sc in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[sc])
            merged = merged.drop(columns=[sc])
    return merged


def topological_order(nodes: Sequence[str], edges: pd.DataFrame, area_lookup: Dict[str, float]) -> Tuple[List[str], bool]:
    node_set = set(nodes)
    indeg = {n: 0 for n in node_set}
    adj: Dict[str, List[str]] = defaultdict(list)
    for row in edges.itertuples(index=False):
        u = normalize_id(row.FROM_STATION)
        v = normalize_id(row.TO_STATION)
        if u in node_set and v in node_set and u != v:
            adj[u].append(v)
            indeg[v] += 1
    queue = [n for n in node_set if indeg[n] == 0]
    queue.sort(key=lambda n: (area_lookup.get(n, float("inf")), n))
    dq = deque(queue)
    order: List[str] = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in sorted(adj.get(u, [])):
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    acyclic = len(order) == len(node_set)
    if not acyclic:
        remaining = sorted(node_set - set(order), key=lambda n: (area_lookup.get(n, float("inf")), n))
        order.extend(remaining)
    return order, acyclic


def build_graph_tables(node_static: pd.DataFrame, edges: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    station_to_graphs: Dict[str, Set[str]] = defaultdict(set)
    for row in node_static[["GRAPH_ID", "STATION_ID"]].itertuples(index=False):
        station_to_graphs[row.STATION_ID].add(row.GRAPH_ID)

    # 若原边表没有BASIN_ID，根据节点归属推断。
    inferred_rows: List[dict] = []
    for row in edges.itertuples(index=False):
        graph = normalize_id(row.GRAPH_ID)
        if not graph:
            common = station_to_graphs.get(row.FROM_STATION, set()) & station_to_graphs.get(row.TO_STATION, set())
            if len(common) == 1:
                graph = next(iter(common))
        record = row._asdict()
        record["GRAPH_ID"] = graph
        inferred_rows.append(record)
    edges = pd.DataFrame(inferred_rows)
    edges = edges[edges["GRAPH_ID"] != ""].copy()

    catalogs: List[pd.DataFrame] = []
    edge_outputs: List[pd.DataFrame] = []
    graph_qc: List[dict] = []

    for graph_id, nodes_df in node_static.groupby("GRAPH_ID", sort=True):
        graph_edges = edges[edges["GRAPH_ID"] == graph_id].copy()
        nodes = nodes_df["STATION_ID"].tolist()
        area_lookup = dict(zip(nodes_df["STATION_ID"], nodes_df["UPSTREAM_AREA_KM2"]))
        order, acyclic = topological_order(nodes, graph_edges, area_lookup)
        index_map = {sid: idx for idx, sid in enumerate(order)}

        cat = nodes_df.set_index("STATION_ID").loc[order].reset_index()
        cat["NODE_INDEX"] = cat["STATION_ID"].map(index_map).astype(int)
        role_upper = cat["ROLE"].astype(str).str.upper()
        cat["IS_OUTLET"] = (
            (cat["STATION_ID"] == cat["OUTLET_ID"]) |
            role_upper.str.contains("OUTLET", na=False) |
            role_upper.str.contains("出口", na=False)
        ).astype(int)
        catalogs.append(cat)

        valid_edge = graph_edges[
            graph_edges["FROM_STATION"].isin(index_map) & graph_edges["TO_STATION"].isin(index_map)
        ].copy()
        valid_edge["FROM_NODE"] = valid_edge["FROM_STATION"].map(index_map).astype(int)
        valid_edge["TO_NODE"] = valid_edge["TO_STATION"].map(index_map).astype(int)
        edge_outputs.append(valid_edge)
        graph_qc.append({
            "GRAPH_ID": graph_id,
            "NODE_COUNT": len(cat),
            "EDGE_COUNT": len(valid_edge),
            "TOPOLOGY_ACYCLIC": int(acyclic),
            "OUTLET_COUNT": int(cat["IS_OUTLET"].sum()),
            "GRAPH_QC": "ACCEPT" if acyclic and int(cat["IS_OUTLET"].sum()) >= 1 else "REVIEW",
        })

    catalog = pd.concat(catalogs, ignore_index=True) if catalogs else pd.DataFrame()
    edge_topology = pd.concat(edge_outputs, ignore_index=True) if edge_outputs else pd.DataFrame()
    graph_qc_df = pd.DataFrame(graph_qc)

    node_features = ["GRAPH_ID", "BASIN_ID", "NODE_INDEX", "STATION_ID", "OUTLET_ID", "ROLE", "IS_OUTLET", "STATIC_QC"] + list(NODE_FEATURE_ALIASES)
    node_static_out = catalog[[c for c in node_features if c in catalog.columns]].copy()

    edge_cols = [
        "GRAPH_ID", "FROM_NODE", "TO_NODE", "FROM_STATION", "TO_STATION"
    ] + list(EDGE_FEATURE_ALIASES)
    edge_static_out = edge_topology[[c for c in edge_cols if c in edge_topology.columns]].copy()
    return catalog, edge_topology, node_static_out, edge_static_out.merge(graph_qc_df[["GRAPH_ID", "GRAPH_QC"]], on="GRAPH_ID", how="left")


# -----------------------------------------------------------------------------
# 洪水事件和划分
# -----------------------------------------------------------------------------

def canonicalize_events(path: Path) -> pd.DataFrame:
    df = read_table(path)
    aliases = {
        "EVENT_ID": ["EVENT_ID", "event_id", "FINAL_EVENT_ID", "rain_event_id", "HYDRO_EVENT_ID"],
        "GRAPH_ID": ["GRAPH_ID", "BASIN_ID", "basin_id"],
        "OUTLET_ID": ["OUTLET_ID", "outlet_id", "STATION_ID"],
        "RAIN_START": ["RAIN_START", "rain_start"],
        "RAIN_END": ["RAIN_END", "rain_end"],
        "HYDRO_START": [
            "HYDRO_START", "hydro_start", "RESPONSE_START", "response_start",
            "RISE_START", "rise_start",
        ],
        "PEAK_TIME": ["PEAK_TIME", "peak_time"],
        "HYDRO_END": [
            "HYDRO_END", "hydro_end", "EVENT_END", "event_end",
            "RECESSION_END", "recession_end",
        ],
        "SAMPLE_START": ["SAMPLE_START", "sample_start"],
        "SAMPLE_END": ["SAMPLE_END", "sample_end"],
        "EVENT_TYPE": ["EVENT_TYPE", "event_type", "HYDRO_CLASS", "hydro_class", "CLASS"],
        "EVENT_GRADE": ["EVENT_GRADE", "event_grade", "GRADE", "grade"],
        "COMPOUND_EVENT": ["COMPOUND_EVENT", "compound_event", "IS_COMPOUND"],
        "PEAK_COUNT": ["PEAK_COUNT", "peak_count"],
        "SOURCE_RAIN_EVENT_IDS": ["SOURCE_RAIN_EVENT_IDS", "source_rain_event_ids"],
        "SOURCE_RAIN_EVENT_COUNT": ["SOURCE_RAIN_EVENT_COUNT", "source_rain_event_count"],
    }
    required_fields = {
        "EVENT_ID", "GRAPH_ID", "OUTLET_ID", "RAIN_START", "RAIN_END",
        "HYDRO_START", "PEAK_TIME", "HYDRO_END", "SAMPLE_START", "SAMPLE_END",
        "EVENT_TYPE", "EVENT_GRADE", "COMPOUND_EVENT", "PEAK_COUNT",
        "SOURCE_RAIN_EVENT_IDS", "SOURCE_RAIN_EVENT_COUNT",
    }
    out = pd.DataFrame(index=df.index)
    for canonical, names in aliases.items():
        required = canonical in required_fields
        col = resolve_column(df, names, "事件字段-{}".format(canonical), required=required)
        if col:
            out[canonical] = df[col]
        else:
            out[canonical] = np.nan

    out["GRAPH_ID"] = out["GRAPH_ID"].map(normalize_id)
    out["BASIN_ID"] = out["GRAPH_ID"]
    out["OUTLET_ID"] = out["OUTLET_ID"].map(normalize_id)
    for col in ["RAIN_START", "RAIN_END", "HYDRO_START", "PEAK_TIME", "HYDRO_END", "SAMPLE_START", "SAMPLE_END"]:
        out[col] = parse_local_time_series(out[col])
    out["SAMPLE_START"] = out["SAMPLE_START"].where(out["SAMPLE_START"].notna(), out["RAIN_START"] - pd.Timedelta(hours=24))
    out["SAMPLE_END"] = out["SAMPLE_END"].where(out["SAMPLE_END"].notna(), out["HYDRO_END"])
    out["SAMPLE_END"] = out["SAMPLE_END"].where(out["SAMPLE_END"].notna(), out["RAIN_END"] + pd.Timedelta(hours=48))
    out["EVENT_TYPE"] = out["EVENT_TYPE"].fillna("HYDRO_FLOOD").astype(str).str.strip().str.upper()
    out["EVENT_GRADE"] = out["EVENT_GRADE"].fillna("").astype(str).str.strip().str.upper()
    out["COMPOUND_EVENT"] = out["COMPOUND_EVENT"].map(parse_bool_value).astype(bool)
    out["PEAK_COUNT"] = to_numeric(out["PEAK_COUNT"])
    out["SOURCE_RAIN_EVENT_IDS"] = out["SOURCE_RAIN_EVENT_IDS"].fillna("").astype(str).str.strip()
    out["SOURCE_RAIN_EVENT_COUNT"] = to_numeric(out["SOURCE_RAIN_EVENT_COUNT"])

    # 缺少事件ID时自动生成稳定ID。
    missing_id = out["EVENT_ID"].isna() | (out["EVENT_ID"].astype(str).str.strip() == "")
    out["EVENT_ID"] = out["EVENT_ID"].astype(str).str.strip()
    for idx in out.index[missing_id]:
        stamp = out.at[idx, "RAIN_START"]
        stamp_text = stamp.strftime("%Y%m%d%H") if pd.notna(stamp) else "unknown"
        out.at[idx, "EVENT_ID"] = "{}_{}".format(out.at[idx, "GRAPH_ID"], stamp_text)
    out = out.dropna(subset=[
        "RAIN_START", "RAIN_END", "HYDRO_START", "PEAK_TIME", "SAMPLE_START", "SAMPLE_END"
    ])
    out = out[(out["GRAPH_ID"] != "") & (out["OUTLET_ID"] != "")].copy()

    duplicate_ids = out[out.duplicated("EVENT_ID", keep=False)]["EVENT_ID"].tolist()
    if duplicate_ids:
        raise ValueError("事件EVENT_ID不唯一：{}".format(duplicate_ids[:20]))

    invalid_time = (
        (out["RAIN_START"] > out["RAIN_END"])
        | (out["SAMPLE_START"] > out["SAMPLE_END"])
        | (out["HYDRO_START"] > out["PEAK_TIME"])
        | (out["HYDRO_END"].notna() & (out["PEAK_TIME"] > out["HYDRO_END"]))
        | (out["PEAK_TIME"] < out["SAMPLE_START"])
        | (out["PEAK_TIME"] > out["SAMPLE_END"])
    )
    if invalid_time.any():
        bad = out.loc[invalid_time, [
            "EVENT_ID", "HYDRO_START", "PEAK_TIME", "HYDRO_END", "SAMPLE_START", "SAMPLE_END"
        ]].head(20).to_dict("records")
        raise ValueError("模型事件时间契约无效：{}".format(bad))

    counts = out["SOURCE_RAIN_EVENT_COUNT"]
    peak_counts = out["PEAK_COUNT"]
    non_integer = counts.isna() | (counts < 1) | (counts % 1 != 0)
    non_integer |= peak_counts.isna() | (peak_counts < 1) | (peak_counts % 1 != 0)
    source_id_counts = out["SOURCE_RAIN_EVENT_IDS"].map(
        lambda value: len([item for item in value.split(";") if item.strip()])
    )
    invalid_compound = (
        non_integer
        | (source_id_counts != counts.fillna(-1))
        | (peak_counts != counts)
        | (out["COMPOUND_EVENT"] != (counts > 1))
    )
    if invalid_compound.any():
        bad = out.loc[invalid_compound, [
            "EVENT_ID", "COMPOUND_EVENT", "PEAK_COUNT",
            "SOURCE_RAIN_EVENT_IDS", "SOURCE_RAIN_EVENT_COUNT",
        ]].head(20).to_dict("records")
        raise ValueError("复合事件字段不一致：{}".format(bad))
    out["PEAK_COUNT"] = peak_counts.astype(int)
    out["SOURCE_RAIN_EVENT_COUNT"] = counts.astype(int)
    out = out.reset_index(drop=True)
    return out


def filter_events(events: pd.DataFrame, event_types: Sequence[str], grades: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    type_set = {x.upper() for x in event_types}
    grade_set = {x.upper() for x in grades}
    keep_type = events["EVENT_TYPE"].isin(type_set) if type_set else pd.Series(True, index=events.index)
    if grade_set and events["EVENT_GRADE"].ne("").any():
        keep_grade = events["EVENT_GRADE"].isin(grade_set)
    else:
        keep_grade = pd.Series(True, index=events.index)
    keep = keep_type & keep_grade
    selected = events[keep].copy()
    excluded = events[~keep].copy()
    excluded["EXCLUSION_REASON"] = np.where(~keep_type[~keep], "EVENT_TYPE", "EVENT_GRADE")
    return selected.reset_index(drop=True), excluded.reset_index(drop=True)


def assign_splits(events: pd.DataFrame, train_fraction: float, val_fraction: float) -> pd.DataFrame:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction必须位于0和1之间。")
    if not 0 <= val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("val_fraction无效，且train+val必须小于1。")
    out = events.copy()
    sort_time = out["PEAK_TIME"].where(out["PEAK_TIME"].notna(), out["RAIN_START"])
    out["_SORT_TIME"] = sort_time
    out = out.sort_values(["_SORT_TIME", "GRAPH_ID", "EVENT_ID"]).reset_index(drop=True)
    n = len(out)
    n_train = max(1, int(math.floor(n * train_fraction))) if n else 0
    n_val = int(math.floor(n * val_fraction))
    if n >= 3 and val_fraction > 0 and n_val == 0:
        n_val = 1
    if n_train + n_val >= n and n >= 2:
        n_train = max(1, n - n_val - 1)
    out["SPLIT"] = "TEST"
    out.loc[: n_train - 1, "SPLIT"] = "TRAIN"
    if n_val > 0:
        out.loc[n_train : n_train + n_val - 1, "SPLIT"] = "VALIDATION"
    out["SPLIT_REASON"] = "GLOBAL_CHRONOLOGICAL_EVENT_SPLIT"
    out["EVENT_YEAR"] = out["_SORT_TIME"].dt.year
    return out.drop(columns=["_SORT_TIME"])


# -----------------------------------------------------------------------------
# 节点逐时降雨
# -----------------------------------------------------------------------------

@dataclass
class RainSource:
    path: Path
    station_hint: str = ""
    graph_hint: str = ""
    start_hint: Optional[pd.Timestamp] = None
    end_hint: Optional[pd.Timestamp] = None


def infer_id_from_filename(path: Path, candidate_ids: Set[str]) -> str:
    stem = path.stem
    tail = normalize_id(stem.rsplit("_", 1)[-1])
    if tail in candidate_ids:
        return tail
    for sid in sorted(candidate_ids, key=len, reverse=True):
        if re.search(r"(?<![0-9A-Za-z]){}(?![0-9A-Za-z])".format(re.escape(sid)), stem, flags=re.I):
            return sid
    return ""


def discover_rain_sources(rain_dir: Path, node_ids: Set[str], graph_ids: Set[str]) -> List[RainSource]:
    index_files = [
        p for p in rain_dir.rglob("*.csv")
        if "index" in p.name.lower() and "node" in p.name.lower()
    ]
    sources: List[RainSource] = []
    seen: Set[Path] = set()

    for idx_path in sorted(index_files):
        try:
            idx = read_table(idx_path)
            path_col = resolve_column(idx, ["FILE", "FILE_PATH", "CSV_PATH", "OUTPUT_FILE", "PATH"], "降雨索引-文件路径", required=False)
            station_col = resolve_column(idx, ["STATION_ID", "NODE_ID", "station_id", "node_id"], "降雨索引-节点", required=False)
            graph_col = resolve_column(idx, ["BASIN_ID", "GRAPH_ID", "basin_id"], "降雨索引-流域", required=False)
            start_col = resolve_column(idx, ["START_TIME", "start_time", "DATA_START"], "降雨索引-开始", required=False)
            end_col = resolve_column(idx, ["END_TIME", "end_time", "DATA_END"], "降雨索引-结束", required=False)
            if path_col is None:
                continue
            for _, values in idx.iterrows():
                raw_path = values.get(path_col, "")
                if pd.isna(raw_path) or not str(raw_path).strip():
                    continue
                p = Path(str(raw_path))
                if not p.is_absolute():
                    p = (idx_path.parent / p).resolve()
                if not p.exists() or p in seen:
                    continue
                sid = normalize_id(values.get(station_col, "")) if station_col else ""
                gid = normalize_id(values.get(graph_col, "")) if graph_col else ""
                start = pd.to_datetime(values.get(start_col), errors="coerce") if start_col else pd.NaT
                end = pd.to_datetime(values.get(end_col), errors="coerce") if end_col else pd.NaT
                sources.append(RainSource(p, sid, gid, start if pd.notna(start) else None, end if pd.notna(end) else None))
                seen.add(p)
        except Exception as exc:
            LOG.warning("读取节点降雨索引失败，转为扫描数据文件：%s；%s", idx_path, exc)

    # 索引不完整时扫描CSV。跳过明确的流域汇总、事件和索引表。
    for path in sorted(rain_dir.rglob("*.csv")):
        lname = path.name.lower()
        if path in seen or "index" in lname or "candidate" in lname or "event" in lname:
            continue
        if "basin_hourly" in lname and "node" not in lname:
            continue
        try:
            header = read_table(path, nrows=5)
            time_col = resolve_column(
                header, ["TIMESTAMP", "OBS_TIME", "TIME", "TM", "DATETIME"],
                "降雨文件-时间", required=False
            )
            start_col = resolve_column(
                header, ["start_time", "START_TIME", "RAIN_START", "interval_start"],
                "降雨文件-区间开始", required=False
            )
            end_col = resolve_column(
                header, ["end_time", "END_TIME", "RAIN_END", "interval_end"],
                "降雨文件-区间结束", required=False
            )
            rain_col = resolve_column(
                header, [
                    "area_rain_mm", "AREA_RAIN_MM", "RAIN_MM", "NODE_RAIN_MM",
                    "PRECIP_MM", "P", "RAINFALL"
                ],
                "降雨文件-雨量", required=False
            )
            node_col = resolve_column(
                header, ["STATION_ID", "NODE_ID", "station_id", "node_id"],
                "降雨文件-节点", required=False
            )
            sid_hint = infer_id_from_filename(path, node_ids)
            has_time = bool(time_col or (start_col and end_col))
            if has_time and rain_col and (node_col or sid_hint):
                sources.append(RainSource(path, sid_hint, ""))
                seen.add(path)
        except Exception:
            continue

    if not sources:
        raise FileNotFoundError(
            "在{}中没有识别到节点逐时降雨文件。需要包含时间、节点编号和雨量字段，或文件名包含STATION_ID。"
            .format(rain_dir)
        )
    LOG.info("识别到%d个节点逐时降雨数据源。", len(sources))
    return sources


def load_node_rain(
    sources: Sequence[RainSource],
    needed_nodes: Set[str],
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取第11步节点稀疏小时面雨量。

    当前工作流的真实格式为：
      basin_id,outlet_id,node_id,start_time,end_time,area_rain_mm,...

    第10步已经仅保留 interval_hours≈1 的正雨量区间，第11步固定权重聚合后仍以
    [start_time, end_time) 表示一个小时。因此模型时间戳使用 start_time；未出现的
    节点-小时按第10/11步设计解释为0雨量，而不是缺失值。
    """
    frames: List[pd.DataFrame] = []
    coverage_rows: List[dict] = []
    seen_nodes: Set[str] = set()
    interval_rows = 0
    legacy_rows = 0
    skipped_nonhourly = 0

    for i, source in enumerate(sources, 1):
        try:
            df = read_table(source.path)
            node_col = resolve_column(
                df, ["STATION_ID", "NODE_ID", "station_id", "node_id"],
                "节点降雨-节点", required=False
            )
            if node_col:
                sid = df[node_col].map(normalize_id)
            else:
                hint = source.station_hint or infer_id_from_filename(source.path, needed_nodes)
                if not hint:
                    raise ValueError("节点降雨文件既没有node_id/STATION_ID，文件名也无法识别节点编号。")
                sid = pd.Series(hint, index=df.index, dtype="object")

            # 第11步真实字段优先：start_time/end_time + area_rain_mm。
            start_col = resolve_column(
                df, ["start_time", "START_TIME", "RAIN_START", "interval_start"],
                "节点降雨-开始时间", required=False
            )
            end_col = resolve_column(
                df, ["end_time", "END_TIME", "RAIN_END", "interval_end"],
                "节点降雨-结束时间", required=False
            )
            interval_rain_col = resolve_column(
                df, ["area_rain_mm", "AREA_RAIN_MM"],
                "节点降雨-区间面雨量", required=False
            )

            if start_col and end_col and interval_rain_col:
                start_time = parse_local_time_series(df[start_col])
                end_time = parse_local_time_series(df[end_col])
                duration_h = (end_time - start_time).dt.total_seconds() / 3600.0
                hourly = duration_h.between(0.99, 1.01, inclusive="both")
                bad = int((duration_h.notna() & ~hourly).sum())
                skipped_nonhourly += bad
                if bad:
                    LOG.warning("%s 有%d条非1小时区间，已跳过。", source.path.name, bad)
                rain = to_numeric(df[interval_rain_col])
                part = pd.DataFrame({
                    "STATION_ID": sid,
                    # 与第12步 rain_start 的时间定义保持一致，不向后平移1小时。
                    "TIMESTAMP": floor_hour(start_time),
                    "INTERVAL_END": end_time,
                    "RAIN_MM": rain,
                })
                part = part[hourly.fillna(False)].copy()
                interval_rows += len(part)
                source_mode = "STEP11_START_END_AREA_RAIN"
            else:
                # 向后兼容旧版单时间戳格式。
                time_col = resolve_column(
                    df, ["TIMESTAMP", "OBS_TIME", "TIME", "TM", "DATETIME"],
                    "节点降雨-时间"
                )
                rain_col = resolve_column(
                    df, ["RAIN_MM", "NODE_RAIN_MM", "PRECIP_MM", "P", "RAINFALL"],
                    "节点降雨-雨量"
                )
                time = parse_local_time_series(df[time_col])
                rain = to_numeric(df[rain_col])
                part = pd.DataFrame({
                    "STATION_ID": sid,
                    "TIMESTAMP": floor_hour(time),
                    "INTERVAL_END": pd.NaT,
                    "RAIN_MM": rain,
                })
                legacy_rows += len(part)
                source_mode = "LEGACY_TIMESTAMP_RAIN"

            part = part[part["STATION_ID"].isin(needed_nodes)]
            part = part[(part["TIMESTAMP"] >= min_time) & (part["TIMESTAMP"] <= max_time)]
            part = part.dropna(subset=["TIMESTAMP", "RAIN_MM"])
            part = part[part["RAIN_MM"] >= 0].copy()

            if not part.empty:
                frames.append(part[["STATION_ID", "TIMESTAMP", "RAIN_MM"]])
                seen_nodes.update(part["STATION_ID"].unique().tolist())
                for sid_value, group in part.groupby("STATION_ID"):
                    # 这里只记录“正雨量记录范围”用于审计；动态表不再用它裁剪RAIN_MASK。
                    coverage_rows.append({
                        "STATION_ID": sid_value,
                        "SOURCE_FILE": str(source.path),
                        "SOURCE_MODE": source_mode,
                        "FIRST_RECORDED_RAIN_HOUR": group["TIMESTAMP"].min(),
                        "LAST_RECORDED_RAIN_HOUR": group["TIMESTAMP"].max(),
                        "NONZERO_OR_RECORDED_HOURS": int(group["TIMESTAMP"].nunique()),
                        "SPARSE_MISSING_MEANS_ZERO": 1,
                    })
        except Exception as exc:
            LOG.warning("跳过无法读取的节点降雨文件：%s；%s", source.path, exc)
        if i == 1 or i % 20 == 0 or i == len(sources):
            LOG.info("读取节点稀疏小时降雨 [%d/%d]", i, len(sources))

    if not frames:
        raise ValueError(
            "节点降雨数据在目标节点和事件时间范围内没有记录。"
            "第11步应使用start_time/end_time/area_rain_mm格式。"
        )

    rain = pd.concat(frames, ignore_index=True)
    # 第11步按 basin_id,node_id,start_time,end_time 应唯一；若存在重复文件，取均值并记录警告。
    dup_count = int(rain.duplicated(["STATION_ID", "TIMESTAMP"], keep=False).sum())
    if dup_count:
        LOG.warning("节点降雨发现%d条重复节点-小时记录，将按小时取均值。", dup_count)
    rain = rain.groupby(["STATION_ID", "TIMESTAMP"], as_index=False)["RAIN_MM"].mean()
    rain["RAIN_MM"] = rain["RAIN_MM"].clip(lower=0)

    coverage = pd.DataFrame(coverage_rows)
    if not coverage.empty:
        coverage = coverage.groupby("STATION_ID", as_index=False).agg(
            FIRST_RECORDED_RAIN_HOUR=("FIRST_RECORDED_RAIN_HOUR", "min"),
            LAST_RECORDED_RAIN_HOUR=("LAST_RECORDED_RAIN_HOUR", "max"),
            RAIN_SOURCE_FILE_COUNT=("SOURCE_FILE", "nunique"),
            RAIN_RECORDED_HOURS=("NONZERO_OR_RECORDED_HOURS", "sum"),
            SPARSE_MISSING_MEANS_ZERO=("SPARSE_MISSING_MEANS_ZERO", "max"),
        )

    missing_nodes = sorted(set(needed_nodes) - seen_nodes)
    if missing_nodes:
        LOG.warning(
            "有%d个模型节点在第11步稀疏表中从未出现正雨量记录：%s。"
            "这些节点的RAIN_MASK将保持0，请核对第09/11步权重与雨量覆盖。",
            len(missing_nodes), ", ".join(missing_nodes[:30])
        )
    LOG.info(
        "节点降雨读取完成：有效节点%d/%d，正雨量节点-小时%d；第11步区间记录%d，旧格式记录%d，跳过非1小时%d。",
        len(seen_nodes), len(needed_nodes), len(rain), interval_rows, legacy_rows, skipped_nonhourly
    )
    return rain, coverage


# -----------------------------------------------------------------------------
# 河道站水文数据
# -----------------------------------------------------------------------------

def discover_hydro_files(hydro_dir: Path, needed_nodes: Set[str]) -> Tuple[List[Tuple[Path, str]], pd.DataFrame]:
    selected: List[Tuple[Path, str]] = []
    audit: List[dict] = []
    all_files = [p for p in hydro_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".xlsx", ".xls"}]
    for path in sorted(all_files):
        sid = infer_id_from_filename(path, needed_nodes)
        if sid:
            selected.append((path, sid))
            status = "SELECTED"
        else:
            status = "SKIPPED_NON_MODEL_NODE"
        audit.append({"FILE": str(path), "PARSED_STATION_ID": sid, "STATUS": status})
    if not selected:
        raise FileNotFoundError("未从河道文件名中识别到任何模型节点文件。文件名应包含‘站名_站号’。")
    LOG.info("河道目录共%d个文件，按文件名筛选模型节点文件%d个。", len(all_files), len(selected))
    return selected, pd.DataFrame(audit)


def load_hydro(
    files: Sequence[Tuple[Path, str]],
    needed_nodes: Set[str],
    min_time: pd.Timestamp,
    max_time: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    audit: List[dict] = []
    for i, (path, sid_hint) in enumerate(files, 1):
        try:
            df = read_table(path)
            station_col = resolve_column(df, ["STCD", "STATION_ID", "station_id", "站码", "测站编码"], "河道数据-站号", required=False)
            time_col = resolve_column(df, ["TM", "OBS_TIME", "TIME", "DATETIME", "时间", "观测时间"], "河道数据-时间")
            flow_col = resolve_column(df, ["Q", "FLOW", "flow", "流量"], "河道数据-流量", required=False)
            level_col = resolve_column(df, ["Z", "WATER_LEVEL", "water_level", "水位"], "河道数据-水位", required=False)
            fetch_col = resolve_column(df, ["记录抓取时间", "FETCH_TIME", "fetch_time", "抓取时间"], "河道数据-抓取时间", required=False)
            sid_series = df[station_col].map(normalize_id) if station_col else pd.Series(sid_hint, index=df.index)
            # 文件名是预处理流程中的主要站号；内部站号不一致时只保留提示，不改变归属。
            internal = sorted(set(sid_series.dropna()) - {""})
            mismatch = bool(internal and sid_hint not in internal)
            time = parse_local_time_series(df[time_col])
            part = pd.DataFrame({
                "STATION_ID": sid_hint,
                "TIMESTAMP_RAW": time,
                "FLOW": to_numeric(df[flow_col]) if flow_col else np.nan,
                "WATER_LEVEL": to_numeric(df[level_col]) if level_col else np.nan,
            })
            # Frozen data rule: negative discharge from station 611E2950 is an
            # invalid observation, not zero flow and not grounds for deleting
            # the station.  Other stations are not silently reinterpreted; if
            # a retained negative reaches model windows, the strict loader will
            # reject the candidate and force an explicit data decision.
            negative_flow_observed = part["FLOW"].notna() & (part["FLOW"] < 0)
            negative_flow_invalid = negative_flow_observed & (
                part["STATION_ID"].isin(NEGATIVE_FLOW_AS_MISSING_STATIONS)
            )
            negative_flow_raw_count = int(negative_flow_invalid.sum())
            negative_flow_retained_raw_count = int((negative_flow_observed & ~negative_flow_invalid).sum())
            part["_FLOW_WAS_NEGATIVE_INVALID"] = negative_flow_invalid.astype(np.int8)
            part["_FLOW_NEGATIVE_RETAINED"] = (negative_flow_observed & ~negative_flow_invalid).astype(np.int8)
            part.loc[negative_flow_invalid, "FLOW"] = np.nan
            if fetch_col:
                part["FETCH_TIME"] = parse_local_time_series(df[fetch_col])
            else:
                part["FETCH_TIME"] = part["TIMESTAMP_RAW"]
            part = part.dropna(subset=["TIMESTAMP_RAW"])
            part = part[(part["TIMESTAMP_RAW"] >= min_time - pd.Timedelta(hours=1)) & (part["TIMESTAMP_RAW"] <= max_time + pd.Timedelta(hours=1))]
            negative_flow_in_range_count = int(part["_FLOW_WAS_NEGATIVE_INVALID"].sum())
            negative_flow_retained_in_range_count = int(part["_FLOW_NEGATIVE_RETAINED"].sum())
            part["TIMESTAMP"] = floor_hour(part["TIMESTAMP_RAW"])
            part["COMPLETENESS"] = part[["FLOW", "WATER_LEVEL"]].notna().sum(axis=1)
            part = part.sort_values(["TIMESTAMP", "COMPLETENESS", "FETCH_TIME"])
            part = part.drop_duplicates(["STATION_ID", "TIMESTAMP"], keep="last")
            negative_flow_selected_count = int(part["_FLOW_WAS_NEGATIVE_INVALID"].sum())
            negative_flow_retained_selected_count = int(part["_FLOW_NEGATIVE_RETAINED"].sum())
            frames.append(part[["STATION_ID", "TIMESTAMP", "FLOW", "WATER_LEVEL"]])
            audit.append({
                "FILE": str(path),
                "STATION_ID": sid_hint,
                "ROW_COUNT_RAW": len(df),
                "ROW_COUNT_USED": len(part),
                "INTERNAL_ID_MISMATCH": int(mismatch),
                "HAS_FLOW_FIELD": int(flow_col is not None),
                "HAS_LEVEL_FIELD": int(level_col is not None),
                "NEGATIVE_FLOW_RAW_AS_MISSING": negative_flow_raw_count,
                "NEGATIVE_FLOW_IN_RANGE_AS_MISSING": negative_flow_in_range_count,
                "NEGATIVE_FLOW_SELECTED_AS_MISSING": negative_flow_selected_count,
                "NEGATIVE_FLOW_RAW_RETAINED": negative_flow_retained_raw_count,
                "NEGATIVE_FLOW_IN_RANGE_RETAINED": negative_flow_retained_in_range_count,
                "NEGATIVE_FLOW_SELECTED_RETAINED": negative_flow_retained_selected_count,
                "STATUS": "LOADED",
            })
        except Exception as exc:
            audit.append({"FILE": str(path), "STATION_ID": sid_hint, "STATUS": "ERROR", "ERROR": str(exc)})
            LOG.warning("河道文件读取失败：%s；%s", path, exc)
        if i == 1 or i % 20 == 0 or i == len(files):
            LOG.info("读取河道水文 [%d/%d]", i, len(files))
    if not frames:
        raise ValueError("没有加载到可用河道水文记录。")
    hydro = pd.concat(frames, ignore_index=True)
    hydro = hydro[hydro["STATION_ID"].isin(needed_nodes)]
    hydro = hydro.groupby(["STATION_ID", "TIMESTAMP"], as_index=False).agg(
        FLOW=("FLOW", "last"),
        WATER_LEVEL=("WATER_LEVEL", "last"),
    )
    return hydro, pd.DataFrame(audit)


# -----------------------------------------------------------------------------
# 动态文件、目标变量和样本索引
# -----------------------------------------------------------------------------

def event_intervals_by_graph(events: pd.DataFrame) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    result: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for row in events.itertuples(index=False):
        result[row.GRAPH_ID].append((pd.Timestamp(row.SAMPLE_START).floor("h"), pd.Timestamp(row.SAMPLE_END).ceil("h")))
    return result


def make_dynamic_tables(
    catalog: pd.DataFrame,
    events: pd.DataFrame,
    rain: pd.DataFrame,
    rain_coverage: pd.DataFrame,
    hydro: pd.DataFrame,
    dynamic_dir: Optional[Path],
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    intervals = event_intervals_by_graph(events)
    # 第10/11步是“仅保存正雨量”的稀疏产品；缺失的节点-小时代表0雨量。
    # 只要该节点在第11步稀疏数据中出现过，就在所有事件样本小时把降雨视为有效。
    rain_valid_nodes: Set[str] = set()
    if not rain_coverage.empty and "STATION_ID" in rain_coverage.columns:
        rain_valid_nodes = set(rain_coverage["STATION_ID"].map(normalize_id)) - {""}
    rain_idx = rain.set_index(["STATION_ID", "TIMESTAMP"])["RAIN_MM"]
    hydro_idx = hydro.set_index(["STATION_ID", "TIMESTAMP"])[["FLOW", "WATER_LEVEL"]]

    dynamic: Dict[str, pd.DataFrame] = {}
    qc_rows: List[dict] = []
    if dynamic_dir is not None:
        dynamic_dir.mkdir(parents=True, exist_ok=True)

    for graph_id, nodes_df in catalog.groupby("GRAPH_ID", sort=True):
        times = timestamps_from_intervals(intervals.get(graph_id, []))
        if len(times) == 0:
            continue
        node_ids = nodes_df.sort_values("NODE_INDEX")["STATION_ID"].tolist()
        multi = pd.MultiIndex.from_product([times, node_ids], names=["TIMESTAMP", "STATION_ID"])
        df = pd.DataFrame(index=multi).reset_index()
        df["GRAPH_ID"] = graph_id
        node_index_map = dict(zip(nodes_df["STATION_ID"], nodes_df["NODE_INDEX"]))
        df["NODE_INDEX"] = df["STATION_ID"].map(node_index_map).astype(int)

        keys = pd.MultiIndex.from_frame(df[["STATION_ID", "TIMESTAMP"]])
        df["RAIN_MM"] = rain_idx.reindex(keys).to_numpy()
        df["FLOW"] = hydro_idx["FLOW"].reindex(keys).to_numpy()
        df["WATER_LEVEL"] = hydro_idx["WATER_LEVEL"].reindex(keys).to_numpy()

        # 第10/11步明确采用“正雨量稀疏存储”：没有记录的小时隐式贡献0。
        # 因此不能用“首次正雨~末次正雨”作为覆盖期，否则事件前的干旱小时会被误判为缺失。
        df["RAIN_MASK"] = df["STATION_ID"].isin(rain_valid_nodes).astype(np.int8)
        fill_zero = df["RAIN_MM"].isna() & (df["RAIN_MASK"] == 1)
        df.loc[fill_zero, "RAIN_MM"] = 0.0
        df["FLOW_MASK"] = df["FLOW"].notna().astype(np.int8)
        df["WATER_LEVEL_MASK"] = df["WATER_LEVEL"].notna().astype(np.int8)

        cols = [
            "GRAPH_ID", "TIMESTAMP", "NODE_INDEX", "STATION_ID",
            "RAIN_MM", "FLOW", "WATER_LEVEL", "RAIN_MASK", "FLOW_MASK", "WATER_LEVEL_MASK"
        ]
        df = df[cols].sort_values(["TIMESTAMP", "NODE_INDEX"]).reset_index(drop=True)
        path = (
            dynamic_dir / "graph_{}_hourly.csv".format(safe_filename(graph_id))
            if dynamic_dir is not None
            else None
        )
        if path is not None:
            write_csv(df, path)
        dynamic[graph_id] = df

        outlet_nodes = nodes_df[nodes_df["IS_OUTLET"] == 1]["STATION_ID"].tolist()
        outlet = outlet_nodes[0] if outlet_nodes else ""
        outlet_df = df[df["STATION_ID"] == outlet] if outlet else pd.DataFrame()
        qc_rows.append({
            "GRAPH_ID": graph_id,
            "DYNAMIC_FILE": str(path) if path is not None else "",
            "NODE_COUNT": len(node_ids),
            "TIMESTAMP_COUNT": len(times),
            "ROW_COUNT": len(df),
            "RAIN_VALID_FRACTION": float(df["RAIN_MASK"].mean()) if len(df) else np.nan,
            "OUTLET_ID": outlet,
            "OUTLET_FLOW_COVERAGE": float(outlet_df["FLOW_MASK"].mean()) if len(outlet_df) else 0.0,
            "OUTLET_LEVEL_COVERAGE": float(outlet_df["WATER_LEVEL_MASK"].mean()) if len(outlet_df) else 0.0,
        })
        if path is not None:
            LOG.info("输出动态图 %s：%d节点，%d小时，%d行。", graph_id, len(node_ids), len(times), len(df))
    return dynamic, pd.DataFrame(qc_rows)


def choose_target_by_graph(dynamic_qc: pd.DataFrame, mode: str, min_flow_coverage: float) -> pd.DataFrame:
    rows: List[dict] = []
    mode = mode.upper()
    for row in dynamic_qc.itertuples(index=False):
        if mode == "FLOW":
            target = "FLOW"
        elif mode == "WATER_LEVEL":
            target = "WATER_LEVEL"
        else:
            target = "FLOW" if float(row.OUTLET_FLOW_COVERAGE) >= min_flow_coverage else "WATER_LEVEL"
        coverage = float(row.OUTLET_FLOW_COVERAGE if target == "FLOW" else row.OUTLET_LEVEL_COVERAGE)
        rows.append({
            "GRAPH_ID": row.GRAPH_ID,
            "OUTLET_ID": row.OUTLET_ID,
            "TARGET_VARIABLE": target,
            "TARGET_COVERAGE": coverage,
            "TARGET_SELECTION_REASON": "REQUESTED" if mode != "AUTO" else ("FLOW_COVERAGE_OK" if target == "FLOW" else "FLOW_INCOMPLETE_USE_LEVEL"),
        })
    return pd.DataFrame(rows)


def build_sample_index(
    events: pd.DataFrame,
    dynamic: Dict[str, pd.DataFrame],
    target_map: pd.DataFrame,
    history_hours: int,
    forecast_hours: int,
    step_hours: int,
    min_target_coverage: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_lookup = target_map.set_index("GRAPH_ID").to_dict("index")
    rows: List[dict] = []
    rejected: List[dict] = []
    sample_counter = 1
    rejection_counter = 1

    def reject(
        event: object,
        reason: str,
        *,
        outlet: str = "",
        target: str = "",
        issue: object = pd.NaT,
        target_start: object = pd.NaT,
        target_end: object = pd.NaT,
        target_coverage: object = np.nan,
    ) -> None:
        nonlocal rejection_counter
        rejected.append({
            "REJECTION_ID": "R{:09d}".format(rejection_counter),
            "SAMPLE_ID": "",
            "EVENT_ID": event.EVENT_ID,
            "GRAPH_ID": event.GRAPH_ID,
            "OUTLET_ID": outlet,
            "FORECAST_TIME": issue,
            "TARGET_START": target_start,
            "TARGET_END": target_end,
            "TARGET_VARIABLE": target,
            "TARGET_COVERAGE": target_coverage,
            "MIN_TARGET_COVERAGE": min_target_coverage,
            "REASON": reason,
            "SPLIT": event.SPLIT,
        })
        rejection_counter += 1

    for event in events.itertuples(index=False):
        graph_id = event.GRAPH_ID
        graph_dynamic = dynamic.get(graph_id)
        target_info = target_lookup.get(graph_id)
        if graph_dynamic is None or not target_info:
            reject(event, "MISSING_DYNAMIC_OR_TARGET")
            continue
        outlet = normalize_id(event.OUTLET_ID) or normalize_id(target_info["OUTLET_ID"])
        target = target_info["TARGET_VARIABLE"]
        mask_col = "FLOW_MASK" if target == "FLOW" else "WATER_LEVEL_MASK"
        out_df = graph_dynamic[graph_dynamic["STATION_ID"] == outlet].set_index("TIMESTAMP")
        if out_df.empty:
            reject(event, "OUTLET_NOT_IN_DYNAMIC", outlet=outlet, target=target)
            continue

        first_issue = max(
            pd.Timestamp(event.SAMPLE_START).ceil("h") + pd.Timedelta(hours=history_hours),
            pd.Timestamp(event.RAIN_START).floor("h"),
        )
        event_end = pd.Timestamp(event.HYDRO_END) if pd.notna(event.HYDRO_END) else pd.Timestamp(event.SAMPLE_END)
        last_issue = min(
            pd.Timestamp(event.SAMPLE_END).floor("h") - pd.Timedelta(hours=forecast_hours),
            event_end.floor("h"),
        )
        if first_issue > last_issue:
            reject(event, "EVENT_WINDOW_TOO_SHORT", outlet=outlet, target=target)
            continue

        for issue in pd.date_range(first_issue, last_issue, freq="{}h".format(step_hours)):
            target_start = issue + pd.Timedelta(hours=1)
            target_end = issue + pd.Timedelta(hours=forecast_hours)
            target_times = pd.date_range(target_start, target_end, freq="h")
            masks = out_df[mask_col].reindex(target_times).fillna(0)
            target_coverage = float(masks.mean()) if len(masks) else 0.0
            if target_coverage < min_target_coverage:
                reject(
                    event,
                    "TARGET_COVERAGE_BELOW_THRESHOLD",
                    outlet=outlet,
                    target=target,
                    issue=issue,
                    target_start=target_start,
                    target_end=target_end,
                    target_coverage=target_coverage,
                )
                continue
            input_start = issue - pd.Timedelta(hours=history_hours - 1)
            rows.append({
                "SAMPLE_ID": "S{:09d}".format(sample_counter),
                "EVENT_ID": event.EVENT_ID,
                "GRAPH_ID": graph_id,
                "OUTLET_ID": outlet,
                "INPUT_START": input_start,
                "FORECAST_TIME": issue,
                "TARGET_START": target_start,
                "TARGET_END": target_end,
                "HISTORY_HOURS": history_hours,
                "FORECAST_HOURS": forecast_hours,
                "TARGET_VARIABLE": target,
                "TARGET_COVERAGE": target_coverage,
                "SPLIT": event.SPLIT,
            })
            sample_counter += 1

    sample_columns = [
        "SAMPLE_ID", "EVENT_ID", "GRAPH_ID", "OUTLET_ID", "INPUT_START",
        "FORECAST_TIME", "TARGET_START", "TARGET_END", "HISTORY_HOURS",
        "FORECAST_HOURS", "TARGET_VARIABLE", "TARGET_COVERAGE", "SPLIT",
    ]
    samples = pd.DataFrame(rows, columns=sample_columns)
    rejected_df = pd.DataFrame(rejected, columns=SAMPLE_REJECTION_COLUMNS)

    event_ids = set(events["EVENT_ID"].astype(str))
    sampled_event_ids = set(samples["EVENT_ID"].astype(str)) if not samples.empty else set()
    rejected_event_ids = set(rejected_df["EVENT_ID"].astype(str)) if not rejected_df.empty else set()
    unaccounted = sorted(event_ids - sampled_event_ids - rejected_event_ids)
    if unaccounted:
        raise RuntimeError("final事件既无样本也无拒绝记录：{}".format(unaccounted[:20]))
    return samples, rejected_df


# -----------------------------------------------------------------------------
# 事件过程合并与水位基准质控
# -----------------------------------------------------------------------------

def target_hours_from_samples(samples: pd.DataFrame) -> pd.DatetimeIndex:
    hours = pd.DatetimeIndex([])
    for row in samples.itertuples(index=False):
        part = pd.date_range(pd.Timestamp(row.TARGET_START), pd.Timestamp(row.TARGET_END), freq="h")
        hours = hours.union(part)
    return hours.sort_values()


def tukey_outer_fence(values: pd.Series) -> Optional[Tuple[float, float, float, float, float]]:
    numeric = to_numeric(values).dropna()
    if len(numeric) < 4:
        return None
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    return (
        q1,
        q3,
        iqr,
        q1 - TUKEY_OUTER_FENCE_MULTIPLIER * iqr,
        q3 + TUKEY_OUTER_FENCE_MULTIPLIER * iqr,
    )


def build_event_overlap_audit(
    events: pd.DataFrame,
    samples: pd.DataFrame,
    dynamic: Dict[str, pd.DataFrame],
    target_map: pd.DataFrame,
    forecast_hours: int,
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """用正式目标小时和实测峰时审计不同EVENT_ID是否属于同一水文过程。"""
    target_lookup = target_map.set_index("GRAPH_ID").to_dict("index")
    event_lookup = events.set_index("EVENT_ID")
    event_info: Dict[str, dict] = {}

    for (graph_id, event_id), event_samples in samples.groupby(["GRAPH_ID", "EVENT_ID"], sort=True):
        if event_id not in event_lookup.index:
            raise ValueError("sample_index引用未知EVENT_ID={}".format(event_id))
        event = event_lookup.loc[event_id]
        target = str(target_lookup[graph_id]["TARGET_VARIABLE"]).upper()
        station = normalize_id(event.OUTLET_ID)
        graph_dynamic = dynamic.get(graph_id)
        if graph_dynamic is None:
            continue
        hours = target_hours_from_samples(event_samples)
        value_col = "FLOW" if target == "FLOW" else "WATER_LEVEL"
        mask_col = value_col + "_MASK"
        observed = graph_dynamic[
            (graph_dynamic["STATION_ID"] == station)
            & (graph_dynamic["TIMESTAMP"].isin(hours))
            & (graph_dynamic[mask_col] == 1)
        ][["TIMESTAMP", value_col]].copy()
        observed[value_col] = to_numeric(observed[value_col])
        observed = observed.dropna(subset=["TIMESTAMP", value_col]).sort_values("TIMESTAMP")
        peak_time = pd.NaT
        peak_value = np.nan
        if not observed.empty:
            peak_value = float(observed[value_col].max())
            peak_time = observed.loc[observed[value_col] == peak_value, "TIMESTAMP"].min()
        event_info[str(event_id)] = {
            "graph_id": str(graph_id),
            "station_id": station,
            "target_variable": target,
            "split": str(event.SPLIT).upper(),
            "target_hours": frozenset(pd.Timestamp(x) for x in hours),
            "valid_hours": frozenset(pd.Timestamp(x) for x in observed["TIMESTAMP"]),
            "target_start": hours.min(),
            "target_end": hours.max(),
            "hydro_start": event.HYDRO_START,
            "hydro_end": event.HYDRO_END,
            "peak_time": peak_time,
            "peak_value": peak_value,
        }

    rows: List[dict] = []
    for graph_id in sorted({x["graph_id"] for x in event_info.values()}):
        ordered_ids = sorted(
            [eid for eid, info in event_info.items() if info["graph_id"] == graph_id],
            key=lambda eid: (event_info[eid]["target_start"], eid),
        )
        for left_index, event_a in enumerate(ordered_ids):
            a = event_info[event_a]
            for right_index in range(left_index + 1, len(ordered_ids)):
                event_b = ordered_ids[right_index]
                b = event_info[event_b]
                interval_overlap = b["target_start"] <= a["target_end"]
                adjacent = right_index == left_index + 1
                if not interval_overlap and not adjacent:
                    break
                shared = a["valid_hours"] & b["valid_hours"]
                overlap_hours = len(shared)
                denominator = min(len(a["valid_hours"]), len(b["valid_hours"]))
                overlap_fraction = overlap_hours / denominator if denominator else np.nan
                official_overlap = np.nan
                if all(pd.notna(x) for x in [a["hydro_start"], a["hydro_end"], b["hydro_start"], b["hydro_end"]]):
                    official_overlap = max(
                        0.0,
                        (
                            min(pd.Timestamp(a["hydro_end"]), pd.Timestamp(b["hydro_end"]))
                            - max(pd.Timestamp(a["hydro_start"]), pd.Timestamp(b["hydro_start"]))
                        ).total_seconds() / 3600.0,
                    )
                hydro_gap = np.nan
                if pd.notna(a["hydro_end"]) and pd.notna(b["hydro_start"]):
                    hydro_gap = (
                        pd.Timestamp(b["hydro_start"]) - pd.Timestamp(a["hydro_end"])
                    ).total_seconds() / 3600.0
                same_peak = bool(
                    pd.notna(a["peak_time"])
                    and pd.notna(b["peak_time"])
                    and pd.Timestamp(a["peak_time"]) == pd.Timestamp(b["peak_time"])
                )
                must_merge = bool(
                    (overlap_hours > 0 and same_peak)
                    or (pd.notna(official_overlap) and official_overlap > 0)
                )
                cross_split = a["split"] != b["split"]
                if must_merge and cross_split:
                    status = "CROSS_SPLIT_LEAKAGE"
                    reason = "SAME_CONTINUOUS_RESPONSE_CROSSES_PROVISIONAL_SPLIT"
                    action = "MERGE_PROCESS_THEN_RERUN_DETERMINISTIC_SPLIT"
                elif must_merge:
                    status = "MUST_MERGE"
                    reason = (
                        "SHARED_OBSERVED_PEAK_AND_TARGET_HOURS"
                        if overlap_hours > 0 and same_peak
                        else "OFFICIAL_HYDRO_WINDOWS_OVERLAP"
                    )
                    action = "MERGE_PROCESS_THEN_REBUILD_ALL_EVENT_DEPENDENCIES"
                elif overlap_hours > 0:
                    status = "REVIEW"
                    reason = "SHARED_TARGET_HOURS_WITH_DIFFERENT_OBSERVED_PEAKS"
                    action = "REVIEW_CONTINUOUS_HYDROGRAPH"
                elif pd.notna(hydro_gap) and 0 <= hydro_gap <= forecast_hours:
                    status = "REVIEW"
                    reason = "HYDRO_GAP_NOT_LONGER_THAN_FORECAST_HORIZON"
                    action = "REVIEW_RECESSION_COMPLETION"
                else:
                    status = "OK"
                    reason = "NO_SHARED_RESPONSE_EVIDENCE"
                    action = "NONE"
                rows.append({
                    "GRAPH_ID": graph_id,
                    "EVENT_ID_A": event_a,
                    "EVENT_ID_B": event_b,
                    "target_station_id": a["station_id"],
                    "target_variable": a["target_variable"],
                    "hydro_start_A": a["hydro_start"],
                    "hydro_end_A": a["hydro_end"],
                    "hydro_start_B": b["hydro_start"],
                    "hydro_end_B": b["hydro_end"],
                    "overlap_hours": overlap_hours,
                    "overlap_fraction": overlap_fraction,
                    "official_hydro_overlap_hours": official_overlap,
                    "hydro_gap_hours": hydro_gap,
                    "same_observed_peak_time": same_peak,
                    "peak_time_A": a["peak_time"],
                    "peak_time_B": b["peak_time"],
                    "observed_peak_A": a["peak_value"],
                    "observed_peak_B": b["peak_value"],
                    "split_A": a["split"],
                    "split_B": b["split"],
                    "cross_split": cross_split,
                    "status": status,
                    "reason": reason,
                    "suggested_action": action,
                })
    return pd.DataFrame(rows, columns=EVENT_OVERLAP_COLUMNS), event_info


def merge_event_components(
    events: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    event_info: Dict[str, dict],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    parent = {str(event_id): str(event_id) for event_id in events["EVENT_ID"]}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    merge_pairs = overlap_audit[
        overlap_audit["status"].isin(["MUST_MERGE", "CROSS_SPLIT_LEAKAGE"])
    ]
    for row in merge_pairs.itertuples(index=False):
        union(str(row.EVENT_ID_A), str(row.EVENT_ID_B))

    components: Dict[str, List[str]] = defaultdict(list)
    for event_id in parent:
        components[find(event_id)].append(event_id)
    lookup = events.set_index("EVENT_ID")
    merged_rows: List[pd.Series] = []
    event_to_survivor: Dict[str, str] = {}
    reduction_by_graph: Dict[str, int] = defaultdict(int)
    merged_component_count = 0

    for component in components.values():
        ordered = sorted(
            component,
            key=lambda eid: (pd.Timestamp(lookup.loc[eid, "RAIN_START"]), eid),
        )
        survivor = ordered[0]
        for event_id in ordered:
            event_to_survivor[event_id] = survivor
        rows = lookup.loc[ordered]
        result = rows.iloc[0].copy()
        result["EVENT_ID"] = survivor
        result["RAIN_START"] = rows["RAIN_START"].min()
        result["RAIN_END"] = rows["RAIN_END"].max()
        result["HYDRO_START"] = rows["HYDRO_START"].min()
        result["HYDRO_END"] = rows["HYDRO_END"].max()
        result["SAMPLE_START"] = rows["SAMPLE_START"].min()
        result["SAMPLE_END"] = rows["SAMPLE_END"].max()

        peak_candidates = []
        for event_id in ordered:
            info = event_info.get(event_id, {})
            if pd.notna(info.get("peak_time", pd.NaT)) and pd.notna(info.get("peak_value", np.nan)):
                peak_candidates.append((float(info["peak_value"]), pd.Timestamp(info["peak_time"])))
        if peak_candidates:
            best_value = max(value for value, _time in peak_candidates)
            result["PEAK_TIME"] = min(time for value, time in peak_candidates if value == best_value)
        else:
            result["PEAK_TIME"] = rows["PEAK_TIME"].min()

        grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "": 4}
        grades = [str(value).upper() for value in rows["EVENT_GRADE"]]
        result["EVENT_GRADE"] = max(grades, key=lambda value: grade_rank.get(value, 99))
        rain_ids: List[str] = []
        for value in rows["SOURCE_RAIN_EVENT_IDS"]:
            for item in str(value).split(";"):
                item = item.strip()
                if item and item not in rain_ids:
                    rain_ids.append(item)
        result["SOURCE_RAIN_EVENT_IDS"] = ";".join(rain_ids)
        result["SOURCE_RAIN_EVENT_COUNT"] = len(rain_ids)
        result["PEAK_COUNT"] = len(rain_ids)
        result["COMPOUND_EVENT"] = len(rain_ids) > 1
        result["SOURCE_EVENT_IDS"] = ";".join(ordered)
        result["SOURCE_EVENT_COUNT"] = len(ordered)
        result["EVENT_MERGE_STATUS"] = "MERGED" if len(ordered) > 1 else "UNCHANGED"
        merged_rows.append(result)
        if len(ordered) > 1:
            merged_component_count += 1
            reduction_by_graph[str(result["GRAPH_ID"])] += len(ordered) - 1

    merged = pd.DataFrame(merged_rows).reset_index(drop=True)
    merged = merged.sort_values(["PEAK_TIME", "GRAPH_ID", "EVENT_ID"]).reset_index(drop=True)
    merge_audit = overlap_audit.copy()
    merge_audit["MERGED_EVENT_ID"] = merge_audit["EVENT_ID_A"].map(event_to_survivor)
    same_component = (
        merge_audit["EVENT_ID_A"].map(event_to_survivor)
        == merge_audit["EVENT_ID_B"].map(event_to_survivor)
    )
    merge_audit["MERGE_ACTION"] = np.where(same_component, "MERGED", "NOT_MERGED")
    summary = {
        "event_count_before_merge": int(len(events)),
        "event_count_after_merge": int(len(merged)),
        "event_reduction_count": int(len(events) - len(merged)),
        "merged_component_count": int(merged_component_count),
        "must_merge_pair_count": int((overlap_audit["status"] == "MUST_MERGE").sum()),
        "provisional_cross_split_pair_count": int(
            (overlap_audit["status"] == "CROSS_SPLIT_LEAKAGE").sum()
        ),
        "event_reduction_by_graph": dict(sorted(reduction_by_graph.items())),
    }
    return merged, merge_audit, summary


def build_water_level_reference_audit(
    events: pd.DataFrame,
    samples: pd.DataFrame,
    dynamic: Dict[str, pd.DataFrame],
    target_map: pd.DataFrame,
) -> pd.DataFrame:
    """只用暂定TRAIN事件识别站内水位基准断裂，不用VAL/TEST反向筛数据。"""
    event_lookup = events.set_index("EVENT_ID")
    records: List[dict] = []
    for target in target_map.itertuples(index=False):
        if str(target.TARGET_VARIABLE).upper() != "WATER_LEVEL":
            continue
        graph_id = str(target.GRAPH_ID)
        station = normalize_id(target.OUTLET_ID)
        graph_dynamic = dynamic.get(graph_id)
        if graph_dynamic is None:
            continue
        observed = graph_dynamic[
            (graph_dynamic["STATION_ID"] == station)
            & (graph_dynamic["WATER_LEVEL_MASK"] == 1)
        ].set_index("TIMESTAMP")["WATER_LEVEL"]
        observed = to_numeric(observed).dropna().sort_index()
        graph_samples = samples[samples["GRAPH_ID"] == graph_id]
        for event_id, event_samples in graph_samples.groupby("EVENT_ID", sort=True):
            hours = target_hours_from_samples(event_samples)
            values = observed[observed.index.isin(hours)]
            event = event_lookup.loc[event_id]
            records.append({
                "GRAPH_ID": graph_id,
                "STATION_ID": station,
                "EVENT_ID": event_id,
                "SPLIT": str(event.SPLIT).upper(),
                "TARGET_HOUR_COUNT": int(len(values)),
                "MIN_Z": float(values.min()) if len(values) else np.nan,
                "MAX_Z": float(values.max()) if len(values) else np.nan,
                "MEDIAN_Z": float(values.median()) if len(values) else np.nan,
            })

    audit = pd.DataFrame(records)
    if audit.empty:
        return pd.DataFrame(columns=WATER_LEVEL_REFERENCE_COLUMNS)
    output: List[dict] = []
    for (graph_id, station), group in audit.groupby(["GRAPH_ID", "STATION_ID"], sort=True):
        train = group[(group["SPLIT"] == "TRAIN") & group["MEDIAN_Z"].notna()]
        fence = tukey_outer_fence(train["MEDIAN_Z"])
        for row in group.itertuples(index=False):
            outside = False
            q1 = q3 = iqr = low = high = np.nan
            if fence is not None:
                q1, q3, iqr, low, high = fence
                outside = bool(
                    row.SPLIT == "TRAIN"
                    and pd.notna(row.MIN_Z)
                    and pd.notna(row.MAX_Z)
                    and (row.MAX_Z < low or row.MIN_Z > high)
                )
            nontrain_outside = bool(
                row.SPLIT != "TRAIN"
                and fence is not None
                and pd.notna(row.MIN_Z)
                and pd.notna(row.MAX_Z)
                and (row.MAX_Z < low or row.MIN_Z > high)
            )
            if outside:
                status = "FAIL"
                reason = "TRAIN_EVENT_WATER_LEVEL_REFERENCE_SHIFT"
                action = "EXCLUDE_EVENT_AND_MASK_SOURCE_WINDOW"
            elif nontrain_outside:
                status = "REVIEW"
                reason = "NONTRAIN_EVENT_OUTSIDE_TRAIN_REFERENCE_FENCE"
                action = "NONE_DO_NOT_FILTER_VALIDATION_OR_TEST"
            elif fence is None:
                status = "REVIEW"
                reason = "INSUFFICIENT_TRAIN_EVENTS_FOR_REFERENCE_FENCE"
                action = "NONE"
            else:
                status = "OK"
                reason = "NONE"
                action = "NONE"
            record = row._asdict()
            record.update({
                "TRAIN_EVENT_COUNT": int(len(train)),
                "TRAIN_MEDIAN_Q1": q1,
                "TRAIN_MEDIAN_Q3": q3,
                "TRAIN_MEDIAN_IQR": iqr,
                "TRAIN_OUTER_FENCE_LOW": low,
                "TRAIN_OUTER_FENCE_HIGH": high,
                "OUTSIDE_TRAIN_OUTER_FENCE": int(outside or nontrain_outside),
                "QC_STATUS": status,
                "QC_REASON": reason,
                "ACTION": action,
                "MASKED_SOURCE_ROW_COUNT": 0,
            })
            output.append(record)
    return pd.DataFrame(output, columns=WATER_LEVEL_REFERENCE_COLUMNS)


def apply_water_level_reference_exclusions(
    hydro: pd.DataFrame,
    events: pd.DataFrame,
    audit: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    cleaned = hydro.copy()
    updated = audit.copy()
    event_lookup = events.set_index("EVENT_ID")
    excluded_ids = set(
        updated.loc[updated["ACTION"] == "EXCLUDE_EVENT_AND_MASK_SOURCE_WINDOW", "EVENT_ID"].astype(str)
    )
    for event_id in sorted(excluded_ids):
        event = event_lookup.loc[event_id]
        station = normalize_id(event.OUTLET_ID)
        mask = (
            (cleaned["STATION_ID"] == station)
            & (cleaned["TIMESTAMP"] >= pd.Timestamp(event.SAMPLE_START).floor("h"))
            & (cleaned["TIMESTAMP"] <= pd.Timestamp(event.SAMPLE_END).ceil("h"))
            & cleaned["WATER_LEVEL"].notna()
        )
        count = int(mask.sum())
        cleaned.loc[mask, "WATER_LEVEL"] = np.nan
        updated.loc[updated["EVENT_ID"] == event_id, "MASKED_SOURCE_ROW_COUNT"] = count
    return cleaned, updated, excluded_ids


# -----------------------------------------------------------------------------
# 标准化参数
# -----------------------------------------------------------------------------

def stats_record(values: pd.Series) -> Dict[str, object]:
    v = to_numeric(values).dropna()
    if len(v) == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    std = float(v.std(ddof=0))
    if std == 0:
        std = 1.0
    return {
        "count": int(len(v)),
        "mean": float(v.mean()),
        "std": std,
        "min": float(v.min()),
        "max": float(v.max()),
    }


def rows_in_intervals(timestamps: pd.Series, intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=timestamps.index)
    for start, end in merge_intervals(intervals):
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def compute_normalization(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    dynamic: Dict[str, pd.DataFrame],
    node_static: pd.DataFrame,
    edge_static: pd.DataFrame,
) -> Dict[str, object]:
    train_samples = samples[samples["SPLIT"] == "TRAIN"].copy()
    train_events = events[events["SPLIT"] == "TRAIN"].copy()
    dynamic_values: Dict[str, List[pd.Series]] = defaultdict(list)

    # 只使用训练样本的输入时间窗。
    for graph_id, group in train_samples.groupby("GRAPH_ID"):
        df = dynamic.get(graph_id)
        if df is None:
            continue
        intervals = list(zip(group["INPUT_START"], group["FORECAST_TIME"]))
        mask = rows_in_intervals(df["TIMESTAMP"], intervals)
        part = df[mask]
        dynamic_values["RAIN_MM"].append(part.loc[part["RAIN_MASK"] == 1, "RAIN_MM"])
        dynamic_values["FLOW"].append(part.loc[part["FLOW_MASK"] == 1, "FLOW"])
        dynamic_values["WATER_LEVEL"].append(part.loc[part["WATER_LEVEL_MASK"] == 1, "WATER_LEVEL"])

    dynamic_stats = {}
    for feature in ["RAIN_MM", "FLOW", "WATER_LEVEL"]:
        vals = pd.concat(dynamic_values[feature], ignore_index=True) if dynamic_values[feature] else pd.Series(dtype=float)
        dynamic_stats[feature] = stats_record(vals)

    train_graphs = set(train_events["GRAPH_ID"])
    node_part = node_static[node_static["GRAPH_ID"].isin(train_graphs)]
    edge_part = edge_static[edge_static["GRAPH_ID"].isin(train_graphs)]
    node_stats = {feature: stats_record(node_part[feature]) for feature in NODE_FEATURE_ALIASES if feature in node_part.columns}
    edge_stats = {feature: stats_record(edge_part[feature]) for feature in EDGE_FEATURE_ALIASES if feature in edge_part.columns}

    return {
        "computed_from_split": "TRAIN",
        "scope": "TRAIN_ONLY",
        "features": dynamic_stats,
        "node_static": node_stats,
        "edge_static": edge_stats,
    }


def hourly_absolute_differences(series: pd.Series) -> pd.Series:
    ordered = series.sort_index()
    time_delta = ordered.index.to_series().diff()
    differences = ordered.diff().abs()
    return differences[time_delta == pd.Timedelta(hours=1)].dropna()


def build_water_level_station_audit(
    events: pd.DataFrame,
    samples: pd.DataFrame,
    dynamic: Dict[str, pd.DataFrame],
    target_map: pd.DataFrame,
    normalization: Dict[str, object],
) -> pd.DataFrame:
    """按正式split输出可由validator复核的站级水位范围与基准一致性。"""
    columns = [
        "station_id", "graph_ids", "split", "valid_count", "min_z", "max_z",
        "mean_z", "std_z", "train_min_z", "train_max_z",
        "out_of_train_range_count", "out_of_train_range_fraction",
        "normalization_train_min_z", "normalization_train_max_z",
        "out_of_normalization_range_count", "out_of_normalization_range_fraction",
        "max_abs_delta_z", "train_jump_outer_fence_m", "suspicious_jump_count",
        "train_event_count", "train_reference_shift_event_count",
        "train_reference_shift_event_ids", "max_train_event_median_shift_m",
        "normalization_computed_from_split", "qc_status", "qc_reason",
    ]
    level_stats = normalization["features"]["WATER_LEVEL"]
    normalization_min = level_stats.get("min")
    normalization_max = level_stats.get("max")
    event_lookup = events.set_index("EVENT_ID")
    rows: List[dict] = []

    for target in target_map.itertuples(index=False):
        if str(target.TARGET_VARIABLE).upper() != "WATER_LEVEL":
            continue
        graph_id = str(target.GRAPH_ID)
        station = normalize_id(target.OUTLET_ID)
        graph_dynamic = dynamic.get(graph_id)
        graph_samples = samples[samples["GRAPH_ID"] == graph_id]
        if graph_dynamic is None or graph_samples.empty:
            continue
        values = graph_dynamic[
            (graph_dynamic["STATION_ID"] == station)
            & (graph_dynamic["WATER_LEVEL_MASK"] == 1)
        ].set_index("TIMESTAMP")["WATER_LEVEL"]
        values = to_numeric(values).dropna().sort_index()

        event_records: List[dict] = []
        split_values: Dict[str, pd.Series] = {}
        for event_id, event_samples in graph_samples.groupby("EVENT_ID", sort=True):
            hours = target_hours_from_samples(event_samples)
            observed = values[values.index.isin(hours)]
            if len(observed):
                event_records.append({
                    "EVENT_ID": event_id,
                    "SPLIT": str(event_lookup.loc[event_id, "SPLIT"]).upper(),
                    "EVENT_TIME": hours.min(),
                    "MIN": float(observed.min()),
                    "MAX": float(observed.max()),
                    "MEDIAN": float(observed.median()),
                })
        for split, split_samples in graph_samples.groupby("SPLIT", sort=True):
            hours = pd.DatetimeIndex([])
            for _event_id, event_samples in split_samples.groupby("EVENT_ID", sort=True):
                hours = hours.union(target_hours_from_samples(event_samples))
            split_values[str(split).upper()] = values[values.index.isin(hours)]

        train_values = split_values.get("TRAIN", pd.Series(dtype=float))
        train_min = float(train_values.min()) if len(train_values) else np.nan
        train_max = float(train_values.max()) if len(train_values) else np.nan
        train_differences = hourly_absolute_differences(train_values)
        jump_fence = tukey_outer_fence(train_differences)
        jump_threshold = jump_fence[4] if jump_fence is not None else np.nan
        event_frame = pd.DataFrame(event_records)
        train_events = (
            event_frame[event_frame["SPLIT"] == "TRAIN"].sort_values("EVENT_TIME")
            if not event_frame.empty
            else pd.DataFrame()
        )
        shift_ids: List[str] = []
        maximum_shift = np.nan
        if not train_events.empty:
            median_fence = tukey_outer_fence(train_events["MEDIAN"])
            if median_fence is not None:
                low, high = median_fence[3], median_fence[4]
                shift_ids = sorted(train_events.loc[
                    (train_events["MAX"] < low) | (train_events["MIN"] > high), "EVENT_ID"
                ].astype(str))
            shifts = train_events["MEDIAN"].diff().abs().dropna()
            if len(shifts):
                maximum_shift = float(shifts.max())

        for split in ["TRAIN", "VALIDATION", "TEST"]:
            if split not in split_values:
                continue
            observed = split_values[split]
            count = len(observed)
            outside_station = (
                (observed < train_min) | (observed > train_max)
                if pd.notna(train_min) and pd.notna(train_max)
                else pd.Series(False, index=observed.index)
            )
            outside_normalization = (
                (observed < float(normalization_min)) | (observed > float(normalization_max))
                if normalization_min is not None and normalization_max is not None
                else pd.Series(False, index=observed.index)
            )
            differences = hourly_absolute_differences(observed)
            suspicious = (
                differences > float(jump_threshold)
                if pd.notna(jump_threshold)
                else pd.Series(False, index=differences.index)
            )
            reasons: List[str] = []
            status = "OK"
            if shift_ids:
                status = "FAIL"
                reasons.append("TRAIN_WATER_LEVEL_REFERENCE_SHIFT_EVENTS=" + ";".join(shift_ids))
            if status != "FAIL" and int(outside_station.sum()):
                status = "REVIEW"
                reasons.append("OUTSIDE_STATION_TRAIN_RANGE")
            if status != "FAIL" and int(outside_normalization.sum()):
                status = "REVIEW"
                reasons.append("OUTSIDE_GLOBAL_TRAIN_NORMALIZATION_RANGE")
            if status != "FAIL" and int(suspicious.sum()):
                status = "REVIEW"
                reasons.append("HOURLY_JUMP_EXCEEDS_TRAIN_TUKEY_OUTER_FENCE")
            rows.append({
                "station_id": station,
                "graph_ids": graph_id,
                "split": split,
                "valid_count": int(count),
                "min_z": float(observed.min()) if count else np.nan,
                "max_z": float(observed.max()) if count else np.nan,
                "mean_z": float(observed.mean()) if count else np.nan,
                "std_z": float(observed.std(ddof=0)) if count else np.nan,
                "train_min_z": train_min,
                "train_max_z": train_max,
                "out_of_train_range_count": int(outside_station.sum()),
                "out_of_train_range_fraction": int(outside_station.sum()) / count if count else np.nan,
                "normalization_train_min_z": normalization_min,
                "normalization_train_max_z": normalization_max,
                "out_of_normalization_range_count": int(outside_normalization.sum()),
                "out_of_normalization_range_fraction": int(outside_normalization.sum()) / count if count else np.nan,
                "max_abs_delta_z": float(differences.max()) if len(differences) else np.nan,
                "train_jump_outer_fence_m": jump_threshold,
                "suspicious_jump_count": int(suspicious.sum()),
                "train_event_count": int(len(train_events)),
                "train_reference_shift_event_count": int(len(shift_ids)),
                "train_reference_shift_event_ids": ";".join(shift_ids),
                "max_train_event_median_shift_m": maximum_shift,
                "normalization_computed_from_split": normalization["computed_from_split"],
                "qc_status": status,
                "qc_reason": ";".join(reasons) if reasons else "NONE",
            })
    return pd.DataFrame(rows, columns=columns)


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="整理山洪预报模型数据集（Python 3）")
    p.add_argument("--project-root", required=True, help="湖南项目根目录")
    p.add_argument("--workflow-dir", default="", help="MERIT_workflow目录；默认 project-root/Arcgis/MERIT_workflow")
    p.add_argument("--hydro-data", default="", help="全部河道站文件夹；默认 project-root/归档_正确解压/河道")
    p.add_argument("--node-rain-dir", default="", help="第11步节点逐时降雨目录；默认自动发现")
    p.add_argument("--events-csv", default="", help="第13步 final_flood_events.csv；默认自动发现")
    p.add_argument("--edges-csv", default="", help="第06步 edges.csv；默认自动发现")
    p.add_argument("--node-static-csv", default="", help="最终节点静态属性CSV；默认优先第15步final")
    p.add_argument("--edge-static-csv", default="", help="边静态属性CSV；默认第14步结果")
    p.add_argument("--output-dir", default="", help="输出目录；默认 MERIT_workflow/16_model_dataset")
    p.add_argument("--event-types", default="HYDRO_FLOOD", help="纳入事件类型，逗号分隔")
    p.add_argument("--event-grades", default="A,B", help="纳入事件等级，逗号分隔；为空表示不限")
    p.add_argument("--history-hours", type=int, default=24)
    p.add_argument("--forecast-hours", type=int, default=6)
    p.add_argument("--sample-step-hours", type=int, default=1)
    p.add_argument("--train-fraction", type=float, default=0.70)
    p.add_argument("--validation-fraction", type=float, default=0.15)
    p.add_argument("--target-variable", choices=["AUTO", "FLOW", "WATER_LEVEL"], default="AUTO")
    p.add_argument("--min-flow-target-coverage", type=float, default=0.70)
    p.add_argument("--min-target-coverage", type=float, default=0.80)
    p.add_argument(
        "--water-level-reference-shift-policy",
        choices=["EXCLUDE_EVENT", "FAIL"],
        default="EXCLUDE_EVENT",
        help=(
            "TRAIN事件水位基准断裂处理：EXCLUDE_EVENT仅排除不可恢复事件并掩膜其源水位窗；"
            "FAIL保留QC后终止。不会自动平移水位或删除整站。"
        ),
    )
    p.add_argument("--allow-missing-static", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    sources = discover_sources(args)

    if sources.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError("输出目录已存在，请使用 --overwrite：{}".format(sources.output_dir))
        shutil.rmtree(sources.output_dir)
    sources.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(sources.output_dir / "metadata" / "build_log.txt")

    LOG.info("开始整理模型数据集。")
    LOG.info("项目根目录：%s", sources.project_root)
    LOG.info("工作流目录：%s", sources.workflow_dir)
    LOG.info("节点静态属性：%s", sources.node_static_csv)
    LOG.info("边静态属性：%s", sources.edge_static_csv if sources.edge_static_csv else "未找到")
    LOG.info("边拓扑：%s", sources.edges_csv)
    LOG.info("最终洪水事件：%s", sources.events_csv)
    LOG.info("节点逐时降雨目录：%s", sources.rain_dir)
    LOG.info("河道站目录：%s", sources.hydro_dir)

    graph_dir = sources.output_dir / "graph"
    dynamic_dir = sources.output_dir / "dynamic"
    events_dir = sources.output_dir / "events"
    metadata_dir = sources.output_dir / "metadata"
    qc_dir = sources.output_dir / "qc"
    for d in [graph_dir, dynamic_dir, events_dir, metadata_dir, qc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. 静态属性和图结构
    node_static_raw, missing_static = canonicalize_node_static(sources.node_static_csv, args.allow_missing_static)
    edges_raw = canonicalize_edges(sources.edges_csv)
    edges_raw = merge_edge_static(edges_raw, sources.edge_static_csv)
    catalog, edge_topology, node_static, edge_static = build_graph_tables(node_static_raw, edges_raw)
    write_csv(catalog[[
        "GRAPH_ID", "BASIN_ID", "NODE_INDEX", "STATION_ID", "OUTLET_ID", "ROLE", "IS_OUTLET", "STATIC_QC"
    ]], graph_dir / "node_catalog.csv")
    write_csv(edge_topology[[
        c for c in ["GRAPH_ID", "FROM_NODE", "TO_NODE", "FROM_STATION", "TO_STATION"] if c in edge_topology.columns
    ]], graph_dir / "edge_topology.csv")
    write_csv(node_static, graph_dir / "node_static_attributes.csv")
    write_csv(edge_static, graph_dir / "edge_static_attributes.csv")
    LOG.info("图结构完成：%d个图，%d个节点，%d条边。", catalog["GRAPH_ID"].nunique(), len(catalog), len(edge_topology))

    # 2. 事件筛选。正式split必须等事件合并和不可恢复水位事件排除后再执行。
    events_all = canonicalize_events(sources.events_csv)
    events_selected, events_excluded = filter_events(
        events_all,
        parse_csv_list(args.event_types),
        parse_csv_list(args.event_grades),
    )
    graph_set = set(catalog["GRAPH_ID"])
    valid_graph = events_selected["GRAPH_ID"].isin(graph_set)
    missing_graph_events = events_selected[~valid_graph].copy()
    if not missing_graph_events.empty:
        missing_graph_events["EXCLUSION_REASON"] = "GRAPH_NOT_IN_NODE_CATALOG"
        events_excluded = pd.concat([events_excluded, missing_graph_events], ignore_index=True, sort=False)
    events_selected = events_selected[valid_graph].copy()
    if events_selected.empty:
        raise ValueError("事件筛选后为空，请检查 --event-types 和 --event-grades。")

    # 3. 降雨和水文数据。先按未合并事件总范围读取一次，后续所有重建共用同一原始表。
    needed_nodes = set(catalog["STATION_ID"])
    graph_ids = set(catalog["GRAPH_ID"])
    min_time = events_selected["SAMPLE_START"].min().floor("h")
    max_time = events_selected["SAMPLE_END"].max().ceil("h")

    rain_sources = discover_rain_sources(sources.rain_dir, needed_nodes, graph_ids)
    rain, rain_coverage = load_node_rain(rain_sources, needed_nodes, min_time, max_time)
    write_csv(rain_coverage, qc_dir / "rain_source_coverage.csv")

    hydro_files, hydro_selection = discover_hydro_files(sources.hydro_dir, needed_nodes)
    write_csv(hydro_selection, qc_dir / "hydro_file_selection.csv")
    hydro, hydro_load_audit = load_hydro(hydro_files, needed_nodes, min_time, max_time)
    write_csv(hydro_load_audit, qc_dir / "hydro_load_audit.csv")

    # 4. 暂定split仅用于TRAIN水位基准审计；不作为正式输出。
    provisional_events = assign_splits(events_selected, args.train_fraction, args.validation_fraction)
    provisional_dynamic, provisional_dynamic_qc = make_dynamic_tables(
        catalog, provisional_events, rain, rain_coverage, hydro, None
    )
    provisional_target_map = choose_target_by_graph(
        provisional_dynamic_qc, args.target_variable, args.min_flow_target_coverage
    )
    provisional_samples, _ = build_sample_index(
        provisional_events,
        provisional_dynamic,
        provisional_target_map,
        args.history_hours,
        args.forecast_hours,
        args.sample_step_hours,
        args.min_target_coverage,
    )
    if provisional_samples.empty:
        raise ValueError("未生成任何暂定样本，无法执行水位基准和事件过程审计。")
    water_reference_audit = build_water_level_reference_audit(
        provisional_events,
        provisional_samples,
        provisional_dynamic,
        provisional_target_map,
    )
    hydro, water_reference_audit, reference_excluded_ids = apply_water_level_reference_exclusions(
        hydro, provisional_events, water_reference_audit
    )
    write_csv(water_reference_audit, qc_dir / "water_level_reference_event_audit.csv")
    if reference_excluded_ids and args.water_level_reference_shift_policy == "FAIL":
        raise ValueError(
            "TRAIN水位事件存在不可恢复的站内基准断裂：{}。"
            "QC已写出；如确认采用事件级排除，请使用默认EXCLUDE_EVENT策略重新运行。"
            .format(sorted(reference_excluded_ids))
        )
    if reference_excluded_ids:
        bad_events = events_selected[events_selected["EVENT_ID"].isin(reference_excluded_ids)].copy()
        bad_events["EXCLUSION_REASON"] = "WATER_LEVEL_REFERENCE_DATUM_SHIFT_UNRECOVERABLE"
        bad_events["EXCLUSION_DETAIL"] = "TRAIN_EVENT_OUTSIDE_STATION_EVENT_MEDIAN_TUKEY_OUTER_FENCE"
        events_excluded = pd.concat([events_excluded, bad_events], ignore_index=True, sort=False)
        events_selected = events_selected[~events_selected["EVENT_ID"].isin(reference_excluded_ids)].copy()
        LOG.warning(
            "排除%d场不可恢复水位基准事件并掩膜其源水位窗：%s。整站保留。",
            len(reference_excluded_ids), ", ".join(sorted(reference_excluded_ids))
        )

    # 5. 在清洁观测上生成暂定样本，按确定性强证据合并同一连续洪水过程。
    provisional_events = assign_splits(events_selected, args.train_fraction, args.validation_fraction)
    provisional_dynamic, provisional_dynamic_qc = make_dynamic_tables(
        catalog, provisional_events, rain, rain_coverage, hydro, None
    )
    provisional_target_map = choose_target_by_graph(
        provisional_dynamic_qc, args.target_variable, args.min_flow_target_coverage
    )
    provisional_samples, _ = build_sample_index(
        provisional_events,
        provisional_dynamic,
        provisional_target_map,
        args.history_hours,
        args.forecast_hours,
        args.sample_step_hours,
        args.min_target_coverage,
    )
    premerge_overlap, premerge_event_info = build_event_overlap_audit(
        provisional_events,
        provisional_samples,
        provisional_dynamic,
        provisional_target_map,
        args.forecast_hours,
    )
    merged_events, event_merge_audit, event_merge_summary = merge_event_components(
        events_selected, premerge_overlap, premerge_event_info
    )
    events_split = assign_splits(merged_events, args.train_fraction, args.validation_fraction)
    write_csv(event_merge_audit, qc_dir / "event_merge_audit.csv")
    LOG.info(
        "事件过程合并：合并前%d，合并后%d，连通组%d，减少%d。",
        event_merge_summary["event_count_before_merge"],
        event_merge_summary["event_count_after_merge"],
        event_merge_summary["merged_component_count"],
        event_merge_summary["event_reduction_count"],
    )

    # 6. 用最终事件重新生成全部依赖：split、动态图、target map和sample index。
    write_csv(events_all, events_dir / "flood_events_all.csv")
    write_csv(events_split, events_dir / "flood_events_final.csv")
    write_csv(events_split[[
        "EVENT_ID", "GRAPH_ID", "OUTLET_ID", "EVENT_YEAR", "EVENT_TYPE", "EVENT_GRADE", "SPLIT", "SPLIT_REASON"
    ]], events_dir / "data_split.csv")
    write_csv(events_excluded, qc_dir / "event_exclusion.csv")

    dynamic, dynamic_qc = make_dynamic_tables(
        catalog, events_split, rain, rain_coverage, hydro, dynamic_dir
    )
    write_csv(dynamic_qc, qc_dir / "dynamic_coverage.csv")
    target_map = choose_target_by_graph(dynamic_qc, args.target_variable, args.min_flow_target_coverage)
    write_csv(target_map, events_dir / "target_variable_by_graph.csv")
    samples, rejected_samples = build_sample_index(
        events_split,
        dynamic,
        target_map,
        args.history_hours,
        args.forecast_hours,
        args.sample_step_hours,
        args.min_target_coverage,
    )
    if samples.empty:
        raise ValueError("最终事件未生成任何有效训练样本，请检查事件时段和出口水文完整性。")
    write_csv(samples, events_dir / "sample_index.csv")
    write_csv(rejected_samples, qc_dir / "sample_rejection.csv")

    postmerge_overlap, _ = build_event_overlap_audit(
        events_split, samples, dynamic, target_map, args.forecast_hours
    )
    write_csv(postmerge_overlap, qc_dir / "event_hydrograph_overlap.csv")
    residual_merge = postmerge_overlap[
        postmerge_overlap["status"].isin(["MUST_MERGE", "CROSS_SPLIT_LEAKAGE"])
    ]
    if not residual_merge.empty:
        raise RuntimeError(
            "事件合并后仍存在强重复证据，拒绝输出正式数据：{}"
            .format(residual_merge[["EVENT_ID_A", "EVENT_ID_B", "status"]].head(20).to_dict("records"))
        )

    # 7. 标准化统计和最终水位站级QC，只用正式TRAIN。
    normalization = compute_normalization(samples, events_split, dynamic, node_static, edge_static)
    with (metadata_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump(normalization, f, ensure_ascii=False, indent=2)
    water_level_station_audit = build_water_level_station_audit(
        events_split, samples, dynamic, target_map, normalization
    )
    write_csv(water_level_station_audit, qc_dir / "water_level_station_audit.csv")
    water_fail = water_level_station_audit[water_level_station_audit["qc_status"] == "FAIL"]
    if not water_fail.empty:
        raise RuntimeError(
            "最终TRAIN仍存在水位基准断裂，拒绝输出正式数据：{}"
            .format(sorted(set(water_fail["station_id"].astype(str))))
        )

    quality_summary = {
        "dataset_contract_version": "5-event-zqc",
        "event": event_merge_summary,
        "water_level": {
            "reference_excluded_event_count": int(len(reference_excluded_ids)),
            "reference_excluded_event_ids": sorted(reference_excluded_ids),
            "final_fail_station_count": int(water_fail["station_id"].nunique()) if not water_fail.empty else 0,
            "final_review_row_count": int((water_level_station_audit["qc_status"] == "REVIEW").sum()),
            "reference_shift_rule": (
                "Provisional TRAIN event median Tukey outer fences "
                "(Q1-3*IQR, Q3+3*IQR); exclude only when the complete event target range "
                "lies outside a fence. VALIDATION/TEST are never filtered by this rule."
            ),
        },
        "strict_failure_counts": {
            "residual_must_merge": int((postmerge_overlap["status"] == "MUST_MERGE").sum()),
            "residual_cross_split_leakage": int(
                (postmerge_overlap["status"] == "CROSS_SPLIT_LEAKAGE").sum()
            ),
            "water_level_fail_stations": int(water_fail["station_id"].nunique()) if not water_fail.empty else 0,
        },
    }
    with (qc_dir / "dataset_quality_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(quality_summary, f, ensure_ascii=False, indent=2)

    feature_schema = {
        # Rainfall is an explicit forcing tensor, not duplicated in dynamic_node_features.
        "dynamic_features": ["FLOW", "WATER_LEVEL"],
        "rainfall_storage": "step11_sparse_positive_hour_intervals; missing node-hour within model windows means 0 mm",
        "rainfall_timestamp_convention": "TIMESTAMP = start_time of [start_time,end_time) 1-hour interval",
        "dynamic_masks": ["RAIN_MASK", "FLOW_MASK", "WATER_LEVEL_MASK"],
        "node_static_features": list(NODE_FEATURE_ALIASES),
        "edge_static_features": list(EDGE_FEATURE_ALIASES),
        "history_hours": args.history_hours,
        "forecast_hours": args.forecast_hours,
        "sample_step_hours": args.sample_step_hours,
        "target_mode": args.target_variable,
        "time_zone": "Asia/Shanghai",
        "dynamic_layout": "long: GRAPH_ID,TIMESTAMP,NODE_INDEX,STATION_ID,features,masks",
        "physical_features": {
            "incremental_area_km2": {
                "source": "log_incremental_area",
                "transform": "log1p",
                "unit": "km2",
            }
        },
    }
    with (metadata_dir / "feature_schema.json").open("w", encoding="utf-8") as f:
        json.dump(feature_schema, f, ensure_ascii=False, indent=2)

    source_manifest = {
        "dataset_contract_version": "5-event-zqc",
        "generator_script": file_fingerprint(Path(__file__).resolve()),
        "project_root": str(sources.project_root),
        "workflow_dir": str(sources.workflow_dir),
        "hydro_dir": str(sources.hydro_dir),
        "rain_dir": str(sources.rain_dir),
        "events_csv": file_fingerprint(sources.events_csv),
        "edges_csv": file_fingerprint(sources.edges_csv),
        "node_static_csv": file_fingerprint(sources.node_static_csv),
        "edge_static_csv": file_fingerprint(sources.edge_static_csv) if sources.edge_static_csv else None,
        "missing_node_static_features": missing_static,
        "event_merge_summary": event_merge_summary,
        "water_level_reference_excluded_event_ids": sorted(reference_excluded_ids),
        "parameters": vars(args),
    }
    # Path对象转文本。
    source_manifest["parameters"] = {k: str(v) if isinstance(v, Path) else v for k, v in source_manifest["parameters"].items()}
    with (metadata_dir / "source_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(source_manifest, f, ensure_ascii=False, indent=2)

    summary_rows = [
        {"ITEM": "graph_count", "VALUE": int(catalog["GRAPH_ID"].nunique())},
        {"ITEM": "node_count", "VALUE": int(len(catalog))},
        {"ITEM": "edge_count", "VALUE": int(len(edge_topology))},
        {"ITEM": "source_event_count", "VALUE": int(len(events_all))},
        {"ITEM": "water_level_reference_excluded_event_count", "VALUE": int(len(reference_excluded_ids))},
        {"ITEM": "event_count_before_merge", "VALUE": int(event_merge_summary["event_count_before_merge"])},
        {"ITEM": "event_merge_reduction_count", "VALUE": int(event_merge_summary["event_reduction_count"])},
        {"ITEM": "merged_event_component_count", "VALUE": int(event_merge_summary["merged_component_count"])},
        {"ITEM": "event_count", "VALUE": int(len(events_split))},
        {"ITEM": "train_event_count", "VALUE": int((events_split["SPLIT"] == "TRAIN").sum())},
        {"ITEM": "validation_event_count", "VALUE": int((events_split["SPLIT"] == "VALIDATION").sum())},
        {"ITEM": "test_event_count", "VALUE": int((events_split["SPLIT"] == "TEST").sum())},
        {"ITEM": "sample_count", "VALUE": int(len(samples))},
        {"ITEM": "train_sample_count", "VALUE": int((samples["SPLIT"] == "TRAIN").sum())},
        {"ITEM": "validation_sample_count", "VALUE": int((samples["SPLIT"] == "VALIDATION").sum())},
        {"ITEM": "test_sample_count", "VALUE": int((samples["SPLIT"] == "TEST").sum())},
        {"ITEM": "rejected_candidate_window_count", "VALUE": int(len(rejected_samples))},
        {"ITEM": "event_without_sample_count", "VALUE": int(
            len(set(events_split["EVENT_ID"]) - set(samples["EVENT_ID"]))
        )},
        {"ITEM": "dynamic_row_count", "VALUE": int(sum(len(x) for x in dynamic.values()))},
        {"ITEM": "missing_static_feature_count", "VALUE": int(len(missing_static))},
    ]
    write_csv(pd.DataFrame(summary_rows), metadata_dir / "dataset_summary.csv")

    LOG.info("模型数据集整理完成：%s", sources.output_dir)
    LOG.info(
        "源事件%d；水位基准排除%d；事件合并减少%d；正式事件%d。",
        len(events_all), len(reference_excluded_ids),
        event_merge_summary["event_reduction_count"], len(events_split)
    )
    LOG.info("事件%d场，样本%d个；TRAIN/VALIDATION/TEST=%d/%d/%d。",
             len(events_split), len(samples),
             int((samples["SPLIT"] == "TRAIN").sum()),
             int((samples["SPLIT"] == "VALIDATION").sum()),
             int((samples["SPLIT"] == "TEST").sum()))


if __name__ == "__main__":
    main()
