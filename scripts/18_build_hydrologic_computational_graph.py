#!/usr/bin/env python3
"""Build the formal 10 km2 MERIT hydrologic computational graph.

This is a self-contained orchestration step.  It reads the frozen 33 study
basins and existing MERIT/rainfall/DEM/CISC/CCAM facts, builds topology before
mapping any observations, validates the complete product in a staging
directory, and atomically publishes ``project/_hydrologic_graph_v1``.

Computational semantics
-----------------------
* MERIT UPA >= 10 km2 defines channel cells.
* Channel sources, confluences and the basin outlet define computational nodes.
* The channel path between adjacent computational nodes is a directed edge.
* Every basin cell follows MERIT D8 to its first computational node.  Those
  cells form that node's mutually exclusive local/unit catchment.
* ``incremental_area_km2`` is local runoff area. ``upstream_area_km2`` is the
  accumulated sum of this node and all upstream unit areas.
* Hydrologic stations are mapped only after topology is frozen and never
  participate in graph construction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from contextlib import ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import site
import sys
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window, shapes
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from shapely import make_valid, voronoi_polygons
from shapely.geometry import LineString, MultiPoint, Point, box, mapping, shape
from shapely.ops import nearest_points, unary_union


ACTIVE_34 = (
    "Q_61113300", "Q_61115200", "Q_61116401", "Q_611E0340",
    "Q_611E0360", "Q_611E0380", "Q_611E0405", "Q_611E1000",
    "Q_611E1500", "Q_611E2185", "Q_611E2370", "Q_611E2480",
    "Q_611E2524", "Q_611E2550", "Q_611E2600", "Q_611E2620",
    "Q_611E2645", "Q_61205550", "Q_612E1800", "Q_612E1810",
    "Q_612E1900", "Q_612E1950", "Q_612E2390", "Q_612E2460",
    "Q_612E2600", "Q_612E2820", "Q_61304550", "Q_61309200",
    "Q_61310900", "Q_613E2810", "Q_613E2870", "Q_61402050",
    "Q_61512000", "Q_62306134",
)
EXCLUDED_GRAPH = "Q_61512000"
EXCLUDED_STATION = "61512000"
ACTIVE_33 = tuple(value for value in ACTIVE_34 if value != EXCLUDED_GRAPH)
CHANNEL_THRESHOLD_KM2 = 10.0
VALID_D8 = frozenset((1, 2, 4, 8, 16, 32, 64, 128))
D8_OFFSET = {
    1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
    16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1),
}
GEOD = Geod(ellps="WGS84")


def parse_args() -> argparse.Namespace:
    workflow = Path(__file__).resolve().parent
    root = workflow.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basins", type=Path, default=workflow / "00f_q_small_basin_spatial/full_catchments.gpkg")
    parser.add_argument("--study-nodes", type=Path, default=workflow / "00f_q_small_basin_spatial/nodes.csv")
    parser.add_argument("--active-dataset", type=Path, default=root / "project/_model_dataset_v7_event_multitask")
    parser.add_argument("--upa", type=Path, default=workflow / "02_merit/MERIT_Hydro_v1.0.1_upa_work.tif")
    parser.add_argument("--flow-direction", type=Path, default=workflow / "02_merit/MERIT_Hydro_v1.0.1_dir_work_arcgis.tif")
    parser.add_argument("--merit-manifest", type=Path, default=workflow / "02_merit/merit_workarea_manifest.json")
    parser.add_argument("--rain-master", type=Path, default=workflow / "08_rain_station_master/rain_station_master.csv")
    parser.add_argument("--rain-sparse-index", type=Path, default=workflow / "10q_q_small_basin_rain_sparse/selected_rain_sparse_index.csv")
    parser.add_argument("--dem", type=Path, default=root / "Arcgis/projected_hydrology/project_dem.tif")
    parser.add_argument("--project-flow-dir", type=Path, default=root / "Arcgis/projected_hydrology/flow_dir.tif")
    parser.add_argument("--flow-acc", type=Path, default=root / "Arcgis/projected_hydrology/flow_accumu.tif")
    parser.add_argument("--cisc", type=Path, default=root / "CISC2022_StationRiver/cisc2022.tif")
    parser.add_argument("--cisc-valid-mask", type=Path, default=root / "CISC2022_StationRiver/valid_mask_2022.tif")
    parser.add_argument("--ccam-root", type=Path, default=root / "Arcgis/CCAM")
    parser.add_argument("--output-dir", type=Path, default=root / "project/_hydrologic_graph_v1")
    parser.add_argument("--rain-buffer-km", type=float, default=30.0)
    parser.add_argument("--area-closure-fail-pct", type=float, default=1.0)
    parser.add_argument("--station-snap-fail-m", type=float, default=5000.0)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{label} missing: {result}")
    return result


def norm_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raster_pair_is_aligned(upa_ds, dir_ds) -> bool:
    return bool(
        upa_ds.crs == dir_ds.crs and upa_ds.transform == dir_ds.transform
        and upa_ds.width == dir_ds.width and upa_ds.height == dir_ds.height
        and upa_ds.bounds == dir_ds.bounds and upa_ds.crs is not None
        and upa_ds.crs.to_epsg() == 4326
    )


def raster_covers_geometry(dataset, bounds) -> bool:
    minx, miny, maxx, maxy = bounds
    tolerance = min(abs(dataset.transform.a), abs(dataset.transform.e)) * 0.25
    return bool(
        dataset.bounds.left <= minx + tolerance
        and dataset.bounds.bottom <= miny + tolerance
        and dataset.bounds.right >= maxx - tolerance
        and dataset.bounds.top >= maxy - tolerance
    )


def manifest_tile_pairs(manifest_path: Path) -> list[tuple[str, Path, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    tile_files = manifest.get("tile_files", {})
    upa = {Path(p).stem.removesuffix("_upa"): require_file(Path(p), "MERIT UPA tile") for p in tile_files.get("upa", [])}
    direction = {Path(p).stem.removesuffix("_dir"): require_file(Path(p), "MERIT DIR tile") for p in tile_files.get("dir", [])}
    if not upa or set(upa) != set(direction):
        raise ValueError("MERIT manifest UPA/DIR tile keys are empty or mismatched")
    return [(f"manifest_tile:{key}", upa[key], direction[key]) for key in sorted(upa)]


def downstream_cell(cell: int, direction: np.ndarray) -> int | None:
    height, width = direction.shape
    row, col = divmod(int(cell), width)
    offset = D8_OFFSET.get(int(direction[row, col]))
    if offset is None:
        return None
    rr, cc = row + offset[0], col + offset[1]
    return rr * width + cc if 0 <= rr < height and 0 <= cc < width else None


def topological_order(nodes: set[int], edges: list[tuple[int, int]]) -> list[int]:
    incoming = {node: 0 for node in nodes}
    outgoing: dict[int, list[int]] = {node: [] for node in nodes}
    for source, target in edges:
        if source == target or source not in nodes or target not in nodes:
            raise ValueError("invalid topology edge")
        outgoing[source].append(target)
        incoming[target] += 1
    queue = sorted(node for node, degree in incoming.items() if degree == 0)
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in sorted(outgoing[node]):
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(nodes):
        raise ValueError("computational graph is not a DAG")
    return order


def longest_hops(nodes: set[int], edges: list[tuple[int, int]], outlet: int) -> int:
    order = topological_order(nodes, edges)
    hops = {node: 0 for node in nodes}
    for source in order:
        for edge_source, target in edges:
            if edge_source == source:
                hops[target] = max(hops[target], hops[source] + 1)
    if outlet != order[-1]:
        raise ValueError("outlet is not the unique terminal node")
    return int(hops[outlet])


def line_length_m(coordinates: list[tuple[float, float]]) -> float:
    total = 0.0
    for first, second in zip(coordinates[:-1], coordinates[1:]):
        _, _, distance = GEOD.inv(first[0], first[1], second[0], second[1])
        total += float(distance)
    return total


def resolve_node_labels(basin_mask: np.ndarray, direction: np.ndarray, boundary_label: dict[int, int]) -> tuple[np.ndarray, int]:
    inside = set(np.flatnonzero(basin_mask.ravel()).tolist())
    cache = dict(boundary_label)
    for start in inside:
        if start in cache:
            continue
        path: list[int] = []
        seen: set[int] = set()
        current = start
        label = -1
        while True:
            if current in cache:
                label = int(cache[current])
                break
            if current not in inside or current in seen:
                break
            seen.add(current)
            path.append(current)
            nxt = downstream_cell(current, direction)
            if nxt is None:
                break
            current = nxt
        for cell in path:
            cache[cell] = label
    labels = np.full(basin_mask.size, -1, dtype=np.int32)
    for cell in inside:
        labels[cell] = int(cache.get(cell, -1))
    unassigned = int((labels[np.fromiter(inside, dtype=np.int64)] < 0).sum())
    return labels.reshape(basin_mask.shape), unassigned


def polygonize_units(labels: np.ndarray, transform, basin_geometry, count: int) -> list:
    parts: dict[int, list] = defaultdict(list)
    for geometry, value in shapes(labels.astype(np.int32), mask=labels >= 0, transform=transform):
        parts[int(value)].append(shape(geometry))
    units = []
    for index in range(count):
        if not parts[index]:
            raise ValueError(f"node {index} has no unit cells")
        geom = make_valid(unary_union(parts[index]).intersection(basin_geometry))
        if geom.is_empty:
            raise ValueError(f"node {index} unit became empty after basin clip")
        units.append(geom)
    # Centre-cell rasterization omits only thin boundary slivers. Assign every
    # omitted polygon component to its nearest unit so the published units form
    # a complete, non-overlapping partition of the authoritative basin polygon.
    covered = unary_union(units)
    missing = make_valid(basin_geometry.difference(covered))
    if not missing.is_empty:
        components = list(missing.geoms) if hasattr(missing, "geoms") else [missing]
        for component in components:
            if component.is_empty or component.area == 0:
                continue
            index = min(range(count), key=lambda idx: units[idx].distance(component))
            units[index] = make_valid(units[index].union(component))
    return units


def build_basin_graph(graph_id: str, outlet_row: pd.Series, basin_row: pd.Series, upa_ds, dir_ds, raster_source: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    basin_geometry = make_valid(basin_row.geometry)
    window = geometry_window(upa_ds, [mapping(basin_geometry)], pad_x=1, pad_y=1)
    transform = upa_ds.window_transform(window)
    upa = upa_ds.read(1, window=window).astype(np.float64)
    direction = dir_ds.read(1, window=window)
    basin_mask = geometry_mask([mapping(basin_geometry)], out_shape=upa.shape, transform=transform, invert=True, all_touched=False)
    if not basin_mask.any():
        raise ValueError(f"{graph_id}: basin contains no MERIT cell")
    global_row, global_col = upa_ds.index(float(outlet_row.SNAP_LON), float(outlet_row.SNAP_LAT))
    outlet_row_local = int(global_row - window.row_off)
    outlet_col_local = int(global_col - window.col_off)
    if not (0 <= outlet_row_local < upa.shape[0] and 0 <= outlet_col_local < upa.shape[1]):
        raise ValueError(f"{graph_id}: outlet outside raster window")
    if not basin_mask[outlet_row_local, outlet_col_local]:
        raise ValueError(f"{graph_id}: outlet outside basin mask")
    height, width = upa.shape
    outlet_cell = outlet_row_local * width + outlet_col_local
    valid = np.isfinite(upa)
    if upa_ds.nodata is not None:
        valid &= upa != float(upa_ds.nodata)
    channel = set(np.flatnonzero((basin_mask & valid & (upa >= CHANNEL_THRESHOLD_KM2)).ravel()).tolist())

    if not channel or channel == {outlet_cell}:
        boundaries = {outlet_cell}
        sources = {outlet_cell}
        confluences: set[int] = set()
        reach_paths: list[tuple[int, int, list[int]]] = []
    else:
        if outlet_cell not in channel:
            raise ValueError(f"{graph_id}: outlet is below the fixed 10 km2 channel threshold")
        incoming: dict[int, list[int]] = {cell: [] for cell in channel}
        down: dict[int, int] = {}
        for cell in channel:
            nxt = downstream_cell(cell, direction)
            if nxt in channel:
                down[cell] = int(nxt)
                incoming[int(nxt)].append(cell)
        connected: set[int] = set()
        stack = [outlet_cell]
        while stack:
            cell = stack.pop()
            if cell in connected:
                continue
            connected.add(cell)
            stack.extend(incoming.get(cell, ()))
        if connected != channel:
            raise ValueError(f"{graph_id}: {len(channel-connected)} threshold channel cells do not reach outlet")
        indegree = {cell: len(incoming.get(cell, ())) for cell in connected}
        sources = {cell for cell, degree in indegree.items() if degree == 0}
        confluences = {cell for cell, degree in indegree.items() if degree >= 2}
        boundaries = sources | confluences | {outlet_cell}
        reach_paths = []
        for source in sorted(boundaries - {outlet_cell}):
            path = [source]
            current = source
            seen = {source}
            while current != outlet_cell:
                nxt = down.get(current)
                if nxt is None or nxt in seen:
                    raise ValueError(f"{graph_id}: broken/cyclic MERIT channel path")
                path.append(nxt)
                seen.add(nxt)
                current = nxt
                if current in boundaries:
                    reach_paths.append((source, current, path))
                    break

    cell_edges = [(source, target) for source, target, _ in reach_paths]
    order = topological_order(boundaries, cell_edges)
    if sum(node not in {outlet_cell} and all(source != node for source, _ in cell_edges) for node in boundaries):
        raise ValueError(f"{graph_id}: non-outlet terminal node")
    if [node for node in boundaries if all(source != node for source, _ in cell_edges)] != [outlet_cell]:
        raise ValueError(f"{graph_id}: graph does not have exactly one outlet")
    node_index = {cell: index for index, cell in enumerate(order)}
    node_id = {cell: f"{graph_id}_N{index + 1:04d}" for index, cell in enumerate(order)}
    labels, unassigned = resolve_node_labels(basin_mask, direction, {cell: node_index[cell] for cell in boundaries})
    if unassigned:
        raise ValueError(f"{graph_id}: {unassigned} basin cells do not drain to a computational node")
    unit_geometries = polygonize_units(labels, transform, basin_geometry, len(order))
    area_series = gpd.GeoSeries(unit_geometries, crs=4326).to_crs(6933).area / 1e6

    accumulated = {cell: float(area_series.iloc[node_index[cell]]) for cell in order}
    outgoing = {source: target for source, target in cell_edges}
    for cell in order:
        if cell in outgoing:
            accumulated[outgoing[cell]] += accumulated[cell]

    nodes: list[dict] = []
    units: list[dict] = []
    for cell in order:
        row, col = divmod(cell, width)
        lon, lat = rasterio.transform.xy(transform, row, col, offset="center")
        index = node_index[cell]
        kind = "OUTLET" if cell == outlet_cell else ("CONFLUENCE" if cell in confluences else "SOURCE")
        incoming_count = sum(target == cell for _, target in cell_edges)
        out_count = int(cell != outlet_cell)
        common = {
            "GRAPH_ID": graph_id, "NODE_ID": node_id[cell], "NODE_INDEX": index,
            "OUTLET_ID": norm_id(outlet_row.OUTLET_ID), "NODE_TYPE": kind,
            "IS_OUTLET": int(cell == outlet_cell), "LON": float(lon), "LAT": float(lat),
            "IN_DEGREE": incoming_count, "OUT_DEGREE": out_count,
            "MERIT_UPA_KM2": float(upa[row, col]),
            "incremental_area_km2": float(area_series.iloc[index]),
            "upstream_area_km2": float(accumulated[cell]),
            "MERIT_RASTER_SOURCE": raster_source,
        }
        nodes.append(common)
        units.append({**common, "geometry": unit_geometries[index]})

    edges: list[dict] = []
    for edge_index, (source, target, path) in enumerate(reach_paths):
        coordinates = []
        for cell in path:
            row, col = divmod(cell, width)
            lon, lat = rasterio.transform.xy(transform, row, col, offset="center")
            coordinates.append((float(lon), float(lat)))
        length = line_length_m(coordinates)
        if not math.isfinite(length) or length <= 0:
            raise ValueError(f"{graph_id}: nonpositive reach length")
        edges.append({
            "GRAPH_ID": graph_id, "EDGE_ID": f"{graph_id}_E{edge_index + 1:04d}",
            "FROM_NODE_ID": node_id[source], "TO_NODE_ID": node_id[target],
            "FROM_NODE_INDEX": node_index[source], "TO_NODE_INDEX": node_index[target],
            "REACH_LENGTH_M": length, "geometry": LineString(coordinates),
        })

    basin_area = float(gpd.GeoSeries([basin_geometry], crs=4326).to_crs(6933).area.iloc[0] / 1e6)
    unit_union = unary_union(unit_geometries)
    unit_total = float(area_series.sum())
    overlap_area = max(0.0, unit_total - float(gpd.GeoSeries([unit_union], crs=4326).to_crs(6933).area.iloc[0] / 1e6))
    missing_area = float(gpd.GeoSeries([basin_geometry.difference(unit_union)], crs=4326).to_crs(6933).area.iloc[0] / 1e6)
    closure_pct = 100.0 * (unit_total - basin_area) / basin_area
    metrics = {
        "GRAPH_ID": graph_id, "OUTLET_ID": norm_id(outlet_row.OUTLET_ID),
        "NODE_COUNT": len(nodes), "EDGE_COUNT": len(edges),
        "SOURCE_COUNT": len(sources), "CONFLUENCE_COUNT": len(confluences),
        "MAX_HOPS": longest_hops(boundaries, cell_edges, outlet_cell),
        "BASIN_AREA_KM2": basin_area, "TOTAL_UNIT_AREA_KM2": unit_total,
        "AREA_CLOSURE_ERROR_PCT": closure_pct, "OVERLAP_AREA_KM2": overlap_area,
        "MISSING_AREA_KM2": missing_area, "IS_DAG": 1, "OUTLET_CONNECTED": 1,
        "MERIT_RASTER_SOURCE": raster_source,
    }
    return nodes, edges, units, metrics


def map_observation_stations(study_nodes: pd.DataFrame, node_catalog: pd.DataFrame, edge_gdf: gpd.GeoDataFrame, station_snap_fail_m: float) -> pd.DataFrame:
    node_points = gpd.GeoDataFrame(node_catalog.copy(), geometry=gpd.points_from_xy(node_catalog.LON, node_catalog.LAT), crs=4326).to_crs(32649)
    reaches = edge_gdf.to_crs(32649)
    records = []
    for graph_id in ACTIVE_33:
        stations = study_nodes[study_nodes.GRAPH_ID.eq(graph_id)].copy()
        graph_nodes = node_points[node_points.GRAPH_ID.eq(graph_id)]
        graph_edges = reaches[reaches.GRAPH_ID.eq(graph_id)]
        outlet_node = graph_nodes[graph_nodes.IS_OUTLET.eq(1)]
        if len(outlet_node) != 1:
            raise ValueError(f"{graph_id}: expected one computational outlet")
        for station in stations.itertuples(index=False):
            point = gpd.GeoSeries([Point(float(station.SNAP_LON), float(station.SNAP_LAT))], crs=4326).to_crs(32649).iloc[0]
            if int(station.IS_OUTLET) == 1 or graph_edges.empty:
                mapped = outlet_node.iloc[0]
                snap_point = mapped.geometry
                distance = float(point.distance(snap_point))
                edge_id = ""
                method = "FORCED_OUTLET_NODE" if int(station.IS_OUTLET) == 1 else "SINGLE_NODE_GRAPH"
            else:
                edge_index = min(graph_edges.index, key=lambda idx: graph_edges.at[idx, "geometry"].distance(point))
                edge = graph_edges.loc[edge_index]
                line = edge.geometry
                along = float(line.project(point))
                snap_point = line.interpolate(along)
                distance = float(point.distance(snap_point))
                # A station immediately upstream of the channel-initiation point
                # observes the source unit; elsewhere the reach drains to TO_NODE.
                mapped_id = edge.FROM_NODE_ID if along <= 50.0 else edge.TO_NODE_ID
                mapped = graph_nodes[graph_nodes.NODE_ID.eq(mapped_id)].iloc[0]
                edge_id = edge.EDGE_ID
                method = "NEAREST_REACH_TO_SOURCE" if along <= 50.0 else "NEAREST_REACH_TO_DOWNSTREAM_UNIT"
            snapped_wgs = gpd.GeoSeries([snap_point], crs=32649).to_crs(4326).iloc[0]
            records.append({
                "GRAPH_ID": graph_id, "STATION_ID": norm_id(station.STATION_ID),
                "STATION_ROLE": station.ROLE, "IS_OUTLET_STATION": int(station.IS_OUTLET),
                "OBSERVATION_LON": float(station.SNAP_LON), "OBSERVATION_LAT": float(station.SNAP_LAT),
                "MAPPED_NODE_ID": mapped.NODE_ID, "MAPPED_NODE_INDEX": int(mapped.NODE_INDEX),
                "MAPPED_EDGE_ID": edge_id, "GRAPH_SNAP_LON": float(snapped_wgs.x),
                "GRAPH_SNAP_LAT": float(snapped_wgs.y), "SNAP_DISTANCE_M": distance,
                "MAPPING_METHOD": method,
            })
    result = pd.DataFrame(records).sort_values(["GRAPH_ID", "IS_OUTLET_STATION", "STATION_ID"])
    if len(result) != len(study_nodes) or result.STATION_ID.eq(EXCLUDED_STATION).any():
        raise ValueError("station observation mapping is incomplete or contains excluded station")
    if (result.SNAP_DISTANCE_M > station_snap_fail_m).any():
        bad = result.loc[result.SNAP_DISTANCE_M > station_snap_fail_m, ["GRAPH_ID", "STATION_ID", "SNAP_DISTANCE_M"]]
        raise ValueError(f"station-to-graph snap exceeds {station_snap_fail_m} m:\n{bad}")
    outlet_map = result[result.IS_OUTLET_STATION.eq(1)].merge(node_catalog[["GRAPH_ID", "NODE_ID", "IS_OUTLET"]], left_on=["GRAPH_ID", "MAPPED_NODE_ID"], right_on=["GRAPH_ID", "NODE_ID"], validate="one_to_one")
    if len(outlet_map) != 33 or not outlet_map.IS_OUTLET.eq(1).all():
        raise ValueError("an outlet station was not mapped to the computational outlet node")
    return result


def build_rain_weights(
    units: gpd.GeoDataFrame,
    rain_master_path: Path,
    buffer_km: float,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rain = pd.read_csv(rain_master_path, dtype={"station_id": str}, encoding="utf-8-sig")
    required = {"station_id", "longitude", "latitude", "first_time", "last_time", "record_count", "qc_status"}
    if not required.issubset(rain.columns):
        raise ValueError(f"rain master missing fields: {sorted(required-set(rain.columns))}")
    rain = rain[rain.qc_status.eq("ACCEPT")].copy()
    rain["longitude"] = pd.to_numeric(rain.longitude, errors="coerce")
    rain["latitude"] = pd.to_numeric(rain.latitude, errors="coerce")
    rain["record_count"] = pd.to_numeric(rain.record_count, errors="coerce").fillna(0)
    rain = rain.dropna(subset=["longitude", "latitude"])
    rain_gdf = gpd.GeoDataFrame(rain, geometry=gpd.points_from_xy(rain.longitude, rain.latitude), crs=4326).to_crs(32649)
    unit_projected = units.to_crs(32649)
    # Reprojection can expose tiny ring defects at raster-cell seams.  Repair
    # per unit; a full-study union is unnecessary merely to select gauges and
    # is much more susceptible to GEOS topology failures.
    unit_projected["geometry"] = unit_projected.geometry.map(
        lambda geom: make_valid(geom) if geom.is_valid else make_valid(geom.buffer(0))
    )
    study = box(*unit_projected.total_bounds).buffer(buffer_km * 1000.0)
    rain_gdf = rain_gdf[rain_gdf.geometry.intersects(study)].copy()
    rain_gdf = rain_gdf.sort_values(["record_count", "station_id"], ascending=[False, True])
    rain_gdf["xy"] = list(zip(rain_gdf.geometry.x.round(3), rain_gdf.geometry.y.round(3)))
    rain_gdf = rain_gdf.drop_duplicates("xy").sort_values("station_id").reset_index(drop=True)
    if len(rain_gdf) < 3:
        raise ValueError("fewer than three accepted rain gauges around study basins")
    extent = box(*study.bounds)
    cells = list(voronoi_polygons(MultiPoint(list(rain_gdf.geometry)), extend_to=extent, ordered=True).geoms)
    voronoi = gpd.GeoDataFrame({"rain_station_id": rain_gdf.station_id.values}, geometry=cells, crs=32649)
    spatial_index = voronoi.sindex
    rows = []
    minimum_coverage = 1.0
    for unit in unit_projected.itertuples(index=False):
        geom = unit.geometry
        area = float(geom.area)
        candidates = spatial_index.query(geom, predicate="intersects")
        overlaps = []
        for candidate in candidates:
            overlap = float(geom.intersection(voronoi.geometry.iloc[candidate]).area)
            if overlap > 1e-3:
                overlaps.append((str(voronoi.rain_station_id.iloc[candidate]), overlap))
        covered = sum(value for _, value in overlaps)
        coverage = covered / area if area > 0 else 0.0
        minimum_coverage = min(minimum_coverage, coverage)
        if not overlaps or coverage < 0.999:
            raise ValueError(f"{unit.GRAPH_ID}/{unit.NODE_ID}: Thiessen coverage={coverage:.6f}")
        for station_id, overlap in overlaps:
            rows.append({
                "GRAPH_ID": unit.GRAPH_ID, "NODE_ID": unit.NODE_ID,
                "rain_station_id": station_id, "weight": overlap / covered,
            })
    weights = pd.DataFrame(rows)
    weight_error = float((weights.groupby(["GRAPH_ID", "NODE_ID"]).weight.sum() - 1.0).abs().max())
    if weight_error > 1e-9:
        raise ValueError(f"rainfall weights do not sum to one: {weight_error}")
    metadata = rain_gdf.drop(columns=["geometry", "xy"]).copy()
    lookup = metadata.set_index("station_id")
    coverage_rows = []
    for (graph_id, node_id), group in weights.groupby(["GRAPH_ID", "NODE_ID"], sort=True):
        selected = lookup.loc[group.rain_station_id]
        source_first = pd.to_datetime(selected.first_time, errors="coerce")
        source_last = pd.to_datetime(selected.last_time, errors="coerce")
        if source_first.isna().any() or source_last.isna().any():
            raise ValueError(f"{graph_id}/{node_id}: rain source time metadata missing")
        coverage_rows.append({
            "GRAPH_ID": graph_id, "NODE_ID": node_id,
            # The frozen v7 source contract materializes every unrecorded
            # Step11 graph-node hour as zero with RAIN_MASK=1 across the global
            # timeline.  Gauge first/last times are retained as provenance, not
            # intersected into an artificial all-gauge common interval.
            "VALID_START": valid_start.strftime("%Y-%m-%d %H:%M:%S"),
            "VALID_END": valid_end.strftime("%Y-%m-%d %H:%M:%S"),
            "SOURCE_FIRST_TIME_MIN": source_first.min().strftime("%Y-%m-%d %H:%M:%S"),
            "SOURCE_LAST_TIME_MAX": source_last.max().strftime("%Y-%m-%d %H:%M:%S"),
            "GAUGE_COUNT": len(group), "WEIGHT_SUM": float(group.weight.sum()),
            "ZERO_SEMANTICS": "ABSENT_SPARSE_ROW_WITHIN_VALID_PERIOD_IS_0_MM",
        })
    return weights, pd.DataFrame(coverage_rows), {
        "gauge_count": int(weights.rain_station_id.nunique()),
        "weight_count": len(weights), "minimum_thiessen_coverage": minimum_coverage,
        "maximum_weight_sum_error": weight_error,
    }


def aggregate_rainfall(weights: pd.DataFrame, sparse_index_path: Path, rain_dir: Path) -> tuple[dict[str, int], int]:
    sparse_index = pd.read_csv(sparse_index_path, dtype=str, encoding="utf-8-sig")
    sparse_index["station_id"] = sparse_index.station_id.map(norm_id)
    sparse_lookup = sparse_index.set_index("station_id").sparse_file.to_dict()
    target_map: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for row in weights.itertuples(index=False):
        target_map[norm_id(row.rain_station_id)].append((row.GRAPH_ID, row.NODE_ID, float(row.weight)))
    rain_dir.parent.mkdir(parents=True, exist_ok=True)
    db_path = rain_dir.parent / "rain_aggregation.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=OFF")
        cursor.execute("PRAGMA synchronous=OFF")
        cursor.execute("CREATE TABLE c (graph_id TEXT,node_id TEXT,start_time TEXT,end_time TEXT,value REAL)")
        insert = "INSERT INTO c VALUES (?,?,?,?,?)"
        missing_sparse = 0
        for number, station_id in enumerate(sorted(target_map), 1):
            file_value = sparse_lookup.get(station_id)
            if not file_value or not Path(file_value).is_file():
                # The sparse product contains only positive one-hour rainfall;
                # a gauge with no sparse file contributes zero throughout its
                # valid period and therefore needs no inserted record.
                missing_sparse += 1
                continue
            frame = pd.read_csv(file_value, usecols=["start_time", "end_time", "rainfall_mm", "interval_hours"], encoding="utf-8-sig")
            frame["rainfall_mm"] = pd.to_numeric(frame.rainfall_mm, errors="coerce")
            frame["interval_hours"] = pd.to_numeric(frame.interval_hours, errors="coerce")
            frame = frame[frame.rainfall_mm.gt(0) & frame.interval_hours.sub(1.0).abs().le(0.01)]
            if frame.empty:
                continue
            base = list(frame[["start_time", "end_time", "rainfall_mm"]].itertuples(index=False, name=None))
            for graph_id, node_id, weight in target_map[station_id]:
                batch = [(graph_id, node_id, start, end, float(value) * weight) for start, end, value in base]
                cursor.executemany(insert, batch)
            if number % 50 == 0:
                connection.commit()
        connection.commit()
        cursor.execute("CREATE INDEX c_idx ON c(graph_id,start_time,node_id)")
        connection.commit()
        rain_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for graph_id in ACTIVE_33:
            query = "SELECT graph_id,node_id,start_time,end_time,SUM(value) AS rain_mm FROM c WHERE graph_id=? GROUP BY graph_id,node_id,start_time,end_time HAVING SUM(value)>0 ORDER BY start_time,node_id"
            frame = pd.read_sql_query(query, connection, params=(graph_id,))
            frame.columns = ["GRAPH_ID", "NODE_ID", "START_TIME", "END_TIME", "RAIN_MM"]
            frame.to_csv(rain_dir / f"graph_{graph_id}_hourly_sparse.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
            counts[graph_id] = len(frame)
    finally:
        connection.close()
        if db_path.exists():
            db_path.unlink()
    return counts, missing_sparse


def compute_static_attributes(units: gpd.GeoDataFrame, nodes: pd.DataFrame, edges: gpd.GeoDataFrame, args, workflow: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    step14 = load_module("step14_static_helpers", workflow / "14_compute_dem_cisc_static_attributes_py3.py")
    # The project venv intentionally stays lean, while the same Python 3.13
    # installation already provides xarray/scipy in the user's standard site.
    # Reuse that installed read-only NetCDF stack without installing packages.
    if importlib.util.find_spec("xarray") is None:
        user_site = Path(site.getusersitepackages())
        if user_site.is_dir() and str(user_site) not in sys.path:
            sys.path.append(str(user_site))
    step15 = load_module("step15_static_helpers", workflow / "15_compute_ccam_static_attributes_py3_fallback.py")
    if step15.xr is None:
        raise RuntimeError("PDEP.nc requires the existing xarray/scipy installation, but it is unavailable to this interpreter")
    dem_path = require_file(args.dem, "project DEM")
    project_dir_path = require_file(args.project_flow_dir, "projected flow direction")
    acc_path = require_file(args.flow_acc, "projected flow accumulation")
    cisc_path = require_file(args.cisc, "CISC")
    valid_path = require_file(args.cisc_valid_mask, "CISC valid mask")
    ccam_root = args.ccam_root.expanduser().resolve()
    ksat_paths = [require_file(ccam_root / "soil_ksat" / f"log_k_s_l{i}", f"CCAM Ksat L{i}") for i in range(1, 4)]
    pdep_path = require_file(ccam_root / "soil_profile/PDEP.nc", "CCAM PDEP")
    igbp_path = require_file(ccam_root / "land_cover/processed_igbp.tif", "CCAM IGBP")

    units_wgs = units.to_crs(4326)
    total_bounds = units_wgs.total_bounds
    bounds = (total_bounds[0] - 0.05, total_bounds[1] - 0.05, total_bounds[2] + 0.05, total_bounds[3] + 0.05)
    spec = step15.detect_binary_grid(ksat_paths[0])
    indices = step15.crop_indices(bounds, spec, "north_to_south", pad=1)
    k_arrays, k_transforms = [], []
    for path in ksat_paths:
        layer_spec = step15.detect_binary_grid(path)
        array, transform = step15.read_binary_window(path, layer_spec, indices, "little", "north_to_south")
        array[(~np.isfinite(array)) | (array < -10) | (array > 10)] = np.nan
        k_arrays.append(array)
        k_transforms.append(transform)
    pdep, pdep_transform, _, _, _ = step15.load_pdep_subset(pdep_path, "PDEP1", bounds)
    pdep[(~np.isfinite(pdep)) | (pdep < 0) | (pdep > 10000)] = np.nan

    node_rows = []
    fallback_count = 0
    coarse_count = 0
    with rasterio.open(dem_path) as dem_ds, rasterio.open(acc_path) as acc_ds, rasterio.open(project_dir_path) as raw_dir_ds, rasterio.open(cisc_path) as cisc_ds, rasterio.open(valid_path) as raw_valid_ds, rasterio.open(igbp_path) as igbp_ds:
        dir_context = None
        if step14.check_raster_grid_compat(acc_ds, raw_dir_ds):
            flow_dir_ds = raw_dir_ds
        else:
            dir_context = WarpedVRT(raw_dir_ds, crs=acc_ds.crs, transform=acc_ds.transform, width=acc_ds.width, height=acc_ds.height, resampling=Resampling.nearest)
            flow_dir_ds = dir_context
        valid_context = None
        if step14.check_raster_grid_compat(cisc_ds, raw_valid_ds):
            valid_ds = raw_valid_ds
        else:
            valid_context = WarpedVRT(raw_valid_ds, crs=cisc_ds.crs, transform=cisc_ds.transform, width=cisc_ds.width, height=cisc_ds.height, resampling=Resampling.nearest)
            valid_ds = valid_context
        units_dem = units.to_crs(dem_ds.crs)
        units_flow = units.to_crs(acc_ds.crs)
        units_cisc = units.to_crs(cisc_ds.crs)
        threshold_cells = int(math.ceil(2.0e6 / abs(float(acc_ds.transform.a * acc_ds.transform.e))))
        try:
            for position in range(len(units)):
                source = units.iloc[position]
                geom_wgs = units_wgs.geometry.iloc[position]
                dem_stats = step14.calc_dem_stats(dem_ds, units_dem.geometry.iloc[position])
                flow_data = step14.read_flow_window(acc_ds, flow_dir_ds, units_flow.geometry.iloc[position], 5000.0)
                if flow_data is None:
                    raise ValueError(f"{source.GRAPH_ID}/{source.NODE_ID}: projected flow raster does not overlap unit")
                acc, fdir, acc_valid, poly_mask, transform = flow_data
                flow_stats = step14.mean_flow_distance_and_density(acc, fdir, acc_valid, poly_mask, transform, threshold_cells)
                flow_stats["drainage_density_km_per_km2"] = flow_stats["stream_length_km"] / float(source.incremental_area_km2)
                cisc_stats = step14.calc_cisc_stats(cisc_ds, valid_ds, units_cisc.geometry.iloc[position], 1.0)

                k_values, k_coverages, k_counts, methods = [], [], [], []
                for array, transform in zip(k_arrays, k_transforms):
                    value = step15.continuous_stats(array, transform, geom_wgs, -10, 10)
                    method = "POLYGON_MEAN"
                    if not np.isfinite(value["mean"]):
                        fallback = step15.continuous_centroid_nearest_fallback(array, transform, geom_wgs, -10, 10, 50.0)
                        value["mean"] = fallback["value"]
                        method = fallback["method"]
                    k_values.append(value["mean"])
                    k_coverages.append(value["coverage"])
                    k_counts.append(value["count"])
                    methods.append(method)
                pdep_stats = step15.continuous_stats(pdep, pdep_transform, geom_wgs, 0, 10000)
                pdep_method = "POLYGON_MEAN"
                if not np.isfinite(pdep_stats["mean"]):
                    fallback = step15.continuous_centroid_nearest_fallback(pdep, pdep_transform, geom_wgs, 0, 10000, 50.0)
                    pdep_stats["mean"] = fallback["value"]
                    pdep_method = fallback["method"]
                igbp_stats = step15.igbp_stats(igbp_ds, geom_wgs)
                igbp_method = "POLYGON_MODE_FRACTION"
                if not np.isfinite(igbp_stats["forest_fraction"]):
                    fallback = step15.igbp_centroid_nearest_fallback(igbp_ds, geom_wgs, 50.0)
                    igbp_stats["forest_fraction"] = fallback["forest_fraction"]
                    igbp_method = fallback["method"]
                used_fallback = any(method != "POLYGON_MEAN" for method in methods) or pdep_method != "POLYGON_MEAN" or igbp_method != "POLYGON_MODE_FRACTION"
                coarse = min(k_counts) < 3 or pdep_stats["count"] < 3 or igbp_stats["valid_count"] < 3
                fallback_count += int(used_fallback)
                coarse_count += int(coarse and not used_fallback)
                soil_log = (5 * k_values[0] + 10 * k_values[1] + 15 * k_values[2]) / 30.0
                record = {
                    "GRAPH_ID": source.GRAPH_ID, "NODE_ID": source.NODE_ID,
                    "NODE_INDEX": int(source.NODE_INDEX), "OUTLET_ID": source.OUTLET_ID,
                    "NODE_TYPE": source.NODE_TYPE, "IS_OUTLET": int(source.IS_OUTLET),
                    "incremental_area_km2": float(source.incremental_area_km2),
                    "upstream_area_km2": float(source.upstream_area_km2),
                    "MERIT_UPA_KM2": float(source.MERIT_UPA_KM2),
                    "log_incremental_area": math.log1p(float(source.incremental_area_km2)),
                    "log_upstream_area": math.log1p(float(source.upstream_area_km2)),
                    "mean_hillslope_flow_distance_m": flow_stats["mean_hillslope_flow_distance_m"],
                    "mean_slope_deg": dem_stats["mean_slope_deg"],
                    "elevation_mean_m": dem_stats["elevation_mean_m"],
                    "elevation_std_m": dem_stats["elevation_std_m"],
                    "drainage_density_km_per_km2": flow_stats["drainage_density_km_per_km2"],
                    "soil_log_ksat_0_30cm": soil_log,
                    "soil_profile_depth_cm": pdep_stats["mean"],
                    "forest_fraction": igbp_stats["forest_fraction"],
                    "impervious_fraction": cisc_stats["impervious_fraction"],
                    "dem_valid_fraction": dem_stats["dem_valid_fraction"],
                    "flow_distance_valid_fraction": flow_stats["hillslope_distance_valid_fraction"],
                    "cisc_valid_fraction": cisc_stats["cisc_valid_fraction"],
                    "ksat_valid_fraction_min": min(k_coverages),
                    "pdep_valid_fraction": pdep_stats["coverage"],
                    "igbp_valid_fraction": igbp_stats["coverage"],
                    "CCAM_FALLBACK_USED": int(used_fallback),
                    "STATIC_QC": "REVIEW_SPATIAL_FALLBACK" if used_fallback else ("REVIEW_COARSE_SUPPORT" if coarse else "ACCEPT"),
                }
                core = [
                    "log_incremental_area", "log_upstream_area", "mean_hillslope_flow_distance_m",
                    "mean_slope_deg", "elevation_std_m", "drainage_density_km_per_km2",
                    "soil_log_ksat_0_30cm", "soil_profile_depth_cm", "forest_fraction", "impervious_fraction",
                ]
                if any(not np.isfinite(float(record[field])) for field in core):
                    missing = [field for field in core if not np.isfinite(float(record[field]))]
                    raise ValueError(f"{source.GRAPH_ID}/{source.NODE_ID}: missing core static attributes {missing}")
                node_rows.append(record)
        finally:
            if dir_context is not None:
                dir_context.close()
            if valid_context is not None:
                valid_context.close()

    node_static = pd.DataFrame(node_rows).sort_values(["GRAPH_ID", "NODE_INDEX"])
    node_points = gpd.GeoDataFrame(nodes.copy(), geometry=gpd.points_from_xy(nodes.LON, nodes.LAT), crs=4326)
    with rasterio.open(dem_path) as dem_ds:
        elevation = step14.sample_dem_at_nodes(node_points, "NODE_ID", dem_ds)
    edge_rows = []
    clamp_count = 0
    for edge in edges.itertuples(index=False):
        from_elev = float(elevation.get(edge.FROM_NODE_ID, np.nan))
        to_elev = float(elevation.get(edge.TO_NODE_ID, np.nan))
        if not np.isfinite(from_elev) or not np.isfinite(to_elev):
            raise ValueError(f"{edge.EDGE_ID}: missing DEM endpoint elevation")
        raw_drop = from_elev - to_elev
        raw_slope = raw_drop / float(edge.REACH_LENGTH_M)
        slope = max(raw_slope, 1e-6)
        clamped = raw_slope <= 0
        clamp_count += int(clamped)
        edge_rows.append({
            "GRAPH_ID": edge.GRAPH_ID, "EDGE_ID": edge.EDGE_ID,
            "FROM_NODE_ID": edge.FROM_NODE_ID, "TO_NODE_ID": edge.TO_NODE_ID,
            "FROM_NODE_INDEX": int(edge.FROM_NODE_INDEX), "TO_NODE_INDEX": int(edge.TO_NODE_INDEX),
            "reach_length_m": float(edge.REACH_LENGTH_M),
            "from_elevation_m": from_elev, "to_elevation_m": to_elev,
            "reach_elevation_drop_m": raw_drop, "reach_slope_raw_m_per_m": raw_slope,
            "reach_slope_m_per_m": slope,
            "SLOPE_QC": "CLAMPED_TO_1E-6" if clamped else "ACCEPT",
        })
    edge_static = pd.DataFrame(edge_rows).sort_values(["GRAPH_ID", "FROM_NODE_INDEX"])
    return node_static, edge_static, {"ccam_fallback_nodes": fallback_count, "ccam_coarse_nodes": coarse_count, "edge_slope_clamped": clamp_count}


def describe(values: Iterable[float]) -> str:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return "n=0"
    return f"median={np.median(array):.3f}, p10={np.quantile(array,.1):.3f}, p90={np.quantile(array,.9):.3f}, min={array.min():.3f}, max={array.max():.3f}"


def write_report(path: Path, metrics: pd.DataFrame, nodes: pd.DataFrame, edges: pd.DataFrame, units: gpd.GeoDataFrame, mapping_frame: pd.DataFrame, coverage: pd.DataFrame, rain_counts: dict[str, int], rain_info: dict, static_info: dict) -> None:
    per_graph = metrics.copy()
    per_graph["STATIONS"] = per_graph.GRAPH_ID.map(mapping_frame.groupby("GRAPH_ID").size())
    per_graph["MAX_STATION_SNAP_M"] = per_graph.GRAPH_ID.map(mapping_frame.groupby("GRAPH_ID").SNAP_DISTANCE_M.max())
    per_graph["RAIN_ROWS"] = per_graph.GRAPH_ID.map(rain_counts)
    headers = ["GRAPH_ID", "NODE_COUNT", "EDGE_COUNT", "MAX_HOPS", "BASIN_AREA_KM2", "TOTAL_UNIT_AREA_KM2", "AREA_CLOSURE_ERROR_PCT", "STATIONS", "MAX_STATION_SNAP_M", "RAIN_ROWS"]
    table = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers)-1)) + "|"]
    for row in per_graph[headers].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table.append("| " + " | ".join(values) + " |")
    static_review = int(node_static_review := 0)
    if "STATIC_QC" in nodes.columns:
        static_review = int((nodes.STATIC_QC != "ACCEPT").sum())
    text = f"""# Hydrologic Computational Graph v1 — BUILD AND QC

## 构建规则

- Study basins：固定 34 个正式研究流域中排除 `{EXCLUDED_GRAPH}` / station `{EXCLUDED_STATION}`，最终 33 个。
- Channelization：MERIT Hydro upstream area `UPA >= 10 km²`，不测试其他阈值。
- Topology：仅由 MERIT D8 河网的河源、汇流点和唯一 outlet 决定；水文站不参与拓扑。
- Node/unit：每个 computational node 对应直接汇入该节点、互不重叠的 unit catchment。
- Edge/reach：每条 edge 是相邻 computational nodes 之间的真实 MERIT D8 river reach。
- 面积语义：`incremental_area_km2` 是 local runoff area；`upstream_area_km2` 是本节点及全部上游 units 的累计面积；`MERIT_UPA_KM2` 是独立的 MERIT 像元上游面积。
- Rainfall：重新用新 unit polygon 与现有雨量站 Thiessen 面积交集计算固定权重；有效期内未出现在稀疏文件中的小时明确为 `0 mm`。
- Static：DEM/flow accumulation/CISC 和 CCAM Ksat/PDEP/IGBP 均按新 unit polygon 重新聚合。粗分辨率 CCAM 无中心像元时使用已有 centroid/nearest fallback，并保留标记。
- Spatial units：经纬度为 EPSG:4326；距离/河长为 m；面积为 km²；坡度为 m/m；高程为 m。

## 总体统计

- Basins: **{len(metrics)}**
- Computational nodes / unit catchments: **{len(nodes)}**
- Directed edges / river reaches: **{len(edges)}**
- Node count: {describe(metrics.NODE_COUNT)}
- Edge count: {describe(metrics.EDGE_COUNT)}
- MAX_HOPS: {describe(metrics.MAX_HOPS)}
- Unit area (km²): {describe(units.incremental_area_km2)}
- Reach length (m): {describe(edges.REACH_LENGTH_M)}
- Station observations mapped: **{len(mapping_frame)}**（outlet 33；internal {len(mapping_frame)-33}）
- Station snap distance (m): {describe(mapping_frame.SNAP_DISTANCE_M)}
- Rain gauges used: **{rain_info['gauge_count']}**；Thiessen minimum coverage: **{rain_info['minimum_thiessen_coverage']*100:.6f}%**
- Published positive-rain sparse rows: **{sum(rain_counts.values())}**；node validity records: **{len(coverage)}**
- Static review flags: **{static_review}**；CCAM fallback nodes: **{static_info['ccam_fallback_nodes']}**；coarse-support nodes: **{static_info['ccam_coarse_nodes']}**
- Nonpositive raw DEM reach slopes clamped to `1e-6 m/m`: **{static_info['edge_slope_clamped']}**（raw slope retained）。

## 每个 basin

{chr(10).join(table)}

## 完整 QC

- Graph count = 33；excluded basin/station absent: PASS
- All graphs directed DAG with one connected outlet: PASS
- Unit count equals node count: PASS
- Unit catchments overlap: max **{metrics.OVERLAP_AREA_KM2.max():.12g} km²**
- Unit completeness gap: max **{metrics.MISSING_AREA_KM2.max():.12g} km²**
- Area closure error: max absolute **{metrics.AREA_CLOSURE_ERROR_PCT.abs().max():.10g}%**
- All 33 outlet stations map to their final outlet node: PASS
- All observation mappings within configured distance: PASS
- Rainfall Thiessen coverage and weight sums: PASS
- Every node has rainfall validity bounds; sparse zero semantics recorded: PASS
- Core node static attributes have no missing/nonfinite values: PASS
- Edge length and routing slope have no missing/nonpositive values: PASS
- Formal directory was published only after all checks completed: PASS

**FINAL QC STATUS: PASS**
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    workflow = Path(__file__).resolve().parent
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing formal output: {output}")
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    required_paths = {
        "basins": require_file(args.basins, "study basin polygons"),
        "study_nodes": require_file(args.study_nodes, "study station nodes"),
        "active_catalog": require_file(args.active_dataset.expanduser().resolve() / "graph/node_catalog.csv", "frozen active node catalog"),
        "upa": require_file(args.upa, "MERIT UPA"),
        "direction": require_file(args.flow_direction, "MERIT flow direction"),
        "manifest": require_file(args.merit_manifest, "MERIT manifest"),
        "rain_master": require_file(args.rain_master, "rain station master"),
        "rain_sparse_index": require_file(args.rain_sparse_index, "rain sparse index"),
    }
    active = pd.read_csv(required_paths["active_catalog"], dtype=str, encoding="utf-8-sig")
    actual = set(active.GRAPH_ID.map(norm_id))
    if actual != set(ACTIVE_34):
        raise ValueError(f"frozen active graph contract changed; missing={sorted(set(ACTIVE_34)-actual)}, extra={sorted(actual-set(ACTIVE_34))}")
    feature_schema_path = require_file(args.active_dataset.expanduser().resolve() / "metadata/feature_schema.json", "frozen feature schema")
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8-sig"))
    rain_valid_start = pd.Timestamp(feature_schema["global_time_start"])
    rain_valid_end = pd.Timestamp(feature_schema["global_time_end"])
    if rain_valid_start >= rain_valid_end or "unrecorded" not in str(feature_schema.get("rain_zero_semantics", "")).lower():
        raise ValueError("frozen rainfall global-time/zero-semantics contract is unavailable")
    study_nodes = pd.read_csv(required_paths["study_nodes"], dtype=str, encoding="utf-8-sig")
    study_nodes["GRAPH_ID"] = study_nodes.GRAPH_ID.map(norm_id)
    study_nodes["STATION_ID"] = study_nodes.STATION_ID.map(norm_id)
    study_nodes["OUTLET_ID"] = study_nodes.OUTLET_ID.map(norm_id)
    study_nodes = study_nodes[study_nodes.GRAPH_ID.isin(ACTIVE_33) & study_nodes.QC_STAT.eq("ACCEPT") & study_nodes.USE_NODE.eq("1")].copy()
    for field in ["IS_OUTLET", "SNAP_LON", "SNAP_LAT"]:
        study_nodes[field] = pd.to_numeric(study_nodes[field], errors="raise")
    if study_nodes.GRAPH_ID.nunique() != 33 or int(study_nodes.IS_OUTLET.sum()) != 33 or study_nodes.STATION_ID.eq(EXCLUDED_STATION).any():
        raise ValueError("study station input does not cover exactly 33 non-excluded outlet basins")
    outlet_rows = study_nodes[study_nodes.IS_OUTLET.eq(1)].set_index("GRAPH_ID")

    basins = gpd.read_file(required_paths["basins"], layer="full_catchments", engine="pyogrio")
    if basins.crs is None or basins.crs.to_epsg() != 4326 or basins.HYDRO_NODE_ID.duplicated().any():
        raise ValueError("full catchments must be unique EPSG:4326 HYDRO_NODE_ID polygons")
    basins["HYDRO_NODE_ID"] = basins.HYDRO_NODE_ID.map(norm_id)
    basin_lookup = basins.set_index("HYDRO_NODE_ID")
    fallback = manifest_tile_pairs(required_paths["manifest"])

    stage.mkdir(parents=True)
    try:
        all_nodes, all_edges, all_units, graph_metrics = [], [], [], []
        with ExitStack() as stack:
            sources = []
            for label, upa_path, dir_path in [("work_raster", required_paths["upa"], required_paths["direction"]), *fallback]:
                upa_ds = stack.enter_context(rasterio.open(upa_path))
                dir_ds = stack.enter_context(rasterio.open(dir_path))
                if not raster_pair_is_aligned(upa_ds, dir_ds):
                    raise ValueError(f"unaligned MERIT UPA/DIR pair: {label}")
                sources.append((label, upa_ds, dir_ds))
            for number, graph_id in enumerate(ACTIVE_33, 1):
                outlet = outlet_rows.loc[graph_id]
                hydro_node = norm_id(outlet.HYDRO_NODE_ID)
                if hydro_node not in basin_lookup.index:
                    raise ValueError(f"{graph_id}: missing full catchment {hydro_node}")
                basin = basin_lookup.loc[hydro_node]
                available = [source for source in sources if raster_covers_geometry(source[1], basin.geometry.bounds)]
                if not available:
                    raise ValueError(f"{graph_id}: no MERIT raster pair fully covers basin")
                label, upa_ds, dir_ds = available[0]
                nodes, edges, units, metrics = build_basin_graph(graph_id, outlet, basin, upa_ds, dir_ds, label)
                all_nodes.extend(nodes); all_edges.extend(edges); all_units.extend(units); graph_metrics.append(metrics)
                print(f"[graph {number:02d}/33] {graph_id}: nodes={len(nodes)} edges={len(edges)} hops={metrics['MAX_HOPS']}", flush=True)

        node_catalog = pd.DataFrame(all_nodes).sort_values(["GRAPH_ID", "NODE_INDEX"])
        edge_gdf = gpd.GeoDataFrame(all_edges, geometry="geometry", crs=4326)
        units_gdf = gpd.GeoDataFrame(all_units, geometry="geometry", crs=4326).sort_values(["GRAPH_ID", "NODE_INDEX"])
        metrics = pd.DataFrame(graph_metrics).sort_values("GRAPH_ID")
        if len(metrics) != 33 or len(node_catalog) != len(units_gdf) or len(edge_gdf) != len(node_catalog) - 33:
            raise ValueError("graph/node/unit/edge cardinality contract failed")
        if not metrics.IS_DAG.eq(1).all() or not metrics.OUTLET_CONNECTED.eq(1).all():
            raise ValueError("DAG/outlet connectivity QC failed")
        if metrics.AREA_CLOSURE_ERROR_PCT.abs().max() > args.area_closure_fail_pct:
            raise ValueError("unit catchment area closure exceeds tolerance")
        if metrics.OVERLAP_AREA_KM2.max() > 1e-6 or metrics.MISSING_AREA_KM2.max() > 1e-6:
            raise ValueError("unit catchment overlap/completeness QC failed")

        station_mapping = map_observation_stations(study_nodes, node_catalog, edge_gdf, args.station_snap_fail_m)
        rain_weights, rain_coverage, rain_info = build_rain_weights(
            units_gdf, required_paths["rain_master"], args.rain_buffer_km,
            rain_valid_start, rain_valid_end,
        )
        rain_counts, missing_sparse = aggregate_rainfall(rain_weights, required_paths["rain_sparse_index"], stage / "rainfall/node_hourly_rain_sparse")
        rain_info["gauges_without_positive_sparse_file"] = missing_sparse
        if set(rain_counts) != set(ACTIVE_33) or len(rain_coverage) != len(node_catalog):
            raise ValueError("rainfall output does not cover every graph/node")

        node_static, edge_static, static_info = compute_static_attributes(units_gdf, node_catalog, edge_gdf, args, workflow)
        if len(node_static) != len(node_catalog) or len(edge_static) != len(edge_gdf):
            raise ValueError("static attribute cardinality mismatch")
        if node_static.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).isna().any().any():
            raise ValueError("node static numeric field contains missing/nonfinite data")
        if edge_static[["reach_length_m", "reach_slope_m_per_m"]].replace([np.inf, -np.inf], np.nan).isna().any().any() or (edge_static[["reach_length_m", "reach_slope_m_per_m"]] <= 0).any().any():
            raise ValueError("edge length/slope is missing or nonpositive")

        graph_dir = stage / "graph"
        graph_dir.mkdir(parents=True)
        node_catalog.to_csv(graph_dir / "node_catalog.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        edge_gdf.drop(columns="geometry").to_csv(graph_dir / "edge_topology.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        node_static.to_csv(graph_dir / "node_static_attributes.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        edge_static.to_csv(graph_dir / "edge_static_attributes.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        station_mapping.to_csv(graph_dir / "station_observation_mapping.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        units_gdf.to_file(graph_dir / "unit_catchments.gpkg", layer="unit_catchments", driver="GPKG", engine="pyogrio")
        rain_coverage.to_csv(stage / "rainfall/node_rainfall_coverage.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
        # Use node static (which contains STATIC_QC) in the report.
        write_report(stage / "BUILD_AND_QC.md", metrics, node_static, edge_gdf, units_gdf, station_mapping, rain_coverage, rain_counts, rain_info, static_info)

        expected = {
            "graph/node_catalog.csv", "graph/edge_topology.csv", "graph/node_static_attributes.csv",
            "graph/edge_static_attributes.csv", "graph/station_observation_mapping.csv",
            "graph/unit_catchments.gpkg", "rainfall/node_rainfall_coverage.csv", "BUILD_AND_QC.md",
        }
        if not expected.issubset({str(path.relative_to(stage)).replace("\\", "/") for path in stage.rglob("*") if path.is_file()}):
            raise ValueError("required formal outputs are incomplete")
        if "FINAL QC STATUS: PASS" not in (stage / "BUILD_AND_QC.md").read_text(encoding="utf-8"):
            raise ValueError("final report did not record PASS")
        stage.rename(output)
        print(json.dumps({
            "status": "PASS", "output": str(output), "graphs": 33,
            "nodes": len(node_catalog), "edges": len(edge_gdf),
            "stations": len(station_mapping), "rain_sparse_rows": sum(rain_counts.values()),
        }, ensure_ascii=False, indent=2))
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    main()
