# Model Dataset v8 — Hydrologic Computational Graph

## 构建结果

- Dataset: `E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan\project\_model_dataset_v8_hydrologic_graph`
- Spatial source: `project/_hydrologic_graph_v1`（原样只读复用）
- Temporal source: `project/_model_dataset_v7_event_multitask`（只读继承 event、split、sample origin 和 Q/Z）
- Graphs: **33**
- Computational nodes: **237**
- Directed river-reach edges: **204**
- Observation stations: **39**（outlet **33**；internal **6**）
- Events: **2807**（TRAIN 1178；VALIDATION 836；TEST 793）
- Samples: **279574**（TRAIN 124016；VALIDATION 83401；TEST 72157）
- Tensor files: **33** graph-grouped compressed NPZ；总大小 **14.60 MiB**。

## Physics forcing 与 sparse observations

- `history_rain`: `[S, Nnode, 24]`, float32, mm
- `future_rain`: `[S, Nnode, 6]`, float32, mm
- `node_static`: `[Nnode, 10]`, float32
- `incremental_area_km2`: `[Nnode]`, float32；它是 local runoff/unit catchment area，不是 upstream area
- `edge_index`: `[2, Nedge]`, int64
- `edge_static`: `[Nedge, 2]`, float32；字段为 `reach_length_m`, `reach_slope_m_per_m`
- `obs_station_id`: `[Nobs]`；`obs_node_index`: `[Nobs]`，允许多个真实站映射到同一 computational node
- `q_history`, `z_history`: `[S, Nobs, 24]`, float32；缺测为 NaN，独立 boolean mask
- `q_target`: `[S, Nobs, 6]`, float32，原始 Q（m³/s）
- `z_target`: `[S, Nobs, 6]`, float32，保持既有目标语义 `ΔZ(t+h)=Z(t+h)-Z(t0)`
- 没有站点的 computational node 不存在任何 Q/Z 数组槽位；Q/Z 从未作为 `[Nnode,T]` dynamic feature 构建，也未以0伪造。
- 雨量是 physics forcing；冻结有效时间内稀疏文件无正雨量记录的小时为真实 `0 mm`。

## Supervision / forcing coverage

| split | samples | Q history | Z history | Q target | ΔZ target | rain forcing | positive rain |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN | 124016 | 58.993% | 88.548% | 59.846% | 86.542% | 100.000% | 29.037% |
| VALIDATION | 83401 | 72.540% | 89.306% | 73.144% | 87.033% | 100.000% | 27.624% |
| TEST | 72157 | 75.131% | 92.991% | 75.773% | 90.680% | 100.000% | 26.880% |

Q/Z coverage 的分母是各 split 的 `sample × Nobs × hours`；不同任务独立缺测，mask=0 的任务不贡献监督。Rain forcing 在全部 sample/node/hour 上完整且有限。

## TRAIN-only normalization

- `metadata/dataset_contract.json` 内的 `normalization` 全部标记 `computed_from_split=TRAIN`。
- Rain statistics：仅使用 TRAIN sample 中实际暴露的 history+future forcing。
- Q/Z statistics：按 observation station 独立拟合，仅使用 TRAIN sample exposure；无该任务数据的站保留 `available=false`，不伪造 scale。
- `z_target` scale 按 TRAIN-only ΔZ 拟合。
- Node/edge static statistics：仅使用至少有一个 TRAIN sample 的 graph；VALIDATION/TEST 不参与拟合。

## QC

- 33 graph 集合与 `_hydrologic_graph_v1` 一致，`Q_61512000/61512000` 不存在：PASS
- 237 nodes、204 edges、39 station mappings 与 QC-PASS 空间事实一致：PASS
- 每图 DAG、唯一 outlet、edge endpoint、station-node index 一致：PASS
- 33 个出口观测站全部映射到各自 final outlet node：PASS
- v8 的 SAMPLE_ID/EVENT_ID/FORECAST_TIME/SPLIT 与 v7 的33图子集逐条一致：PASS
- history=24 h、forecast=6 h，事件和 split 未重建、未重分：PASS
- Rain tensor 全部有限、非负，forcing coverage=100%：PASS
- Q/Z 在 mask=1 时有限，在 mask=0 时为 NaN；未向无站节点扩展：PASS
- Node/edge static 与 incremental area 无 missing/nonfinite：PASS
- 所有 normalization/statistics 仅由 TRAIN 拟合：PASS
- Staging 完整验证后才原子发布：PASS

**FINAL QC STATUS: PASS**
