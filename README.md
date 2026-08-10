# 湖南省多站河网短时洪水训练与测试

本项目已经按湖南正式 `_model_dataset` 接口接通，不再把真实训练伪装成 synthetic 流程。它支持多张不同节点数的河网、按事件划分 TRAIN/VALIDATION/TEST、按河网自动选择流量或水位目标，以及 E1–E4 四组神经/物理消融实验。

默认任务是用前 24 小时预测未来 1–6 小时。正式训练只读取湖南数据；浙江微调尚未启用。

## 1. 环境

建议使用 Python 3.11 及独立虚拟环境。在 `project` 目录执行：

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

当前正式 CSV 数据层在 Windows 上要求 `num_workers: 0`，以免多个 worker 复制整省动态张量。TRAIN 与 VALIDATION 会在同一进程共享只读动态张量。

GPU 服务器上 `device: auto` 会自动选择 CUDA。正式配置默认仍保持 `amp: false`；确认服务器 GPU/PyTorch 支持后，可在所用的 `hunan_*.yaml` 中覆盖为 `amp: true`。AMP 的动态 loss-scale 溢出会由 `GradScaler` 跳过该步并降低 scale；FP32 训练遇到非有限梯度仍立即报错。

## 2. 唯一认可的正式目录

默认位置是 `project/_model_dataset`，也可通过 `--dataset-root` 指定绝对路径。

```text
_model_dataset/
├─ graph/
│  ├─ node_catalog.csv
│  ├─ edge_topology.csv
│  ├─ node_static_attributes.csv
│  └─ edge_static_attributes.csv
├─ dynamic/
│  ├─ graph_<BASIN_ID>_hourly.csv
│  └─ ...
├─ events/
│  ├─ flood_events_all.csv
│  ├─ flood_events_final.csv
│  ├─ data_split.csv
│  ├─ sample_index.csv
│  └─ target_variable_by_graph.csv
├─ metadata/
│  ├─ feature_schema.json
│  ├─ normalization_stats.json
│  ├─ dataset_summary.csv
│  ├─ source_manifest.json
│  └─ build_log.txt
└─ qc/
   ├─ dynamic_coverage.csv
   ├─ event_exclusion.csv
   ├─ hydro_file_selection.csv
   ├─ hydro_load_audit.csv
   ├─ rain_source_coverage.csv
   └─ sample_rejection.csv
```

任何正式文件缺失、主外键不一致、时间不连续、单位不合法、QC 拒绝记录仍进入样本、或 split 存在冲突时，程序都会终止，不会回退到 synthetic 或静默填补。

## 3. 表字段契约

### 图与静态属性

`graph/node_catalog.csv`：

```text
GRAPH_ID,BASIN_ID,NODE_INDEX,STATION_ID,OUTLET_ID,ROLE,IS_OUTLET
```

同一图的 `NODE_INDEX` 必须从 0 连续递增，只能有一个 `IS_OUTLET=1`，且该站必须等于 `OUTLET_ID`。

`graph/edge_topology.csv`：

```text
GRAPH_ID,FROM_NODE,TO_NODE,FROM_STATION,TO_STATION
```

方向必须为直接上游 `FROM` → 直接下游 `TO`。图必须是能汇入唯一出口的 DAG。当前正式物理路由没有分流比例，因此出度大于 1 的分汊节点会被拒绝；多个上游汇入一个下游是支持的。

`graph/node_static_attributes.csv` 必须含下列 10 项且每个节点恰好一行：

```text
GRAPH_ID,STATION_ID,
log_incremental_area,log_upstream_area,
mean_hillslope_flow_distance_m,mean_slope_deg,elevation_std_m,
drainage_density_km_per_km2,soil_log_ksat_0_30cm,
soil_profile_depth_cm,forest_fraction,impervious_fraction
```

`graph/edge_static_attributes.csv`：

```text
GRAPH_ID,FROM_STATION,TO_STATION,
reach_length_km,reach_slope_m_per_m
```

加载器会把 `reach_length_km` 明确转换为 m。`feature_schema.json` 也必须声明上述两个源字段；`reach_length_m` 仅是模型内部名称，不能作为正式 CSV/schema 输入。河长必须大于 0，坡降必须大于等于 0。正式输入不要求、也不会读取 `channel_width_m`。

### 逐时动态数据

每个河网严格读取：

```text
dynamic/graph_<BASIN_ID>_hourly.csv
```

文件内只能包含与该 `BASIN_ID` 对应的一个 `GRAPH_ID`，字段为：

```text
GRAPH_ID,TIMESTAMP,STATION_ID,
RAIN_MM,FLOW,WATER_LEVEL,
RAIN_MASK,FLOW_MASK,WATER_LEVEL_MASK
```

每个出现的整点必须为该图全部节点各提供一行。缺测值可留空，但对应 mask 必须为 0；mask 为 1 的值必须有限。`RAIN_MM` 单位为 mm/h，`FLOW` 为 m³/s，`WATER_LEVEL` 为 m。

### 事件、目标和划分

`events/flood_events_final.csv`：

```text
EVENT_ID,GRAPH_ID,BASIN_ID,OUTLET_ID,
RAIN_START,RAIN_END,HYDRO_START,PEAK_TIME,HYDRO_END,
SAMPLE_START,SAMPLE_END,EVENT_TYPE,EVENT_GRADE,
COMPOUND_EVENT,PEAK_COUNT,
SOURCE_RAIN_EVENT_IDS,SOURCE_RAIN_EVENT_COUNT
```

主实验只加载 `EVENT_TYPE=HYDRO_FLOOD` 且 `EVENT_GRADE=A/B`。`flood_events_all.csv` 至少应含 `EVENT_ID,GRAPH_ID`，并覆盖 final 表中的全部事件。

`events/data_split.csv`：

```text
EVENT_ID,GRAPH_ID,EVENT_YEAR,EVENT_GRADE,SPLIT,SPLIT_REASON
```

`SPLIT` 只能是 `TRAIN`、`VALIDATION`、`TEST`。同一事件不能跨集合；每个河网还会检查 TRAIN → VALIDATION → TEST 的时间顺序。

`events/sample_index.csv`：

```text
SAMPLE_ID,EVENT_ID,GRAPH_ID,OUTLET_ID,
INPUT_START,FORECAST_TIME,TARGET_END,
HISTORY_HOURS,FORECAST_HOURS,TARGET_VARIABLE,SPLIT
```

时间定义为：

```text
历史输入 = [INPUT_START, FORECAST_TIME]，共 HISTORY_HOURS 个整点
预测目标 = FORECAST_TIME 后第 1...FORECAST_HOURS 小时
TARGET_END - FORECAST_TIME = FORECAST_HOURS
```

模型内部把出口标签展开为 `[F,N]`，但只有出口节点的目标 mask 为真。

`events/target_variable_by_graph.csv` 至少必须含：

```text
GRAPH_ID,TARGET_VARIABLE
```

可附加 `BASIN_ID,OUTLET_ID`。目标只能是 `FLOW`、`WATER_LEVEL` 或 `BOTH`，并必须与该图所有 `sample_index` 行一致。正式配置默认 `target_variable: AUTO`，以此表为权威来源。

## 4. 两个必需 JSON 契约

`metadata/feature_schema.json` 必须明确 10/2/2 维特征顺序（节点静态/边静态/动态），以及如何从对数面积恢复物理 km²。程序绝不会猜测 `log_incremental_area` 的对数底数。

```json
{
  "dynamic_features": ["FLOW", "WATER_LEVEL"],
  "node_static_features": [
    "log_incremental_area",
    "log_upstream_area",
    "mean_hillslope_flow_distance_m",
    "mean_slope_deg",
    "elevation_std_m",
    "drainage_density_km_per_km2",
    "soil_log_ksat_0_30cm",
    "soil_profile_depth_cm",
    "forest_fraction",
    "impervious_fraction"
  ],
  "edge_static_features": [
    "reach_length_km",
    "reach_slope_m_per_m"
  ],
  "physical_features": {
    "incremental_area_km2": {
      "source": "log_incremental_area",
      "transform": "log1p",
      "unit": "km2"
    }
  }
}
```

`transform` 支持 `ln`、`log1p`、`log10`。如果静态表另含未变换的 `incremental_area_km2`，可将 `source` 写为该列并使用 `unit: km2`。

`metadata/normalization_stats.json` 必须显式声明只由 TRAIN 计算：

```json
{
  "computed_from_split": "TRAIN",
  "features": {
    "RAIN_MM": {"mean": 0.0, "std": 1.0, "min": 0.0, "max": 100.0},
    "FLOW": {"mean": 20.0, "std": 15.0, "min": 0.0, "max": 500.0},
    "WATER_LEVEL": {"mean": 2.0, "std": 0.8, "min": -1.0, "max": 10.0}
  }
}
```

FLOW/WATER_LEVEL 的 TRAIN 标准差还用于把联合 Q/Z Huber 损失无量纲化，避免把 m³/s 与 m 直接相加。训练 loss 与早停 loss 因此是标准化空间的无量纲 Huber；下列评估指标仍在物理空间计算：

- 全部有效标签的逐时 MAE，以及 MAE、RMSE、带符号 bias、NSE、KGE；
- `q_sample_peak_mae` 与 `q_sample_peak_bias`，其中 bias 为预测峰值减观测峰值；
- `q_sample_relative_peak_bias`，即 `(预测峰值-观测峰值)/观测峰值` 的比率，不乘 100，观测峰值为 0 时不计；
- `q_sample_peak_timing_mae_hours` 与 `q_sample_peak_timing_bias_hours`，带符号值为正表示预测滞后、为负表示预测提前；
- `q_sample_relative_volume_bias`，保留洪量高估/低估方向。

正式 `evaluate.py` 还输出基于现有滑动窗口的 `EVENT_ID`/`GRAPH_ID` 等权宏平均，并在 `window_group_metrics` 中保留逐事件、逐河网明细。字段名刻意包含 `window`：同一目标时刻若出现在多个预测窗口中仍会重复计入，不能把这些值解释为去重后的完整洪水过程指标。NSE/KGE 等无定义的组不会进入对应宏平均，实际参与组数写在 `*_defined_count`。

训练验证和独立评估还会生成去重后的真实事件/河网/水位站诊断。真实事件按正式 `EVENT_ID` 聚合；同一事件、站点和目标时刻只保留最短预见期（最新起报）的预测。逐站 ΔZ 使用该 sample 历史窗内、截至 `FORECAST_TIME` 最后一个有效实测水位为基准，不读取未来真实水位做校正。完整口径、字段与输出目录见 [docs/validation_diagnostics.md](docs/validation_diagnostics.md)。

当前数据契约没有权威预警阈值、连续业务发报记录或平水负事件，因此暂不计算高流量分层 NSE/KGE、POD/CSI/HSS/F1/FAR 和有效预见期；在补齐业务定义前不会用任意分位数冒充业务阈值。

`source_manifest.json` 应记录数据构建版本、源文件及动态文件校验和。checkpoint 会绑定站点顺序、目标映射和核心契约文件 SHA-256；数据重建后不匹配会拒绝评估，防止站点参数错配。

## 5. QC 门禁

所有列出的 QC 文件必须存在且是带表头的 UTF-8 CSV。

- `event_exclusion.csv` 至少含 `EVENT_ID`；表中事件不得出现在已加载样本。
- `sample_rejection.csv` 必须包含 `REJECTION_ID,SAMPLE_ID,EVENT_ID,GRAPH_ID,OUTLET_ID,FORECAST_TIME,TARGET_START,TARGET_END,TARGET_VARIABLE,TARGET_COVERAGE,MIN_TARGET_COVERAGE,REASON,SPLIT`。每个 final 事件必须至少拥有一个有效 sample，或在该表中有明确拒绝记录；低于未来目标覆盖率阈值的候选窗口逐窗记录。
- 其余 coverage/audit 表会检查可读性并在预检报告中给出行数。

`dataset_summary.csv` 必须是带表头 CSV，`source_manifest.json` 必须是有效 JSON，`build_log.txt` 必须是 UTF-8 文本。

训练前还必须运行确定性事件/水位审计：

```powershell
python audit_dataset_quality.py --dataset-root "D:\path\to\_model_dataset"
```

它只生成证据，不重写事件、split 或样本：

- `qc/event_hydrograph_overlap.csv`：同一图/出口的相邻或时间重叠事件对。只有“共享有效目标时刻且共享同一实测峰时”，或正式 hydro window 确实重叠时，才标记 `MUST_MERGE`；仅样本时段重叠但峰不同标记 `REVIEW`。同一连续过程跨 split 标记 `CROSS_SPLIT_LEAKAGE`。
- `qc/water_level_station_audit.csv`：按目标站和 split 输出物理水位范围、TRAIN 站级范围、TRAIN 全局 normalization 范围、逐时跳变和站内基准一致性。TRAIN 事件中若整场水位范围落在站级事件中位数 Tukey outer fence 之外，判为基准断裂并标记 `FAIL`，不自动删除站点或事件。
- `qc/dataset_quality_audit_summary.json`：合并连通组、预计事件数变化、不重新划分时的暂定 split 数量和严格失败计数；正式数量必须在合并后的真实事件层重新执行原 deterministic split 得到。

`validate_dataset.py` 会重新计算这些审计，不会只信任已有 QC 文件。`strict_validation=true` 时，只要存在 `MUST_MERGE`、`CROSS_SPLIT_LEAKAGE`、TRAIN 水位基准断裂，或 normalization 与 TRAIN 输入窗口重算不一致，就会终止。可用 `--qc-output-dir` 在失败前保留本次审计证据。

仓库现已纳入权威上游构建器 `scripts/16_build_model_dataset_v3.py` 及其运行说明 `docs/README_16_build_model_dataset.md`。该构建器在正式 split 前使用同一出口的有效目标小时、实测峰时和正式 hydro window 形成确定性合并连通组；随后重新生成 final/all events、split、sample index、normalization、summary、QC 和 manifest。TRAIN 内不可恢复的水位基准断裂按事件级排除并留痕，不自动平移水位，也不删除整站。旧 `_model_dataset_v4_candidate` 仍保留为审计证据，不能继续训练。

## 6. 未来降雨策略

物理预测的目标期需要一个明确的降雨假设，项目不会悄悄读取未来实测值。

- `persistence`：默认。各节点使用最后一个历史有效雨量持平外推，未来 `RAIN_MASK=0`，表示它不是实测或预报产品。
- `zero`：未来雨量置 0，mask 为 0。
- `observed_hindcast`：读取目标期实测雨量，只适合作为“完美降雨已知”的回算上限实验，不能当作业务预报成绩。

若以后接入数值天气预报，可在相同位置扩展独立 forecast forcing 字段和模式。

## 7. 预检、训练和独立测试

先做只读预检：

```powershell
python validate_dataset.py --config configs/hunan_e4.yaml --dataset-root "D:\path\to\_model_dataset" --qc-output-dir "D:\path\to\_model_dataset\qc" --output outputs/dataset_validation.json
```

报告必须显示 `"status": "VALID"` 才进入训练。

训练 E4：

```powershell
python train_hunan.py --config configs/hunan_e4.yaml --dataset-root "D:\path\to\_model_dataset"
```

事件与水位修复后的下一次全新 E4 使用
`configs/hunan_e4_event_zqc_v1.yaml`。它只继承现有正式 E4 并改用新的
checkpoint/log 名，不会覆盖 `hunan_e4_diagnostics_*`：

```powershell
python train_hunan.py --config configs/hunan_e4_event_zqc_v1.yaml --dataset-root "D:\path\to\_model_dataset_v5_event_zqc"
```

必须先确认同一新数据目录的 validator 输出 `status=VALID`；不要用旧 best
checkpoint 评估重建后的事件定义。

若同名 log/checkpoint 已存在，全新训练会拒绝覆盖。确认要从头重跑时显式增加 `--overwrite`；续训不要使用该参数。

best/last checkpoint 均先写入同目录临时文件，完成 flush/fsync 后通过原子替换发布；保存或替换失败不会覆盖上一份完整 checkpoint。

从最后一次完整状态续训：

```powershell
python train_hunan.py --config configs/hunan_e4.yaml --dataset-root "D:\path\to\_model_dataset" --resume outputs/hunan_e4_best.last.pt
```

独立 TEST：

```powershell
python evaluate.py --config configs/hunan_e4.yaml --dataset-root "D:\path\to\_model_dataset" --checkpoint outputs/hunan_e4_best.pt --output outputs/hunan_e4_test.json
```

需要对 best checkpoint 重跑验证集详细诊断时：

```powershell
python evaluate.py --config configs/hunan_e4.yaml --dataset-root "D:\path\to\_model_dataset" --checkpoint outputs/hunan_e4_best.pt --split VALIDATION --output outputs/hunan_e4_validation.json
```

训练只用 TRAIN 拟合、VALIDATION 选最佳 checkpoint；TEST 不参与拟合和早停。评估只加载模型权重，不恢复 optimizer。未训练河网、站点映射、核心数据契约、求解器积分契约或物理参数边界发生变化都会被拒绝。

默认配置也指向 `project/_model_dataset`，若数据放在该处可省略 `--dataset-root`。

新版多目标 E4 单次训练使用独立配置和输出，不覆盖旧实验：

```powershell
python train_hunan.py --config configs/hunan_e4_multitask.yaml --dataset-root "D:\path\to\_model_dataset"
```

它采用事件—流域平衡 Q point loss、6 h 峰值/洪量 loss、绝对 Z 与真正逐小时
first-difference Z loss，并以越大越好的综合 VALIDATION score 选择 best/早停。
准确公式、去重口径和默认权重见
[docs/multitask_training_and_hpo.md](docs/multitask_training_and_hpo.md)。

Optuna 是独立、显式启用的后续入口；`hyperparameter_optimization.enabled`
默认是 `false`，普通训练不会启动搜索。本轮不应把 HPO 当作新版单次训练的
默认下一步。

## 8. E1–E4 实验

正式配置采用分层继承：`base.yaml` 只保存共享默认值，`hunan_e4.yaml` 在其上覆盖湖南数据契约和 E4 运行参数，E1–E3 再继承 `hunan_e4.yaml` 并仅切换产流/汇流模块。不需要另外的旧 `e1_pure_ai.yaml`–`e4_full_physics.yaml` 文件。

| 配置 | 产流 | 汇流 |
|---|---|---|
| `configs/hunan_e1_pure_ai.yaml` | Pure LSTM | Directed GNN |
| `configs/hunan_e2_physics_runoff.yaml` | Water-balance LSTM | Directed GNN |
| `configs/hunan_e3_physics_routing.yaml` | Pure LSTM | Kinematic wave |
| `configs/hunan_e4.yaml` | Water-balance LSTM | Kinematic wave |

旧基线四组共享完全相同的数据、split、目标、损失、评价和 checkpoint 规则。
未来多目标 HPO 只允许在 E4 搜索一次共享参数；得到真实最优结果后，E1–E4
冻结同一套 loss/权重、hidden dim、lr、weight decay、seed、epoch、早停和
checkpoint 规则，只切换上表两个 physics mode，禁止分别调参。TEST 不参与
任何选择或剪枝。

## 9. 模型与安全约束

- `GraphEventBatch` 使用 `[B,H,N,D]` / `[B,T,N]`，不同河网不 padding；同一个 batch 只含一张图。
- 全省站点使用稳定的全局 `station_index`，Q–Z 观测参数不会因河网节点数不同而错位。
- 对数静态面积只用于神经特征，水量换算单独使用反变换后的 `node_area_km2`。
- 历史 Q/Z/降雨 mask 会作为模型输入；缺测 0 与真实 0 可区分。
- 运动波使用可微后向 Euler 单调非线性求解、守恒蓄量和直接上游汇入；CFL 仅报告显式方法所需的等价子步数，不再控制积分或中止短河段高流量样本。非有限值或隐式方程残差超限仍会明确报错。
- 数据没有实测河宽时，运动波根据河长、坡降及边两端节点属性学习有界的 `effective width`；Manning n 同样是有界可学习参数。二者是由路由目标校准的等效水力参数，不应表述为实测河宽或实测糙率。
- 水位观测头对非负 Q 和河道蓄量结构单调，并按全局站点索引取参数。
- 物理库容从每个样本窗口开始 warm-up；默认 24 小时。慢响应流域应通过实验加长 HISTORY_HOURS。

当前模型没有闸坝调度、分洪比例、回水、潮汐边界或连续跨事件状态缓存。含这些过程的河网不能在未建模的情况下解释为纯自然河道结果。

当前 JSON 同时报告全部有效出口标签的微平均和按 `EVENT_ID`/`GRAPH_ID` 等权的窗口宏平均。由于滑动窗口可能重叠，论文定稿前仍应在业务时序定义完成后另做目标时刻去重和完整洪水过程复核。

## 10. 调试模式与浙江数据

`profile_model.py` 和单元测试复用上表四份湖南配置，但只生成内存中的 synthetic fixture 检查结构和吞吐；它们不产生正式科研结果。`base.yaml` 是继承基底，不是湖南正式训练入口。

浙江正式适配器尚未实现。当前版本不会把 synthetic 数据当作浙江微调，也不会让同一个 loader 同时充当训练和验证。待浙江整理为相同目录契约后，再增加独立预训练权重加载、浙江事件级微调 split 和独立 TEST。
