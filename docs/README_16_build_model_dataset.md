# 第16步：整理山洪预报模型数据集（事件合并与水位基准 QC 版）

权威脚本：`16_build_model_dataset_v3.py`

本版仍保持原模型数据契约和实验设置，但会在正式 split 之前完成两项源头修复：

1. 按实测水文过程合并被多个候选降雨事件重复切分的 `EVENT_ID`；
2. 识别 TRAIN 内不可恢复的站内水位基准断裂，仅排除受影响事件，不删站、不自动平移水位。

模型架构、24→6 h、Q/Z 特征、静态属性、优化器、loss 权重及其他训练设置均不在本脚本中修改。

## 1. 运行环境

使用普通 Python 3，不使用 ArcMap Python 2.7。

```powershell
python -m pip install pandas numpy openpyxl
```

## 2. 推荐运行命令

不要覆盖旧候选数据。直接输出新的 `_model_dataset_v5_event_zqc`：

```powershell
python "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan\Arcgis\MERIT_workflow\16_build_model_dataset_v3.py" --project-root "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan" --output-dir "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan\project\_model_dataset_v5_event_zqc" --overwrite
```

脚本自动寻找：

- `06_topology\edges.csv`
- `11_node_rain` 下的节点逐时降雨文件和索引
- `13_final_flood_events\final_flood_events.csv`
- `14_static_attributes_dem_cisc\edge_static_attributes_dem.csv`
- `15_static_attributes_ccam\node_static_attributes_final.csv`
- `归档_正确解压\河道` 下的全部河道站文件

若要先检查水位异常、发现后立即停止，不进行事件级排除：

```powershell
python "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan\Arcgis\MERIT_workflow\16_build_model_dataset_v3.py" --project-root "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan" --output-dir "E:\FSN-PostDoc\other-projects\Flash flood prediction\Hunan\project\_model_dataset_v5_event_zqc" --water-level-reference-shift-policy FAIL --overwrite
```

默认 `EXCLUDE_EVENT` 策略才用于生成下一轮可训练数据。

## 3. 固定样本设置

- 主事件：`HYDRO_FLOOD`
- 事件等级：A、B
- 历史输入：24 h
- 预测时效：未来 1–6 h
- 样本步长：1 h
- 正式划分：完成事件合并和水位事件排除后，按事件时间顺序 70%/15%/15%
- 目标变量：`AUTO`；出口流量覆盖率达到 70% 时选择流量，否则选择水位
- 目标 6 h 完整率低于 80% 的窗口不进入样本索引
- 标准化：只使用最终 TRAIN 输入窗口重新计算

需要固定预测变量时可继续使用：

```powershell
--target-variable FLOW
```

或：

```powershell
--target-variable WATER_LEVEL
```

## 4. 事件定义与合并规则

### 4.1 一个正式事件与四类数据的关系

本数据集把“同一 `GRAPH_ID`、同一出口站的一次连续洪水响应过程”作为正式事件，而不是把一段降雨或一行流量/水位记录单独当作事件：

- **流域/河网**：`GRAPH_ID` 固定事件发生在哪一张河网图；图内节点、上下游拓扑和静态属性不因事件改变。
- **降雨**：一场正式事件可以由一个或多个候选降雨过程触发。`SOURCE_RAIN_EVENT_IDS` 保存来源，`RAIN_START–RAIN_END` 保存这些来源的时间并集；真正进入模型的是该图各节点逐小时 `RAIN_MM`，按样本的 24 h 历史窗口用时间戳读取，不是只给事件附一个总雨量。
- **流量/水位**：`HYDRO_START–HYDRO_END` 描述同一出口的连续响应过程，`PEAK_TIME` 来自该图最终目标变量的实测过程。`AUTO` 模式按整张图出口的流量覆盖率固定选择 `FLOW` 或 `WATER_LEVEL`，不会在同一图的不同事件间来回切换目标。
- **训练样本**：事件是划分 TRAIN/VALIDATION/TEST 的最小单位；一个事件内部再按 1 h 滑动生成多个“24 h 历史 → 未来 1–6 h”样本。同一事件的所有样本只能属于同一 split。

因此，事件合并后不是把某一场降雨硬贴给新的 `EVENT_ID`。脚本会合并降雨来源及时间窗，并按合并后的 `SAMPLE_START–SAMPLE_END` 重新从第11步节点逐时降雨表和河道逐时表构建动态数据，再重新生成全部滑动样本。历史输入只读取截至 `FORECAST_TIME` 的实测降雨；未来 1–6 h 降雨仍遵守模型配置的 forcing 策略，不会因事件合并而偷看未来实测雨量。

### 4.2 自动合并判据

同一 `GRAPH_ID + OUTLET_ID` 内，只有以下确定性证据之一成立才自动合并：

1. 两个事件共享有效目标小时，且这些目标小时中的实测主洪峰时刻完全相同；
2. 两个正式 `HYDRO_START—HYDRO_END` 水文窗确实重叠。

仅目标时段重叠但实测峰时不同，或两个水文过程间隔不超过 6 h，只标记 `REVIEW`，不自动合并。事件编号相邻、洪峰数值相同但峰时不同等弱证据不用于自动合并。

合并连通组使用最早原 `EVENT_ID` 作为新事件 ID，并新增：

- `SOURCE_EVENT_IDS`
- `SOURCE_EVENT_COUNT`
- `EVENT_MERGE_STATUS`

雨量、水文和样本时间窗取并集；主峰时刻按合并目标小时中的实测最大值重新确定。随后重新执行 deterministic split，并重建全部依赖文件。

当前 800 事件候选数据的只读重演结果为：

- 15 条强合并关系；
- 13 个合并连通组；
- 合并减少 14 个事件。

## 5. 水位基准断裂规则

规则只使用暂定 TRAIN，不使用 VALIDATION/TEST 反向筛数据：

1. 对每个水位目标站计算每场 TRAIN 事件有效目标小时的中位水位；
2. 使用 Tukey outer fence：`Q1 - 3×IQR` 至 `Q3 + 3×IQR`；
3. 只有整场事件的最小值—最大值范围完全落在 fence 外，才判定为站内水位基准断裂；
4. 默认排除该事件，并把该事件源时段的水位置为缺测；站点及其他年份全部保留；
5. VALIDATION/TEST 超出 TRAIN 范围只标记 `REVIEW`，绝不据此删事件。

真实候选数据会明确定位：

- `611G3160 / B016_F0013`
- `611G3160 / B016_F0014`

其水位约为 0–4 m，而同站其他事件约为 573–579 m。脚本不会猜测或添加约 573 m 的偏移量，因为缺少权威测站基准换算关系；这两场事件按“不可恢复基准混用”进入事件级排除 QC。

因此，按当前候选数据预计：

- 原始事件：800
- 水位基准事件排除：2
- 事件合并减少：14
- 正式事件：784
- 重新划分后 TRAIN/VALIDATION/TEST 事件数：548/117/119

这些数量必须以本机实际重建日志和 `dataset_summary.csv` 为最终依据。

## 6. 输出结构

```text
_model_dataset_v5_event_zqc\
├─ graph\
│  ├─ node_catalog.csv
│  ├─ edge_topology.csv
│  ├─ node_static_attributes.csv
│  └─ edge_static_attributes.csv
├─ dynamic\
│  ├─ graph_<BASIN_ID>_hourly.csv
│  └─ ...
├─ events\
│  ├─ flood_events_all.csv
│  ├─ flood_events_final.csv
│  ├─ data_split.csv
│  ├─ sample_index.csv
│  └─ target_variable_by_graph.csv
├─ metadata\
│  ├─ feature_schema.json
│  ├─ normalization_stats.json
│  ├─ dataset_summary.csv
│  ├─ source_manifest.json
│  └─ build_log.txt
└─ qc\
   ├─ event_merge_audit.csv
   ├─ event_hydrograph_overlap.csv
   ├─ water_level_reference_event_audit.csv
   ├─ water_level_station_audit.csv
   ├─ dataset_quality_audit_summary.json
   ├─ dynamic_coverage.csv
   ├─ event_exclusion.csv
   ├─ hydro_file_selection.csv
   ├─ hydro_load_audit.csv
   ├─ rain_source_coverage.csv
   └─ sample_rejection.csv
```

`event_merge_audit.csv` 保存合并前证据与最终归并 ID；`event_hydrograph_overlap.csv` 是合并后的再审计。若后者仍出现 `MUST_MERGE` 或 `CROSS_SPLIT_LEAKAGE`，脚本会直接失败。

`water_level_reference_event_audit.csv` 保存 TRAIN 事件级基准判断、fence、处理动作和实际掩膜行数；`water_level_station_audit.csv` 是最终 split 下的站级复核。最终仍有水位 `FAIL` 站时脚本会失败。

## 7. 动态文件格式

每个候选河网组一个长表：

```text
GRAPH_ID
TIMESTAMP
NODE_INDEX
STATION_ID
RAIN_MM
FLOW
WATER_LEVEL
RAIN_MASK
FLOW_MASK
WATER_LEVEL_MASK
```

文件只保存最终事件涉及的合并时间区间，不复制滑动样本。

## 8. 构建后的验证和训练

把新目录放到服务器项目根目录后执行：

```bash
cd /home/zheda/Flash_flood_prediction
git pull --ff-only origin main

python audit_dataset_quality.py --dataset-root _model_dataset_v5_event_zqc

python validate_dataset.py \
  --config configs/hunan_e4_event_zqc_v1.yaml \
  --dataset-root _model_dataset_v5_event_zqc \
  --qc-output-dir _model_dataset_v5_event_zqc/qc \
  --output outputs/hunan_e4_event_zqc_v1_dataset_validation.json
```

只有 validator 通过后才运行：

```bash
python train_hunan.py \
  --config configs/hunan_e4_event_zqc_v1.yaml \
  --dataset-root _model_dataset_v5_event_zqc
```

新数据必须从头训练，不能沿用旧 checkpoint。
