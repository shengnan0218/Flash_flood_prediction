# Flash Flood Prediction

湖南中小流域 1–6 h 流量预报。项目使用同一份事件数据，比较普通与物理约束的产流、河网汇流模块。

```text
72 h rainfall + node static
            ↓
       runoff LSTM
            ↓
      directed graph routing
            ↓
 Q0-anchored route reliability gate
            ↓
          Q(t+1…t+6)
```

Q0 只在最后一层作为观测锚点；不会修改 LSTM 隐状态、坡面蓄水或河道蓄水。

## Installation

```bash
conda create -n flashflood python=3.11 -y
conda activate flashflood
pip install -r requirements.txt
```

GPU 训练需要安装与服务器 CUDA 对应的 PyTorch。

## Data contract

默认数据集目录：

```text
_model_dataset_v11_72h_event_balanced/
├── metadata/dataset_contract.json
├── graph/station_observation_mapping.csv
├── samples/sample_index.csv
└── tensors/*.npz
```

- 历史降雨：72 h；未来降雨：6 h；
- Q/Z 观测历史：24 h；预测目标：未来 1–6 h Q；
- 节点静态属性：10 项；边属性：河段长度、河段坡降；
- TRAIN、VALIDATION、TEST 为既定事件划分；所有统计量只拟合于 TRAIN；
- `observed_hindcast` 表示未来降雨使用观测雨量，因而评估的是产流—汇流结构的理想上限。

其他数据目录可用 `--dataset-root` 指定：

```bash
python train_hunan.py \
  --config configs/e4_water_balance_lstm_muskingum_gnn.yaml \
  --dataset-root /path/to/_model_dataset_v11_72h_event_balanced
```

## Model components

### Runoff LSTM

- `pure_lstm`：以标准化的 `log1p(rain)` 和静态属性预测单位面积径流深，再按增量面积换算为 m³/s。
- `water_balance_lstm`：快、慢蓄水库均以 mm 表示；其控制量显式使用当前降雨、快慢蓄水和静态属性。每时步满足：

\[
P_t + S_{t-1} = Q_{lat,t} + L_t + S_t
\]

其中 (L_t) 是有界的未观测损失/深层补给通量，不会把全部降雨强制转化为出口径流。

### Graph routing

- `pure_gnn`：无物理约束的有向图消息传递对照。
- `muskingum_gnn`：在河网拓扑上逐河段执行可微 Muskingum 路由。每条河段只有一个有效 travel-time 参数，它由河段长度、坡降和节点静态属性区域化，并且只允许相对物理先验作有界修正。外部时间步固定为 1 h，而部分河段的有效 travel time 超过数小时，因此采用 `X=0` 的线性蓄水库 Muskingum 特例；这避免正 `X` 在长河段上产生负递推系数。

该物理路由不学习河宽、河深或 Manning 糙率；它们无法由现有流量监督可靠反演。路由模块提供 travel-time 与逐时质量守恒诊断。

### Output gate

对有 Q0 观测的样本，最终输出为：

\[
\hat Q_{t+h}=\max\left(0,Q_0+g_{t+h}(Q_{route,t+h}-Q_{route,t})\right),\quad g\in[0,1]
\]

小型 MLP 仅预测 (g)，没有自由加性残差。因此它只能在 persistence 与完整路由增量之间调节可信度，不能绕过产流—汇流主干重新生成流量。Q0 缺失时回退为路由流量。

Z 由 TRAIN-only 线性 rating curve 从 Q 派生，只用于报告。

## Four experiments

| Experiment | Runoff | Routing | Config |
|---|---|---|---|
| E1 | Pure LSTM | Pure graph GNN | `configs/e1_pure_lstm_pure_gnn.yaml` |
| E2 | Mass-conserving LSTM | Pure graph GNN | `configs/e2_water_balance_lstm_pure_gnn.yaml` |
| E3 | Pure LSTM | Muskingum graph routing | `configs/e3_pure_lstm_muskingum_gnn.yaml` |
| E4 | Mass-conserving LSTM | Muskingum graph routing | `configs/e4_water_balance_lstm_muskingum_gnn.yaml` |

四组共享数据、采样、损失、优化器、Q0 门控和训练预算；只改变产流和汇流模块是否采用物理约束。

## Training

默认训练 E4：

```bash
python train_hunan.py --overwrite
```

显式运行四组：

```bash
python train_hunan.py --config configs/e1_pure_lstm_pure_gnn.yaml --overwrite
python train_hunan.py --config configs/e2_water_balance_lstm_pure_gnn.yaml --overwrite
python train_hunan.py --config configs/e3_pure_lstm_muskingum_gnn.yaml --overwrite
python train_hunan.py --config configs/e4_water_balance_lstm_muskingum_gnn.yaml --overwrite
```

训练预算固定为 30 epochs。checkpoint 不按 `val_loss` 选择，而按 VALIDATION 的 **station-macro median persistence skill** 选择；日志同时报告：

- 总体 Q NSE/KGE；
- Q0 有效子集的 persistence skill 与 ΔQ NSE；
- station-macro mean/median skill、优于 persistence 的站点比例；
- 1–6 h 每个提前量的 persistence skill。

旧 checkpoint 与本版本不兼容，必须从零训练。`--resume` 仅适用于同一版本、同一数据合同的 `*.last.pt`。

## Evaluation

先只在 VALIDATION 选模型：

```bash
python evaluate.py \
  --config configs/e4_water_balance_lstm_muskingum_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_muskingum_gnn_best.pt \
  --split VALIDATION
```

使用三分解定位问题：

```bash
python evaluate_decomposition.py \
  --config configs/e4_water_balance_lstm_muskingum_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_muskingum_gnn_best.pt \
  --split VALIDATION \
  --output-dir outputs/e4_validation_decomposition
```

三分解固定在同一 Q0 有效子集上比较：

1. `persistence`：Q0 保持不变；
2. `full_route`：完整应用物理路由产生的 ΔQ；
3. `gated_route`：应用模型学习到的门控 ΔQ，即最终输出。

只有冻结 VALIDATION 最优实验后，才运行一次 TEST：

```bash
python evaluate.py \
  --config configs/e4_water_balance_lstm_muskingum_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_muskingum_gnn_best.pt \
  --split TEST
```

## Pre-flight tests

```bash
pytest -q
pytest -q tests/test_hydrologic_model.py tests/test_output_decomposition.py
```

模型测试覆盖：四组前向/反向、守恒产流的逐时水量闭合、零雨零产流、travel-time 随河长变化、Muskingum 连续性残差、Q0 门控和未来 Z 不泄漏。

## Normalization

- 降雨：`log1p` 后使用 TRAIN 均值/标准差；
- 节点及神经网络边属性：TRAIN-only 标准化后裁剪至 `[-5, 5]`；
- Muskingum 先验：原始河段长度和坡降，不使用标准化量；
- Q loss、Q0 门控：TRAIN-only per-station Q 均值与尺度；
- 高流量阈值、rating curve：仅由 TRAIN 拟合。
