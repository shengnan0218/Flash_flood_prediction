# Flash Flood Prediction

面向湖南中小流域的 1–6 h 山洪流量预测项目。模型以计算单元降雨、节点静态属性和河网拓扑为输入，采用统一的：

```text
产流 LSTM → 河网 GNN → 全连接输出层 → 未来 6 h 流量 Q
```

项目提供普通与物理约束产流、汇流模块的四组严格对照实验。

## 1. 环境安装

推荐使用 Python 3.11 和独立 Conda 环境：

```bash
conda create -n flashflood python=3.11 -y
conda activate flashflood
pip install -r requirements.txt
```

主要依赖为 PyTorch 2.2+、NumPy、Pandas、PyYAML 和 pytest。GPU 训练时应安装与服务器 CUDA 版本匹配的 PyTorch。

## 2. 数据目录

默认数据集目录为：

```text
_model_dataset_v11_72h_event_balanced/
├── metadata/
│   └── dataset_contract.json
├── graph/
│   └── station_observation_mapping.csv
├── samples/
│   └── sample_index.csv
└── tensors/
    └── *.npz
```

数据固定为：

- 降雨历史 72 h；
- Q/Z 观测历史 24 h；
- 未来降雨与预测目标 6 h；
- 10 个节点静态属性；
- 2 个边静态属性：河段长度和河段坡降；
- TRAIN、VALIDATION、TEST 按事件划分。

默认路径写在 `configs/base.yaml`。数据位于其他位置时，通过 `--dataset-root` 指定，无需修改配置：

```bash
python train_hunan.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --dataset-root /path/to/_model_dataset_v11_72h_event_balanced
```

## 3. 模型组成

### 3.1 产流模块

- `pure_lstm`：普通 LSTM，根据标准化降雨和节点静态属性直接预测各节点侧向产流；
- `water_balance_lstm`：水量平衡 LSTM，将降雨划分到快、慢两个蓄水库，并通过连续时间退水率产生侧向流量。

72 h 历史降雨和未来 6 h 降雨连续输入产流模块，历史时段用于暖启动产流状态。

### 3.2 汇流模块

- `pure_gnn`：按有向河网拓扑进行非物理消息传递；
- `kinematic_wave_gnn`：采用可学习河宽和曼宁系数的隐式运动波求解器进行河道汇流。

运动波模块使用河段长度、坡降以及上下游节点静态属性，并逐时维持河道蓄量和质量平衡诊断。

### 3.3 全连接输出层

输出层以路由流量变化为基础。当预报起点 Q0 有观测时，基准预测为：

```text
Qbase(t) = Q0 + Qroute(t) - Qroute(0)
```

小型 MLP 根据路由结果、Q0、站点流量尺度、预报时效和出口静态属性给出有界修正，最终输出非负 Q。Q0 不用于修改 LSTM 隐状态或河道蓄量。

水位 Z 不作为训练目标，由 TRAIN 数据拟合的固定站点线性 rating curve 从预测 Q 推导，用于评价与结果输出。

## 4. 四组实验

| 实验 | 产流模块 | 汇流模块 | 配置文件 |
|---|---|---|---|
| E1 | 普通 LSTM | 普通 GNN | `configs/e1_pure_lstm_pure_gnn.yaml` |
| E2 | 水量平衡 LSTM | 普通 GNN | `configs/e2_water_balance_lstm_pure_gnn.yaml` |
| E3 | 普通 LSTM | 运动波 GNN | `configs/e3_pure_lstm_kinematic_wave_gnn.yaml` |
| E4 | 水量平衡 LSTM | 运动波 GNN | `configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml` |

四组实验共享数据、静态属性、输出层、损失、采样、优化器和评价流程，只改变 `runoff_mode` 与 `routing_mode`。

## 5. 训练

默认运行 E4：

```bash
python train_hunan.py
```

运行指定实验：

```bash
python train_hunan.py --config configs/e1_pure_lstm_pure_gnn.yaml
python train_hunan.py --config configs/e2_water_balance_lstm_pure_gnn.yaml
python train_hunan.py --config configs/e3_pure_lstm_kinematic_wave_gnn.yaml
python train_hunan.py --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml
```

只训练一个指定河网：

```bash
python train_hunan.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --graph-id B001
```

默认训练设置：

- 100 epochs，不启用 early stopping；
- AdamW，学习率 `1e-3`，weight decay `1e-5`；
- batch size 16；
- TRAIN 使用事件均衡、LOW/RISING/PEAK/RECESSION 阶段分层采样；
- checkpoint 按 VALIDATION loss 最小值选择；
- loss 为 Q point loss、TRAIN-only 高流量加权 loss 和 6 h volume loss。

每组实验会生成三个 checkpoint 和一个训练日志。例如 E4：

```text
outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.pt
outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.last.pt
outputs/e4_water_balance_lstm_kinematic_wave_gnn_final.pt
outputs/e4_water_balance_lstm_kinematic_wave_gnn_train.csv
```

- `*_best.pt`：VALIDATION loss 最优模型；
- `*.last.pt`：最新 epoch 的完整状态，用于续训；
- `*_final.pt`：第 100 epoch 模型；
- `*_train.csv`：逐 epoch 训练与验证指标。

已有同名输出时程序默认拒绝覆盖。重新开始并覆盖旧输出：

```bash
python train_hunan.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --overwrite
```

从最新完整状态续训：

```bash
python train_hunan.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --resume outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.last.pt
```

`--resume` 与 `--overwrite` 不能同时使用。

## 6. 评价

TEST 评价：

```bash
python evaluate.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.pt \
  --split TEST
```

VALIDATION 评价：

```bash
python evaluate.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.pt \
  --split VALIDATION
```

指定评价目录和顶层 JSON：

```bash
python evaluate.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.pt \
  --split TEST \
  --output-dir outputs/e4_test_evaluation \
  --output outputs/e4_test_result.json
```

评价结果包括：

- 总体、站点和河网 Q 指标；
- station/event 级指标；
- 1–6 h lead-time 指标；
- persistence 与 Delta-Q 对照；
- 固定提前量事件洪峰、洪峰时刻和事件 NSE；
- rating curve 推导水位及其覆盖率、外推范围审计。

未指定 `--output-dir` 时，结果写入 checkpoint 同目录下的：

```text
<checkpoint_name>_<split>_evaluation/
```

## 7. 测试

运行全部测试：

```bash
pytest -q
```

只验证当前四组模型：

```bash
pytest -q tests/test_hydrologic_model.py
```

该测试覆盖四组模型的前向与反向传播、输出形状与有限性、Q0 输出锚定，以及未来 Z target 不进入预测过程。

## 8. 项目结构

```text
configs/                         四组实验及共享配置
data/                            batch 数据结构、设备和拓扑工具
datasets/                        数据集读取、校验和事件均衡 sampler
models/hydrologic_model.py       统一 LSTM–GNN–FC 模型
models/runoff/                   普通/水量平衡产流实现
models/routing/                  普通 GNN 与运动波 GNN
losses/hydrologic_loss.py        Q-only 训练目标
trainers/                        训练循环、checkpoint 和评价聚合
metrics/                         站点、事件、时效及洪峰指标
scripts/training.py              配置、数据、模型的统一装配入口
scripts/rating.py                TRAIN-only 站点 rating curve 拟合
train_hunan.py                   训练入口
evaluate.py                      VALIDATION/TEST 评价入口
tests/                           数据、物理模块和模型测试
outputs/                         checkpoint、日志和评价结果
```

## 9. 主要配置项

共享参数位于 `configs/base.yaml`，四组实验配置通过 `_base_` 继承。常用参数包括：

| 配置项 | 含义 |
|---|---|
| `runoff_mode` | `pure_lstm` 或 `water_balance_lstm` |
| `routing_mode` | `pure_gnn` 或 `kinematic_wave_gnn` |
| `hidden_dim` | LSTM/GNN 隐藏维度 |
| `batch_size` | 同一 graph mini-batch 大小 |
| `data.dataset_root` | 模型数据集目录 |
| `data.future_rainfall_mode` | `observed_hindcast`、`zero` 或 `persistence` |
| `physical_bounds` | 河宽与曼宁系数范围 |
| `solver` | 运动波时间步、空间步长和隐式迭代设置 |
| `loss` | Q 点、高流量和体积损失权重 |
| `training` | epoch、输出路径和梯度裁剪 |

修改实验参数时，应同时修改 `training.checkpoint`、`training.final_checkpoint` 和 `training.log_csv`，保证不同实验输出互不覆盖。
