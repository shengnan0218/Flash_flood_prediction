# Flash Flood Prediction

本仓库只保留一条模型主线：

```text
72 h 降雨 + 节点静态属性
        ↓
产流 LSTM
        ↓
有向河网 GNN
        ↓
Q0 锚定的小型全连接层
        ↓
未来 6 h 流量 Q
```

模型不包含 observation encoder、hidden-state correction、上游残差传播或河道蓄量校正。过去 24 h 的 Q/Z 不进入 LSTM/GNN；最后一个有效 Q0 仅用于输出层构造 `Q0 + (Qroute(t) - Qroute(0))`。水位 Z 仅由 TRAIN 数据拟合的固定 rating curve 从 Q 推导，不参与训练。

## 四组严格对照

| 实验 | 产流 | 汇流 | 配置 |
|---|---|---|---|
| E1 | 普通 LSTM | 普通有向 GNN | `configs/e1_pure_lstm_pure_gnn.yaml` |
| E2 | 水量平衡 LSTM | 普通有向 GNN | `configs/e2_water_balance_lstm_pure_gnn.yaml` |
| E3 | 普通 LSTM | 运动波 GNN | `configs/e3_pure_lstm_kinematic_wave_gnn.yaml` |
| E4 | 水量平衡 LSTM | 运动波 GNN | `configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml` |

四组实验共享相同的数据、输入、输出头、损失、采样、优化器和评价代码，只改变两个物理开关。

## 运行

```bash
python train_hunan.py --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml

python evaluate.py \
  --config configs/e4_water_balance_lstm_kinematic_wave_gnn.yaml \
  --checkpoint outputs/e4_water_balance_lstm_kinematic_wave_gnn_best.pt \
  --split TEST
```

其他实验只需替换配置文件。每组配置有独立 checkpoint、训练日志和评价输出，避免互相覆盖。

## 固定数据契约

- 降雨历史：72 h
- Q/Z 观测历史：24 h（当前模型只读取末时刻 Q0/Z0，用于输出锚定和水位换算）
- 预报时长：6 h
- 节点静态属性：当前 10 项，暂不修改
- 边静态属性：河段长度、坡降
- normalization、洪水阈值和 rating curve：只能由 TRAIN 拟合
- TRAIN：事件均衡、阶段分层采样；VALIDATION/TEST：完整样本

## 核心文件

- `models/hydrologic_model.py`：唯一模型实现
- `models/runoff/water_balance_continuous.py`：水量平衡 LSTM
- `models/routing/kinematic_wave_optimized.py`：运动波 GNN
- `losses/hydrologic_loss.py`：统一 Q-only 损失
- `scripts/training.py`：唯一训练/评价装配入口
- `tests/test_hydrologic_model.py`：四组前向/反向、无状态校正、无未来 Z 泄漏测试

旧版模型、配置、训练器、损失、测试、诊断结果和状态校正模块已删除，不再提供兼容入口。
