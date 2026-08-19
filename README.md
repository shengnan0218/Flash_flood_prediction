# Flash Flood Prediction — Hunan Formal V10 / V9 / V8

本仓库当前正式数据事实层为冻结的 **V8 hydrologic computational graph dataset**：

```text
_model_dataset_v8_hydrologic_graph/
```

正式训练、评价和 preflight 入口只支持 **V10、V9、V8**。P2、P3 以及 V8 之前的实验配置和专用运行路径已经退役，不再作为正式工作流。

## 1. 当前正式版本

### V10 — 默认正式模型

配置：

```text
configs/hunan_e4_v10.yaml
```

V10 的职责划分是：

```text
24 h rainfall/history
        ↓
water-balance LSTM runoff
        ↓
optimized kinematic-wave routing
        ↓
1–6 h Q forecast                 ← 唯一学习/监督目标
        ↓
TRAIN-only station rating curve
        ↓
Q-derived stage
        ↓
final-history-bin Z residual anchoring
```

关键约束：

- 主模型只学习未来 **Q**；没有独立 neural Z head。
- future Z 不进入 loss，不参与 checkpoint selection，也没有梯度回传到 Q 模型。
- 保留 V9 的 24 h sequential warm-up、质量守恒的状态订正和优化后的 kinematic-wave routing。
- 历史 Q/Z 仍可作为起报状态同化输入；“不用 Z 做未来监督”不等于“丢弃历史 Z 状态信息”。
- station-specific rating curve 只由冻结数据 **TRAIN** 中唯一物理目标时刻的同时有效 Q/Z 拟合：`Z = aQ + b`。
- 水位输出为：

```text
Delta-Z_hat = f_station(Q_future_hat) - f_station(Q0_analysis)
Z_hat       = Z0_observed + Delta-Z_hat
```

- `Q0_analysis` 优先使用最后一个 history 小时 bin 内保留的 Q 观测；缺测时使用 V9 状态同化后的物理/model Q0。
- `Z0_observed` 必须来自最后一个 history 小时 bin；不向前搜索更早 Z，不使用未来 Z。
- 冻结 V8 的小时标签表示 hourly bin。最后一个 Q0/Z0 是该 bin 内保留下来的代表观测，**不保证是 bin 末端的精确瞬时整点值**。
- rating curve 不对超出 TRAIN Q 范围的预测做静默截断；最终评价会显式报告 rating extrapolation 比例。

### V9 — 保留可复现

保留：

```text
configs/hunan_v9_base.yaml
configs/hunan_e1_v9.yaml
configs/hunan_e2_v9.yaml
configs/hunan_e3_v9.yaml
configs/hunan_e4_v9.yaml
```

V9 的模型、loss、trainer、评价和测试均保留，用于完整复现实验和与 V10 对照。

### V8 — 保留可复现

保留原 E1–E4 配置：

```text
configs/hunan_e1_pure_ai.yaml
configs/hunan_e2_physics_runoff.yaml
configs/hunan_e3_physics_routing.yaml
configs/hunan_e4.yaml
```

V8 的冻结 dataset contract、loader、模型和评价逻辑不由 V10 重建或覆盖。

## 2. V10 数据边界

V10 **不重建 V8 数据集**，只在 loader 层建立只读任务视图。

冻结 V8 sample 可因 Q 或 Z 任一任务有效而存在。因为 V10 是 Q-only：

- TRAIN：只保留冻结 `sample_index.csv` 中 `Q_TARGET_VALID_COUNT > 0` 的样本进入学习；
- VALIDATION：同样只保留有 Q target 的样本用于 Q-only model selection；
- TEST 最终评价：保留完整冻结 TEST split，Q 和 derived stage 各自使用自己的 truth mask。

这不是重新划分数据：EVENT_ID、SPLIT、FORECAST_TIME、tensor row、graph topology 和观测值全部沿用冻结 V8；preflight 会同时报告 frozen sample count、active Q-supervised count 和被任务视图排除的数量。

## 3. TRAIN-only rating curve

V10 在启动时从冻结 dataset root 读取 TRAIN，并按：

```text
(STATION_ID, physical target unix hour)
```

去重重叠 forecast windows。若同一物理时刻的重复 Q/Z 值冲突，直接失败，不静默选取。

rating curve 拟合本身不要求 Q0，因为它估计的是站点 Q–Z 关系，而不是 forecast-origin availability。正式配置要求所有 outlet station 都有足够的 TRAIN-only Q/Z 配对；否则 preflight 失败。

rating 参数是 model buffer，不是 trainable parameter。checkpoint 绑定 rating artifact SHA；评价时如果 dataset/rating artifact 已变化会拒绝加载为同一实验。

## 4. 时间和 forcing 语义

当前湖南冻结数据：

```text
history duration: 24 h
forecast duration: 6 h
forcing step:      1 h
target step:       1 h
future rainfall:   observed_hindcast
```

降雨时间戳表示 `[start, end)` 小时区间起点。hydro 小时标签是 hourly bin 标签；当前冻结处理保留 bin 内代表观测。forecast origin 是最后一个 history bin 的结束边界；没有对整个 Q/Z tensor 做额外 1 h shift。

`observed_hindcast` 是当前湖南实验的既定 forcing 条件，不应解释为业务上已知未来降雨。未来业务预测需另行接入降雨预报 forcing，并作为新的实验条件报告。

## 5. V10 loss 与训练

V10 只包含：

```text
q_point_weight  = 1.0
q_peak_weight   = 0.25
q_volume_weight = 0.25
```

Q 误差使用 TRAIN-only per-station Q scale。没有 Z level、Z slope、Q–Z consistency 或其他 future-Z loss。

正式配置：

```text
batch_size: 16
epochs: 100
early_stopping: false
optimizer: AdamW
lr: 0.001
```

checkpoint selection 固定为 **Q-only validation loss**。

## 6. 运行顺序

服务器同步：

```bash
cd ~/Flash_flood_prediction
git fetch origin
git reset --hard origin/main
```

训练前必须先做只读 preflight：

```bash
python validate_dataset.py --output outputs/hunan_e4_v10_preflight.json
```

只有报告最后为：

```json
"status": "VALID"
```

才进入正式训练。

启动 V10：

```bash
nohup python train_hunan.py > hunan_e4_v10_train.log 2>&1 &
```

查看训练：

```bash
ps -eo pid,etime,%cpu,%mem,cmd | grep "python train_hunan.py" | grep -v grep
tail -f hunan_e4_v10_train.log
```

默认输出：

```text
outputs/hunan_e4_v10_best.pt
outputs/hunan_e4_v10_final.pt
outputs/hunan_e4_v10_train.csv
```

最终 TEST：

```bash
python evaluate.py --checkpoint outputs/hunan_e4_v10_best.pt --split TEST
```

V10 final evaluation 输出至少包括：

- global Q metrics；
- station Q / derived-stage metrics；
- graph metrics；
- event × station metrics；
- 1–6 h lead-time metrics；
- corrected-stage coverage；
- TRAIN rating range extrapolation audit；
- rating artifact / evaluation-view audit。

## 7. V8 / V9 复现

V8/V9 不再是默认入口，但可以通过显式 `--config` 复现：

```bash
python train_hunan.py --config configs/hunan_e4_v9.yaml
python train_hunan.py --config configs/hunan_e4.yaml
```

对应评价也显式提供相同 config：

```bash
python evaluate.py --config configs/hunan_e4_v9.yaml --checkpoint <checkpoint> --split TEST
```

V10 的新增代码不得反向修改 V8/V9 模型架构、loss 或配置语义。

## 8. 代码审计门禁

正式合并前 CI 同时检查：

- V10 config contract；
- TRAIN-only rating calibration 与 overlap dedup；
- Q-supervised read-only dataset view；
- V10 state_dict 中不存在 independent Z head；
- rating curve 不可训练；
- derived stage 已从 autograd detach；
- rating intercept 对 corrected Delta-Z 不产生影响；
- 缺 Z0 时不向前搜索；
- 缺 observed Q0 时使用 model/assimilated Q0；
- negative Q fail-fast；
- Q-only loss 不受任何 Z 输出影响；
- V9 state assimilation tests；
- optimized kinematic-wave 与原 solver 的输出/梯度等价性；
- V8/V9 相关正式测试继续通过。

如果 real-data preflight、checkpoint compatibility 或任何物理/时间契约不一致，程序应失败，而不是回退、填补或静默修正。
