# Flash Flood Prediction — Hunan Formal V11 / V10 / V9 / V8

当前默认正式模型为 **V11**。V8、V9、V10 的模型、配置和运行路径继续保留，用于严格复现和对照；P2、P3 以及更早的专用实验路径已经退役。

## 1. V11 为什么存在

V10 的只读泛化审计显示：总体 Q 指标较好，但 TRAIN→VALIDATION 的动态响应泛化明显下降；尤其 `Delta-Q` 和相对 Q0 persistence 的增益明显弱于 TRAIN。同时，原 `q_peak_loss` 实际约束的是每个滑动 6 h window 的局部最大值，并不等同于真正洪水事件洪峰。

V11 因此**不改 V10 的可训练模型架构**，只针对状态暴露、TRAIN sampling 和 loss 定义做三项有明确物理/统计含义的修改。

## 2. V11 正式设计

配置：

```text
configs/hunan_e4_v11.yaml
```

主流程：

```text
72 h antecedent rainfall
        ↓
water-balance LSTM runoff + optimized kinematic-wave routing warm-up
        ↓
last 24 h Q/Z observation context only
        ↓
forecast-origin mass-aware state assimilation
        ↓
1–6 h Q forecast                    ← 唯一学习/监督目标
        ↓
TRAIN-only station rating curve
        ↓
Q-derived stage + final-history-bin Z anchor
```

关键约束：

- rainfall physical warm-up 从 24 h 扩为 **72 h**，用于恢复更长的前期湿润/蓄水状态；
- Q/Z observation encoder **严格仍为 24 h**，不能因为扩展 rainfall history 而同时扩到 72 h，以免强化站点历史/persistence shortcut；
- runoff、optimized kinematic-wave routing、V10 mass-aware forecast-origin assimilation、Q-only forecast 和 rating-derived Z 架构保持不变；
- future Z 不进入 loss，不参与 checkpoint selection，也没有梯度回传到 Q；
- `Q0_analysis` 与 `Z0_observed` 仍遵守最后一个 history hourly bin 的 V10 语义，不伪称为精确 bin-end 瞬时观测；
- rating curve 仍只由 TRAIN 中唯一物理时刻 Q/Z 拟合；
- TEST 最终评价仍保留完整冻结 split，不因为 Q-only 训练过滤掉可评价 derived stage 的样本。

## 3. V11 数据层：从冻结 V8 派生，不改 V8

冻结 V8 数据事实层继续保留：

```text
_model_dataset_v8_hydrologic_graph/
```

V11 新建独立派生数据集：

```text
_model_dataset_v11_72h_event_balanced/
```

构建脚本：

```text
scripts/20_build_hydrologic_graph_model_dataset_v11.py
```

V11 **严格继承 V8 的 33 graphs、2,807 events、279,574 SAMPLE_ID、EVENT_ID、SPLIT 和 FORECAST_TIME**。Q/Z、静态属性、拓扑和未来 6 h target 均继承冻结 V8；只重新从 authoritative computational-unit rainfall source 构造 72 h antecedent rainfall tensor。

构建命令示例：

```bash
python scripts/20_build_hydrologic_graph_model_dataset_v11.py \
  --v8-dataset _model_dataset_v8_hydrologic_graph \
  --hydrologic-graph _hydrologic_graph_v1 \
  --output-dir _model_dataset_v11_72h_event_balanced
```

如果 hydrologic-graph rainfall 源不在默认路径，必须显式给出 `--hydrologic-graph`。

**不能对 V8 最早样本之前缺少的 48 h rainfall 静默补 0。** Builder 会逐 node 检查 authoritative rainfall valid period；如果无法覆盖某个 V8 origin 所需的完整 72 h antecedent period，直接失败。它还会重新计算未来 6 h rainfall，并与冻结 V8 future-rain tensor 做 `1e-6` 数值一致性校验，防止时间轴错位。

## 4. Event-balanced TRAIN

V11 不再 exhaustive 地把所有逐小时 sliding forecast origins 都送入每个 epoch。

TRAIN 的基本平衡单位是 **EVENT_ID**：

```text
每个 event / epoch：最多 8 个不重复 forecast origins

优先：
2 × LOW
2 × RISING
2 × PEAK
2 × RECESSION
```

若某一 phase 候选不足，从该 event 尚未选择的其他有效 origins 补齐；event 总候选不足 8 时全部使用，不制造重复样本。

phase 标签只用于 TRAIN sampling。其定义基于该 event 的 outlet Q：

- `PEAK`：该 6 h target window 的 outlet max ≥ event outlet peak 的 80%；
- `LOW`：window max ≤ event outlet Q 中位数；
- 其余若 window 内 Q 上升则 `RISING`；否则 `RECESSION`。

VALIDATION 不使用 event-balanced subsampling：model selection 仍评价完整的 Q-supervised VALIDATION view。TEST 使用完整冻结 TEST view。

## 5. V11 loss

V10 的 sliding-window maximum peak term 被正式删除。V11 loss：

```text
q_point_weight     = 1.0
q_high_flow_weight = 0.25
q_volume_weight    = 0.25
```

`q_point` 继续使用 TRAIN-only per-station Q scale 的 Huber error。

`q_high_flow` 使用每站只从 TRAIN、按 `(STATION_ID, physical target hour)` 去重后计算的 P80/P99：

```text
Q < P80     : 不进入额外 high-flow term
P80→P99     : multiplier 从 1 平滑增加到 3
Q >= P99    : multiplier = 3
```

它直接加强真正高流量物理时刻，而不是把任意 6 h window 的局部 maximum 当成一次洪峰。

`q_volume` 保留 V10 的 6 h duration-normalized mean-Q / volume bias 定义。

所有 Q normalization、P80/P99 threshold 和 rating calibration 都严格 TRAIN-only。

## 6. V11 evaluation

正式 V11 evaluation 保留 V10 的：

- pooled / station / outlet / graph Q metrics；
- derived `Delta-Z` 和 anchored absolute Z；
- rating extrapolation；
- event × station 和 lead-time metrics。

此外增加两类真正用于泛化判断的指标：

```text
1–6 h Q0 persistence baseline
skill over persistence
Delta-Q NSE/RMSE
```

以及按固定提前量重建事件过程的：

```text
fixed-lead event peak magnitude
peak ratio / relative error
peak timing error
fixed-lead event NSE
```

因此 V11 不再用训练中的 window-max loss 代替真正 event-peak forecast skill。

## 7. V11 preflight

数据构建完成后，训练前必须先运行：

```bash
python validate_dataset.py --output outputs/hunan_e4_v11_preflight.json
```

preflight 会同时验证：

- 72 h rainfall tensor / 24 h Q-Z history / 6 h forecast；
- antecedent rainfall 无 valid-period 外 zero padding；
- TRAIN/VALIDATION/TEST event 无泄漏；
- Q-only TRAIN/VALIDATION view 与完整 frozen split 的关系；
- 实际 epoch-0 event-balanced sampling 无重复、每 event ≤8、batch 不跨 graph；
- phase contract；
- P80/P99 threshold 的 TRAIN-only unique-physical-hour provenance 与 outlet coverage；
- rating 的 TRAIN-only provenance；
- 没有独立 neural Z head，rating 不是 trainable parameter；
- 没有残留 window `q_peak_loss`；
- 最终 TEST view 不被 Q-supervision filter 缩减。

只有最终：

```json
"status": "VALID"
```

才启动正式训练。

## 8. 正式训练与评价

服务器同步：

```bash
cd ~/Flash_flood_prediction
git fetch origin
git reset --hard origin/main
```

启动 V11：

```bash
nohup python -u train_hunan.py \
  --config configs/hunan_e4_v11.yaml \
  > hunan_e4_v11_train.log 2>&1 &
```

默认输出：

```text
outputs/hunan_e4_v11_best.pt
outputs/hunan_e4_v11_final.pt
outputs/hunan_e4_v11_train.csv
```

正式 VALIDATION/TEST：

```bash
python evaluate.py \
  --config configs/hunan_e4_v11.yaml \
  --checkpoint outputs/hunan_e4_v11_best.pt \
  --split VALIDATION
```

模型方案确定前优先使用 VALIDATION；TEST 保留为最终报告。

## 9. V10 / V9 / V8 保留复现

V10：

```text
configs/hunan_e4_v10.yaml
```

V9：

```text
configs/hunan_v9_base.yaml
configs/hunan_e1_v9.yaml
configs/hunan_e2_v9.yaml
configs/hunan_e3_v9.yaml
configs/hunan_e4_v9.yaml
```

V8：

```text
configs/hunan_e1_pure_ai.yaml
configs/hunan_e2_physics_runoff.yaml
configs/hunan_e3_physics_routing.yaml
configs/hunan_e4.yaml
```

显式 `--config` 仍可复现这些版本；V11 不修改其模型、loss、trainer、dataset contract 或配置本体。
