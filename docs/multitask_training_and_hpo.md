# E4 多目标损失、验证选择与共享超参数优化

## 1. 向后兼容与入口

`configs/hunan_e4.yaml` 保持旧实验：

\[
L_{legacy}=2\,\operatorname{Huber}(Q/\sigma_Q)+
\operatorname{Huber}(Z/\sigma_Z).
\]

新实验使用 `configs/hunan_e4_multitask.yaml`，输出到
`outputs/hunan_e4_multitask_v1_*`。两个尺度 \(\sigma_Q,\sigma_Z\) 均来自
`normalization_stats.json` 中明确标记为 TRAIN 的标准差。VALIDATION/TEST
不会重算 loss scale。

严格对照实验使用 `configs/hunan_e4_multitask_qnorm_v1.yaml`。它保持模型、
优化器、batch、全部多任务权重及 composite selection 不变，只将三个 Q loss
项的 \(\sigma_Q\) 改为逐图 TRAIN scale，并将 patience 设为 100 以完整执行
epoch 0--99。best checkpoint 仍按原 composite score 选择，不使用最后一轮
替代。既有 `hunan_e4_multitask.yaml` 继续使用全局 Q scale。

## 2. TRAIN 事件—流域平衡

对 graph \(g\) 中 event \(e\) 的一个滑动窗口 sample，原始权重为：

\[
w_{g,e,s}^{raw}=\frac{1}{N_G N_{E(g)} N_{S(g,e)}}.
\]

最后乘以 TRAIN sample 总数，使全体 sample weight 平均值为 1。于是每个
graph 总权重相等；同一 graph 内每个 event 总权重相等；一个 event 增加
五倍窗口只会把该 event 的固定总权重分给更多窗口，不会自动增加五倍梯度
贡献。该权重仅用于 Q point、peak、volume 三项。

## 3. 多目标损失

总损失固定为：

\[
L=2L_Q+L_Z,
\]

\[
L_Q=L_{q,point}+\lambda_{peak}L_{q,peak}
+\lambda_{volume}L_{q,volume},
\]

\[
L_Z=L_{z,level}+\lambda_{slope}L_{z,slope}.
\]

- `q_point`：每个 sample 内先对有效 Q target 的 TRAIN-scale Huber 求均值，
  再应用事件—流域权重。
- `q_peak`：未来 6 h 有效 mask 内预测峰与实测峰之差除以 \(\sigma_Q\)，
  使用平方误差。
- `q_volume`：未来有效小时平均 Q 之差除以 \(\sigma_Q\)，使用平方误差。
  采用平均 Q 可避免缺测导致有效小时数不同而产生无意义体积尺度差异；1 h
  固定步长下它与有效时段累计量约束等价。
- `z_level`：绝对水位的 TRAIN-scale mask-aware Huber。
- `z_slope`：真正的 first difference。第一预测小时减去 history 中最近一个
  有效实测 Z；history 无有效 Z 时关闭第一小时 slope mask。后续小时计算
  \(Z_h-Z_{h-1}\)，只有相邻两个 target 都有效时才参与。预测和实测使用
  完全相同的差分定义，不使用未来实测作校正。

默认 \(\lambda_{peak}=\lambda_{volume}=\lambda_{slope}=0.25\)。训练 CSV
分别记录 total、Q total/point/peak/volume、Z total/level/slope 及有效数量。

逐图对照配置对每个含 FLOW 监督的 graph 先按
`GRAPH_ID + outlet target timestamp` 去除滑窗重复，再以 TRAIN 物理 Q
计算 population std（`ddof=0`）：

\[
\sigma_{Q,g}^{used}=\max(\operatorname{std}(Q_{TRAIN,g}),1.0\;m^3/s).
\]

不足两个有效唯一时刻会按 GRAPH_ID 立即失败，不回退全局 std。这个 scale
只进入 Q point/peak/volume 的 error scaling；模型输出、history/target、运动波
路由和全部正式指标仍为物理单位。VALIDATION/TEST 会重新构造只读 TRAIN 视图
取得相同 scale，不在被评估 split 上拟合。训练开始时完整统计写入
`outputs/hunan_e4_multitask_qnorm_v1_q_scales.json`，并随 checkpoint config
保存、在 resume/evaluation 时核对。审计文件列出全部 TRAIN graph；
WATER_LEVEL-only graph 明确标为 `NOT_APPLICABLE_NO_FLOW_SUPERVISION`，不为
不存在的 Q loss 伪造或回退 scale。

## 4. VALIDATION selection score

正式 VALIDATION 先复用既有去重规则：同一
`EVENT_ID/station/target timestamp` 只保留最短 lead（并列时使用 lexical
`SAMPLE_ID`）。令：

\[
S_{eff}(x)=\frac{clip(x,-1,1)+1}{2},\qquad
S_{err}(e;s)=\frac{1}{1+e/s}.
\]

六项 skill 为：

1. graph median Q NSE：\(S_{eff}\)；
2. graph median Q KGE：\(S_{eff}\)；
3. event median absolute relative peak error：\(S_{err}(e;1)\)；
4. event median absolute relative volume error：\(S_{err}(e;1)\)；
5. station median absolute-Z MAE：\(S_{err}(e;\sigma_Z)\)；
6. station median first-difference-Z MAE：\(S_{err}(e;\sigma_Z)\)。

最终分数为：

\[
S=0.35S_{NSE}+0.15S_{KGE}+0.20S_{peak}+0.10S_{volume}
+0.10S_{Z}+0.10S_{dZ}.
\]

`validation_selection_score` 越大越好，并且同时用于 best checkpoint、early
stopping、Optuna objective 和 MedianPruner report。旧 `val_loss` 继续记录，
但在新配置中不再选择模型。若某轮所有 graph 的 NSE/KGE 都因零方差而无定义，
对应 skill 明确记为 0、`defined` 标志记为 0，且原权重不重分配；任何 peak、
volume 或 Z 误差 component 无定义时则明确失败。

## 5. Optuna 框架

普通 `train_hunan.py` 不导入或启动搜索。配置默认
`hyperparameter_optimization.enabled: false`，独立入口还要求显式
`--enable`，因此不会因安装 Optuna 而误启动。

搜索器为 `TPESampler(seed=42)`；剪枝器为：

```text
MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
```

仅搜索 `learning_rate`、`weight_decay`、`hidden_dim`、`q_peak_weight`、
`q_volume_weight`、`z_slope_weight`。E4 mode、Q:Z、point/level 权重、batch、
history、forecast、数据 split、物理边界、Manning n、CFL、dx、坡降下限、
积分格式、隐式迭代和运动波求解参数均禁止搜索。代码中的 allow-list 会拒绝
多余或缺失参数。

每个 trial 只构造 TRAIN/VALIDATION loader，输出隔离在
`outputs/hpo_e4_multitask_v1/trial_XXXX/`。只有真实完成研究后才导出
`best_params.json`、`best_trial_summary.json`、`study_trials.csv` 和共享配置
片段；当前仓库不包含伪造的最优参数。

## 6. E1–E4 冻结原则

未来只用 E4 搜索一次共享参数。得到真实 best params 后，E1–E4 必须冻结
同一数据集、split、seed、loss、loss weights、hidden dim、lr、weight decay、
batch size、epochs、early stopping、checkpoint selection 和评价流程；四组只
切换 runoff/routing physics mode。TEST 只在参数、epoch 和 checkpoint 全部由
TRAIN/VALIDATION 冻结后分别执行，绝不参与搜索、剪枝或选择。
