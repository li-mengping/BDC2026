#goal 用 cross-fitted checkpoint 规则复核无股票身份特征候选并形成可推广决策

当前状态：`achieved`。候选通过 cross-fitted Pareto 门禁，生产 round 由四折曲线固定为 127；推广工作进入下一 loop。

## 最终状态（定义 Done）

只有以下条件全部成立，本 loop 才能写为 `achieved`：

1. 有/无 `instrument` 两个变体在同一四折、参数和逐轮指标上可比。
2. 每个被评估折的 checkpoint 只由其余三折选择，被评估折不参与选择。
3. 结果包含均值、最差折、标准差、正收益率、选择 round、耗时和数据哈希。
4. 仅当候选形成 Pareto 改善时才允许推广；失败证据不得覆盖生产配置。
5. 状态、证据、日志和 handoff 可由新 Agent 独立恢复。

## 验证方式

| 条件 | 命令或证据 | 通过标准 |
| --- | --- | --- |
| 1-3 | `python -m code.experiments.xgb_crossfit_checkpoint`，timeout 30 分钟 | 四个 outer fold 均有 other-fold round 来源和指标 |
| 4 | `evidence/crossfit_checkpoint.json` | 决策规则与生产 diff 一致 |
| 5 | `python .agents/scripts/validate_loop.py` | loop-001 与 loop-002 全部可解析 |

长命令每 5 分钟检查日志；10 分钟无 CPU/GPU 或日志进展时终止并记录。实验通常应在 10 分钟内完成。

## 约束条件

- 只使用冻结数据和调用方已激活的兼容环境，不得联网或访问 holdout 选型。
- 不改变标签、切分、参数或组合策略，确保只审计 checkpoint 与身份特征。
- 不删除失败证据，不把同折峰值分数写成 cross-fitted 分数。

## 明确非目标

- 本 loop 不进行 HPO、模型集成或 Docker 发布。
- 不根据已观察 holdout 推广候选。
- 不把 implemented、validated 和 achieved 混写。

## 阻塞条件

- 逐轮指标无法按 fold 对齐。
- CUDA/依赖故障导致两个变体不可比。
- 两次实现尝试仍无法证明被评估折没有参与 round 选择。

## 最终审计

- [x] 重读 Done 与约束
- [x] 运行 cross-fitted 实验与 loop 校验
- [x] 核对 holdout 未加载
- [x] 核对实验阶段未改生产配置
- [x] 所有本 loop 条件已满足

## 迭代优先级

1. 选择器无泄漏证明。
2. 两变体可比证据。
3. 推广或拒绝决策。
4. 轮换到 Top5 objective。
