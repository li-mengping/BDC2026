#goal 用无 holdout 的 cross-fitted 门禁判断 Top5 对齐 NDCG 是否优于正式 pairwise

当前状态：`achieved`。已完成可比判断；NDCG 候选因最差折、方差和 RankIC 回归被拒绝，生产配置未修改。

## 最终状态（定义 Done）

1. pairwise 与一个 NDCG top-k 候选使用相同数据、特征、参数预算和 folds。
2. NDCG 训练标签为周内 0-30 relevance，评估仍用原始周收益。
3. checkpoint 对每个 outer fold 只由其他 folds 选择。
4. 有完整 Pareto 决策；失败不得修改生产配置。

## 验证方式

- `python -m code.experiments.xgb_top5_objective`，timeout 30 分钟；输出四折 cross-fitted 指标与逐轮证据。
- `python .agents/scripts/validate_loop.py`；四个 loop 均可解析。

## 约束条件

- 不加载 holdout，不做 HPO，不改窗口、特征或组合策略。
- 候选必须同时满足 mean/worst 不下降、std 不上升。
- 运行超过 10 分钟无进展时终止并审计。

## 明确非目标

- 不在本 loop 推广模型族集成。
- 不用 NDCG 指标替代真实收益门禁。

## 阻塞条件

- XGBoost NDCG 不接受周内 relevance 契约。
- 两个目标无法在同一 folds/round 预算下比较。

## 最终审计

- [x] 标签转换测试通过
- [x] holdout 未加载
- [x] cross-fitted 选择证明完整
- [x] 生产配置未被失败候选修改

## 迭代优先级

1. 标签与目标契约。
2. cross-fitted 比较。
3. 推广或停止。
