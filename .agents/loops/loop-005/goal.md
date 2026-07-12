#goal 在统一 cross-fitted 契约下判断 LightGBM LambdaRank 是否形成模型族 Pareto 改善

当前状态：`achieved`。已完成模型族判断；LightGBM 因均值、最差折、方差和 RankIC 回归被拒绝。

## 最终状态（定义 Done）

1. LightGBM 与冻结 XGBoost baseline 使用相同数据、39 特征、folds、160 rounds 和原始收益评估。
2. checkpoint 对每个 outer fold 只由其他 folds 选择，holdout 未加载。
3. 形成 mean/worst/std/RankIC/耗时的 Pareto 决策。
4. 失败候选不修改生产配置；成功候选进入最新 validation 二次门禁。

## 验证方式

- `python -m code.experiments.lightgbm_family_crossfit`，timeout 30 分钟；输出四折证据。
- `python .agents/scripts/validate_loop.py`；全部 loop 可解析。

## 约束条件

- 只比较一个现有配置候选，不做 LightGBM HPO。
- 不使用 NDCG 训练指标作为验收，不加载 holdout。
- 超过 10 分钟无进展则终止并审计。

## 明确非目标

- 本 loop 不做模型集成或 CatBoost。
- 不修改正式训练入口。

## 阻塞条件

- LightGBM 无法在当前既有环境运行。
- baseline evidence 与候选 folds 不一致。

## 最终审计

- [x] baseline 来源可追踪
- [x] holdout 未加载
- [x] Pareto 决策完整
- [x] 生产配置未被失败候选修改

## 迭代优先级

1. 可比性。
2. cross-fitted 结果。
3. 推广或轮换 CatBoost。
