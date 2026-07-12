#goal 在统一 cross-fitted 契约下判断 CatBoost YetiRankPairwise 是否形成模型族 Pareto 改善

当前状态：`achieved`。已完成最后一个异构树族判断；CatBoost 因方差、正收益率和 RankIC 回归被拒绝，触发结构方向停止规则。

## 最终状态（定义 Done）

1. CatBoost 与冻结 XGBoost baseline 使用相同数据、39 特征、folds、160 rounds 和原始收益评估。
2. checkpoint 对每个 outer fold 只由其他 folds 选择，holdout 未加载。
3. 形成 mean/worst/std/RankIC/耗时的 Pareto 决策。
4. 失败候选不修改生产配置；若失败则模型结构族方向停止。

## 验证方式

- `python -m code.experiments.catboost_family_crossfit`，timeout 30 分钟；输出四折证据。
- `python .agents/scripts/validate_loop.py`；全部 loop 可解析。

## 约束条件

- 只比较一个现有 CatBoost 配置，不做 HPO 或 ensemble。
- 不加载 holdout，不改变生产配置。
- 超过 10 分钟无进展则检查 GPU/进程/日志并终止异常运行。

## 明确非目标

- 不创建新的神经网络或 NAS 搜索空间。
- 不用 CatBoost 自带排序指标替代原始收益。

## 阻塞条件

- CatBoost GPU 在当前 RTX 3090/驱动不可运行。
- 两个模型无法共享 folds 或特征契约。

## 最终审计

- [x] GPU 设备明确为 0
- [x] holdout 未加载
- [x] Pareto 决策完整
- [x] 连续无 Pareto 改善停止规则真实执行

## 迭代优先级

1. 环境与可比性。
2. cross-fitted 结果。
3. 冻结结构方向或推广。
