---
name: bdc2026-experiment
description: 在冻结数据和滚动验证门禁下设计、执行并记录 BDC2026 单假设模型实验、结构比较和有限 HPO。需要优化模型时使用。
---

# BDC2026 Experiment

每轮只登记一个可证伪假设，先做低成本 smoke，再做同折完整验证。比较必须使用比赛绝对组合收益；excess、RankIC 和资源指标用于诊断和 Pareto 门禁。

优先尝试标签、表示、时序/横截面结构、排序目标、checkpoint 和组合器；只有完整 Pareto 胜出后才允许受限 Optuna。holdout 只在验证选型后观察，不得早停或排序。

运行前冻结配置，运行后保存日志、指标、运行时、哈希和反思。连续两轮无 Pareto 改善或四轮无新证据时停止。禁止联网训练、覆盖 champion、无边界 HPO 或使用绝对路径。
