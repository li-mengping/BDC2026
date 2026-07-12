# BDC2026 Agent 唯一入口

## 启动顺序

1. 运行 `python .agents/agentctl.py doctor`。
2. 从 `doctor` 输出读取 `active_goal` 与 `current_loop`，不得从文档猜测当前状态。
3. 读取对应 Goal、loop 的 `state/loop.json`、`plan.md`、`handoff.md` 和 evidence index。
4. 按任务路由读取一个仓库 skill；不要一次加载全部 references。

## 事实源

- 比赛口径：`.agents/constraints/competition.md` 与 `requirements-matrix.md`
- 当前审计：`.agents/audit/initial-audit.md`、`model-structure-audit.md`、`external-research.md`
- 当前 Goal 与循环：由 `agentctl doctor` 从机器可读状态发现
- 历史证据：loop-001 至 loop-007，只读
- 控制器：`.agents/agentctl.py` 与 `.agents/schemas/`

## Skill 路由

- Goal 创建、轮换、关闭：`bdc2026-goal`
- 数据、口径、代码和模型审计：`bdc2026-audit`
- 代码实现与测试：`bdc2026-develop`
- 模型实验与 Pareto 决策：`bdc2026-experiment`
- 独立性声明、审查与反思：`bdc2026-review`
- Docker、bundle 与远端提交：`bdc2026-release`

## 不变量

- 所有路径必须仓库相对；运行时使用当前 Python 或显式环境变量覆盖。
- 不读取 2026-06-29 及之后行情做训练、验证或选型。
- 绝对 Top5 组合收益是主指标；excess 与 RankIC 是诊断项。
- 执行者与审查者角色分开；同一 Agent 顺序执行时标记“非独立复核”。
- achieved Goal 不回写，只能由新 Goal 通过 `supersedes` 纠偏。
- 原始 CSV、模型、output、temp 和 dist 不提交 Git。
- 每轮实验单独提交；失败证据不覆盖生产配置。
- 所有提交使用用户规定的完整 Agent commit 模板。
