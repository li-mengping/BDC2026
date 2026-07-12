---
name: bdc2026-develop
description: 驱动 BDC2026 的审计、实现、实验、测试和交接闭环。需要由 Agent 协作开发或修复模型代码时使用。
---

# BDC2026 Develop

按“事实 -> 假设 -> 最小改动 -> 验证 -> 交接”推进，唯一状态入口是 `.agents/agentctl.py`。

## 执行顺序

1. `doctor` 检查环境、技能和相对路径约束。
2. 由 auditor 输出缺陷和证据缺口；由 model-architect 提出一个结构假设。
3. implementer 只实现已批准方向；experiment-runner 注册、执行并记录一次实验。
4. skeptic-reviewer 独立检查可比性、泄漏、Pareto 和失败证据。
5. 通过验证后交给 release-controller，提交前运行 `commit check`。

## 交接契约

交接必须包含 inputs、outputs、verification、limitations、next_action、证据路径和执行者。单 Agent 扮演多角色时不得声称独立审查。

## 负路径

禁止无边界 HPO、覆盖生产配置、静默重试、联网训练、绝对路径、本机用户名和未记录的模型效果。失败实验只更新状态和 evidence。
