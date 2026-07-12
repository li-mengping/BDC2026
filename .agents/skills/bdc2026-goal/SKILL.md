---
name: bdc2026-goal
description: 管理 BDC2026 仓库内可恢复、可验证的 Goal 状态与证据链。用户要求创建、轮换、审计、关闭或恢复模型优化目标时使用。
---

# BDC2026 Goal

Goal 是仓库内的机器可读契约，不是聊天摘要。优先准备仓库相对 JSON 合同并运行 `python .agents/agentctl.py goal create --contract <path>`。合同必须包含 objective、success_criteria、verification（命令和通过标准）、constraints、non_goals、blockers、iteration_priority 和 final_audit。

## 状态规则

允许 `planned -> active -> evidence_ready -> validated -> achieved`，也允许 `blocked`、`rejected`。已 `achieved` 的 Goal 不回写，只能由 `goal supersede` 产生新 Goal。任何完成状态都必须有 evidence；不要用未来计划冒充证据。

## 工作流

1. 运行 `doctor`，按输出的 `active_goal` 和 `current_loop` 读取 Goal、约束、handoff 和最近反思。
2. 明确 Done、验证命令、非目标、阻塞条件和预算。
3. 每次迁移运行 `goal audit`，失败时保留状态并报告缺口。
4. 在新 loop 中记录实验和反思，结束时只引用已落盘证据。

## 负路径

禁止修改旧 loop 结论、把单折结果称为完成、跳过状态迁移、使用绝对路径或引入本机专用规则。跨角色复核必须在 handoff 中标注真实执行者。
