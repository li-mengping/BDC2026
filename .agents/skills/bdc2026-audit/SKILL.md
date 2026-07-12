---
name: bdc2026-audit
description: 审计 BDC2026 比赛约束、沪深行情、收益标签、验证切分、代码入口和复现证据。接管仓库、修改数据口径或开始模型实验前使用。
---

# BDC2026 Audit

读取 `.agents/constraints/`、当前 v2 Goal、loop 状态和数据 manifest；将事实、假设、证据缺口分开落盘。

检查比赛公式（第一个评估日开盘到第五个评估日开盘）、完整周样本政策、特征时点、重复键、停牌缺口、哈希和输出格式。训练与预测不可联网。

禁止训练、改写生产配置、把未运行结论写成 validated 或使用绝对本机路径。检查细节见 `references/audit-gates.md`。
