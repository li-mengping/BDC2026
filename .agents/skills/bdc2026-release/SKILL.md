---
name: bdc2026-release
description: 验证 BDC2026 候选的离线训练预测、确定性、CSV、Docker、时限、体积、数据和模型哈希。准备提交冻结或发布审计时使用。
---

# BDC2026 Release

检查当前 Goal、manifest、模型元数据和提交说明，再在断网容器中从训练开始双跑。校验 UTF-8 `stock_id,weight`、股票数、非负权重、权重和、时间和镜像体积。

缺失任一实际证据时只能报告 `candidate`、`blocked` 或 `not completed`，不得写 release-ready。发布检查不修改模型或训练配置；所有路径使用仓库相对路径。
