#goal 推广 cross-fitted 无股票身份候选并完成正式离线双跑

当前状态：`unmet`。正式最新 validation 和已观察 holdout 显著回退，候选已拒绝并恢复生产配置；不执行无意义的双跑推广。

## 最终状态（定义 Done）

1. 正式配置排除 `instrument`，固定 127 boosting rounds。
2. metadata 记录 38 个特征、无 `instrument`、`best_iteration=126` 和 cross-fitted 选择规则。
3. 正式训练/预测成功，结果 CSV 合规；holdout 只报告不选型。
4. 两次离线运行的模型、metadata 和结果哈希一致，时限合规。
5. loop 状态、证据和 handoff 可恢复。

## 验证方式

| 条件 | 命令或证据 | 通过标准 |
| --- | --- | --- |
| 1-3 | `python -m code.models.xgboost.train` 与 `predict`，timeout 30 分钟 | metadata 与 CSV 满足上述契约 |
| 4 | `python .agents/scripts/release_verify.py`，timeout 30 分钟 | 所有本机离线 gates 为 true |
| 5 | `python .agents/scripts/validate_loop.py` | 三个 loop 均可解析 |

## 约束条件

- 不使用 holdout 选择特征、round 或参数。
- 不改变标签、窗口、树参数和 Top5 组合策略。
- 失败时恢复生产配置必须通过新提交，不改写实验证据。

## 明确非目标

- 本 loop 不比较新目标函数或模型族。
- 不把已观察 holdout 恢复为独立门禁。
- 不把宿主机验证替代 Docker 验证；Docker 属后续发布 loop。

## 阻塞条件

- 新 artifact 无法确定性复现。
- 固定 127 rounds 与 metadata/predict 语义不一致。
- 正式 validation 出现无法解释的数据或指标回归。

## 最终审计

- [x] 候选配置、metadata、artifact 一致
- [ ] 候选双跑哈希一致（因核心效果门禁先失败而停止）
- [x] holdout 未参与选择，仅用于选择后回归审计
- [x] 状态与证据更新，生产配置已恢复

## 迭代优先级

1. 配置与 artifact 一致性。
2. 正式训练预测。
3. 离线双跑。
4. 轮换到目标函数结构实验。
