# Handoff

先读 `goal.md` 与 `../loop-002/evidence/crossfit_checkpoint.json`。生产配置只能使用无 `instrument` 和 127 rounds；不得回到 validation 峰值选择。

推广已拒绝：较早 rolling crossfit 虽改善，但最新 8 周 validation 和随后报告的 holdout 均明显回退。生产 YAML 已恢复原始 39 特征与 validation checkpoint 规则。下一 loop 只比较一个 Top5 对齐 objective，不延伸 identity/round 方向。
