# Handoff

先读 `goal.md`、`state/progress.json`、`state/direction_proposals.json` 和 `../loop-001/evidence/stock_identity_ablation.json`。不得把 loop-001 的逐折峰值指标当作本 loop 的最终结果。

本 loop 已 achieved。下一 loop 应先推广 `exclude_features=[instrument]` 与 cross-validation 固定 127 rounds，完成正式训练双跑；随后比较一个 Top5 对齐的 ranking objective。不得重新用 validation 峰值替代 127 rounds。
