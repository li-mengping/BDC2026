# Handoff

读取 `goal.md`、loop-003 的拒绝证据和本 loop evidence。不得修改正式 YAML，除非 NDCG 同时通过 cross-fitted 与最新 validation 门禁。

NDCG 已拒绝，不允许对该目标继续 HPO。下一 loop 比较一个 LightGBM LambdaRank 模型族候选，必须复用相同 folds、原始收益评估和 cross-fitted checkpoint。
