# 实验契约

每个实验记录：ID、父基线、数据哈希、代码 commit、假设、参数、四折指标、最差折、标准差、正收益折比例、训练时间、预测时间、模型大小、状态和停止原因。

主 Pareto 维度：最大化 Top5 超额收益均值与最差折，最小化标准差、训练时间和模型体积。只有相同数据、标签和切分的结果可以直接比较。

holdout 不属于选型指标。任何已参与过滤、排序、早停或人工选择的 holdout 都必须标记为“已观察辅助证据”，不得恢复为独立门禁。

方向 proposal 必须包含 `hypothesis`、`local_evidence_needed`、`first_action`、`acceptance_gate`、`stop_rule`、`risk`、`anti_pattern_avoided` 和 `status`。
