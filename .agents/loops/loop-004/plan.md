# Loop 004 计划

1. 构造 pairwise raw-return 与 NDCG relevance 两个可比训练矩阵。
2. 保存逐轮原始收益指标。
3. leave-one-fold-out 选择 checkpoint 并执行 Pareto 门禁。
4. 更新状态；成功才进入正式推广，失败则轮换模型族方向。
