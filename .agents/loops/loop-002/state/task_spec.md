# 任务规格

使用 leave-one-fold-out checkpoint selection 复核删除 `instrument` 的结构候选。选择 round 时只允许读取另外三个 rolling fold 的逐轮 Top5 excess；outer fold 只用于一次评估。

本地证据：`../loop-001/evidence/stock_identity_ablation.json`、`data/manifest.json`。

验收：候选 cross-fitted mean/worst 不下降、std 不上升且至少一项严格改善。停止：不构成 Pareto 即拒绝并转向 Top5 objective。
