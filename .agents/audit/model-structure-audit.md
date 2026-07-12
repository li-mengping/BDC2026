# BDC2026 模型结构审计 V2

## 1. 审计问题与判定边界

本文件回答“模型是否只是在调参”和“哪里可能有结构性突破”，不把一次高分当作结论。每个方向必须同时说明：输入表示、归纳偏置、训练目标、选择器、组合映射、成本和失败语义。所有比较使用相同冻结数据、相同滚动折、相同 target cutoff 和相同绝对收益评分器；excess、RankIC 和 Top5 precision 仅作诊断。

## 2. 原官方 Transformer 的结构复原要求

官方方案（以比赛材料可复核的模型/评分描述为准）应被视为待重建的 baseline，而不是可直接复制的实现：

1. 日线面板按股票和时间形成固定历史窗口；时间编码器提取单股时序表示。
2. 股票表示在横截面上交互或聚合，输出每只股票的排序分数。
3. 训练目标是未来周标签，推理只使用目标周开始前可见的窗口。
4. 组合层按分数选不超过 5 只股票并校验权重。

仓库不得复制无 LICENSE 的上游源码。E0 应按公开契约重新实现最小 adapter，记录上游 URL、版本/commit、参数差异和不可复原部分；只有在评分器等价、时间切分等价、输入字段等价后，才能把该模型称为“官方 baseline”。

## 3. 当前 XGBoost 结构图与缺陷

```text
冻结日线 -> 39 个手工/横截面特征 -> 按目标周分组
         -> XGBoost rank:pairwise -> validation checkpoint
         -> direct Top5 -> 等权 result.csv
```

### 3.1 结构性问题

| 位置 | 问题 | 为什么不是单纯参数问题 | 下一步 |
| --- | --- | --- | --- |
| 表示 | 绝对价格、成交量和横截面派生量混用 | 量纲和阶段阈值可能改变排序规则 | robust rank/z-score 与因果归一化 E1 |
| 身份 | `instrument` 是整数连续值 | 树会把编号大小解释成有序关系，也可能记忆身份 | 明确禁用/类别/embedding 三种契约再消融 |
| 时间 | 只将窗口压成表格特征 | 无法表达跨日模式、波动簇和事件顺序 | causal TCN/GRU E2 |
| 横截面 | 每只股票独立得分，缺少同周交互 | 无法直接表达拥挤、相对强弱和行业替代 | permutation-equivariant attention E3 |
| 目标 | pairwise 覆盖整个横截面 | 业务只支付 Top5 的排序质量 | ListMLE/LambdaLoss/Top5 objective E4 |
| 选择 | 每折在同一 validation 找峰值 | 选择器本身成为高方差模型 | cross-fitted/统一 checkpoint |
| 组合 | 历史 Top10+均值方差路径多余且易失败 | 后处理目标与模型排序不一致 | direct Top5 快路径 |

### 3.2 当前实现的非结构风险

- XGBoost、LightGBM、CatBoost 各自复制数据、metrics、checkpoint 和 artifact 逻辑，跨族实验不满足单一控制变量。
- 训练元数据同时保存 holdout 分数，旧脚本存在读入后用于组合决策的路径；新的实验不得再加载已观察 holdout。
- 生产 artifact 由配置名和字符串路径拼接，若配置与 metadata 漂移，预测可能加载错误轮次；必须由 `ModelArtifact` 契约校验。
- 训练/预测脚本和发布说明曾使用不同 Python 版本；容器 digest、锁文件和说明必须一起审计。

## 4. 可证伪结构方向

### E0：同口径 baseline

实现官方 Transformer adapter、当前 XGBoost champion、朴素等权/动量策略。输出每周绝对组合收益、最差折、波动、正收益率、RankIC、时间、显存和体积。若官方 baseline 无法按公开契约复现，记录失败原因，不用 XGBoost 的 excess 分数替代。

### E1：表示与身份

在同一 ranker 上比较横截面 robust rank、z-score、量纲归一化以及禁用/类别/embedding 的 `instrument`。一次只改变一个表示因素；任何候选若只改善单周或只改善 excess，拒绝。

当前状态：XGBoost 的 ordered identity、disabled、date-local rank、median/MAD robust z-score、native categorical 已完成 4 折 nested walk-forward。没有变体保持主收益均值同时改善全部门禁，故不晋升；embedding 只属于神经结构，尚未形成与这五个树模型变体同口径的完整证据。

### E2：因果时序编码器

使用小型 causal TCN 或 GRU，输入原始窗口和 mask，禁止未来 padding 泄漏。先固定参数做 smoke，再与 XGBoost 同折比较；若耗时/显存不在预算内，方向直接停止。

### E3：置换等变横截面交互

单股时间编码后，对同一周股票集合执行 DeepSets 或 cross-stock attention。训练时随机置换股票顺序，输出应满足同样置换。至少提供 permutation test 作为结构 guard；否则不能称为横截面模型。

### E4：目标对齐

只在最佳表示/结构上比较 pairwise、ListMLE/LambdaLoss 和 Top5-weighted objective。标签仍为原始收益，评价仍为绝对组合收益；relevance 转换必须可逆记录，不能以 NDCG 分数代替比赛收益。

### E5：OOF 残差互补

先计算 OOF 预测 rank 与收益残差相关性，再决定是否 rank ensemble。相关性高或单模型没有 Pareto 优势时不集成；“模型更多”不是通过条件。

### E6：受限 HPO

只对 E0-E5 中完整 Pareto 胜出者运行最多 16 个 Optuna trial，基线入队，完成两个 fold 后才允许 pruning。搜索空间必须在 `direction_proposals.json` 中登记；不得用 FLAML/NAS 扩张搜索面来掩盖结构证据不足。

## 5. 统一模型 spine 契约

```text
MarketPanel -> EvaluationWindow -> FoldSpec -> RankDataset
            -> RankerAdapter -> MetricReport -> Portfolio -> ModelArtifact
```

- `MarketPanel` 唯一拥有数据清单、时点和成分股快照。
- `EvaluationWindow` 唯一拥有 T+1/T+5 标签和完整周政策。
- `FoldSpec` 唯一拥有 rolling/holdout 划分，禁止模型自行切分。
- `RankDataset` 唯一拥有分组、mask、feature schema 和 label。
- `RankerAdapter` 只负责 fit/predict/save/load；不读取 holdout、不写组合文件。
- `MetricReport` 统一绝对收益主指标与诊断指标。
- `Portfolio` 只负责 Top5、权重和 CSV 校验。
- `ModelArtifact` 记录配置哈希、数据清单哈希、fold、checkpoint 和版本。

## 6. Pareto 门禁与停止规则

候选必须满足：绝对收益均值不降、最差折不降、波动不升，且至少一个严格改善；同时训练/预测时限、峰值显存、模型体积和 permutation/离线 guard 通过。连续两个结构方向及一次受限 HPO 无完整 Pareto 改善时停止，并标注搜索空间、预算和未覆盖结构，不能宣称全局最优。

## 7. 证据状态机

每个方向依次为 `proposed -> smoke -> rolling -> reviewed -> promoted|rejected`。失败方向保留输入配置、日志、指标、失败类别和反思；生产配置只有 `promoted` 才能改变。单 Agent 顺序扮演执行者和审查者时，记录为“非独立复核”，不得声称双人审查。
