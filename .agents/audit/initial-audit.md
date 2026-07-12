# BDC2026 深度初始审计与纠偏记录

审计版本：V2（2026-07-12）

审计范围：比赛公式、原始实现、当前生产路径、模型结构和可复现实验。

证据等级：`FACT` 为仓库或比赛材料可直接复核，`INFERENCE` 为基于事实的解释，`TODO` 必须通过新实验或官方复核后才能升级。

## 1. 结论先行

当前生产路径是“冻结行情 -> 周样本 -> 39 个日线派生特征 -> XGBoost `rank:pairwise` -> direct Top5 等权”。它已经能离线双跑，但过去的“已闭环”结论不能覆盖以下事实：

1. 原始方案的官方 Transformer baseline 没有在同一数据、同一折和同一绝对收益评分器下复现，因此 XGBoost 的提升不能称为相对官方 baseline 的提升。
2. 训练选择长期使用 Top5 **excess**，比赛实际评分是权重加权的绝对收益；二者可能给出相反的候选。
3. holdout 曾被读入训练/报告与组合决策链；即使后续脚本改为 validation-only，已经观察过的 holdout 不能再次作为独立门禁。
4. checkpoint、模型族、股票身份和候选池各有独立实现，产生重复 ranking spine；跨模型结果不能直接归因于归纳偏置。
5. 固定股票池回看历史存在 survivorship bias；`instrument` 作为整数连续特征还可能让树学习身份记忆。
6. 发布说明、Python 版本和实际 Docker 基础镜像曾不一致，且最初缺少根级 `readme.md`，这会让“能跑”与“可交付”脱钩。

因此，本仓库的可信结论应表述为：在声明的滚动折、特征空间和预算内，XGBoost pairwise + direct Top5 是当前已验证的候选；不是数学全局最优，也不是官方 baseline 的已证提升。

## 2. 比赛真实口径（必须与实现分层）

### 2.1 三层事实

| 层 | 事实 | 允许的实现责任 |
| --- | --- | --- |
| 比赛公式 | 对目标周的 5 个评估交易日，单票为第 1 日开盘买入至第 5 日开盘卖出 | 标签函数只实现 `(open_5 - open_1) / open_1` |
| 官方评分器行为 | 每只股票取最后 5 行，使用第 1 行与第 5 行的开盘价；组合为权重加权和 | 评分等价测试必须固定行序、重复行和边界错误 |
| 训练样本政策 | 默认只保留周一至周五五个交易日齐全的自然周 | 样本过滤可以排除节假日周，但不能改写比赛公式 |

“周一开盘至周五开盘”只是在目标周恰好是周一至周五时的日历化称呼；通用契约必须写成 `T+1 open -> T+5 open`。周五收盘价不是标签端点，收盘扰动不得改变标签。

### 2.2 数据截止与泄漏

训练/验证的特征时点必须早于目标周第一个交易日；冻结数据的最后日期为 2026-06-26。任何包含 2026-06-29 及之后行情的训练、验证或特征生成都必须失败，而不是静默截断。输入窗口跨停牌周、重复主键、非六位股票代码和缺失开盘价均属于数据错误。

## 3. 原始实现的缺陷链

以下是第一次审计时识别的根因，而非仅列出后来修复的文件：

| 根因 | 可见症状 | 影响 | 修复/仍存状态 |
| --- | --- | --- | --- |
| 入口与依赖未形成契约 | 脚本、模型模块和说明文档分别演化，无法从根目录确认生产入口 | 新 Agent 可能运行错误模块或错误环境 | 入口测试与离线双跑已补；需继续由结构测试守住 |
| 标签语义依赖“自然周名称” | 文档曾写周一到周五收盘/开盘混用，未绑定评分器的第 1/第 5 行 | 训练目标与比赛分数错位 | 统一 `t1_open_to_t5_open`，增加评分等价测试 |
| 在线下载与训练耦合 | 下载脚本动态成分股、依赖联网和平台日期格式 | 无法复现，且有赛后数据泄漏路径 | 数据清单冻结；训练/预测断网失败 |
| 评估目标漂移 | 训练优化 Top5 excess，提交评价绝对收益；组合扫描另有 Top10+均值方差 | 选择的模型不一定提高真实分数 | 绝对收益升为主指标，excess 仅诊断；direct Top5 已验证 |
| holdout 被当作选择信号 | 训练对象同时构造 validation 与 holdout 并在实验链报告选择 | holdout 不再是独立测试 | 历史结果保留但 supersede；新 loop 禁止读取选型 |
| checkpoint 逐折峰值选择 | 每折在同一小 validation 上扫描 100+ rounds | 多重比较过拟合、方差被低估 | 已做 cross-fit 证据；生产仍需明确统一 selector |
| 结构重复 | XGBoost 自有 metrics/checkpoint，LightGBM/CatBoost 另有 `ranking_common.py` | 模型差异与数据管线差异混淆 | V2 共享 spine 已落地；通用数据对象已接入，模型 adapter 迁移仍未完成 |
| 股票身份混用 | `instrument` 为整数，既像类别又按连续数值进入树 | 可能记忆股票历史均值，面对成分变化脆弱 | 消融有回退证据；暂不删除，需先定义类别/embedding 契约 |
| survivorship bias | 使用当前 300 股集合回看历史 | 历史样本缺少退市/调入股票 | 作为已知限制；除非取得报备的历史成分快照，不伪装为无偏 |
| 发布契约漂移 | 说明曾写 Python 3.12，Docker 已验证 Python 3.10；根目录缺 `readme.md` | 交付方无法复现已验证环境 | 固定 Python 3.10 digest，新增根 readme 与 bundle 门禁 |
| Python 包名碰撞 | 顶层业务包名为 `code`，会遮蔽标准库同名模块 | 在不受支持的 Python 3.13 下 pytest debugger 导入失败 | 项目明确支持 3.10-3.12；后续若迁移包名必须作为独立兼容性重构，不能在模型实验中顺手改 |

## 4. 当前模型与效果证据

- 模型：XGBoost `rank:pairwise`，输入窗口 12 周，手工日线/横截面特征；输出按模型分数直接取 5 只等权。
- validation Top5 excess 从 `0.027972` 提升到 `0.034520` 的证据只说明相对于仓库旧组合策略的滚动改善。
- 已观察 holdout `0.052557` 不能再作为独立选择门禁；所有后续候选只能使用冻结 validation/OOF 证据。
- NDCG、LightGBM LambdaRank、CatBoost 和受限 Optuna 均未形成同时改善均值、最差折、稳定性和 RankIC 的 Pareto 结果，故没有进入生产。
- 宿主与 Docker 均完成断网双跑，平台内模型/metadata/result 哈希一致；Windows 与 Linux 数值序列不宣称字节哈希一致。

### 4.1 V2 nested walk-forward 基线

`loop-008` 的 E1 使用 4 个 outer folds、每折 4 个评估周；checkpoint 只在该折 outer-train 尾部 4 周选择，再用完整 outer-train 重训。当前 raw ordered identity XGBoost 在 16 个 outer 周上的绝对 Top5 均值为 `0.024884`，最差周 `-0.021126`，标准差 `0.033709`，正收益率 `0.6875`，RankIC `0.093640`。

该结果是当前更可信的结构比较基线，但仍是 expanding walk-forward，不是独立 holdout：较早 outer 周可进入后续折训练历史。`identity_disabled` 提高最差周、正收益率和 RankIC，却把均值降至 `0.020973`；rank、robust z-score 和 categorical identity 也没有形成完整 Pareto 改善，因此 E1 全部保持 hold。XGBoost 峰值显存仍缺失，不能据此宣称完整资源 Pareto 或发布就绪。

## 5. 结构突破口（优先级而非承诺）

1. **官方 baseline 重建（P0）**：按公开训练/评分契约重新实现 Transformer adapter，不复制无 LICENSE 源码；在同一冻结数据、fold、绝对收益和 direct Top5 下得到可比较的 E0。
2. **统一数据与评估 spine（P0）**：`MarketPanel -> EvaluationWindow -> FoldSpec -> RankDataset -> RankerAdapter -> MetricReport -> Portfolio -> ModelArtifact`，模型只提供 adapter。
3. **因果时序编码（P1）**：先做低成本 temporal-convolution/GRU smoke，验证原始序列是否包含 39 个特征遗漏的信息。
4. **置换等变横截面交互（P1）**：单股编码后使用 cross-stock attention 或 DeepSets，禁止股票顺序成为信号。
5. **目标对齐（P1）**：比较 pairwise、ListMLE/LambdaLoss、Top5-weighted objective；评价始终使用原始绝对组合收益。
6. **残差互补集成（P2）**：仅在 OOF 误差相关性足够低时做 rank average；模型数量本身不是结构改进。

所有方向先单折低轮次 smoke，后完整滚动；连续两个结构方向和一次受限 HPO 没有完整 Pareto 改善即停止，结论限定为“在已声明搜索空间与预算内停止”。

## 6. 审计遗留项

- `TODO-E0`：建立官方 Transformer、XGBoost、朴素等权/动量三个 baseline 的同口径表。
- `TODO-DATA`：获得并报备历史沪深 300 成分股快照后，重估 survivorship 影响。
- `TODO-SPINE`：共享对象已建立；迁移各模型到统一 `RankerAdapter` 前，不得把跨族比较写成完全公平实验。
- `TODO-METRIC`：把绝对收益、最差折、波动、正收益率、RankIC、时延、显存和体积写入同一 MetricReport。

## 7. 外部研究证据登记规则

本审计不把搜索摘要或无许可证代码当作事实。外部研究必须在 `.agents/audit/external-research.md` 登记来源 URL、访问日期、上游版本/提交、适用性、许可证和未采用原因；研究资料不得包含私有源码、日志或凭据。当前登记文件明确“尚无被训练直接使用的外部研究”。
