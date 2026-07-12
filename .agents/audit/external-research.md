# 外部研究证据登记

本文件是外部资料的唯一登记入口，不是训练数据或模型事实源。检索使用 AnySearch 公共网页搜索与 GitHub 公共 API，查询不包含私有源码、日志、数据或本机路径。任何资料进入方向设计前必须登记来源、版本、许可证、适用性和拒绝边界。

## 已登记来源

| ID | 来源 | 访问日期 | 版本/许可证 | 适用性与决定 |
| --- | --- | --- | --- | --- |
| EXT-001 | [THU-BDC2026 官方基准](https://github.com/Sherlock1956/THU-BDC2026) | 2026-07-12 | `7fdf2dc3f9a61afe955ca52497a48f4c47542517`；未检测到许可证 | 核对评分器、标签和公开结构；不复制源码，E0 只做 clean-room 契约复现 |
| EXT-002 | [Qlib CSI300 benchmarks](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md) | 2026-07-12 | `d5379c520f66a39953bad76234a7019a72796fd0`；MIT | TCN、GRU、Transformer、Localformer 在 Alpha158 上没有普适赢家；据此先做低成本 TCN，再验证 attention，不引入 Qlib 运行时 |
| EXT-003 | [MASTER](https://github.com/SJTU-DMTai/MASTER) / [AAAI 2024 论文](https://ojs.aaai.org/index.php/AAAI/article/view/27767) | 2026-07-12 | `de8f58557096abde4216a701b35fc4368158d111`；MIT | 市场引导特征选择及交替时序/横截面聚合形成 E3 假设；其公开 validation processor 差异同时证明处理契约必须统一 |
| EXT-004 | [TRA, KDD 2021](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA) | 2026-07-12 | 随 Qlib commit；MIT | 多市场 regime 只作为切片失败后的后续假设，不预先扩张 mixture |
| EXT-005 | [ACT, 2026](https://arxiv.org/html/2604.20204) | 2026-07-12 | arXiv preprint；不适用作预训练来源 | temporal crosstalk 只作为后续研究假设；晚于比赛预训练截止，不使用代码、权重或外部产物 |
| EXT-101 | [Goal-Driven](https://github.com/MAX0MAX/goal-driven) | 2026-07-12 | `73697b7d4ce41feb2956b05e24fb73396c8e9bc2`；MIT | 采用 Goal contract、progress、fresh verifier 分层；runtime 记录不得增加验收规则 |
| EXT-102 | [Waymark](https://github.com/Y4tacker/Waymark) | 2026-07-12 | `8accf8a860b9ec6041cf4e34dabe8965d1fedb7c`；GitHub 未检测到许可证 | 采用 durable claim、evidence index、verification record 思想；不复制实现，不引入 SQLite 或常驻服务 |
| EXT-103 | [Agent Handoff](https://github.com/WeirdSky924/agent-handoff-skill) | 2026-07-12 | 公共仓库；仅参考文档协议 | handoff 只保存目标、决定、验证、风险和下一步，不保存聊天转录 |
| EXT-104 | [Agent Spine](https://github.com/netsky-lab/agent-spine) | 2026-07-12 | 搜索结果来源，仓库 API 复核失败 | 仅采用“Codex canonical + Claude thin adapter”的可证伪设计原则，不依赖其代码或声明 |
| EXT-105 | [GS-XGBoost CSI300 动态选股](https://doi.org/10.1145/3785706.3785711) | 2026-07-12 | 2025；论文 DOI；未采用代码/权重 | 支持多阶段滚动测试与组合回测作为外部假设；不把论文收益数字迁移为本项目证据 |
| EXT-106 | [A-share XGBoost forecasting](https://doi.org/10.1145/3724154.3724237) | 2026-07-12 | 2024；论文 DOI；未采用代码/权重 | 提示中期动量、盈利与技术/微观结构特征可进入候选特征审计；必须在冻结数据和同折协议下验证 |
| EXT-107 | [China stock yield forecasting](https://doi.org/10.1080/1540496X.2022.2148464) | 2026-07-12 | 2022；论文 DOI；未采用代码/权重 | 支持技术动量和波动因子作为低成本 E1 假设；不引用外部回测数字 |
| EXT-108 | [Cross-sectional systematic strategies by learning to rank](https://doi.org/10.3905/jfds.2021.1.060) | 2026-07-12 | 2021；论文 DOI；未采用代码/权重 | 支持比较 pointwise、pairwise、listwise 排序目标；最终门禁仍是绝对 Top5 收益 |
| EXT-109 | [XGBoost with Bayesian optimization for stock selection](https://doi.org/10.1109/icimcis68501.2025.11327136) | 2026-07-12 | 2025；论文 DOI；未采用代码/权重 | 只支持受限 Bayesian/HPO 的实验设计，不引入论文的市场、特征或组合口径 |
| EXT-110 | [LambdaRankIC](https://arxiv.org/html/2605.00501) | 2026-07-12 | 2026；arXiv preprint；晚于数据截止 | 记录为未来候选损失函数；不用于当前生产、预训练或效果宣称 |

## 采用规则

- 模型研究只产生 hypothesis，不把外部 benchmark 数字写成本项目效果证据。
- 主办方按绝对组合收益排名，absolute Top5 return 是主选型指标；excess、RankIC、precision 是诊断项。
- 不使用晚于规则截止日期的预训练模型或外部权重。
- 不引入后台服务、数据库或 Node 控制面；仓库控制器保持 Python 标准库、文件状态和相对路径。
- 每次实验在 loop evidence 中引用上述 ID、当前数据 hash、代码 commit 和实际配置。
- AnySearch 查询覆盖沪深 300/XGBoost、Qlib/Alpha158、cross-sectional learning-to-rank、purged walk-forward 与时序/横截面结构；搜索结果只作为 hypothesis，不替代官方评分器或本地运行证据。
