# 比赛要求追踪矩阵 V2

状态定义：`已验证` 必须有可重复命令/证据；`已实现待复验` 表示代码已存在但 V2 门禁尚未完整执行；`待实现` 不得用于完成声明；`历史证据` 只描述旧 loop，不能替代 V2 验收。

| 要求 | V2 状态 | 实施事实源 | 必需验收 |
| --- | --- | --- | --- |
| 第 1/第 5 评估交易日开盘标签 | 已实现待复验 | `code/utils/stock.py`、`code/config.py` | 与官方评分器等价；收盘扰动不变；乱序/少于 5 行失败 |
| 完整周政策与标签分离 | 待实现 | `sample_calendar_policy=full_monday_friday` | 节假日周被排除但公式测试不变 |
| 目标期数据截止 | 待实现 | 数据加载与特征边界 | 2026-06-29 及之后任何训练/验证/特征输入失败 |
| 停牌/缺周不拼接 | 待实现 | `MarketPanel`/窗口生成 | 窗口连续性负路径测试 |
| 官方 Transformer baseline | 待实现 | 独立 adapter 与来源登记 | 不复制无 LICENSE 源码；同折绝对收益 E0 |
| 当前 XGBoost champion | 历史证据 | XGBoost 配置与 loop-001 至 loop-007 | V2 同折绝对收益重跑 |
| 朴素等权/动量 baseline | 待实现 | E0 baseline adapter | 同数据同折 MetricReport |
| 绝对组合收益为主指标 | 待实现 | 共享 MetricReport/选择器 | 候选决策不再由 excess 单独决定 |
| holdout 不参与选择 | 已实现待复验 | validation-only 选择逻辑 | 依赖/访问审计；已观察 holdout 只报告 |
| 共享 ranking spine | 待实现 | `MarketPanel -> ... -> ModelArtifact` | 各模型只实现 adapter；结构/import 测试 |
| direct Top5 快路径 | 已实现待复验 | `code/portfolio/postprocess.py` | 不执行 covariance/均值方差；输出严格 Top5 |
| 固定种子与平台内确定性 | 历史证据 | 配置、训练入口、发布验证 | V2 宿主和 Docker 双跑 hash |
| 离线训练/预测 | 历史证据 | 离线 guard、compose | 从导出 tar load 后 `--network none` 双跑 |
| 训练小于 8 小时 | 历史证据 | Docker/宿主发布 evidence | V2 最大值小于 28800 秒 |
| 预测小于 5 分钟 | 历史证据 | Docker/宿主发布 evidence | V2 最大值小于 300 秒 |
| 提交小于 10GB | 已实现待复验 | `package.sh`、bundle | bundle 与镜像均小于 10GB |
| 结果 schema 与权重 | 已验证 | `code/utils/submission.py` | UTF-8、最多 5 股、非负、总和不超过 1 |
| 数据来源与哈希 | 已验证 | `data/manifest.json` | CSV、成分股、来源、日期、MD5/SHA-256 一致 |
| 必需目录、脚本、readme | 已实现待复验 | 根目录、`tests/test_release_structure.py` | 结构测试与 bundle 内容审计 |
| Python 3.10 固定 digest | 已实现待复验 | `Dockerfile`、`package.sh` | 不含 3.12 或镜像 fallback；实际 image inspect |
| 完整离线 bundle | 已实现待复验 | `package.sh`、被忽略的 `dist/` | tar、compose、data/manifest/constituents、readme、空目录、SHA256SUMS |
| 外部资料合规 | 已实现待复验 | `.agents/audit/external-research.md` | 来源、日期、版本、许可证、适用性、决定齐全 |
| 私有远端仅 `dev` | 历史证据 | 远端发布 evidence | visibility=PRIVATE、default=dev、唯一 head=dev |

## 纠偏说明

- `validation 0.027972 -> 0.034520` 是旧组合到 direct Top5 的 **excess** 历史证据，不等于比赛绝对收益提升，也不等于超过官方 baseline。
- 已观察 holdout `0.052557` 不再是独立验收门禁。
- loop-001 至 loop-007 保留为历史证据；V2 loop 必须显式 supersede 旧“彻底闭环”结论后重新验收。
