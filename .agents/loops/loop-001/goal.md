#goal 在冻结数据和比赛口径下产生可复现、可离线提交且优于复现基线的 BDC2026 候选

## 最终状态

当前状态：`achieved`。全部 Done 条件已由 loop-001 至 loop-007 的滚动实验、停止证据和发布证据满足。

## Done 条件

- 比赛约束全部映射到实现或验证证据。
- 数据来源、成分股、日期范围和哈希被冻结，训练/预测不联网。
- 正式标签严格使用 T+1 开盘至 T+5 开盘。
- 候选在相同滚动切分上形成相对复现基线的 Pareto 改善。
- 两次完整运行结果一致，训练小于 8 小时、预测小于 5 分钟、提交小于 10GB。
- Docker 离线运行生成有效 `result.csv`。

## 验证方式

- `powershell -File .agents/scripts/verify.ps1`，timeout 30 分钟；所有静态、单元、验收和结构检查通过。
- `powershell -File .agents/scripts/run_experiment.ps1 -Mode baseline`，timeout 8 小时 30 分钟；生成完整 manifest、日志和指标。
- `powershell -File .agents/scripts/verify_release.ps1`，timeout 9 小时；离线训练/预测、双跑哈希、时限和体积门禁全部通过。
- 长任务每 10 分钟观察一次日志；无 GPU 利用率、磁盘增长异常或 20 分钟无进度时终止并审计。

## 约束条件

- 使用调用方已激活且满足锁文件的 Python 3.10 环境，不由项目脚本创建新环境。
- 只在数据准备阶段联网；不得上传源码、数据、日志或 secret。
- 每轮实验独立落盘并提交，失败实验不得覆盖生产配置。
- 原始数据和大模型不进入 Git。

## 非目标

- 不安装全局 skill。
- 不创建 `main` 或其他远端分支。
- 不使用未报备外部数据或预训练模型。
- 不以 leaderboard 未知结果替代本地可复现证据。

## 阻塞条件

- 公共数据源无法提供冻结日期范围或成分股快照。
- 依赖无法在 Python 3.10 和目标 CUDA/Windows 上安装。
- Docker 或目标规格硬件不可用，导致最终门禁无法执行。

## 迭代优先级

1. 口径与可运行性。
2. 基线复现。
3. 结构化方向。
4. 有限 HPO。
5. 离线发布验证。

## 最终审计

- 基线与正式候选：validation Top5 excess `0.027972 -> 0.034520`；正式结构为 XGBoost `rank:pairwise` + direct Top5 等权。
- 数据：250,233 行、300 股，SHA-256 `04d7e4da6a9d4915c61d635d1ab24d29e92c00363f4250c01dade8f35f6ccd87`。
- 宿主：训练最大 `72.662s`、预测最大 `11.530s`，model/result 双跑哈希一致。
- Docker：`--network none`，训练 `41.920s`、预测 `4.060s`、镜像 `584324299` bytes，容器内双跑一致。
- 失败方向：Optuna、窗口/因子、identity 消融、Top5 NDCG、LightGBM、CatBoost 均已保留证据；连续无完整 Pareto 改善触发停止规则。
- 远端：私有 `KRPCT/BDC2026`，默认且唯一分支为 `dev`。
- 未满足项：无。已观察 holdout 不再作为独立门禁，跨平台结果不宣称字节一致。
