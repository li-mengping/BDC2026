#goal 冻结最优候选并通过宿主机、Docker 和远端仅 dev 发布门禁

当前状态：`achieved`。宿主机、Docker 与远端仅 `dev` 发布门禁均已通过。

## 最终状态（定义 Done）

1. 正式生产为 XGBoost pairwise + direct Top5，候选 artifact 已清理。
2. 宿主机断网双跑的模型、metadata 和 result 哈希一致。
3. Docker 使用 Python 3.10、精确生产依赖，在 `--network none` 下通过编译/导入/数据/训练/预测审计。
4. Docker 训练小于 8 小时、预测小于 5 分钟、镜像小于 10GB，结果 CSV 合规。
5. 私有 `KRPCT/BDC2026` 只有 `dev` 分支且默认分支为 `dev`。
6. 约束矩阵、总 goal、审计、状态和 handoff 与真实证据一致。

## 验证方式

| 条件 | 命令或证据 | 通过标准 |
| --- | --- | --- |
| 1 | `powershell -File .agents/scripts/cleanup_artifacts.ps1` | 仅 metadata 引用 artifact 保留 |
| 2 | `python .agents/scripts/release_verify.py`，timeout 30 分钟 | 所有宿主 gates 为 true |
| 3-4 | `python .agents/scripts/docker_release_verify.py`，timeout 40 分钟 | 容器代码审计、离线运行、时限和体积全部通过 |
| 5 | `gh repo view`、`git ls-remote --heads` | 私有、default=`dev`、唯一 head=`dev` |
| 6 | `powershell -File .agents/scripts/verify.ps1` | 全测试、所有 loop 和 skill 校验通过 |

长命令每 5 分钟检查输出；Docker 构建 15 分钟无层进展或训练 20 分钟无 GPU/日志进展时终止并审计残留容器。

## 约束条件

- 不新增模型、HPO、数据或全局 skill。
- Docker build 可联网解析已锁定工件，容器训练/预测必须无网络。
- 原始 CSV、模型和输出不提交 Git。
- 远端不得创建 `main` 或其他分支。

## 明确非目标

- 不依据已观察 holdout 继续选择候选。
- 不把宿主机工作区体积替代镜像体积。
- 不把镜像构建成功替代容器内代码和内容审计。

## 阻塞条件

- Docker GPU runtime 不可用。
- GitHub 组织权限不足或仓库已存在且状态冲突。
- 容器离线运行、确定性或时限门禁失败。

## 最终审计

- [x] 清理与宿主双跑通过
- [x] Docker code audit 与 runtime audit 通过
- [x] 镜像/时限/结果门禁通过
- [x] 远端仅 dev 通过
- [x] 所有文档与状态回写
- [x] 所有 Done 条件满足，状态可写为 achieved

## 迭代优先级

1. 依赖与 artifact 冻结。
2. 宿主双跑。
3. Docker 内审计。
4. 远端发布。
5. 最终 goal 审计。
