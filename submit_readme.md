## this is for sbumit, please read code/readme.md only

# 代码说明

## 环境配置

- Python: 3.12
- 依赖管理: uv
- 主要依赖: pandas 2.3.2, numpy 2.3.3, xgboost 3.3.0, ta-lib 0.6.8, tqdm 4.67.1
- 运行设备: XGBoost `device=cuda`（可用时）；运行时由锁定的项目依赖和容器镜像决定，不依赖本机环境名称
- 复现方式: `uv sync --frozen` 后运行根目录脚本

## 数据

- 使用数据文件: `data/stock_data.csv`
- 数据内容: 沪深 300 股票日频行情，字段包含股票代码、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅。
- 未使用额外公开数据、外部词典、embedding 或联网下载数据。
- 训练、验证和预测切分由 `code/utils/validation.py` 按完整自然交易周为单位生成。
- 单个训练和推理样本由模型配置中的 `input_window` 决定；当前 XGBoost 配置为 12 个历史周和 1 个未来标签周
- 标签严格使用目标周周一开盘买入、周五开盘卖出（T+1 open -> T+5 open），由 `code/utils/stock.py` 统一计算
- `data/manifest.json` 冻结行情与成分股哈希；训练和预测发现 manifest 缺失、篡改或无法匹配时直接失败，不联网补数

## 算法

### 特征 `code/features`
- baseline: `158+39`
- 尝试其他不同窗口长度的因子、横截面特征等，效果不是很好
- 后续可尝试深度学习方法提前特征

### 模型 `code/models/xgboost`
主要模型：xgboost ranking model
- LambdaRankIC 自定义目标损失的 XGBoost 模型。
- XGBoost `rank:pairwise` 排序模型。
rankic 选出的排名更为稳定
pairwise 选出的排名上限更高，测试平均值略高

### 后处理 `code/portfolio`
- 单个 pairwise 模型直接取 Top5，避免 Top10 候选池的均值方差筛选引入历史回撤。
- 最终五只股票采用等权；该策略在相同验证和 holdout 上均优于 Top10/15/20 候选池。

### 注意事项

- 输入窗口: 12 个完整交易周。
- validation window: 8 个目标周。
- 生产训练轮数: 200，按照验证集结果（best top5 excess return）选择最佳迭代次数；Optuna 候选仅在滚动前沿证据同时通过 holdout 门禁后晋升
- Notebook 实验必须使用 rolling kfold、每折使用验证周之前全部历史、训练轮数不少于 100。
- 固定随机种子: 42。

## 训练流程

运行:

```bash
bash train.sh
```

## 推理流程

运行:

```bash
bash test.sh
```

## 其他注意事项

```csv
stock_id,weight
600188,0.2
300502,0.2
600089,0.2
600482,0.2
600547,0.2
```
