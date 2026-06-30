# 代码说明

## 环境配置

- Python: 3.12
- 依赖管理: uv
- 主要依赖: pandas 2.3.2, numpy 2.3.3, xgboost 3.3.0, ta-lib 0.6.8, tqdm 4.67.1
- 运行设备: XGBoost `device=cuda`
- 复现方式: `uv sync --frozen` 后运行根目录脚本

## 数据

- 使用数据文件: `data/stock_data.csv`
- 数据内容: 沪深 300 股票日频行情，字段包含股票代码、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅。
- 未使用额外公开数据、外部词典、embedding 或联网下载数据。
- 训练、验证和预测切分由 `code/utils/runtime_split.py` 按完整自然交易周为单位生成。
- 单个训练和推理样本为，(12个历史周,1个未来标签周)

## 算法

### 特征 `code/features`
- baseline: `158+39`
- 尝试其他不同窗口长度的因子、横截面特征等，效果不是很好
- 后续可尝试深度学习方法提前特征

### 模型 `code/ranker`
主要模型：xgboost ranker
- LambdaRankIC 自定义目标损失的 XGBoost 模型。
- XGBoost `rank:pairwise` 排序模型。
rankic 选出的排名更为稳定
pairwise 选出的排名上限更高，测试平均值略高

### 后处理 `code/portfolio`
- 模型预测 Top10，作为候选池（目前是用的单 pairwise 模型）。
- 对候选池做共识排名归一化，把排名分位数映射为预期收益。
- 使用均值方差优化选出最终 Top5 和权重（权重采用均匀分配更稳定）。

### 注意事项

- 输入窗口: 12 个完整交易周。
- validation window: 8 个目标周。
- 训练轮数: 200，按照验证集结果（best top5 return）选择最佳迭代次数
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
