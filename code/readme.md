## Coding Principle
No any fallbacks, compatability, legacy code, tries
No any useless helper functions, normalizers, type checks
No any useless args， we prefer micros
Keep minimal code style, delete all useless or unused variables, functions, classes etc. 
Always write comments(chinese)
Use uv run, keep Modular, Data-Centric, Declarative config, Readability

## 目标任务：预测一个未来交易周，收益最好的 1-5 只股票和持仓比例

核心任务： 排序
数据：日k线数据

输入：单股的历史特征
输出：未来交易周的收益排名
数据样本：(12个历史周,1个未来标签周)

验证：必须重点关注模型在验证集上的实际收益表现，每次训练迭代必须显示loss，验证集收益情况
（目前采用4周作为测试，8周作为验证，后续切换到rolling_cv滚动验证）

Defination:
    stock_week: a natual stock week with 5 stock trading day.
    stock_data: a sample that includes past history weeks as input and a future week for label
    window: the number of week that a stock_data history has

Current ranker:
    xgboost + lambdarankic/pairwise loss

We now use different loss to train different models.
We can gather the top10s from each model.
We do a postprocess to get the final top5 and weights.

Improvement direction:
feature: 
    use different factors/features for different windows 
    DL representation for those features
    cross-section feature between stocks
architecture:
    Mixture of features for one model?
    Mixture of models?
