import numpy as np
import xgboost as xgb


def average_ranks(values: np.ndarray) -> np.ndarray:
    """带 tie 平均名次的升序 rank。"""
    values = np.asarray(values)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks


def ordinal_ranks(values: np.ndarray) -> np.ndarray:
    """论文定义的稳定降序位置 rank：rank 1 是最大值。"""
    order = np.argsort(-np.asarray(values), kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def group_ptr_from_dmatrix(dmatrix: xgb.DMatrix) -> np.ndarray:
    """读取 XGBoost group 边界。"""
    ptr = dmatrix.get_uint_info('group_ptr')
    if len(ptr) <= 1:
        raise ValueError('missing xgboost ranking group')
    return ptr.astype(np.int64)


def group_ptr_from_sizes(groups: list[int]) -> np.ndarray:
    """由 group size 构造边界。"""
    return np.concatenate([[0], np.cumsum(np.asarray(groups, dtype=np.int64))])


def rank_ic_score(pred: np.ndarray, labels: np.ndarray, ptr: np.ndarray) -> float:
    """按 group 计算平均 RankIC。"""
    values = []
    for begin, end in zip(ptr[:-1], ptr[1:]):
        pred_rank = average_ranks(pred[begin:end])
        label_rank = average_ranks(labels[begin:end])
        pred_centered = pred_rank - pred_rank.mean()
        label_centered = label_rank - label_rank.mean()
        denom = np.sqrt(np.sum(pred_centered ** 2) * np.sum(label_centered ** 2))
        if denom > 1e-12:
            values.append(float(np.sum(pred_centered * label_centered) / denom))
    if not values:
        raise ValueError('no valid RankIC group')
    return float(np.mean(values))


def xgb_rank_ic_metric(pred: np.ndarray, dmatrix: xgb.DMatrix) -> tuple[str, float]:
    """XGBoost custom metric 接口。"""
    return 'rank_ic', rank_ic_score(pred, dmatrix.get_label(), group_ptr_from_dmatrix(dmatrix))


def xgb_rank_return_metrics(pred: np.ndarray, dmatrix: xgb.DMatrix) -> list[tuple[str, float]]:
    """训练日志指标：RankIC、Top5 收益、Top10 收益。"""
    ptr = group_ptr_from_dmatrix(dmatrix)
    groups = np.diff(ptr).astype(int).tolist()
    labels = dmatrix.get_label()
    top5 = topk_return_metrics(pred, labels, groups, top_k=5)
    top10 = topk_return_metrics(pred, labels, groups, top_k=10)
    return [
        ('rank_ic', rank_ic_score(pred, labels, ptr)),
        ('top5_return', top5['pred_top5_return_avg']),
        ('top10_return', top10['pred_top10_return_avg']),
    ]


def topk_return_metrics(pred: np.ndarray, labels: np.ndarray, groups: list[int], top_k: int) -> dict[str, float]:
    """计算验证集每组 top-k 等权收益和平均收益。"""
    if top_k <= 0:
        raise ValueError('top_k must be positive')

    pred = np.asarray(pred, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    ptr = group_ptr_from_sizes(groups)

    pred_returns = []
    true_returns = []
    for begin, end in zip(ptr[:-1], ptr[1:]):
        group_pred = pred[begin:end]
        group_label = labels[begin:end]
        if len(group_label) < top_k:
            continue
        pred_returns.append(float(group_label[np.argsort(group_pred)[::-1][:top_k]].mean()))
        true_returns.append(float(group_label[np.argsort(group_label)[::-1][:top_k]].mean()))

    if not pred_returns:
        raise ValueError(f'no valid top{top_k} return group')

    prefix = f'top{top_k}'
    return {
        f'pred_{prefix}_group_returns': pred_returns,
        f'true_{prefix}_group_returns': true_returns,
        f'pred_{prefix}_return_avg': float(np.mean(pred_returns)),
        f'true_{prefix}_return_avg': float(np.mean(true_returns)),
    }


def lambdarankic_objective(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    """LambdaRankIC 自定义目标：在每个日期 group 内构造 RankIC 加权 pairwise 梯度。"""
    labels = dtrain.get_label()
    ptr = group_ptr_from_dmatrix(dtrain)
    grad = np.zeros_like(predt, dtype=np.float32)
    hess = np.zeros_like(predt, dtype=np.float32)

    for begin, end in zip(ptr[:-1], ptr[1:]):
        begin = int(begin)
        end = int(end)
        n = end - begin
        if n < 2:
            hess[begin:end] = 1.0
            continue

        # 论文中 rank 1 是最高收益/最高预测分，tie 由原始顺序稳定打破。
        y = labels[begin:end]
        score = predt[begin:end]
        true_rank = ordinal_ranks(y)
        pred_rank = ordinal_ranks(score)

        local_grad = np.zeros(n, dtype=np.float64)
        local_hess = np.zeros(n, dtype=np.float64)
        denom = n * (n * n - 1.0)

        for high in range(n):
            # 只保留真实收益 high 高于另一只股票的有序 pair。
            label_diff = y[high] - y
            valid = label_diff > 0
            if not np.any(valid):
                continue

            # scale 是论文中 RankIC 对 pair 交换影响的缩放项。
            score_diff = score[high] - score[valid]
            prob = 1.0 / (1.0 + np.exp(np.clip(-score_diff, -50.0, 50.0)))
            scale = 12.0 * np.abs(pred_rank[valid] - pred_rank[high]) * np.abs(true_rank[high] - true_rank[valid]) / denom
            pair_grad = (prob - 1.0) * scale
            pair_hess = np.maximum(2.0 * prob * (1.0 - prob) * scale, 1e-6)

            local_grad[high] += pair_grad.sum()
            local_grad[valid] -= pair_grad
            local_hess[high] += pair_hess.sum()
            local_hess[valid] += pair_hess

        grad[begin:end] = local_grad
        hess[begin:end] = np.maximum(local_hess, 1e-6)

    return grad, hess
