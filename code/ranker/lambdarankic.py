import numpy as np
import xgboost as xgb

from .rankic import group_ptr_from_dmatrix, ordinal_ranks


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
