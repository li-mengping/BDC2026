"""校验比赛最终结果文件。"""

from pathlib import Path

import numpy as np
import pandas as pd


def validate_result_file(path: Path) -> dict:
    frame = pd.read_csv(path, dtype={'stock_id': str})
    if list(frame.columns) != ['stock_id', 'weight']:
        raise ValueError('result columns must be stock_id,weight')
    if not 1 <= len(frame) <= 5:
        raise ValueError('result must contain one to five stocks')
    stock_ids = frame['stock_id'].astype(str).str.zfill(6)
    if not stock_ids.str.fullmatch(r'\d{6}').all():
        raise ValueError('stock_id must contain six digits')
    if stock_ids.duplicated().any():
        raise ValueError('result contains duplicate stocks')
    weights = pd.to_numeric(frame['weight'], errors='coerce').to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError('weights must be finite and non-negative')
    weight_sum = float(weights.sum())
    if weight_sum > 1.0 + 1e-10:
        raise ValueError('weight sum exceeds one')
    return {'stock_count': int(len(frame)), 'weight_sum': weight_sum}
